"""Distributed Stage-A JEPA-WM training on GrandTour."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import checkpoint_dir, data_dir, run_dir
from .data import (
    build_sequence_dataset,
    fetch_missions,
    materialized_missions,
    normalization_stats,
    split_mission_names,
    verify_joint_order_consistency,
)
from .data.grandtour import GrandTourSequenceDataset
from .models import (
    VJEPA21Encoder,
    build_model,
    ensure_vjepa21_checkpoint,
    load_checkpoint,
    model_class,
    save_checkpoint,
)
from .tokens import TokenCache, build_token_cache, cache_fits, load_token_caches, model_inputs
from .utils import init_wandb, run_metadata


def _distributed(min_free_fraction: float) -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("training requires an allocated CUDA/ROCm GPU")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    # Fail in seconds, not after cache building, when another process holds this GPU.
    free, total = torch.cuda.mem_get_info(device)
    print(f"stage=gpu rank={rank} device={local_rank} free_gib={free / 2**30:.1f} total_gib={total / 2**30:.1f}", flush=True)
    if free < min_free_fraction * total:
        raise RuntimeError(
            f"GPU {local_rank} (rank {rank}) already has {(total - free) / 2**30:.1f} GiB in use by another "
            "process; check ROCR_VISIBLE_DEVICES / other jobs on this node (rocm-smi --showpids)"
        )
    return rank, world_size, local_rank, device


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _seed(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _frozen_encoder(config: dict) -> bool:
    """True for the V-JEPA baseline (frozen encoder, checkpoint, token cache); False end to end."""
    return model_class(config["model"]).frozen_visual_encoder


def _write_json(path: Path, value, **kwargs) -> None:
    path.write_text(json.dumps(value, **kwargs) + "\n", encoding="utf-8")


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    caches: list[TokenCache | None] | None,
    device: torch.device,
    precision: torch.dtype,
    max_batches: int,
) -> dict[str, float]:
    """Held-out E1.1 pass: per-step error, persistence baseline, proprio variance."""
    module = getattr(model, "module", model)  # unwrap DDP
    was_training = module.training
    module.eval()
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        inputs, visual_tokens = model_inputs(module, batch, device, caches)
        with torch.autocast(device_type="cuda", dtype=precision):
            metrics = module.evaluate(inputs, visual_tokens)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1
        if max_batches and batches >= max_batches:
            break
    if was_training:
        module.train()
    return {key: value / max(batches, 1) for key, value in totals.items()}


def _check_dataset(
    dataset, data_config: dict, *, name: str, check_drop_rate: bool = True
) -> None:
    """Fail fast on empty, desynced, or mis-ordered data before a long run commits."""
    if not len(dataset):
        raise ValueError(f"{name} dataset is empty")
    if check_drop_rate:
        max_drop_rate = float(data_config.get("max_drop_rate", 0.3))
        rate = float(dataset.stats["drop_rate"])
        if rate > max_drop_rate:
            raise ValueError(
                f"{name} drop_rate {rate:.2%} exceeds max_drop_rate {max_drop_rate:.2%} "
                "-- check sync settings before trusting this data"
            )
    for reader in dataset.readers:
        if not verify_joint_order_consistency(reader.root):
            raise ValueError(
                "joint order mismatch between anymal_state_actuator and "
                f"anymal_state_state_estimator in {reader.mission_dir}"
            )


def prepare(config: dict) -> Path | None:
    """Materialize configured GrandTour data and (frozen-encoder models) the V-JEPA 2.1 checkpoint.

    Idempotent, so it is safe to run interactively via ``quadwm prepare`` and
    again from ``train`` on the same job.
    """
    print("stage=prepare status=starting", flush=True)
    checkpoint_root = checkpoint_dir(config)  # STORAGE_ROOT/checkpoints
    data_cfg = config.get("data", {})
    if data_cfg.get("download", True):
        data_root = data_dir(config)
        print(
            f"stage=data status=ensuring root={data_root} "
            f"missions={data_cfg.get('missions', 'all')}",
            flush=True,
        )
        fetch_missions(
            data_cfg.get("missions"),
            data_root,
            data_cfg.get("download_topics"),
        )
        print("stage=data status=ready", flush=True)
    checkpoint = None
    if _frozen_encoder(config):
        checkpoint = ensure_vjepa21_checkpoint(checkpoint_root)
        print(f"stage=checkpoint status=ready path={checkpoint}", flush=True)
    print("stage=prepare status=complete", flush=True)
    return checkpoint


def _resolve_splits(
    config: dict, run_root: Path, rank: int
) -> tuple[list[str] | None, list[str] | None]:
    """(train, eval) missions; rank 0 records the split in ``splits.json`` for the protocol eval.

    ``None`` train missions means every configured mission; ``None`` eval
    missions means no held-out evaluation.
    """
    data_cfg = config["data"]
    eval_fraction = float(data_cfg.get("eval_fraction", 0.0))
    train_missions = data_cfg.get("missions")
    eval_missions: list[str] | None = data_cfg.get("eval_missions")
    probe_missions: list[str] = data_cfg.get("probe_missions", [])
    if eval_missions is None and eval_fraction > 0:
        available = materialized_missions(data_dir(config), data_cfg.get("missions"))
        if len(available) >= 2:
            # Probe-fit missions are never trained on (shared protocol §1.3 rule 4).
            train_missions, eval_missions, probe_missions = split_mission_names(
                available,
                int(data_cfg.get("split_seed", config.get("seed", 4551))),
                eval_fraction,
                float(data_cfg.get("probe_fraction", 0.0)),
            )
        elif rank == 0:
            print(
                f"stage=split status=skipped available={len(available)} "
                "reason=need at least 2 missions to hold one out",
                flush=True,
            )
    if rank == 0 and eval_missions is not None:
        print(
            f"stage=split train_missions="
            f"{len(train_missions) if train_missions is not None else 'all'} "
            f"eval_missions={len(eval_missions)} probe_missions={len(probe_missions)} "
            f"eval={','.join(eval_missions)}",
            flush=True,
        )
        splits = {"train": train_missions, "eval": eval_missions, "probe": probe_missions}
        _write_json(run_root / "splits.json", splits, indent=2)
    return train_missions, eval_missions


def _sequence_dataset(
    config: dict, missions: list[str] | None, max_sequences: int | None
) -> GrandTourSequenceDataset:
    data_cfg, model_cfg = config["data"], config["model"]
    return build_sequence_dataset(
        data_cfg,
        observation=data_cfg["observation"],
        platform=data_cfg["platform"],
        data_root=data_dir(config),
        context_steps=int(model_cfg["context_steps"]),
        rollout_steps=int(model_cfg["rollout_steps"]),
        tick_hz=float(data_cfg["tick_hz"]),
        control_hz=float(data_cfg["control_hz"]),
        action_frames=int(data_cfg["action_frames"]),
        max_sequences=max_sequences,
        load_images=True,
        missions=missions,
    )


def _build_datasets(
    config: dict, run_root: Path, rank: int
) -> tuple[GrandTourSequenceDataset, GrandTourSequenceDataset | None]:
    """Checked train and held-out datasets, normalized with training-split statistics."""
    data_cfg = config["data"]
    train_missions, eval_missions = _resolve_splits(config, run_root, rank)
    dataset = _sequence_dataset(config, train_missions, data_cfg.get("max_sequences"))
    _check_dataset(dataset, data_cfg, name="sequence")
    if data_cfg.get("normalize", True):
        # Training-split statistics only; eval data reuses them. Stored in the
        # config so every checkpoint carries the transform it was trained with.
        data_cfg["normalization"] = normalization_stats(dataset.readers, int(data_cfg["action_frames"]))
        dataset.normalization = data_cfg["normalization"]
        if rank == 0:
            _write_json(run_root / "normalization.json", data_cfg["normalization"])
    if rank == 0:
        print(
            f"stage=dataset status=ready missions={len(dataset.readers)} "
            f"sequences={len(dataset)}",
            flush=True,
        )
    if not eval_missions:
        return dataset, None
    eval_dataset = _sequence_dataset(config, eval_missions, data_cfg.get("eval_max_sequences"))
    _check_dataset(eval_dataset, data_cfg, name="held-out eval", check_drop_rate=False)
    eval_dataset.normalization = data_cfg.get("normalization")
    if rank == 0:
        print(
            f"stage=eval_dataset status=ready missions={len(eval_dataset.readers)} "
            f"sequences={len(eval_dataset)}",
            flush=True,
        )
    return dataset, eval_dataset


def _token_cache(
    config: dict, datasets: list[GrandTourSequenceDataset], device: torch.device, rank: int, world_size: int
) -> Path | None:
    """Rank 0 builds the frozen-encoder token cache when enabled and it fits on disk.

    Every rank agrees on the outcome and stops loading images when it is
    used. Returns the cache root, or None when training encodes on the fly
    (or the model has no frozen encoder).
    """
    cache_cfg, model_cfg = config.get("cache", {}), config["model"]
    cache_root = Path(cache_cfg.get("root", data_dir(config) / "quadwm-token-cache"))
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    use_cache = cache_cfg.get("mode", "auto") != "off" and _frozen_encoder(config)
    if use_cache and rank == 0:
        use_cache = cache_fits(
            datasets, cache_root, tokens=int(model_cfg["tokens_per_frame"]), dim=int(model_cfg["visual_dim"])
        )
        if use_cache:
            encoder = VJEPA21Encoder(checkpoint_dir(config))
            for dataset in datasets:
                build_token_cache(
                    dataset,
                    encoder,
                    cache_root,
                    device,
                    int(cache_cfg.get("batch_size", 16)),
                    int(model_cfg.get("image_size", 384)),
                )
            del encoder
            torch.cuda.empty_cache()
    if world_size > 1:
        flag = torch.tensor([int(use_cache)], device=device)
        dist.broadcast(flag, src=0)
        use_cache = bool(flag.item())
    _barrier()
    if rank == 0:
        print(f"stage=feature_cache status={'enabled' if use_cache else 'disabled'}", flush=True)
    for dataset in datasets:
        dataset.load_images = not use_cache
    return cache_root if use_cache else None


def _resume_path(config: dict) -> Path | None:
    """``training.resume``: ``last`` is this run's own ``last.pt``; any other value is a path; empty is none."""
    value = config["training"].get("resume")
    if not value:
        return None
    return run_dir(config) / "last.pt" if value == "last" else Path(value)


def _resume(resume_path: Path | None, model, optimizer, rank: int) -> int:
    """Epoch to start from: the checkpoint's when ``resume_path`` exists, else 0."""
    if resume_path is not None and resume_path.is_file():
        start_epoch = load_checkpoint(resume_path, model=model, optimizer=optimizer)
        if rank == 0:
            print(f"stage=resume status=loaded path={resume_path} epoch={start_epoch}", flush=True)
        return start_epoch
    if rank == 0 and resume_path is not None:
        print(f"stage=resume status=fresh no_checkpoint={resume_path}", flush=True)
    return 0


def _start_wandb(config: dict, run_root: Path, metadata: dict):
    """Rank 0 only: start W&B (when enabled) and write ``run_metadata.json``."""
    wandb_run = init_wandb(config, metadata=metadata, run_dir=run_root)
    if wandb_run is None:
        print("stage=wandb status=disabled", flush=True)
    else:
        print(f"stage=wandb status=started url={getattr(wandb_run, 'url', None)}", flush=True)
        metadata["wandb_run_id"] = getattr(wandb_run, "id", "unknown")
        metadata["wandb_url"] = getattr(wandb_run, "url", None)
    _write_json(run_root / "run_metadata.json", metadata, indent=2)
    return wandb_run


class _RunLog:
    """Rank-0 sink for ``metrics.jsonl`` and W&B; a no-op on other ranks."""

    def __init__(self, run_root: Path, wandb_run, enabled: bool):
        self.stream = (run_root / "metrics.jsonl").open("a", encoding="utf-8") if enabled else None
        self.wandb_run = wandb_run

    def record(self, values: dict, step: int, *, to_wandb: bool = True) -> None:
        if self.stream is not None:
            self.stream.write(json.dumps(values | {"step": step, "timestamp": time.time()}) + "\n")
            self.stream.flush()
        if to_wandb and self.wandb_run is not None:
            # Explicit global step so the W&B x-axis matches training
            # progress instead of W&B's internal log counter.
            self.wandb_run.log(values, step=step)

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()


def train(config: dict) -> None:
    train_cfg = config["training"]
    rank, world_size, local_rank, device = _distributed(float(train_cfg.get("min_free_gpu_fraction", 0.5)))
    _seed(int(config.get("seed", 4551)), rank)
    if rank == 0:
        print(
            f"stage=train status=starting device={torch.cuda.get_device_name(device)} "
            f"world_size={world_size}",
            flush=True,
        )
    run_root = run_dir(config)
    if rank == 0:
        run_root.mkdir(parents=True, exist_ok=True)
        _write_json(run_root / "resolved_config.json", config, indent=2, default=str)
        prepare(config)
    _barrier()

    dataset, eval_dataset = _build_datasets(config, run_root, rank)
    cache_root = _token_cache(config, [dataset] + ([eval_dataset] if eval_dataset else []), device, rank, world_size)
    caches = load_token_caches(cache_root, dataset.readers) if cache_root else None
    batch_size = int(train_cfg.get("per_gpu_batch_size", 1))
    eval_loader = eval_caches = None
    if eval_dataset is not None:
        eval_caches = load_token_caches(cache_root, eval_dataset.readers) if cache_root else None
        eval_loader = eval_dataset.loader(batch_size=batch_size, shuffle=False, num_workers=0)

    checkpoint_root = checkpoint_dir(config)
    encode_on_the_fly = cache_root is None and _frozen_encoder(config)
    visual_encoder = VJEPA21Encoder(checkpoint_root) if encode_on_the_fly else None
    model = build_model(config["model"], checkpoint_root, visual_encoder=visual_encoder).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    if rank == 0:
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"stage=model status=ready trainable_parameters={parameters}", flush=True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(train_cfg.get("learning_rate", 5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    start_epoch = _resume(_resume_path(config), model, optimizer, rank)
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = dataset.loader(
        batch_size=batch_size,
        num_workers=int(train_cfg.get("num_workers", 0)),
        sampler=sampler,
    )
    wandb_run = None
    if rank == 0:
        metadata = run_metadata(config) | {
            "rank": rank,
            "world_size": world_size,
            "gpu_name": torch.cuda.get_device_name(device),
        }
        wandb_run = _start_wandb(config, run_root, metadata)
    log = _RunLog(run_root, wandb_run, enabled=rank == 0)

    epochs = int(train_cfg.get("epochs", 10))
    max_steps = int(train_cfg.get("max_steps", 0))
    log_every_steps = max(1, int(train_cfg.get("log_every_steps", 10)))
    eval_every_steps = max(0, int(train_cfg.get("eval_every_steps", 0)))
    eval_max_batches = max(0, int(train_cfg.get("eval_max_batches", 8)))
    accumulation = int(train_cfg.get("gradient_accumulation_steps", 1))
    clip_grad_norm = float(train_cfg.get("clip_grad_norm", 1.0))
    precision = torch.bfloat16 if train_cfg.get("precision", "bf16") == "bf16" else torch.float16
    global_step = 0
    model.train()
    for epoch in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            if max_steps and global_step >= max_steps:
                break
            with torch.autocast(device_type="cuda", dtype=precision):
                inputs, visual_tokens = model_inputs(model, batch, device, caches)
                losses = model(inputs, visual_tokens)
                loss = losses["loss"] / accumulation
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at step {global_step + 1}: {float(loss)}"
                )
            loss.backward()
            if (step + 1) % accumulation != 0:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            metric = {key: value.detach().float().item() for key, value in losses.items()} | {"epoch": epoch}
            log_now = global_step == 1 or global_step % log_every_steps == 0
            log.record(metric, global_step, to_wandb=log_now)
            if rank == 0 and log_now:
                print(
                    f"stage=train epoch={epoch + 1}/{epochs} step={global_step} "
                    + " ".join(f"{key}={value:.6f}" for key, value in metric.items() if key.endswith("loss")),
                    flush=True,
                )
            if eval_loader is None or not eval_every_steps:
                continue
            if global_step == 1 or global_step % eval_every_steps == 0:
                if rank == 0:
                    eval_metrics = _evaluate(model, eval_loader, eval_caches, device, precision, eval_max_batches)
                    log.record({f"eval/{key}": value for key, value in eval_metrics.items()}, global_step)
                    print(
                        f"stage=eval step={global_step} "
                        + " ".join(f"{key}={value:.6f}" for key, value in eval_metrics.items() if "/" not in key),
                        flush=True,
                    )
                _barrier()
        if rank == 0:
            save_checkpoint(run_root / "last.pt", model=model, optimizer=optimizer, epoch=epoch + 1, config=config)
            print(f"stage=checkpoint status=saved path={run_root / 'last.pt'}", flush=True)
        if max_steps and global_step >= max_steps:
            break
    log.close()
    if rank == 0:
        print(f"stage=train status=complete steps={global_step}", flush=True)
    _barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
