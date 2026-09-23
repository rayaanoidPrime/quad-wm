"""Distributed Stage-A JEPA-WM training on GrandTour."""

from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .data import (
    build_sequence_dataset,
    fetch_missions,
    materialized_missions,
    split_mission_names,
)
from .models import (
    VJEPA21Encoder,
    build_model,
    ensure_vjepa21_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from .utils import init_wandb


def _distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")
    if not torch.cuda.is_available():
        raise RuntimeError("training requires an allocated CUDA/ROCm GPU")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _seed(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _prepare_images(images: torch.Tensor, device: torch.device, size: int) -> torch.Tensor:
    batch, frames, channels, height, width = images.shape
    images = images.to(device, non_blocking=True).div(255.0)
    images = images.reshape(batch * frames, channels, height, width)
    images = torch.nn.functional.interpolate(
        images, size=(size, size), mode="bilinear", align_corners=False
    )
    mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((images - mean) / std).reshape(batch, frames, channels, size, size)


class TokenCache:
    def __init__(self, root: Path, mission: str):
        self.root = root
        self.mission = mission
        metadata = json.loads((root / f"{mission}.json").read_text(encoding="utf-8"))
        self.ids = np.asarray(metadata["image_ids"], dtype=np.int64)
        self.lookup = {int(image_id): index for index, image_id in enumerate(self.ids)}
        self.tokens = np.memmap(
            root / f"{mission}.mmap",
            mode="r",
            dtype=np.float16,
            shape=tuple(metadata["shape"]),
        )

    def get(self, image_ids: np.ndarray) -> np.ndarray:
        positions = [self.lookup[int(image_id)] for image_id in image_ids.reshape(-1)]
        return np.asarray(self.tokens[positions]).reshape(
            *image_ids.shape, self.tokens.shape[1], self.tokens.shape[2]
        )


def _cache_valid(root: Path, mission: str, image_ids: list[int], shape: tuple[int, ...]) -> bool:
    metadata_path = root / f"{mission}.json"
    mmap_path = root / f"{mission}.mmap"
    if not metadata_path.is_file() or not mmap_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return metadata["image_ids"] == image_ids and tuple(metadata["shape"]) == shape


@torch.no_grad()
def _build_token_cache(
    dataset,
    encoder,
    cache_root: Path,
    device: torch.device,
    batch_size: int,
    image_size: int,
) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    encoder.eval().to(device)
    for reader_index, reader in enumerate(dataset.readers):
        mission = reader.mission_dir.name
        image_ids = sorted({image_id for index in dataset.index if index[0] == reader_index for image_id in index[1]})
        if not image_ids:
            # A short or fully-invalid mission contributes no sequences, so there
            # is nothing to cache (its reader is simply not used by the loader).
            continue
        first = _prepare_images(
            torch.from_numpy(
                np.stack([reader.load_image(i) for i in image_ids[:1]])
            ).permute(0, 3, 1, 2).unsqueeze(1),
            device,
            image_size,
        )[:, 0]
        first_tokens = encoder(first).float().cpu().numpy().astype(np.float16)
        shape = (len(image_ids), *first_tokens.shape[1:])
        if _cache_valid(cache_root, mission, image_ids, shape):
            continue
        print(
            f"stage=feature_cache mission={mission} images={len(image_ids)} "
            f"shape={tuple(shape)}",
            flush=True,
        )
        temporary = cache_root / f"{mission}.mmap.part"
        metadata_temporary = cache_root / f"{mission}.json.part"
        required_bytes = int(np.prod(shape)) * np.dtype(np.float16).itemsize
        free = shutil.disk_usage(cache_root).free
        if required_bytes > free * 0.9:
            raise RuntimeError(
                f"not enough disk for token cache of {mission}: "
                f"need {required_bytes / 1024**3:.1f} GiB, free {free / 1024**3:.1f} GiB. "
                "Set cache.mode=off or point QUADWM_CACHE_ROOT at a larger filesystem."
            )
        try:
            tokens = np.memmap(temporary, mode="w+", dtype=np.float16, shape=shape)
            for start in range(0, len(image_ids), batch_size):
                ids = image_ids[start : start + batch_size]
                images = np.stack([reader.load_image(image_id) for image_id in ids])
                images = _prepare_images(
                    torch.from_numpy(images).permute(0, 3, 1, 2).unsqueeze(1),
                    device,
                    image_size,
                )[:, 0]
                tokens[start : start + len(ids)] = encoder(images).float().cpu().numpy().astype(np.float16)
            tokens.flush()
            metadata_temporary.write_text(
                json.dumps({"image_ids": image_ids, "shape": list(shape)}) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, cache_root / f"{mission}.mmap")
            os.replace(metadata_temporary, cache_root / f"{mission}.json")
        finally:
            # Never leave a multi-GiB .part behind if the job dies mid-write.
            temporary.unlink(missing_ok=True)
            metadata_temporary.unlink(missing_ok=True)
        print(f"stage=feature_cache mission={mission} status=done", flush=True)


def _cache_fits(datasets, cache_root: Path, tokens: int, dim: int) -> bool:
    ids = sum(
        len({image_id for index in dataset.index if index[0] == reader_index for image_id in index[1]})
        for dataset in datasets
        for reader_index in range(len(dataset.readers))
    )
    required = ids * tokens * dim * 2
    free = shutil.disk_usage(cache_root.parent).free
    print(f"feature cache estimate: {required / 1024**3:.1f} GiB; disk free: {free / 1024**3:.1f} GiB")
    return required < free * 0.8


def _batch_tokens(batch: dict, caches: list[TokenCache | None], device: torch.device) -> torch.Tensor:
    mission_ids = batch["mission_idx"].tolist()
    image_ids = batch["image_ids"].numpy()
    rows = []
    for row, mission_id in enumerate(mission_ids):
        cache = caches[mission_id]
        if cache is None:
            raise RuntimeError(f"no token cache for mission index {mission_id}")
        rows.append(cache.get(image_ids[row]))
    return torch.from_numpy(np.stack(rows)).to(device, non_blocking=True)


def _load_token_caches(cache_root: Path, readers, use_cache: bool) -> list[TokenCache | None]:
    """Per-mission caches, aligned with reader order; None where nothing was cached."""
    if not use_cache:
        return []
    return [
        TokenCache(cache_root, reader.mission_dir.name)
        if (cache_root / f"{reader.mission_dir.name}.json").is_file()
        else None
        for reader in readers
    ]


def _batch_visual_tokens(
    model: torch.nn.Module,
    batch: dict,
    caches: list[TokenCache | None],
    device: torch.device,
    image_size: int,
    use_cache: bool,
) -> torch.Tensor:
    if use_cache:
        return _batch_tokens(batch, caches, device)
    module = model.module if hasattr(model, "module") else model
    images = _prepare_images(batch["images"], device, image_size)
    batch_size, frames = images.shape[:2]
    tokens = module.encode_visual(images.reshape(batch_size * frames, *images.shape[2:]))
    return tokens.reshape(batch_size, frames, *tokens.shape[1:])


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    caches: list[TokenCache | None],
    device: torch.device,
    precision: torch.dtype,
    image_size: int,
    use_cache: bool,
    max_batches: int,
) -> dict[str, float]:
    """Held-out E1.1 pass: per-step error, persistence baseline, proprio variance."""
    module = model.module if hasattr(model, "module") else model
    was_training = module.training
    module.eval()
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        visual_tokens = _batch_visual_tokens(model, batch, caches, device, image_size, use_cache)
        batch["proprio"] = batch["proprio"].to(device, non_blocking=True)
        batch["actions"] = batch["actions"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=precision):
            metrics = module.evaluate(visual_tokens, batch["proprio"], batch["actions"])
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        batches += 1
        if max_batches and batches >= max_batches:
            break
    if was_training:
        module.train()
    return {key: value / max(batches, 1) for key, value in totals.items()}


def train(config: dict) -> None:
    rank, world_size, local_rank, device = _distributed()
    _seed(int(config.get("seed", 4551)), rank)
    if rank == 0:
        print(
            f"stage=train status=starting device={torch.cuda.get_device_name(device)} "
            f"world_size={world_size}",
            flush=True,
        )
    run_root = Path(config.get("run_root", "runs")) / config.get("name", "jepa-baseline")
    checkpoint_root = Path(config.get("checkpoint_root", "checkpoints"))
    data_cfg = config["data"]
    data_root = Path(data_cfg.get("data_root", config.get("data_root", "data/grandtour")))
    model_cfg = {**config["model"], "baseline": True}
    if rank == 0:
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "resolved_config.json").write_text(
            json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
        )
        if data_cfg.get("download", True):
            print("stage=data status=checking", flush=True)
            fetch_missions(
                data_cfg.get("missions"),
                data_root,
                data_cfg.get("download_topics"),
            )
            print("stage=data status=ready", flush=True)
        ensure_vjepa21_checkpoint(checkpoint_root)
    _barrier()

    eval_fraction = float(data_cfg.get("eval_fraction", 0.0))
    train_missions = data_cfg.get("missions")
    eval_missions: list[str] | None = None
    if eval_fraction > 0:
        available = materialized_missions(data_root, data_cfg.get("missions"))
        if len(available) >= 2:
            train_missions, eval_missions = split_mission_names(
                available,
                int(data_cfg.get("split_seed", config.get("seed", 4551))),
                eval_fraction,
            )
        elif rank == 0:
            print(
                f"stage=split status=skipped available={len(available)} "
                "reason=need at least 2 missions to hold one out",
                flush=True,
            )
    if rank == 0 and eval_missions is not None:
        print(
            f"stage=split train_missions={len(train_missions)} "
            f"eval_missions={len(eval_missions)} eval={','.join(eval_missions)}",
            flush=True,
        )

    dataset = build_sequence_dataset(
        data_cfg,
        observation=data_cfg["observation"],
        platform=data_cfg["platform"],
        data_root=data_root,
        context_steps=int(model_cfg["context_steps"]),
        rollout_steps=int(model_cfg["rollout_steps"]),
        tick_hz=float(data_cfg["tick_hz"]),
        control_hz=float(data_cfg["control_hz"]),
        action_frames=int(data_cfg["action_frames"]),
        max_sequences=data_cfg.get("max_sequences"),
        load_images=True,
        missions=train_missions,
    )
    if not len(dataset):
        raise ValueError("sequence dataset is empty")
    if rank == 0:
        print(
            f"stage=dataset status=ready missions={len(dataset.readers)} "
            f"sequences={len(dataset)}",
            flush=True,
        )

    eval_dataset = None
    if eval_missions:
        eval_dataset = build_sequence_dataset(
            data_cfg,
            observation=data_cfg["observation"],
            platform=data_cfg["platform"],
            data_root=data_root,
            context_steps=int(model_cfg["context_steps"]),
            rollout_steps=int(model_cfg["rollout_steps"]),
            tick_hz=float(data_cfg["tick_hz"]),
            control_hz=float(data_cfg["control_hz"]),
            action_frames=int(data_cfg["action_frames"]),
            max_sequences=data_cfg.get("eval_max_sequences"),
            load_images=True,
            missions=eval_missions,
        )
        if not len(eval_dataset):
            raise ValueError("held-out eval dataset is empty")
        if rank == 0:
            print(
                f"stage=eval_dataset status=ready missions={len(eval_dataset.readers)} "
                f"sequences={len(eval_dataset)}",
                flush=True,
            )

    cache_cfg = config.get("cache", {})
    use_cache = cache_cfg.get("mode", "auto") != "off"
    cache_root = Path(cache_cfg.get("root", data_root / "quadwm-token-cache"))
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    image_size = int(model_cfg.get("image_size", 384))
    if use_cache and rank == 0:
        use_cache = _cache_fits(
            [dataset] + ([eval_dataset] if eval_dataset is not None else []),
            cache_root,
            tokens=int(model_cfg["tokens_per_frame"]),
            dim=int(model_cfg["visual_dim"]),
        )
        if use_cache:
            encoder = VJEPA21Encoder(checkpoint_root)
            _build_token_cache(
                dataset,
                encoder,
                cache_root,
                device,
                int(cache_cfg.get("batch_size", 16)),
                image_size,
            )
            if eval_dataset is not None:
                _build_token_cache(
                    eval_dataset,
                    encoder,
                    cache_root,
                    device,
                    int(cache_cfg.get("batch_size", 16)),
                    image_size,
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
    dataset.load_images = not use_cache
    caches = _load_token_caches(cache_root, dataset.readers, use_cache)
    eval_loader = None
    eval_caches: list[TokenCache | None] = []
    if eval_dataset is not None:
        eval_dataset.load_images = not use_cache
        eval_caches = _load_token_caches(cache_root, eval_dataset.readers, use_cache)
        eval_loader = eval_dataset.loader(
            batch_size=int(config["training"].get("per_gpu_batch_size", 1)),
            shuffle=False,
            num_workers=0,
        )
    visual_encoder = None if use_cache else VJEPA21Encoder(checkpoint_root)
    model = build_model(model_cfg, checkpoint_root, visual_encoder=visual_encoder).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=False)
    if rank == 0:
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"stage=model status=ready trainable_parameters={parameters}", flush=True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"].get("learning_rate", 5e-4)),
        weight_decay=float(config["training"].get("weight_decay", 1e-4)),
    )
    resume = config["training"].get("resume")
    resume_path = Path(resume) if resume else None
    if resume_path is not None and resume_path.is_file():
        start_epoch = load_checkpoint(resume_path, model=model, optimizer=optimizer)
        if rank == 0:
            print(f"stage=resume status=loaded path={resume_path} epoch={start_epoch}", flush=True)
    else:
        start_epoch = 0
        if rank == 0 and resume_path is not None:
            print(f"stage=resume status=fresh no_checkpoint={resume_path}", flush=True)
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = dataset.loader(
        batch_size=int(config["training"].get("per_gpu_batch_size", 1)),
        num_workers=int(config["training"].get("num_workers", 0)),
        sampler=sampler,
    )
    metadata = {
        "host": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "git_commit": os.environ.get("QUADWM_GIT_COMMIT", "unknown"),
        "rank": rank,
        "world_size": world_size,
        "visible_devices": os.environ.get("ROCR_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "all")),
        "gpu_name": torch.cuda.get_device_name(device),
    }
    wandb_run = init_wandb(config, metadata=metadata, run_dir=run_root) if rank == 0 else None
    if rank == 0:
        if wandb_run is None:
            print("stage=wandb status=disabled", flush=True)
        else:
            print(
                f"stage=wandb status=started url={getattr(wandb_run, 'url', None)}",
                flush=True,
            )
    if rank == 0:
        if wandb_run is not None:
            metadata["wandb_run_id"] = getattr(wandb_run, "id", "unknown")
            metadata["wandb_url"] = getattr(wandb_run, "url", None)
        (run_root / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    epochs = int(config["training"].get("epochs", 10))
    max_steps = int(config["training"].get("max_steps", 0))
    log_every_steps = max(1, int(config["training"].get("log_every_steps", 10)))
    eval_every_steps = max(0, int(config["training"].get("eval_every_steps", 0)))
    eval_max_batches = max(0, int(config["training"].get("eval_max_batches", 8)))
    accumulation = int(config["training"].get("gradient_accumulation_steps", 1))
    precision = torch.bfloat16 if config["training"].get("precision", "bf16") == "bf16" else torch.float16
    global_step = 0
    metrics_stream = (run_root / "metrics.jsonl").open("a", encoding="utf-8") if rank == 0 else None
    model.train()
    for epoch in range(start_epoch, epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            if max_steps and global_step >= max_steps:
                break
            if use_cache:
                visual_tokens = _batch_tokens(batch, caches, device)
            else:
                batch["images"] = _prepare_images(batch["images"], device, image_size)
                visual_tokens = None
            batch["proprio"] = batch["proprio"].to(device, non_blocking=True)
            batch["actions"] = batch["actions"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=precision):
                if visual_tokens is not None:
                    batch["visual_tokens"] = visual_tokens
                losses = model(batch)
                loss = losses["loss"] / accumulation
            loss.backward()
            if (step + 1) % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"].get("clip_grad_norm", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                metric = {
                    key: value.detach().float().item() for key, value in losses.items()
                } | {"epoch": epoch, "step": global_step, "timestamp": time.time()}
                if rank == 0 and metrics_stream is not None:
                    metrics_stream.write(json.dumps(metric) + "\n")
                    metrics_stream.flush()
                if rank == 0 and (
                    global_step == 1 or global_step % log_every_steps == 0
                ):
                    print(
                        f"stage=train epoch={epoch + 1}/{epochs} step={global_step} "
                        f"loss={metric['loss']:.6f} visual_loss={metric['visual_loss']:.6f} "
                        f"proprio_loss={metric['proprio_loss']:.6f}",
                        flush=True,
                    )
                    if wandb_run is not None:
                        # Explicit global step so the W&B x-axis matches training
                        # progress instead of W&B's internal log counter.
                        wandb_run.log(
                            {k: v for k, v in metric.items() if k not in ("step", "timestamp")},
                            step=global_step,
                        )
                do_eval = (
                    eval_loader is not None
                    and eval_every_steps
                    and (global_step == 1 or global_step % eval_every_steps == 0)
                )
                if do_eval and rank == 0:
                    eval_metrics = _evaluate(
                        model,
                        eval_loader,
                        eval_caches,
                        device,
                        precision,
                        image_size,
                        use_cache,
                        eval_max_batches,
                    )
                    eval_record = {f"eval/{key}": value for key, value in eval_metrics.items()}
                    if metrics_stream is not None:
                        metrics_stream.write(
                            json.dumps(eval_record | {"step": global_step, "timestamp": time.time()})
                            + "\n"
                        )
                        metrics_stream.flush()
                    if wandb_run is not None:
                        wandb_run.log(eval_record, step=global_step)
                    print(
                        f"stage=eval step={global_step} "
                        f"visual_mse={eval_metrics['visual_mse']:.6f} "
                        f"proprio_mse={eval_metrics['proprio_mse']:.6f} "
                        f"persistence_visual_mse={eval_metrics['step_1/persistence_visual_mse']:.6f} "
                        f"proprio_variance={eval_metrics['proprio_variance']:.6f}",
                        flush=True,
                    )
                if do_eval:
                    _barrier()
        if rank == 0:
            save_checkpoint(run_root / "last.pt", model=model, optimizer=optimizer, epoch=epoch + 1, config=config)
            print(f"stage=checkpoint status=saved path={run_root / 'last.pt'}", flush=True)
        if max_steps and global_step >= max_steps:
            break
    if metrics_stream is not None:
        metrics_stream.close()
    if rank == 0 and wandb_run is not None:
        wandb_run.finish()
    if rank == 0:
        print(f"stage=train status=complete steps={global_step}", flush=True)
    _barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
