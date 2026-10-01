"""`quadwm eval`: shared-protocol evaluation of one trained checkpoint on GrandTour.

Runs what real logged data supports (docs/adr/0005):
  - E1.2 / protocol §1: linear + MLP probes fit on frozen latents of the
    probe-fit missions, quality (R², Pearson r) on held-out eval missions.
  - EV5 (EV1 metric on real data): ε_k at the protocol horizons in probe
    space, per component, with the probe floor and a persistence reference.
  - EV7: parameters, open-loop rollout throughput, single-step latency,
    approximate training GPU-hours.
  - The measured ANYmal D gait cycle (protocol §2 says not to copy k = 12).
EV1-sim, EV2, EV3, EV4 and EV6 run in `quadwm sim-eval` (sim_protocol.py).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ..data import build_sequence_dataset
from ..tokens import TokenCache, build_token_cache, cache_fits, load_token_caches, model_inputs
from ..utils import run_metadata
from .common import (
    autocast,
    collect_latents,
    evaluate_probes,
    load_model,
    model_record,
    open_checkpoint,
    precision_dtype,
    write_result,
)
from .metrics import gait_cycle_seconds, select_anchor_windows

NOT_RUN = {
    "EV1_sim_EV2_EV3_EV4_EV6": "simulated evals run separately: `quadwm sim-eval` (docs/adr/0006); "
                               "`quadwm report` pairs both JSONs for Δ_s2r(k)",
}


def _load_splits(run_root: Path, eval_config: dict) -> dict[str, list[str]]:
    path = run_root / "splits.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found: the protocol eval uses the exact mission split the run trained with"
        )
    splits = json.loads(path.read_text(encoding="utf-8"))
    if "probe" not in splits:
        raise ValueError(
            f"{path} has no probe split: this run predates docs/adr/0001 and trained on what are now "
            "probe-fit missions, so it cannot produce protocol probe results. Retrain."
        )
    if not splits["probe"]:
        if eval_config.get("probe_split_fallback") != "eval":
            raise ValueError(f"{path} has an empty probe split; set data.probe_fraction > 0 and retrain")
        # Smoke only: one mission plays every role so the code path runs cheaply.
        splits["probe"] = list(splits["eval"])
    if not splits.get("eval"):
        raise ValueError(f"{path} has no eval missions")
    return splits


def _window_dataset(data_cfg: dict, missions: list[str], eval_config: dict, *, observation: str,
                    load_images: bool, normalization: dict | None = None):
    data_root = Path(data_cfg.get("data_root", "data/grandtour"))
    context_frames = int(eval_config["context_frames"])
    try:
        return build_sequence_dataset(
            data_cfg,
            observation=observation,
            platform=data_cfg["platform"],
            data_root=data_root,
            context_steps=context_frames,
            rollout_steps=max(int(k) for k in eval_config["horizons"]),
            tick_hz=float(data_cfg["tick_hz"]),
            control_hz=float(data_cfg["control_hz"]),
            action_frames=int(data_cfg["action_frames"]),
            max_sequences=None,
            load_images=load_images,
            missions=missions,
            normalization=normalization,
            state_origin=context_frames - 1,  # base position relative to the rollout start
        )
    except KeyError as error:
        raise RuntimeError(
            f"observation {observation!r} topic {error} is missing under {data_root}. Window matching "
            "(eval windows.match_observations) needs every compared model's camera topic materialized; "
            "download it with `quadwm prepare` using that model's config."
        ) from error


def protocol_windows(data_cfg: dict, missions: list[str], eval_config: dict, *, load_images: bool,
                     normalization: dict | None = None):
    """Windows anchored on a shared time grid, matched across every compared observation."""
    own = data_cfg["observation"]
    context_frames = int(eval_config["context_frames"])
    windows_cfg = eval_config["windows"]
    dataset = _window_dataset(data_cfg, missions, eval_config, observation=own, load_images=load_images,
                              normalization=normalization)
    others = [
        _window_dataset(data_cfg, missions, eval_config, observation=observation, load_images=False)
        for observation in windows_cfg["match_observations"] if observation != own
    ]

    def anchors(source, reader_index: int) -> tuple[list[int], np.ndarray]:
        rows = [row for row, (index, _) in enumerate(source.index) if index == reader_index]
        times = source.readers[reader_index].depth_timestamps
        return rows, np.asarray([times[source.index[row][1][context_frames - 1]] for row in rows])

    selected = []
    for reader_index in range(len(dataset.readers)):
        rows, own_anchors = anchors(dataset, reader_index)
        chosen = select_anchor_windows(
            own_anchors,
            [anchors(other, reader_index)[1] for other in others],
            stride_s=float(windows_cfg["anchor_stride_s"]),
            tolerance_s=float(windows_cfg["anchor_tolerance_s"]),
            max_windows=int(windows_cfg["max_per_mission"]),
        )
        selected.extend(rows[i] for i in chosen)
    dataset.select(selected)
    if not len(dataset):
        raise ValueError(f"no protocol windows in missions {missions}")
    return dataset


def _token_caches(model, datasets: dict, cache_cfg: dict, data_root: Path, device: torch.device
                  ) -> dict[str, list[TokenCache | None] | None]:
    """Per-split frozen-token caches (built on first use), or None per split to encode images instead.

    Same cache, settings, and keying as training, so eval reuses training's tokens.
    """
    if not model.frozen_visual_encoder or cache_cfg.get("mode", "auto") == "off":
        return dict.fromkeys(datasets)
    cache_root = Path(cache_cfg.get("root", data_root / "quadwm-token-cache"))
    cache_root.mkdir(parents=True, exist_ok=True)
    if not cache_fits(list(datasets.values()), cache_root, model.tokens_per_frame, model.visual_dim):
        return dict.fromkeys(datasets)
    caches = {}
    for split, dataset in datasets.items():
        build_token_cache(dataset, model.visual_encoder, cache_root, device, int(cache_cfg.get("batch_size", 16)),
                          model.image_size)
        dataset.load_images = False
        caches[split] = load_token_caches(cache_root, dataset.readers, True)
    return caches


def _gait_cycle(readers, tick_hz: float, gait_cfg: dict) -> dict:
    per_mission = {}
    for reader in readers:
        speed = np.linalg.norm(np.asarray(reader.proprio["twist_lin"])[:, :2], axis=1)
        periods = []
        for foot in range(4):
            periods += gait_cycle_seconds(
                reader.proprio_timestamps, reader.proprio["contacts"][:, foot], speed,
                min_speed=float(gait_cfg["min_speed"]),
                min_segment_s=float(gait_cfg["min_segment_s"]),
                period_range_s=tuple(gait_cfg["period_range_s"]),
            )
        if periods:
            per_mission[reader.mission_dir.name] = float(np.median(periods))
    if not per_mission:
        return {"seconds": None, "ticks": None, "per_mission_s": {}}
    seconds = float(np.median(list(per_mission.values())))
    return {"seconds": seconds, "ticks": round(seconds * tick_hz), "tick_hz": tick_hz,
            "per_mission_s": per_mission,
            "note": "protocol §2: substitute this for k=12 via a versioned horizons change, not silently"}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(function, device: torch.device, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        function()
    _synchronize(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        _synchronize(device)
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


@torch.no_grad()
def _compute_parity(model, eval_config: dict, run_root: Path, device: torch.device, precision: torch.dtype) -> dict:
    """EV7 on random inputs of the true shapes (timing does not depend on values).

    Times the same ``model.rollout`` the eval scores, from a full ``context_steps`` context.
    """
    cfg = eval_config["compute"]
    batch, steps = int(cfg["batch_size"]), max(int(k) for k in eval_config["horizons"])
    warmup, repeats = int(cfg["warmup"]), int(cfg["repeats"])
    encoder = model.visual_encoder if model.frozen_visual_encoder else None
    training_only = sum(p.numel() for head in model.training_only_modules() for p in head.parameters())
    encoder_params = sum(p.numel() for p in encoder.parameters()) if encoder is not None else 0
    own = sum(p.numel() for name, p in model.named_parameters() if not name.startswith("visual_encoder."))
    context = model.context_steps
    frames = torch.randn(batch, context, *model.frame_shape, device=device)
    actions = torch.randn(batch, context + steps - 1, model.action_dim, device=device)
    size = model.image_size
    observation = {"images": torch.rand(1, 1, model.image_channels, size, size, device=device),
                   "proprio": torch.randn(1, 1, model.proprio_dim, device=device)}

    def rollout():
        with autocast(device, precision):
            model.rollout(frames, actions, steps)

    def single():  # one new camera frame (through the frozen encoder, if any) -> one predicted frame
        with autocast(device, precision):
            latest = model.encode_frames(*model_inputs(model, observation, device))
            model.rollout(torch.cat((frames[:1, 1:], latest), 1), actions[:1], 1)

    rollout_s = _timed(rollout, device, warmup, repeats)
    single_s = _timed(single, device, warmup, repeats)
    return {
        "parameters_M": {
            "world_model_trainable": own / 1e6,
            "frozen_encoder": encoder_params / 1e6,
            "inference_total": (own - training_only + encoder_params) / 1e6,
            "training_only": training_only / 1e6,
        },
        "rollout_throughput_fps": batch * steps / rollout_s,
        "rollout_batch_size": batch,
        "single_step_latency_ms": single_s * 1000,
        "deployment_latency_ms": None,  # observation -> action needs a policy (EV3)
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "precision": str(precision).removeprefix("torch."),
        "training": training_compute(run_root),
    }


def training_compute(run_root: Path, max_gap_s: float = 600.0) -> dict:
    """Approximate training GPU-hours from metrics.jsonl step timestamps x world size.

    Gaps longer than ``max_gap_s`` (requeue, walltime restarts) are dropped,
    as are evaluation and cache-building time before the first logged step.
    """
    metrics, metadata_path = run_root / "metrics.jsonl", run_root / "run_metadata.json"
    if not metrics.is_file():
        return {"gpu_hours_approx": None}
    stamps = []
    for line in metrics.read_text(encoding="utf-8").splitlines():
        record = json.loads(line) if line.strip() else {}
        if "timestamp" in record and not any(key.startswith("eval/") for key in record):
            stamps.append(float(record["timestamp"]))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    world_size = int(metadata.get("world_size", 1))
    gaps = np.diff(np.sort(stamps)) if len(stamps) > 1 else np.zeros(0)
    seconds = float(gaps[gaps <= max_gap_s].sum())
    return {"gpu_hours_approx": seconds * world_size / 3600, "world_size": world_size,
            "gpu_name": metadata.get("gpu_name"), "logged_steps": len(stamps)}


def evaluate_checkpoint(config: dict, eval_config: dict, checkpoint: Path | None = None,
                        output: Path | None = None) -> dict:
    seed = int(eval_config["seed"])
    run_root, checkpoint, saved, device = open_checkpoint(config, checkpoint, seed)
    trained = saved["config"]
    data_cfg = dict(config["data"])  # paths from this machine; transforms from the checkpoint
    data_cfg["observation"] = trained["data"]["observation"]
    splits = _load_splits(run_root, eval_config)
    horizons = sorted(int(k) for k in eval_config["horizons"])
    context_frames = int(eval_config["context_frames"])
    precision = precision_dtype(eval_config["precision"])
    print(f"stage=eval status=starting checkpoint={checkpoint} epoch={saved.get('epoch')} "
          f"device={device} horizons={horizons}", flush=True)

    model = load_model(saved, Path(config.get("checkpoint_root", "checkpoints"))).to(device).eval()
    datasets = {}
    for split in ("probe", "eval"):
        datasets[split] = protocol_windows(data_cfg, splits[split], eval_config, load_images=True,
                                           normalization=trained["data"].get("normalization"))
        print(f"stage=eval_windows split={split} missions={len(datasets[split].readers)} "
              f"windows={len(datasets[split])}", flush=True)
    caches = _token_caches(model, datasets, config.get("cache", {}), Path(data_cfg["data_root"]), device)
    collected = {
        split: collect_latents(model, dataset, context_frames=context_frames,
                               steps=max(horizons) if split == "eval" else 0,
                               batch_size=int(eval_config["batch_size"]), num_workers=int(eval_config["num_workers"]),
                               device=device, precision=precision, caches=caches[split])
        for split, dataset in datasets.items()
    }
    mission_names = [reader.mission_dir.name for reader in datasets["eval"].readers]
    probes = evaluate_probes(collected, eval_config, mission_names, device).results
    tick_hz = float(data_cfg["tick_hz"])
    result = {
        "protocol": "recipes/shared_evaluation_protocol.md",
        "eval_config": eval_config["name"],
        "domain": "real (GrandTour); eps_k here is protocol EV5",
        "model": model_record(saved, checkpoint, model, int(collected["eval"]["encoded"].shape[-1])),
        "data": {
            "observation": data_cfg["observation"], "tick_hz": tick_hz,
            "eval_missions": sorted(splits["eval"]), "probe_missions": sorted(splits["probe"]),
            "windows": {split: len(dataset) for split, dataset in datasets.items()},
            "context_frames": context_frames,
            "match_observations": eval_config["windows"]["match_observations"],
        },
        "horizons": horizons,
        "horizons_s": [k / tick_hz for k in horizons],
        "probes": probes,
        "gait_cycle": _gait_cycle(datasets["probe"].readers + datasets["eval"].readers, tick_hz,
                                  eval_config["gait"]),
        "not_run": NOT_RUN,
        "metadata": run_metadata(config) | {"eval_seed": seed, "device": str(device)},
    }
    if eval_config["compute"]["enabled"]:
        result["compute"] = _compute_parity(model, eval_config, run_root, device, precision)
    output = write_result(result, output, run_root, checkpoint, eval_config["name"])
    for kind, values in probes.items():
        print(f"stage=eval probe={kind} r2_all={values['quality']['all']['r2']:.3f} "
              + " ".join(f"eps_{k}={values['eps_k'][str(k)]['model']['all']:.3f}"
                         f"(persist={values['eps_k'][str(k)]['persistence']['all']:.3f})" for k in horizons),
              flush=True)
    print(f"stage=eval status=complete output={output}", flush=True)
    return result
