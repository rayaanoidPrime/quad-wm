"""Shared by `quadwm eval` (protocol.py) and `quadwm sim-eval` (sim_protocol.py).

Checkpoint and model loading, latent collection over protocol windows,
probe fitting and scoring (protocol §1 and the ε_k curve), and the result
file. Both evals call these unchanged, so EV5 and EV1-sim share one probe
procedure (needed for Δ_s2r).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..models import VJEPA21Encoder, build_model
from ..tokens import TokenCache, model_inputs
from .metrics import evaluation_sigma, r2_and_pearson, rollout_errors
from .probes import StateProbe, fit_probe
from .rollout import protocol_latents

PRECISIONS = {"bf16": torch.bfloat16, "fp16": torch.float16}


def precision_dtype(name: str) -> torch.dtype:
    if name not in PRECISIONS:
        raise ValueError(f"unknown precision {name!r}; expected one of {sorted(PRECISIONS)}")
    return PRECISIONS[name]


def autocast(device: torch.device, precision: torch.dtype):
    return torch.autocast(device_type=device.type, dtype=precision, enabled=device.type == "cuda")


def open_checkpoint(config: dict, checkpoint: Path | None, seed: int) -> tuple[Path, Path, dict, torch.device]:
    """(run_root, checkpoint path, saved checkpoint, device), seeded."""
    # Resolved exactly like training.train, so eval finds the same run directory.
    run_root = Path(config.get("run_root", "runs")) / config.get("name", "jepa-baseline")
    checkpoint = Path(checkpoint) if checkpoint else run_root / "last.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint {checkpoint} not found")
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cpu":
        print("stage=eval warning=no GPU; running on CPU (slow)", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    normalization = saved["config"]["data"].get("normalization")
    if saved["config"]["data"].get("normalize", True) and normalization is None:
        raise ValueError("checkpoint predates docs/adr/0002 input normalization; retrain")
    return run_root, checkpoint, saved, device


def _strip_encoder(state_dict: dict) -> dict:
    """Runs trained without the token cache saved the frozen V-JEPA weights too; they are reloaded."""
    return {key: value for key, value in state_dict.items() if not key.startswith("visual_encoder.")}


def load_model(saved: dict, checkpoint_root: Path):
    """The checkpoint's world model on CPU, with its frozen encoder (never trained, so reloaded) attached."""
    model = build_model(saved["config"]["model"], checkpoint_root)
    model.load_state_dict(_strip_encoder(saved["model"]))
    if model.frozen_visual_encoder:
        model.visual_encoder = VJEPA21Encoder(checkpoint_root)
    return model


def model_record(saved: dict, checkpoint: Path, model, latent_dim: int) -> dict:
    trained = saved["config"]
    return {
        "name": trained.get("name"), "group": trained.get("wandb", {}).get("group", trained.get("name")),
        "type": trained["model"].get("type", "jepa-baseline"), "seed": trained.get("seed"),
        "checkpoint": str(checkpoint), "epoch": saved.get("epoch"), "latent_dim": latent_dim,
        "context_steps": model.context_steps,
    }


@torch.no_grad()
def collect_latents(model, dataset, *, context_frames: int, steps: int, batch_size: int, num_workers: int,
                    device: torch.device, precision: torch.dtype,
                    caches: list[TokenCache | None] | None = None) -> dict[str, Tensor]:
    """encoded [N, T, D], predicted [N, steps, D] (when ``steps`` > 0), raw states, mission indices.

    ``caches`` supplies frozen-encoder tokens instead of encoding images.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
                        drop_last=False)
    parts: dict[str, list[Tensor]] = {"encoded": [], "predicted": [], "states": [], "mission_idx": []}
    for batch in loader:
        moved, tokens = model_inputs(model, batch, device, caches)
        with autocast(device, precision):
            outputs = protocol_latents(model, moved, context_frames=context_frames, steps=steps,
                                       visual_tokens=tokens)
        parts["encoded"].append(outputs["encoded"].float().cpu())
        if steps:
            parts["predicted"].append(outputs["predicted"].float().cpu())
        parts["states"].append(batch["state"].float())
        parts["mission_idx"].append(torch.as_tensor(batch["mission_idx"]))
    return {key: torch.cat(values) for key, values in parts.items() if values}


def _summary(errors: dict[str, Tensor]) -> tuple[dict[str, float], dict[str, float]]:
    return ({key: float(value.mean()) for key, value in errors.items()},
            {key: float(value.std()) if len(value) > 1 else 0.0 for key, value in errors.items()})


def _score_probe(probe: StateProbe, eval_set: dict, sigma: Tensor, *, context_frames: int, horizons: list[int],
                 mission_names: list[str]) -> dict:
    states = eval_set["states"]
    decoded = probe(eval_set["encoded"])  # [N, T, 40]
    predicted = probe(eval_set["predicted"])  # [N, K, 40]
    quality = r2_and_pearson(decoded.reshape(-1, decoded.shape[-1]), states.reshape(-1, states.shape[-1]))
    curve = {}
    for k in horizons:
        target = states[:, context_frames - 1 + k]
        model_errors = rollout_errors(predicted[:, k - 1], target, sigma)
        mean, std = _summary(model_errors)
        per_mission = {
            mission_names[int(index)]: float(model_errors["all"][eval_set["mission_idx"] == index].mean())
            for index in eval_set["mission_idx"].unique()
        }
        curve[str(k)] = {
            "model": mean,
            "model_std_over_windows": std,
            # Error if the predicted latent were exactly the encoded future: the probe's own floor.
            "encoded_floor": _summary(rollout_errors(decoded[:, context_frames - 1 + k], target, sigma))[0],
            # Probe of z_t held constant: the rollout must beat this.
            "persistence": _summary(rollout_errors(decoded[:, context_frames - 1], target, sigma))[0],
            "per_mission_all": per_mission,
        }
    return {"quality": quality, "eps_k": curve}


@dataclass
class ProbeEval:
    results: dict  # per probe kind: fit, quality, eps_k, sigma (the JSON "probes" block)
    probes: dict[str, StateProbe]
    sigma: Tensor  # per-component eval-set std, floored: the ε_k normalizer


def evaluate_probes(collected: dict[str, dict], eval_config: dict, mission_names: list[str],
                    device: torch.device) -> ProbeEval:
    """Fit every configured probe kind on the probe split; score it on the eval split."""
    sigma = evaluation_sigma(collected["eval"]["states"], float(eval_config["sigma_floor"]))
    horizons = sorted(int(k) for k in eval_config["horizons"])
    probes, results = {}, {}
    for kind in eval_config["probe"]["kinds"]:
        probes[kind], fit = fit_probe(kind, collected["probe"]["encoded"], collected["probe"]["states"],
                                      eval_config["probe"], seed=int(eval_config["seed"]), device=device)
        scores = _score_probe(probes[kind], collected["eval"], sigma, context_frames=int(eval_config["context_frames"]),
                              horizons=horizons, mission_names=mission_names)
        results[kind] = {"fit": fit, **scores, "sigma": sigma.tolist()}
    return ProbeEval(results, probes, sigma)


def finite(value):
    """JSON has no NaN/inf; write them as null."""
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_result(result: dict, output: Path | None, run_root: Path, checkpoint: Path, config_name: str) -> Path:
    """Write to ``output``, or ``<run_root>/eval/<checkpoint>-<config_name>.json``."""
    output = Path(output) if output else run_root / "eval" / f"{checkpoint.stem}-{config_name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(finite(result), indent=2) + "\n", encoding="utf-8")
    return output
