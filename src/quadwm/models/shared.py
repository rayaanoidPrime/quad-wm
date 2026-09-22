"""Model construction and small shared checkpoint helpers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from .jepawm import JEPAWorldModel, VJEPA21_FILENAME

VJEPA21_URL = f"https://dl.fbaipublicfiles.com/vjepa2/{VJEPA21_FILENAME}"


def ensure_vjepa21_checkpoint(checkpoint_root: str | Path) -> Path:
    """Download the official checkpoint once, atomically, to local storage."""

    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / VJEPA21_FILENAME
    if checkpoint.is_file() and checkpoint.stat().st_size > 0:
        return checkpoint
    with tempfile.NamedTemporaryFile(dir=root, suffix=".part", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.hub.download_url_to_file(VJEPA21_URL, str(temporary), progress=True)
        os.replace(temporary, checkpoint)
    finally:
        temporary.unlink(missing_ok=True)
    return checkpoint


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(checkpoint["model"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("epoch", 0))


def build_model(
    wm_config: dict,
    checkpoint_root: str | Path,
    *,
    visual_encoder: torch.nn.Module | None = None,
) -> JEPAWorldModel:
    """Build either the small smoke model or the configured baseline model."""

    baseline = bool(wm_config.get("baseline", False))
    if not baseline:
        return JEPAWorldModel(
            visual_dim=int(wm_config.get("embed_dim", 128)),
            tokens_per_frame=int(wm_config.get("tokens_per_frame", 1)),
            predictor_depth=1,
            predictor_heads=1,
            context_steps=1,
            rollout_context=1,
            encoder=visual_encoder,
        )
    return JEPAWorldModel(
        visual_dim=int(wm_config.get("visual_dim", 768)),
        proprio_dim=int(wm_config.get("proprio_dim", 33)),
        proprio_embed_dim=int(wm_config.get("proprio_embed_dim", 16)),
        action_dim=int(wm_config.get("action_dim", 120)),
        tokens_per_frame=int(wm_config.get("tokens_per_frame", 576)),
        predictor_depth=int(wm_config.get("predictor_depth", 12)),
        predictor_heads=int(wm_config.get("predictor_heads", 16)),
        context_steps=int(wm_config.get("context_steps", 7)),
        rollout_context=int(wm_config.get("rollout_context", 3)),
        encoder=visual_encoder,
    )
