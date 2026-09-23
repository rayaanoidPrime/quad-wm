from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    """Convert common config values into values accepted by W&B."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value



def init_wandb(
    cfg: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
    run_dir: Path | None = None,
):
    """Initialize a W&B run from the repository config.

    Returns ``None`` when logging is disabled. Importing W&B lazily keeps the
    CPU-only path usable in environments that do not install the optional SDK.
    """

    wandb_cfg = cfg.get("wandb", {})
    if not wandb_cfg.get("enabled", False):
        return None

    if run_dir is not None:
        # Set this before importing W&B: newer SDK versions initialize service
        # path defaults during import.
        os.environ.setdefault("WANDB_DIR", str(run_dir))

    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - dependency varies by environment
        raise RuntimeError("W&B logging is enabled, but the wandb package is not installed") from exc

    init_kwargs: dict[str, Any] = {
        "project": wandb_cfg.get("project", "quad-wm"),
        "name": wandb_cfg.get("name"),
        "mode": wandb_cfg.get("mode", os.environ.get("WANDB_MODE", "offline")),
        "config": _jsonable({**cfg, "runtime": metadata or {}}),
    }
    entity = wandb_cfg.get("entity")
    if entity:
        init_kwargs["entity"] = entity
    tags = wandb_cfg.get("tags")
    if tags:
        init_kwargs["tags"] = [_jsonable(tag) for tag in tags]
    for key in ("group", "job_type", "notes"):
        value = wandb_cfg.get(key)
        if value:
            init_kwargs[key] = value
    if run_dir is not None:
        init_kwargs["dir"] = str(run_dir)
    settings_factory = getattr(wandb, "Settings", None)
    settings_fields = getattr(settings_factory, "model_fields", None)
    if settings_factory is not None and (
        settings_fields is None or "start_method" in settings_fields
    ):
        init_kwargs["settings"] = settings_factory(
            start_method=wandb_cfg.get("start_method", "thread")
        )

    return wandb.init(**init_kwargs)



