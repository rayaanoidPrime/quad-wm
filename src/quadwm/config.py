"""Load small YAML experiment configs with shell-style environment defaults."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


def _expand_environment(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name, fallback = match.groups()
        value = os.environ.get(name)
        if value is not None:
            # Also expand a "~" that came in through the environment (e.g.
            # QUADWM_RUN_ROOT="~/..." passed unquoted-expanded through sbatch),
            # so it cannot create a literal ./~ dir inside the checkout.
            return os.path.expanduser(value)
        if fallback is not None:
            # Python never expands "~" in paths the way a shell does, so a
            # "${VAR:-~/x}" default would otherwise create a literal ./~ dir.
            return os.path.expanduser(fallback)
        raise RuntimeError(f"environment variable {name!r} is required by the config")

    return _ENV_PATTERN.sub(replace, text)


def load_config(path: Path | None) -> dict[str, Any]:
    """Load and resolve a YAML config, or return an empty config when omitted."""

    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    resolved_text = _expand_environment(path.read_text(encoding="utf-8"))
    config = yaml.safe_load(resolved_text) or {}
    if not isinstance(config, dict):
        raise TypeError(f"top-level config must be a mapping: {path}")
    config["config_path"] = str(path)
    if "data_config" in config:
        data_path = referenced_path(config, config["data_config"])
        data_text = _expand_environment(data_path.read_text(encoding="utf-8"))
        config["data"] = yaml.safe_load(data_text) or {}
        if not isinstance(config["data"], dict):
            raise TypeError(f"data config must be a mapping: {data_path}")
        config["data"]["config_path"] = str(data_path)
    return config


def referenced_path(config: dict[str, Any], value: str | Path) -> Path:
    """A file another config names (``data_config``, ``protocol_config``, ``sim_config``).

    As given when absolute or present relative to the working directory, else
    relative to the referencing config's own file.
    """
    path = Path(value)
    if not path.is_absolute() and not path.exists() and "config_path" in config:
        path = Path(config["config_path"]).parent / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def run_dir(config: dict[str, Any]) -> Path:
    """``<run_root>/<name>``: where training writes and every eval reads a run."""
    return Path(config.get("run_root", "runs")) / config.get("name", "jepa-baseline")


def checkpoint_dir(config: dict[str, Any]) -> Path:
    """Pretrained-weight storage (the V-JEPA 2.1 checkpoint), shared by every run."""
    return Path(config.get("checkpoint_root", "checkpoints"))


def data_dir(config: dict[str, Any]) -> Path:
    """The materialized GrandTour root: ``data.data_root``, else a top-level ``data_root``."""
    return Path(config.get("data", {}).get("data_root", config.get("data_root", "data/grandtour")))
