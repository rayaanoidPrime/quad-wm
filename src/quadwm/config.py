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
            return value
        if fallback is not None:
            return fallback
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
        data_path = Path(config["data_config"])
        config["data"] = yaml.safe_load(data_path)
        config["data"]["config_path"] = str(data_path)
    return config
