"""Small optional MuJoCo loader used by the simulator compatibility gate."""

from __future__ import annotations

from pathlib import Path

from .base import SimulatorUnavailable


def load_model(model_path: str | Path):
    """Load a MuJoCo model without making MuJoCo a core import-time dependency."""

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise SimulatorUnavailable(
            "MuJoCo is not installed; install the simulator extra before using this adapter"
        ) from exc

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(path)
    return mujoco.MjModel.from_xml_path(str(path))
