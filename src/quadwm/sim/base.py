"""Backend-neutral simulator contract.

Everything a simulator returns uses the same vocabulary as GrandTour
(``MissionReader.sample_state``), so datasets, probes, and the shared
evaluation code never branch on where a trajectory came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class SimulatorUnavailable(RuntimeError):
    """The backend or its assets are not installed on this machine."""


@dataclass(frozen=True)
class TerrainSpec:
    """One tier of the shared terrain suite (protocol §4.3).

    ``kind`` is one of flat | rough | slope | stairs | gaps | steps |
    unilateral_steps (EV6, left side only); ``level``
    is the tier's difficulty in natural units: rough amplitude (m), slope
    angle (deg), stair/step height (m), or gap width (m).
    """

    kind: str = "flat"
    level: float = 0.0
    mirrored: bool = False  # EV6: reflect the terrain left-right


@dataclass(frozen=True)
class DynamicsSpec:
    """Perturbation grid of protocol §5 (EV4); defaults are nominal."""

    base_mass_scale: float = 1.0
    friction_scale: float = 1.0
    latency_ms: float = 0.0


@dataclass
class SimState:
    """Opaque full-state snapshot, restorable for EV2 action replay."""

    physics: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)


class Simulator(Protocol):
    tick_hz: float
    control_hz: float

    def reset(
        self, *, seed: int, terrain: TerrainSpec = TerrainSpec(), dynamics: DynamicsSpec = DynamicsSpec()
    ) -> dict[str, np.ndarray]:
        """Start an episode and return the first tick observation."""

    def step(self, action: np.ndarray) -> dict[str, np.ndarray]:
        """Advance one world-model tick.

        ``action`` is [12 * f] joint-position targets in GrandTour's
        ``JOINT_ORDER``, f = control_hz / tick_hz commands per tick, exactly
        like ``GrandTourSequenceDataset`` actions (before normalization).
        Returns keys: proprio [33], pose_pos, orientation (xyzw), lin_vel,
        ang_vel, gravity, joint_pos, joint_vel, contacts [4], and whichever
        of rgb [H, W, 3] uint8 / depth [H, W] meters the config renders.
        """

    def get_state(self) -> SimState: ...

    def set_state(self, state: SimState) -> None: ...
