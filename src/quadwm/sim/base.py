"""Backend-neutral simulator contracts.

The Track 1 model should consume canonical observations and transitions, never
MuJoCo or Isaac-specific objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class SimulatorUnavailable(RuntimeError):
    """Raised when an optional simulator dependency or asset is unavailable."""


@dataclass(frozen=True)
class Transition:
    observation: Any
    action: Any
    reward: float
    next_observation: Any
    terminated: bool
    truncated: bool


class Simulator(Protocol):
    """Minimal interface required by data collection and evaluation."""

    def reset(self, *, seed: int | None = None) -> Any: ...

    def step(self, action: Any) -> Transition: ...

    def close(self) -> None: ...
