"""Simulator adapters exposed through a small, backend-neutral boundary."""

from .base import Simulator, SimulatorUnavailable

__all__ = ["Simulator", "SimulatorUnavailable"]
