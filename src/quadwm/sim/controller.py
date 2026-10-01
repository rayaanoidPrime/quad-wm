"""Scripted trot: the fixed walking controller behind every simulated eval (docs/adr/0006).

A stand-in for the "fixed pretrained ANYmal D locomotion controller" the
recipes assume. It is a phase-based trot (diagonal pairs half a cycle apart)
around the standing pose: hip flexion sweeps the foot fore-aft, the knee
folds during swing for clearance, and a heading loop lengthens the stride on
one side to hold the commanded yaw. Stride length is the speed knob.

Its output is exactly a GrandTour action: ``12 * f`` joint-position targets
per world-model tick, in ``JOINT_ORDER``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

# Diagonal pairs LF+RH and RF+LH, half a cycle apart. Leg order: LF RF LH RH.
_PHASE = np.array([0.0, np.pi, np.pi, 0.0])
_LEFT = np.array([1.0, -1.0, 1.0, -1.0])  # +1 for left legs
_KNEE = np.array([-1.0, -1.0, 1.0, 1.0])  # knee fold direction (front knees bend the other way)


@dataclass
class TrotController:
    default_pose: np.ndarray
    frames_per_tick: int
    control_hz: float
    frequency_hz: float = 1.5
    knee_lift: float = 0.7
    speed_per_stride: float = 1.6  # m/s per rad of hip sweep, measured on flat ground (kp 300)
    max_stride: float = 0.6
    heading_gain: float = 3.0
    command_mps: float = 0.4
    heading: float = 0.0  # commanded yaw (rad), world frame

    def __post_init__(self):
        self.default_pose = np.asarray(self.default_pose, dtype=np.float64)
        self.time = 0.0

    def reset(self, *, heading: float = 0.0, phase_s: float = 0.0) -> None:
        self.time, self.heading = phase_s, heading

    def stride(self, command_mps: float) -> float:
        return float(np.clip(command_mps / self.speed_per_stride, 0.0, self.max_stride))

    def targets(self, time_s: float, stride: float, turn: float) -> np.ndarray:
        """One 12-d joint target. ``turn`` > 0 lengthens right-side strides (turns left)."""
        phase = 2 * np.pi * self.frequency_hz * time_s + _PHASE
        sides = stride * (1.0 - turn * _LEFT)
        pose = self.default_pose.reshape(4, 3).copy()
        pose[:, 1] += sides * np.cos(phase)
        pose[:, 2] += _KNEE * self.knee_lift * np.maximum(np.sin(phase), 0.0) * (stride > 0)
        return pose.ravel()

    def nominal(self, observation: dict, ticks: int = 1) -> np.ndarray:
        """[ticks, 12 * frames_per_tick] targets from now, without advancing the gait clock.

        Heading feedback uses the current yaw for every returned tick, which
        is what a planner sees as the controller's open-loop continuation.
        """
        yaw = Rotation.from_quat(observation["orientation"]).as_euler("zyx")[0]
        error = np.angle(np.exp(1j * (self.heading - yaw)))
        turn = float(np.clip(self.heading_gain * error, -0.5, 0.5))
        stride = self.stride(self.command_mps)
        times = self.time + np.arange(ticks * self.frames_per_tick) / self.control_hz
        return np.stack([self.targets(time, stride, turn) for time in times]).reshape(ticks, -1)

    def act(self, observation: dict, residual: np.ndarray | None = None) -> np.ndarray:
        """Targets for the next tick, plus an optional per-joint ``residual`` [12] held for the tick."""
        action = self.nominal(observation, 1)[0]
        if residual is not None:
            action = action + np.tile(residual, self.frames_per_tick)
        self.time += self.frames_per_tick / self.control_hz
        return action


def build_controller(sim_config: dict, controller_config: dict | None = None) -> TrotController:
    frames = round(float(sim_config["control_hz"]) / float(sim_config["tick_hz"]))
    return TrotController(
        default_pose=np.asarray(sim_config["default_joint_pos"]),
        frames_per_tick=frames,
        control_hz=float(sim_config["control_hz"]),
        **(controller_config or {}),
    )
