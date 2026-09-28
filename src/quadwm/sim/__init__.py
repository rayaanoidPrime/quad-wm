"""Simulator adapters exposed through a small, backend-neutral boundary."""

from __future__ import annotations

import time

import numpy as np

from .base import DynamicsSpec, SimState, Simulator, SimulatorUnavailable, TerrainSpec

__all__ = [
    "DynamicsSpec",
    "SimState",
    "Simulator",
    "SimulatorUnavailable",
    "TerrainSpec",
    "build_simulator",
    "smoke",
]


def build_simulator(config: dict) -> Simulator:
    backend = config.get("backend", "mujoco")
    if backend == "mujoco":
        from .mujoco_backend import MujocoSimulator

        return MujocoSimulator(config)
    raise SimulatorUnavailable(f"unknown simulator backend {backend!r}")


def smoke(config: dict) -> dict[str, float]:
    """E1.0 gate: hold the default pose on each smoke terrain; report stability and FPS."""
    sim_cfg = config["sim"]
    sim = build_simulator(sim_cfg)
    hold = np.tile(np.asarray(sim_cfg["default_joint_pos"]), sim.frames_per_tick)
    ticks = int(config.get("smoke_ticks", 25))
    report = {}
    for kind in config.get("smoke_terrains", ["flat"]):
        first = sim.reset(seed=int(config.get("seed", 0)), terrain=TerrainSpec(kind=kind))
        start = time.perf_counter()
        for _ in range(ticks):
            observation = sim.step(hold)
        elapsed = time.perf_counter() - start
        height_drop = float(first["pose_pos"][2] - observation["pose_pos"][2])
        report[kind] = {"ticks_per_s": ticks / elapsed, "base_height_drop_m": height_drop}
        print(
            f"stage=sim_smoke terrain={kind} ticks_per_s={ticks / elapsed:.1f} "
            f"base_height_drop_m={height_drop:.3f} contacts={observation['contacts'].tolist()} "
            + " ".join(f"{key}={observation[key].shape}" for key in ("rgb", "depth") if key in observation),
            flush=True,
        )
    return report
