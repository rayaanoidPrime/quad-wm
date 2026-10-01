"""Simulated trajectories in GrandTour's vocabulary, for the simulated protocol evals.

Episodes are rendered once per sim-eval config and cached as ``.npz`` so
every model is scored on identical trajectories (protocol §1.3 rule 5).
``SimWindowDataset`` returns exactly what ``GrandTourSequenceDataset`` does,
so probes and rollouts never branch on where a window came from.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.grandtour import pool_depth, state_vectors
from .base import DynamicsSpec, TerrainSpec

STATE_KEYS = ("pose_pos", "orientation", "lin_vel", "ang_vel", "gravity", "joint_pos", "joint_vel", "contacts")
EPISODE_FORMAT = 1  # bump when collect_episode's output changes, so stale caches are rebuilt


def camera_modality(observation: str) -> str:
    """GrandTour observation tag -> the sim camera modality the model sees."""
    if observation.startswith("depth"):
        return "depth"
    if observation.startswith("rgb"):
        return "rgb"
    raise ValueError(f"no sim camera for observation {observation!r}")


def fallen(observation: dict, *, max_gravity_z: float = -0.5, min_clearance_m: float = 0.2) -> bool:
    """Tilted past ~60 degrees, or the base is within ``min_clearance_m`` of the ground below it."""
    clearance = float(observation["pose_pos"][2]) - float(observation.get("terrain_height", 0.0))
    return bool(observation["gravity"][2] > max_gravity_z or clearance < min_clearance_m)


def frame_images(frames, modality: str, depth_size: int, depth_range) -> torch.Tensor:
    """[T, H, W] depth (m) or [T, H, W, 3] RGB -> model images, as the GrandTour dataset emits them."""
    if modality == "depth":
        return torch.stack([pool_depth(np.asarray(frame, dtype=np.float32), depth_size, tuple(depth_range))
                            for frame in frames])
    return torch.from_numpy(np.asarray(frames)).permute(0, 3, 1, 2).float()


def collect_episode(sim, controller, *, seed: int, terrain: TerrainSpec, ticks: int, command_mps: float,
                    heading_change_rad: float, action_noise_rad: float,
                    dynamics: DynamicsSpec = DynamicsSpec()) -> dict[str, np.ndarray]:
    """Drive the controller (plus Gaussian action noise) for ``ticks`` world-model ticks.

    Halfway through, the commanded heading turns by ``heading_change_rad``
    (random sign) so episodes include turning, not only straight walking.
    """
    rng = np.random.default_rng(seed)
    observation = sim.reset(seed=seed, terrain=terrain, dynamics=dynamics)
    controller.command_mps = command_mps
    controller.reset(heading=0.0, phase_s=float(rng.uniform(0.0, 1.0 / controller.frequency_hz)))
    frames, actions = [], []
    for tick in range(ticks):
        frames.append(observation)
        if tick == ticks - 1:
            break
        if tick == ticks // 2:
            controller.heading += heading_change_rad * rng.choice([-1.0, 1.0])
        action = controller.act(observation)
        action = action + rng.normal(0.0, action_noise_rad, action.shape)
        actions.append(action)
        observation = sim.step(action)
    episode = {key: np.stack([frame[key] for frame in frames]).astype(np.float32) for key in (*STATE_KEYS, "proprio")}
    if "terrain_height" in frames[0]:
        episode["terrain_height"] = np.asarray([frame["terrain_height"] for frame in frames], dtype=np.float32)
    if "depth" in frames[0]:
        episode["depth"] = np.stack([frame["depth"] for frame in frames]).astype(np.float16)
    if "rgb" in frames[0]:
        episode["rgb"] = np.stack([frame["rgb"] for frame in frames])
    episode["actions"] = np.stack(actions).astype(np.float32)
    episode["fallen"] = np.asarray([fallen(frame) for frame in frames])
    return episode


def _episode_specs(spec: dict, split: str) -> list[tuple[str, TerrainSpec, int, float]]:
    low, high = spec["command_mps"]
    specs = []
    for kind, level in spec["terrains"]:
        for seed in spec[f"{split}_seeds"]:
            command = float(np.random.default_rng([seed, 7]).uniform(low, high))
            specs.append((f"{split}-{kind}{level:g}-s{seed}", TerrainSpec(kind=kind, level=float(level)), int(seed),
                          command))
    return specs


def episode_cache_key(sim_config: dict, controller_config: dict, spec: dict, split: str) -> str:
    """Everything that changes an episode, minus machine-local paths."""
    relevant = {
        "format": EPISODE_FORMAT, "split": split, "controller": controller_config,
        "sim": {key: value for key, value in sim_config.items() if key != "mjcf"},
        "episodes": {key: value for key, value in spec.items() if key != "cache_root"},
    }
    return hashlib.sha256(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()[:12]


def episode_set(sim_factory, controller_factory, sim_config: dict, controller_config: dict, spec: dict,
                split: str) -> list[Path]:
    """Paths of the split's cached episodes, collecting (and rendering) them on first use."""
    folder = Path(spec["cache_root"]).expanduser() / f"{split}-{episode_cache_key(sim_config, controller_config, spec, split)}"
    specs = _episode_specs(spec, split)
    paths = [folder / f"{name}.npz" for name, *_ in specs]
    if (folder / "complete.json").is_file():
        return paths
    folder.mkdir(parents=True, exist_ok=True)
    sim, controller = sim_factory(), controller_factory()
    for path, (name, terrain, seed, command) in zip(paths, specs):
        if path.is_file():
            continue
        episode = collect_episode(sim, controller, seed=seed, terrain=terrain, ticks=int(spec["ticks"]),
                                  command_mps=command, heading_change_rad=float(spec["heading_change_rad"]),
                                  action_noise_rad=float(spec["action_noise_rad"]))
        temporary = path.with_name(path.stem + ".part.npz")
        np.savez_compressed(temporary, **episode)
        temporary.replace(path)
        print(f"stage=sim_episodes split={split} episode={name} fell={bool(episode['fallen'].any())}", flush=True)
    (folder / "complete.json").write_text(json.dumps({"episodes": [p.name for p in paths]}) + "\n",
                                          encoding="utf-8")
    return paths


def load_episode(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


class SimWindowDataset(Dataset):
    """Windows of ``context_frames + steps`` ticks from cached episodes, none touching a fall."""

    def __init__(self, episodes: list[dict], *, context_frames: int, steps: int, stride_ticks: int,
                 modality: str, normalization: dict | None, depth_size: int = 64,
                 depth_range=(0.2, 10.0)):
        self.episodes = episodes
        self.context_frames, self.total = context_frames, context_frames + steps
        self.modality, self.depth_size, self.depth_range = modality, depth_size, tuple(depth_range)
        self.normalization = (
            {key: np.asarray(value, dtype=np.float32) for key, value in normalization.items()}
            if normalization else None
        )
        self.index = [
            (episode_index, start)
            for episode_index, episode in enumerate(episodes)
            for start in range(0, len(episode["proprio"]) - self.total + 1, stride_ticks)
            if not episode["fallen"][start : start + self.total].any()
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict:
        episode_index, start = self.index[index]
        episode = self.episodes[episode_index]
        window = slice(start, start + self.total)
        states = [{key: episode[key][tick] for key in STATE_KEYS} for tick in range(start, start + self.total)]
        proprio = episode["proprio"][window]
        actions = episode["actions"][start : start + self.total - 1]
        if self.normalization is not None:
            proprio = (proprio - self.normalization["proprio_mean"]) / self.normalization["proprio_std"]
            actions = (actions - self.normalization["action_mean"]) / self.normalization["action_std"]
        return {
            "proprio": torch.from_numpy(np.asarray(proprio, dtype=np.float32)),
            "actions": torch.from_numpy(np.asarray(actions, dtype=np.float32)),
            "state": torch.from_numpy(state_vectors(states, self.context_frames - 1)),
            "mission_idx": episode_index,
            "image_ids": torch.arange(start, start + self.total),
            "images": frame_images(episode[self.modality][window], self.modality, self.depth_size,
                                   self.depth_range),
        }
