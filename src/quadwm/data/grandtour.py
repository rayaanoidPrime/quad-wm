"""GrandTour dataset interface for quadwm: idempotent download + synced pairs.

Usage (see cli.py):
    fetch_missions(config["data"]["missions"], config["data_root"])
    dataset = build_dataset(
        config["data"], observation=config["wm"]["observation"],
        platform=config["wm"]["platform"], data_root=config["data_root"],
        horizon=config["wm"].get("horizon_steps", 4),
    )
    loader = dataset.loader(batch_size=8)

Known gotchas encoded here (see grandtour_data_guide.md for detail):
  - image_id (array position), not sequence_id, indexes image files on disk.
  - depth PNGs are 16-bit millimeters; converted to meters on load.
  - per-mission camera calibration is read fresh per MissionReader, never
    cached across missions.
  - anymal_state_actuator's 12-joint ordering is documented explicitly by the
    dataset. anymal_state_state_estimator's joint_positions/joint_velocities
    ordering is NOT independently documented -- verify_joint_order_consistency()
    checks it empirically; call it once per mission before trusting either.

Scope note: GrandTourPairDataset uses a single fixed prediction horizon.
The Track 1 recipe's multi-horizon set H_g = {1, 4, 12} for the transition
grounding loss is a straightforward loop over several horizon-configured
instances of this class, not built here.
"""

from __future__ import annotations

import bisect
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio  # v2 API avoids the v3-migration deprecation warning
import numpy as np
import torch
import zarr
from huggingface_hub import list_repo_files, snapshot_download
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader, Dataset

GRANDTOUR_REPO_ID = "leggedrobotics/grand_tour_dataset"

# Confirmed from the anymal_state_actuator topic description.
JOINT_ORDER = [
    "LF_HAA", "LF_HFE", "LF_KFE",
    "RF_HAA", "RF_HFE", "RF_KFE",
    "LH_HAA", "LH_HFE", "LH_KFE",
    "RH_HAA", "RH_HFE", "RH_KFE",
]
FEET = ("LF", "RF", "LH", "RH")

# observation tag -> platform -> {role: topic}. Add rows here, not in yaml,
# so every track's "depth_plus_proprioception" pulls identical topic names.
OBSERVATION_TOPICS = {
    "depth_plus_proprioception": {
        "anymal_d": {
            "depth": "depth_camera_front_upper",
            "proprio": "anymal_state_state_estimator",
            "actuator": "anymal_state_actuator",
        },
    },
    "rgb_plus_proprioception": {
        "anymal_d": {
            "depth": "alphasense_front_center",
            "proprio": "anymal_state_state_estimator",
            "actuator": "anymal_state_actuator",
        },
    },
    # "lidar_plus_proprioception": stretch goal, not wired up yet.
}

_GRAVITY_WORLD = np.array([0.0, 0.0, -1.0])  # world frame, z-up


def resolve_topics(observation: str, platform: str) -> dict[str, str]:
    try:
        return OBSERVATION_TOPICS[observation][platform]
    except KeyError as exc:
        raise ValueError(
            f"No topic mapping for observation={observation!r}, platform={platform!r}. "
            f"Known: {list(OBSERVATION_TOPICS)}"
        ) from exc


def project_gravity(quat_xyzw: np.ndarray) -> np.ndarray:
    """Rotate the world gravity direction into the base frame."""
    rot = Rotation.from_quat(quat_xyzw)  # GrandTour stores (x, y, z, w)
    return rot.inv().apply(_GRAVITY_WORLD)


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def _extract_tars(cache_dir: Path, dest_dir: Path, allow_patterns: list[str] | None) -> None:
    def to_regex(patterns: list[str]) -> re.Pattern:
        parts = [f".*{re.escape(p).replace(r'\*', '.*').replace(r'\?', '.')}$" for p in patterns]
        return re.compile("|".join(parts))

    pattern = to_regex(allow_patterns) if allow_patterns else None
    files = [
        f for f in Path(cache_dir).rglob("*")
        if f.is_file() and (pattern is None or pattern.match(str(f)))
    ]

    for f in [x for x in files if x.suffix == ".tar"]:
        dest = dest_dir / f.relative_to(cache_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(f, "r") as tar:
            tar.extractall(path=dest.parent)

    for f in [x for x in files if x.suffix != ".tar" and x.is_file()]:
        dest = dest_dir / f.relative_to(cache_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)


def _mission_names(missions: list[str] | str | None) -> list[str]:
    if missions is None:
        return []
    if isinstance(missions, str):
        return [mission.strip() for mission in missions.split(",") if mission.strip()]
    return list(missions)


def _mission_ready(mission: Path, topics: list[str] | None) -> bool:
    if not (mission / "data").is_dir():
        return False
    if not topics:
        return True
    try:
        root = zarr.open_group(store=mission / "data", mode="r")
        if any(topic not in root for topic in topics):
            return False
    except (KeyError, OSError, ValueError):
        return False
    camera_topics = tuple(
        topic for topic in topics if topic.startswith(("alpha", "hdr", "depth", "zed"))
    )
    return all((mission / "images" / topic).is_dir() for topic in camera_topics)


def fetch_missions(
    missions: list[str] | str | None,
    data_root: str | Path,
    download_topics: list[str] | None = None,
) -> None:
    """Download selected missions, or every remote mission when selection is None.

    Idempotent: skips a mission when its required topics are already materialized.
    A mission with only a partial topic download is completed on the next run.
    """
    root = Path(data_root).expanduser()
    selected = _mission_names(missions)
    if missions is None:
        remote_files = list_repo_files(repo_id=GRANDTOUR_REPO_ID, repo_type="dataset")
        selected = sorted(
            {
                mission
                for path in remote_files
                if "/" in path
                for mission in [path.split("/", 1)[0]]
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", mission)
            }
        )
    if not selected:
        raise ValueError("GrandTour download selection contains no missions")
    root.mkdir(parents=True, exist_ok=True)
    pending = [m for m in selected if not _mission_ready(root / m, download_topics)]
    if not pending:
        return
    allow_patterns = (
        [
            f"{mission}/*{topic}*"
            for mission in pending
            for topic in download_topics
        ]
        + [f"{mission}/*.yaml" for mission in pending]
        if download_topics
        else [f"{mission}/*" for mission in pending]
    )
    cache_path = snapshot_download(
        repo_id=GRANDTOUR_REPO_ID, allow_patterns=allow_patterns, repo_type="dataset"
    )
    _extract_tars(Path(cache_path), root, allow_patterns)


def _mission_dirs(
    data_root: str | Path,
    missions: list[str] | str | None,
) -> tuple[Path, list[Path]]:
    root = Path(data_root).expanduser()
    names = _mission_names(missions)
    if names:
        mission_dirs = [root / mission for mission in names]
    elif (root / "data").is_dir():
        mission_dirs = [root]
    elif root.is_dir():
        mission_dirs = sorted(
            path for path in root.iterdir() if path.is_dir() and (path / "data").is_dir()
        )
    else:
        mission_dirs = []
    if not mission_dirs:
        raise ValueError(
            f"no GrandTour missions found under {root}; expected <root>/<mission>/data "
            "or <root>/data. Set GRANDTOUR_ROOT to the materialized dataset."
        )
    missing = [path for path in mission_dirs if not (path / "data").is_dir()]
    if missing:
        raise FileNotFoundError(
            "configured GrandTour missions are not materialized: "
            + ", ".join(str(path) for path in missing)
        )
    return root, mission_dirs


# --------------------------------------------------------------------------
# Per-mission reading
# --------------------------------------------------------------------------

@dataclass
class MissionReader:
    """Reads one mission's synced (depth, proprioceptive-state, action) samples."""

    mission_dir: Path
    depth_topic: str
    proprio_topic: str
    actuator_topic: str
    max_gap_s: float = 0.05  # drop a sample if the nearest source is farther than this

    def __post_init__(self) -> None:
        self.root = zarr.open_group(store=self.mission_dir / "data", mode="r")
        self.depth_group = self.root[self.depth_topic]
        self.proprio_group = self.root[self.proprio_topic]
        self.actuator_group = self.root[self.actuator_topic]

        self.depth_timestamps = np.asarray(self.depth_group["timestamp"][:])
        self.proprio_timestamps = np.asarray(self.proprio_group["timestamp"][:])
        self.actuator_timestamps = np.asarray(self.actuator_group["timestamp"][:])

        # Per-mission calibration -- intentionally read fresh, never reused
        # across MissionReader instances.
        self.depth_transform = dict(self.depth_group.attrs.get("transform", {}))
        self.depth_camera_info = dict(self.depth_group.attrs.get("camera_info", {}))

        ext = ".png" if "depth" in self.depth_topic else ".jpeg"
        self._image_dir = self.mission_dir / "images" / self.depth_topic
        self._image_ext = ext

    def __len__(self) -> int:
        return len(self.depth_timestamps)

    def load_depth(self, image_id: int) -> np.ndarray:
        # image_id is the zero-based array position -- NOT
        # depth_group["sequence_id"][image_id], which is the ROS-runtime id
        # and does not index the file on disk.
        path = self._image_dir / f"{image_id:06d}{self._image_ext}"
        depth_mm = imageio.imread(path).astype(np.float32)
        return depth_mm / 1000.0  # mm -> m

    def load_image(self, image_id: int) -> np.ndarray:
        """Load the image at the Zarr array position, not its runtime id."""
        for suffix in (".jpeg", ".jpg", ".png"):
            path = self._image_dir / f"{image_id:06d}{suffix}"
            if path.exists():
                return imageio.imread(path)
        raise FileNotFoundError(f"no image for {self.depth_topic}[{image_id}] in {self._image_dir}")

    @staticmethod
    def _nearest(timestamps: np.ndarray, t: float) -> tuple[int, float]:
        idx = bisect.bisect_left(timestamps, t)
        idx = min(max(idx, 0), len(timestamps) - 1)
        return idx, abs(float(timestamps[idx]) - t)

    def sample_state(self, t: float) -> dict | None:
        p_idx, p_gap = self._nearest(self.proprio_timestamps, t)
        if p_gap > self.max_gap_s:
            return None  # likely a dropped message, not a valid interpolation window
        action = self.sample_action(t)
        if action is None:
            return None

        g = self.proprio_group
        quat = np.asarray(g["pose_orien"][p_idx], dtype=np.float32)  # (x, y, z, w)
        contacts = np.array(
            [g[f"{foot}_FOOT_contact"][p_idx] for foot in FEET], dtype=np.float32
        )
        return {
            "pose_pos": np.asarray(g["pose_pos"][p_idx], dtype=np.float32),
            "lin_vel": np.asarray(g["twist_lin"][p_idx], dtype=np.float32),
            "ang_vel": np.asarray(g["twist_ang"][p_idx], dtype=np.float32),
            "gravity": project_gravity(quat).astype(np.float32),
            "joint_pos": np.asarray(g["joint_positions"][p_idx], dtype=np.float32),
            "joint_vel": np.asarray(g["joint_velocities"][p_idx], dtype=np.float32),
            "contacts": contacts,
            "action": action,
        }

    def sample_action(self, t: float) -> np.ndarray | None:
        a_idx, a_gap = self._nearest(self.actuator_timestamps, t)
        if a_gap > self.max_gap_s:
            return None
        return np.array(
            [
                self.actuator_group[f"{j:02d}_command_position"][a_idx]
                for j in range(len(JOINT_ORDER))
            ],
            dtype=np.float32,
        )

    def sample_action_window(
        self, start_time: float, *, frames: int, control_hz: float
    ) -> np.ndarray | None:
        actions = [
            self.sample_action(start_time + frame / control_hz)
            for frame in range(frames)
        ]
        if any(action is None for action in actions):
            return None
        return np.stack(actions)

    def __getitem__(self, image_id: int) -> dict | None:
        t = float(self.depth_timestamps[image_id])
        state = self.sample_state(t)
        if state is None:
            return None
        return {"depth": self.load_depth(image_id), "timestamp": t, **state}


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

class GrandTourDataset(Dataset):
    """Flat single-frame index across missions. Kept for probe/eval use
    (e.g. the shared evaluation protocol's frozen-model probing); training
    uses GrandTourPairDataset below instead.
    """

    def __init__(self, readers: list[MissionReader]):
        self.readers = readers
        self.index: list[tuple[int, int]] = []
        dropped = 0
        for r_idx, reader in enumerate(readers):
            for image_id in range(len(reader)):
                t = float(reader.depth_timestamps[image_id])
                if reader.sample_state(t) is None:
                    dropped += 1
                    continue
                self.index.append((r_idx, image_id))
        total = dropped + len(self.index)
        self.stats = {
            "total": total, "kept": len(self.index), "dropped": dropped,
            "drop_rate": dropped / total if total else 0.0,
        }

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        r_idx, image_id = self.index[idx]
        s = self.readers[r_idx][image_id]
        return {
            "depth": torch.from_numpy(s["depth"]).unsqueeze(0),
            "pose_pos": torch.from_numpy(s["pose_pos"]),
            "joint_pos": torch.from_numpy(s["joint_pos"]),
            "joint_vel": torch.from_numpy(s["joint_vel"]),
            "lin_vel": torch.from_numpy(s["lin_vel"]),
            "ang_vel": torch.from_numpy(s["ang_vel"]),
            "gravity": torch.from_numpy(s["gravity"]),
            "contacts": torch.from_numpy(s["contacts"]),
            "action": torch.from_numpy(s["action"]),
            "mission_idx": r_idx,
        }

    def loader(self, batch_size: int, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class GrandTourPairDataset(Dataset):
    """(t, t+horizon) pairs within a single mission, for JEPA forward
    prediction and the PSG-JEPA transition-grounding head.

    Pairing happens at the raw image_id level within a mission (not the
    drop-filtered index), so both endpoints get their own independent sync
    check -- and the actual elapsed time between them is checked against the
    expected horizon duration, since a dropped frame in between would
    otherwise silently produce a pair that spans more real time than intended.
    """

    def __init__(
        self, readers: list[MissionReader], horizon: int, rate_hz: float, rate_tol: float = 0.3
    ):
        self.readers = readers
        self.horizon = horizon
        expected_dt = horizon / rate_hz
        self.index: list[tuple[int, int]] = []
        dropped = 0
        for r_idx, reader in enumerate(readers):
            n = len(reader)
            for i in range(n - horizon):
                t0 = float(reader.depth_timestamps[i])
                t1 = float(reader.depth_timestamps[i + horizon])
                if abs((t1 - t0) - expected_dt) > rate_tol * expected_dt:
                    dropped += 1
                    continue
                if reader.sample_state(t0) is None or reader.sample_state(t1) is None:
                    dropped += 1
                    continue
                self.index.append((r_idx, i))
        total = dropped + len(self.index)
        self.stats = {
            "total": total, "kept": len(self.index), "dropped": dropped,
            "drop_rate": dropped / total if total else 0.0,
        }

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        r_idx, i = self.index[idx]
        reader = self.readers[r_idx]
        s0 = reader[i]
        s1 = reader[i + self.horizon]
        out = {
            "depth_t": torch.from_numpy(s0["depth"]).unsqueeze(0),
            "depth_t1": torch.from_numpy(s1["depth"]).unsqueeze(0),
            "action_t": torch.from_numpy(s0["action"]),
        }
        for suffix, s in (("_t", s0), ("_t1", s1)):
            out[f"pose_pos{suffix}"] = torch.from_numpy(s["pose_pos"])
            out[f"joint_pos{suffix}"] = torch.from_numpy(s["joint_pos"])
            out[f"joint_vel{suffix}"] = torch.from_numpy(s["joint_vel"])
            out[f"lin_vel{suffix}"] = torch.from_numpy(s["lin_vel"])
            out[f"ang_vel{suffix}"] = torch.from_numpy(s["ang_vel"])
            out[f"gravity{suffix}"] = torch.from_numpy(s["gravity"])
            out[f"contacts{suffix}"] = torch.from_numpy(s["contacts"])
        return out

    def loader(self, batch_size: int, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
        # num_workers > 0 pickles open zarr groups across worker processes,
        # which is fragile -- keep at 0 for the smoke test; revisit with a
        # worker_init_fn that reopens the store per-worker for full runs.
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class GrandTourSequenceDataset(Dataset):
    """Fixed-length RGB/proprio/action windows for JEPA-WM training."""

    def __init__(
        self,
        readers: list[MissionReader],
        *,
        context_steps: int,
        rollout_steps: int,
        tick_hz: float,
        control_hz: float,
        action_frames: int,
        max_tick_error_s: float = 0.06,
        max_sequences: int | None = None,
        load_images: bool = True,
    ):
        self.readers = readers
        self.context_steps = context_steps
        self.rollout_steps = rollout_steps
        self.total_steps = context_steps + rollout_steps
        self.tick_dt = 1.0 / tick_hz
        self.control_hz = control_hz
        self.action_frames = action_frames
        self.load_images = load_images
        self.index: list[tuple[int, tuple[int, ...]]] = []
        dropped = 0
        for reader_index, reader in enumerate(readers):
            source_dt = (
                float(np.median(np.diff(reader.depth_timestamps)))
                if len(reader) > 1
                else self.tick_dt
            )
            start_stride = max(1, round(self.tick_dt / source_dt))
            for start_id in range(0, len(reader.depth_timestamps), start_stride):
                start_time = reader.depth_timestamps[start_id]
                ids = []
                valid = True
                for step in range(self.total_steps):
                    image_id, error = reader._nearest(
                        reader.depth_timestamps, float(start_time) + step * self.tick_dt
                    )
                    if error > max_tick_error_s:
                        valid = False
                        break
                    ids.append(image_id)
                if not valid or len(set(ids)) != len(ids):
                    dropped += 1
                    continue
                for image_id in ids:
                    if reader.sample_state(float(reader.depth_timestamps[image_id])) is None:
                        valid = False
                        break
                if valid:
                    for image_id in ids[:-1]:
                        if reader.sample_action_window(
                            float(reader.depth_timestamps[image_id]),
                            frames=action_frames,
                            control_hz=control_hz,
                        ) is None:
                            valid = False
                            break
                if valid:
                    self.index.append((reader_index, tuple(ids)))
                    if max_sequences is not None and len(self.index) >= max_sequences:
                        break
                else:
                    dropped += 1
            if max_sequences is not None and len(self.index) >= max_sequences:
                break
        total = dropped + len(self.index)
        self.stats = {
            "total": total,
            "kept": len(self.index),
            "dropped": dropped,
            "drop_rate": dropped / total if total else 0.0,
        }

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict:
        reader_index, image_ids = self.index[index]
        reader = self.readers[reader_index]
        states = [reader.sample_state(float(reader.depth_timestamps[i])) for i in image_ids]
        assert all(state is not None for state in states)
        proprio = np.stack(
            [
                np.concatenate(
                    (
                        state["lin_vel"],
                        state["ang_vel"],
                        state["gravity"],
                        state["joint_pos"],
                        state["joint_vel"],
                    )
                )
                for state in states
            ]
        )
        actions = np.stack(
            [
                reader.sample_action_window(
                    float(reader.depth_timestamps[i]),
                    frames=self.action_frames,
                    control_hz=self.control_hz,
                ).reshape(-1)
                for i in image_ids[:-1]
            ]
        )
        output = {
            "proprio": torch.from_numpy(proprio).float(),
            "actions": torch.from_numpy(actions).float(),
            "mission_idx": reader_index,
            "image_ids": torch.tensor(image_ids, dtype=torch.long),
        }
        if self.load_images:
            observations = np.stack([reader.load_image(i) for i in image_ids])
            output["images"] = torch.from_numpy(observations).permute(0, 3, 1, 2).float()
        return output

    def loader(
        self,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        sampler=None,
    ) -> DataLoader:
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )


def build_dataset(
    data_config: dict,
    observation: str,
    platform: str,
    data_root: str | Path,
    horizon: int = 4,
) -> GrandTourPairDataset:
    topics = resolve_topics(observation, platform)
    max_gap_s = data_config.get("max_gap_ms", 50) / 1000.0
    rate_hz = data_config.get("sync_rate_hz", 15.0)
    _, mission_dirs = _mission_dirs(data_root, data_config.get("missions"))
    readers = [
        MissionReader(
            mission_dir,
            depth_topic=topics["depth"],
            proprio_topic=topics["proprio"],
            actuator_topic=topics["actuator"],
            max_gap_s=max_gap_s,
        )
        for mission_dir in mission_dirs
    ]
    return GrandTourPairDataset(readers, horizon=horizon, rate_hz=rate_hz)


def build_sequence_dataset(
    data_config: dict,
    observation: str,
    platform: str,
    data_root: str | Path,
    *,
    context_steps: int,
    rollout_steps: int,
    tick_hz: float,
    control_hz: float,
    action_frames: int,
    max_sequences: int | None = None,
    load_images: bool = True,
) -> GrandTourSequenceDataset:
    topics = resolve_topics(observation, platform)
    _, mission_dirs = _mission_dirs(data_root, data_config.get("missions"))
    readers = [
        MissionReader(
            mission_dir,
            depth_topic=topics["depth"],
            proprio_topic=topics["proprio"],
            actuator_topic=topics["actuator"],
            max_gap_s=data_config.get("max_gap_ms", 50) / 1000.0,
        )
        for mission_dir in mission_dirs
    ]
    return GrandTourSequenceDataset(
        readers,
        context_steps=context_steps,
        rollout_steps=rollout_steps,
        tick_hz=tick_hz,
        control_hz=control_hz,
        action_frames=action_frames,
        max_sequences=max_sequences,
        load_images=load_images,
    )


def verify_joint_order_consistency(mission_root, n_samples: int = 5, atol: float = 1e-3) -> bool:
    """Check that anymal_state_state_estimator.joint_positions follows the
    same joint order as anymal_state_actuator (confirmed: JOINT_ORDER above).

    state_estimator's own topic description does not restate this ordering --
    run this once per mission before trusting it, rather than assuming it
    matches the actuator topic's documented order.
    """
    est = mission_root["anymal_state_state_estimator"]
    n = len(est["timestamp"])
    for idx in np.linspace(0, n - 1, num=min(n_samples, n), dtype=int):
        est_positions = np.asarray(est["joint_positions"][idx])
        for j in range(len(JOINT_ORDER)):
            actuator_val = mission_root["anymal_state_actuator"][f"{j:02d}_state_joint_position"][idx]
            if not np.isclose(est_positions[j], actuator_val, atol=atol):
                return False
    return True
