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
from huggingface_hub import snapshot_download
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


def project_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotate the world gravity direction into the base frame."""
    w, x, y, z = quat_wxyz
    rot = Rotation.from_quat([x, y, z, w])  # scipy wants (x, y, z, w)
    return rot.inv().apply(_GRAVITY_WORLD)


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def _extract_tars(cache_dir: Path, dest_dir: Path, allow_patterns: list[str]) -> None:
    def to_regex(patterns: list[str]) -> re.Pattern:
        parts = [f".*{re.escape(p).replace(r'\*', '.*').replace(r'\?', '.')}$" for p in patterns]
        return re.compile("|".join(parts))

    pattern = to_regex(allow_patterns)
    files = [f for f in Path(cache_dir).rglob("*") if pattern.match(str(f))]

    for f in [x for x in files if x.suffix == ".tar"]:
        dest = dest_dir / f.relative_to(cache_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(f, "r") as tar:
            tar.extractall(path=dest.parent)

    for f in [x for x in files if x.suffix != ".tar" and x.is_file()]:
        dest = dest_dir / f.relative_to(cache_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)


def fetch_missions(missions: list[str], data_root: str | Path) -> None:
    """Idempotent: skips a mission entirely if its `data/` folder already
    exists on disk. Safe to re-run after an interrupted download -- but note
    this is a coarse completeness check: a mission that was only *partially*
    extracted before an interruption (data/ exists, some topics missing) will
    NOT be re-fetched. Delete that mission's directory to force a redo.
    """
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    pending = [m for m in missions if not (root / m / "data").exists()]
    if not pending:
        return
    allow_patterns = [f"{m}/*" for m in pending]
    cache_path = snapshot_download(
        repo_id=GRANDTOUR_REPO_ID, allow_patterns=allow_patterns, repo_type="dataset"
    )
    _extract_tars(Path(cache_path), root, allow_patterns)


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

    @staticmethod
    def _nearest(timestamps: np.ndarray, t: float) -> tuple[int, float]:
        idx = bisect.bisect_left(timestamps, t)
        idx = min(max(idx, 0), len(timestamps) - 1)
        return idx, abs(float(timestamps[idx]) - t)

    def sample_state(self, t: float) -> dict | None:
        p_idx, p_gap = self._nearest(self.proprio_timestamps, t)
        a_idx, a_gap = self._nearest(self.actuator_timestamps, t)
        if p_gap > self.max_gap_s or a_gap > self.max_gap_s:
            return None  # likely a dropped message, not a valid interpolation window

        g = self.proprio_group
        quat = np.asarray(g["pose_orien"][p_idx], dtype=np.float32)  # (w, x, y, z)
        contacts = np.array(
            [g[f"{foot}_FOOT_contact"][p_idx] for foot in FEET], dtype=np.float32
        )
        # "Action" = commanded joint position, per JOINT_ORDER, from the actuator topic.
        action = np.array(
            [self.actuator_group[f"{j:02d}_command_position"][a_idx] for j in range(len(JOINT_ORDER))],
            dtype=np.float32,
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


def build_dataset(
    data_config: dict,
    observation: str,
    platform: str,
    data_root: str | Path,
    horizon: int = 4,
) -> GrandTourPairDataset:
    topics = resolve_topics(observation, platform)
    missions = data_config.get("missions")
    if missions is None:
        # "split" (train/val/test) resolution needs a manifest mapping split
        # name -> mission list; not built yet. Fail loudly rather than guess.
        raise NotImplementedError(
            "data_config['split'] lookup isn't wired up yet -- "
            "pass an explicit 'missions' list in the data config for now."
        )

    max_gap_s = data_config.get("max_gap_ms", 50) / 1000.0
    rate_hz = data_config.get("sync_rate_hz", 15.0)
    root = Path(data_root)
    readers = [
        MissionReader(
            root / mission,
            depth_topic=topics["depth"],
            proprio_topic=topics["proprio"],
            actuator_topic=topics["actuator"],
            max_gap_s=max_gap_s,
        )
        for mission in missions
    ]
    return GrandTourPairDataset(readers, horizon=horizon, rate_hz=rate_hz)


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