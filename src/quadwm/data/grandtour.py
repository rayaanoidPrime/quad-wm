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
import fnmatch
import random
import re
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio  # v2 API avoids the v3-migration deprecation warning
import numpy as np
import torch
import zarr
from huggingface_hub import list_repo_files, scan_cache_dir, snapshot_download
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
        if f.is_file() and (pattern is None or pattern.match(f.as_posix()))
    ]

    def is_tar(path: Path) -> bool:
        return path.name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2"))

    for f in [x for x in files if is_tar(x)]:
        dest = dest_dir / f.relative_to(cache_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(f, "r:*") as tar:
            members = tar.getmembers()
            relative_archive = f.relative_to(cache_dir)
            mission_name = relative_archive.parts[0]
            mission_prefix = mission_name + "/"
            archive_parent = "/".join(relative_archive.parts[1:-1])
            archive_has_mission_prefix = any(
                member.name == mission_name
                or member.name.startswith(mission_prefix)
                for member in members
            )
            archive_has_parent_prefix = archive_parent and any(
                member.name == archive_parent
                or member.name.startswith(archive_parent + "/")
                for member in members
            )
            if archive_has_mission_prefix:
                extract_root = dest_dir
            elif archive_has_parent_prefix:
                extract_root = dest_dir / mission_name
            else:
                extract_root = dest.parent
            tar.extractall(path=extract_root)

    for f in [x for x in files if not is_tar(x) and x.is_file()]:
        dest = dest_dir / f.relative_to(cache_dir)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)


def _mission_names(missions: list[str] | str | None) -> list[str]:
    if missions is None:
        return []
    if isinstance(missions, str):
        return [mission.strip() for mission in missions.split(",") if mission.strip()]
    return list(missions)


@dataclass(frozen=True)
class MissionReport:
    mission: str
    image_counts: dict[str, int]
    has_data_group: bool
    missing_required_topics: list[str]


def inspect_mission(mission: str | Path) -> MissionReport:
    mission = Path(mission)
    image_counts = {
        topic.name: sum(path.is_file() for path in topic.iterdir())
        for topic in (mission / "images").iterdir()
        if topic.is_dir()
    } if (mission / "images").is_dir() else {}
    present_topics = {
        path.stem for path in (mission / "metadata").glob("*.yaml")
    } if (mission / "metadata").is_dir() else set()
    required_topics = {
        "depth_camera_front_upper",
        "anymal_state_actuator",
        "anymal_state_odometry",
        "anymal_state_state_estimator",
    }
    return MissionReport(
        mission=mission.name,
        image_counts=image_counts,
        has_data_group=(mission / "data").is_dir(),
        missing_required_topics=sorted(required_topics - present_topics),
    )


def inspect_root(root: str | Path, required_topics: list[str] | None = None) -> dict:
    root = Path(root)
    missions = []
    for mission in sorted(path for path in root.iterdir() if path.is_dir()):
        report = inspect_mission(mission)
        item = {
            "mission": report.mission,
            "image_counts": report.image_counts,
            "has_data_group": report.has_data_group,
        }
        if required_topics is not None:
            item["missing_required_topics"] = sorted(
                set(required_topics) & set(report.missing_required_topics)
            )
        missions.append(item)
    return {"mission_count": len(missions), "missions": missions}


def materialize_mission(mission: str | Path) -> int:
    mission = Path(mission)
    archives = sorted(mission.glob("*.tar"))
    for archive in archives:
        with tarfile.open(archive, "r") as tar:
            members = tar.getmembers()
            prefix = mission.name + "/"
            has_prefix = any(
                member.name == mission.name or member.name.startswith(prefix)
                for member in members
            )
            tar.extractall(path=mission.parent if has_prefix else mission)
        archive.unlink()
    return len(archives)


def split_mission_names(names: list[str], seed: int, eval_fraction: float = 0.2) -> tuple[list[str], list[str]]:
    shuffled = sorted(names)
    random.Random(seed).shuffle(shuffled)
    eval_count = max(1, round(len(shuffled) * eval_fraction)) if shuffled else 0
    return shuffled[eval_count:], shuffled[:eval_count]


# GrandTour publishes Zarr v2 stores: topics are sub-groups directly under
# <mission>/data/, but the data root itself has no `.zgroup` marker (an
# implicit v2 group). Zarr 3 requires explicit metadata and refuses to open
# it, so materialize the missing marker before opening.
_IMPLICIT_GROUP_MARKER = '{"zarr_format": 2}\n'


def _open_mission_data(mission: Path):
    """Open a mission's `data/` root, bridging the implicit v2 group gap."""
    data_dir = mission / "data"
    if not (data_dir / ".zgroup").exists() and not (data_dir / "zarr.json").exists():
        (data_dir / ".zgroup").write_text(_IMPLICIT_GROUP_MARKER, encoding="utf-8")
    return zarr.open_group(store=data_dir, mode="r")


def _nearest_indices(
    timestamps: np.ndarray, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest index + absolute gap for many query times (vectorized)."""
    after = np.clip(np.searchsorted(timestamps, times, side="left"), 0, len(timestamps) - 1)
    before = np.clip(after - 1, 0, len(timestamps) - 1)
    choose_before = np.abs(timestamps[before] - times) <= np.abs(timestamps[after] - times)
    index = np.where(choose_before, before, after)
    return index, np.abs(timestamps[index] - times)


def _remote_missions(
    remote_files: list[str], download_topics: list[str] | None
) -> list[str]:
    """Timestamp-named missions that actually contain downloadable topics.

    The repo also stores map-only folders such as
    ``<timestamp>/point_cloud_maps/<timestamp>_dlio.ply``.  Those match the
    timestamp regex but have no training data, so require at least one file
    matching the download patterns (or a ``data/`` subtree when no topics are
    configured).
    """
    missions: set[str] = set()
    for path in remote_files:
        if "/" not in path:
            continue
        mission, relative = path.split("/", 1)
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", mission):
            continue
        if download_topics:
            wanted = any(fnmatch.fnmatch(relative, f"*{topic}*") for topic in download_topics)
        else:
            wanted = relative.startswith("data/")
        if wanted:
            missions.add(mission)
    return sorted(missions)


def _mission_ready(mission: Path, topics: list[str] | None) -> bool:
    if not (mission / "data").is_dir():
        return False
    if not topics:
        return True
    try:
        root = _open_mission_data(mission)
        if any(topic not in root for topic in topics):
            return False
    except (KeyError, OSError, ValueError):
        return False
    camera_topics = tuple(
        topic for topic in topics if topic.startswith(("alpha", "hdr", "depth", "zed"))
    )
    return all((mission / "images" / topic).is_dir() for topic in camera_topics)


def _clear_download_cache() -> None:
    """Delete the GrandTour archives from the Hugging Face cache after extraction.

    ``snapshot_download`` stores the tarballs in the shared ``hub/blobs``
    directory, referenced through the repo's snapshot.  Extracted missions live
    under ``data_root``, so once every pending mission is materialized the
    archives are redundant.  Use the HF cache API so deleting the revision also
    drops the blobs it references, instead of orphaning tens of GiB.
    """
    try:
        cache = scan_cache_dir()
        for repo in cache.repos:
            if repo.repo_id == GRANDTOUR_REPO_ID and repo.repo_type == "dataset":
                cache.delete_revisions(
                    *(revision.commit_hash for revision in repo.revisions)
                ).execute()
                return
    except Exception as exc:  # pragma: no cover - cleanup must not fail the run
        print(f"warning: could not clear GrandTour download cache: {exc}", flush=True)


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
        selected = _remote_missions(remote_files, download_topics)
    if not selected:
        raise ValueError("GrandTour download selection contains no missions")
    root.mkdir(parents=True, exist_ok=True)
    pending = [m for m in selected if not _mission_ready(root / m, download_topics)]
    if not pending:
        # Everything is already materialized, so any archive cache left over
        # from an earlier download is stale -- drop it here too.
        _clear_download_cache()
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
    incomplete = [m for m in pending if not _mission_ready(root / m, download_topics)]
    if incomplete:
        raise RuntimeError(
            "GrandTour extraction did not produce the required layout for: "
            + ", ".join(incomplete)
        )
    _clear_download_cache()


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


def materialized_missions(
    data_root: str | Path, missions: list[str] | str | None = None
) -> list[str]:
    """Names of materialized missions under ``data_root`` (for train/eval splits)."""
    _, mission_dirs = _mission_dirs(data_root, missions)
    return [path.name for path in mission_dirs]


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
        self.root = _open_mission_data(self.mission_dir)
        self.depth_group = self.root[self.depth_topic]
        self.proprio_group = self.root[self.proprio_topic]
        self.actuator_group = self.root[self.actuator_topic]

        self.depth_timestamps = np.asarray(self.depth_group["timestamp"][:])
        self.proprio_timestamps = np.asarray(self.proprio_group["timestamp"][:])
        self.actuator_timestamps = np.asarray(self.actuator_group["timestamp"][:])

        # Preload the state/action arrays once.  Indexing used to read each
        # element straight out of Zarr per candidate sequence, which is orders
        # of magnitude slower than numpy indexing.
        proprio = self.proprio_group
        self.proprio = {
            "pose_pos": np.asarray(proprio["pose_pos"][:], dtype=np.float32),
            "twist_lin": np.asarray(proprio["twist_lin"][:], dtype=np.float32),
            "twist_ang": np.asarray(proprio["twist_ang"][:], dtype=np.float32),
            "pose_orien": np.asarray(proprio["pose_orien"][:], dtype=np.float32),
            "joint_positions": np.asarray(proprio["joint_positions"][:], dtype=np.float32),
            "joint_velocities": np.asarray(proprio["joint_velocities"][:], dtype=np.float32),
            "contacts": np.stack(
                [
                    np.asarray(proprio[f"{foot}_FOOT_contact"][:], dtype=np.float32)
                    for foot in FEET
                ],
                axis=1,
            ),
        }
        actuator = self.actuator_group
        self.actions = np.stack(
            [
                np.asarray(actuator[f"{j:02d}_command_position"][:], dtype=np.float32)
                for j in range(len(JOINT_ORDER))
            ],
            axis=1,
        )

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
        """Index of the timestamp closest to ``t`` (ties go to the earlier one)."""
        after = min(max(bisect.bisect_left(timestamps, t), 0), len(timestamps) - 1)
        before = max(after - 1, 0)
        if abs(float(timestamps[before]) - t) <= abs(float(timestamps[after]) - t):
            return before, abs(float(timestamps[before]) - t)
        return after, abs(float(timestamps[after]) - t)

    def sample_state(self, t: float) -> dict | None:
        p_idx, p_gap = self._nearest(self.proprio_timestamps, t)
        if p_gap > self.max_gap_s:
            return None  # likely a dropped message, not a valid interpolation window
        action = self.sample_action(t)
        if action is None:
            return None

        state = self.proprio
        quat = state["pose_orien"][p_idx]  # (x, y, z, w)
        return {
            "pose_pos": state["pose_pos"][p_idx],
            "lin_vel": state["twist_lin"][p_idx],
            "ang_vel": state["twist_ang"][p_idx],
            "gravity": project_gravity(quat).astype(np.float32),
            "joint_pos": state["joint_positions"][p_idx],
            "joint_vel": state["joint_velocities"][p_idx],
            "contacts": state["contacts"][p_idx],
            "action": action,
        }

    def sample_action(self, t: float) -> np.ndarray | None:
        a_idx, a_gap = self._nearest(self.actuator_timestamps, t)
        if a_gap > self.max_gap_s:
            return None
        return self.actions[a_idx]

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
            if len(reader) == 0:
                continue
            source_dt = (
                float(np.median(np.diff(reader.depth_timestamps)))
                if len(reader) > 1
                else self.tick_dt
            )
            start_stride = max(1, round(self.tick_dt / source_dt))
            start_ids = np.arange(0, len(reader.depth_timestamps), start_stride)
            start_times = reader.depth_timestamps[start_ids].astype(np.float64)
            offsets = np.arange(self.total_steps, dtype=np.float64) * self.tick_dt
            times = start_times[:, None] + offsets[None, :]

            ids, error = _nearest_indices(reader.depth_timestamps, times)
            valid = (error <= max_tick_error_s).all(axis=1)
            valid &= (np.diff(ids, axis=1) > 0).all(axis=1)

            # sample_state validity is timestamp-only: proprio and actuator gaps.
            id_times = reader.depth_timestamps[ids]
            _, proprio_gap = _nearest_indices(reader.proprio_timestamps, id_times)
            _, actuator_gap = _nearest_indices(reader.actuator_timestamps, id_times)
            valid &= (proprio_gap <= reader.max_gap_s).all(axis=1)
            valid &= (actuator_gap <= reader.max_gap_s).all(axis=1)

            # Action windows for all but the final tick.
            window_offsets = np.arange(self.action_frames, dtype=np.float64) / self.control_hz
            window_times = id_times[:, :-1, None] + window_offsets[None, None, :]
            _, window_gap = _nearest_indices(reader.actuator_timestamps, window_times)
            valid &= (window_gap <= reader.max_gap_s).all(axis=(1, 2))

            dropped += int((~valid).sum())
            kept = 0
            for row in ids[valid]:
                self.index.append((reader_index, tuple(int(image_id) for image_id in row)))
                kept += 1
                if max_sequences is not None and kept >= max_sequences:
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
    missions: list[str] | str | None = None,
) -> GrandTourSequenceDataset:
    topics = resolve_topics(observation, platform)
    selected = missions if missions is not None else data_config.get("missions")
    _, mission_dirs = _mission_dirs(data_root, selected)
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
