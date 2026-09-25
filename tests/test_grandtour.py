import tarfile

import numpy as np
import pytest

from quadwm.data.grandtour import (
    inspect_mission,
    inspect_root,
    materialize_mission,
    split_mission_names,
)


def test_nearest_picks_closer_frame_not_next():
    from quadwm.data.grandtour import MissionReader, _nearest_indices

    # Depth frames drift slightly below 0.1s, so a 0.2s tick lands just past
    # frame 2; bisect_left alone would return frame 3 with a ~0.1s error.
    timestamps = np.array([0.0, 0.099999, 0.199998, 0.299997])
    index, gap = MissionReader._nearest(timestamps, 0.2)
    assert index == 2
    assert gap < 1e-3

    times = np.array([0.0, 0.05, 0.2, 0.299997, 10.0])
    vector_index, vector_gap = _nearest_indices(timestamps, times)
    for position, t in enumerate(times):
        scalar_index, scalar_gap = MissionReader._nearest(timestamps, float(t))
        assert vector_index[position] == scalar_index
        assert abs(vector_gap[position] - scalar_gap) < 1e-9


def test_clear_download_cache_passes_commit_hashes(tmp_path, monkeypatch):
    from quadwm.data import grandtour

    class _Revision:
        def __init__(self, commit_hash):
            self.commit_hash = commit_hash

    class _Repo:
        repo_id = grandtour.GRANDTOUR_REPO_ID
        repo_type = "dataset"
        revisions = {_Revision("abc"), _Revision("def")}

    class _Cache:
        repos = [_Repo()]
        deleted = None

        def delete_revisions(self, *hashes):
            self.deleted = hashes
            return self

        def execute(self):
            pass

    cache = _Cache()
    monkeypatch.setattr(grandtour, "scan_cache_dir", lambda: cache)

    grandtour._clear_download_cache()

    # delete_revisions needs the hash strings, not CachedRevisionInfo objects.
    assert set(cache.deleted) == {"abc", "def"}


def test_fetch_missions_clears_cache_only_after_complete_extraction(tmp_path, monkeypatch):
    from quadwm.data import grandtour

    calls = []
    monkeypatch.setattr(grandtour, "_clear_download_cache", lambda: calls.append(True))
    monkeypatch.setattr(grandtour, "snapshot_download", lambda **kwargs: str(tmp_path))
    monkeypatch.setattr(grandtour, "_extract_tars", lambda *args, **kwargs: None)

    # _mission_ready: first call says "pending", second says "materialized".
    ready = iter([False, True])
    monkeypatch.setattr(grandtour, "_mission_ready", lambda *args, **kwargs: next(ready, True))
    grandtour.fetch_missions(["mission-a"], tmp_path)
    assert calls == [True]

    # Already materialized (nothing pending): stale archives still get cleared.
    calls.clear()
    monkeypatch.setattr(grandtour, "_mission_ready", lambda *args, **kwargs: True)
    grandtour.fetch_missions(["mission-a"], tmp_path)
    assert calls == [True]

    # If extraction is still incomplete, keep the archives for resuming.
    calls.clear()
    ready = iter([False, False])
    monkeypatch.setattr(grandtour, "_mission_ready", lambda *args, **kwargs: next(ready, False))
    with pytest.raises(RuntimeError):
        grandtour.fetch_missions(["mission-a"], tmp_path)
    assert calls == []


def test_remote_missions_skips_map_only_folders():
    from quadwm.data.grandtour import _remote_missions

    files = [
        "2024-11-02-17-18-32/data/alphasense_front_center.tar",
        "2024-11-02-17-18-32/metadata/alphasense_front_center.yaml",
        # Timestamp-named folder that only holds a point-cloud map.
        "2024-10-29-09-53-44/point_cloud_maps/2024-10-29-09-53-44_dlio.ply",
        "README.md",
    ]

    assert _remote_missions(files, ["alphasense_front_center"]) == ["2024-11-02-17-18-32"]
    assert _remote_missions(files, None) == ["2024-11-02-17-18-32"]


def test_inspect_mission_reports_topics_and_images(tmp_path):
    mission = tmp_path / "2024-11-02-17-18-32"
    metadata = mission / "metadata"
    images = mission / "images" / "depth_camera_front_upper"
    metadata.mkdir(parents=True)
    images.mkdir(parents=True)
    (mission / "data").mkdir()

    (metadata / "depth_camera_front_upper.yaml").write_text(
        "topic: depth_camera_front_upper\n", encoding="utf-8"
    )
    (metadata / "anymal_state_actuator.yaml").write_text(
        "topic: anymal_state_actuator\n", encoding="utf-8"
    )
    (images / "000001.png").write_bytes(b"not-a-real-image")
    (images / "000002.jpeg").write_bytes(b"not-a-real-image")

    report = inspect_mission(mission)

    assert report.mission == "2024-11-02-17-18-32"
    assert report.image_counts["depth_camera_front_upper"] == 2
    assert report.has_data_group is True
    assert "anymal_state_odometry" in report.missing_required_topics


def test_inspect_root_is_mission_level(tmp_path):
    for name in ("mission-a", "mission-b"):
        (tmp_path / name / "metadata").mkdir(parents=True)

    report = inspect_root(tmp_path, required_topics=[])

    assert report["mission_count"] == 2
    assert {item["mission"] for item in report["missions"]} == {"mission-a", "mission-b"}


def _write_minimal_mission(mission, timestamps):
    import zarr

    root = zarr.open_group(store=mission / "data", mode="w")
    camera = root.create_group("alphasense_front_center")
    camera.create_array("timestamp", data=timestamps)
    state = root.create_group("anymal_state_state_estimator")
    state.create_array("timestamp", data=timestamps)
    state.create_array("pose_pos", data=np.zeros((len(timestamps), 3)))
    state.create_array("twist_lin", data=np.zeros((len(timestamps), 3)))
    state.create_array("twist_ang", data=np.zeros((len(timestamps), 3)))
    state.create_array("pose_orien", data=np.tile([0.0, 0.0, 0.0, 1.0], (len(timestamps), 1)))
    state.create_array("joint_positions", data=np.zeros((len(timestamps), 12)))
    state.create_array("joint_velocities", data=np.zeros((len(timestamps), 12)))
    for foot in ("LF", "RF", "LH", "RH"):
        state.create_array(f"{foot}_FOOT_contact", data=np.zeros(len(timestamps)))
    actuator = root.create_group("anymal_state_actuator")
    actuator.create_array("timestamp", data=timestamps)
    for index in range(12):
        actuator.create_array(f"{index:02d}_command_position", data=np.zeros(len(timestamps)))


def test_sequence_max_sequences_applies_per_mission(tmp_path):
    """A global cap would starve later eval missions and leave them uncacheable."""
    pytest.importorskip("zarr")
    from quadwm.data.grandtour import build_sequence_dataset

    names = ["mission-a", "mission-b"]
    for name in names:
        _write_minimal_mission(tmp_path / name, np.array([0.0, 0.2]))

    dataset = build_sequence_dataset(
        {"missions": names},
        observation="rgb_plus_proprioception",
        platform="anymal_d",
        data_root=tmp_path,
        context_steps=1,
        rollout_steps=1,
        tick_hz=5.0,
        control_hz=50.0,
        action_frames=1,
        max_sequences=1,
        load_images=False,
        missions=names,
    )

    assert len(dataset) == 2
    assert sorted(reader_index for reader_index, _ in dataset.index) == [0, 1]


def test_materialize_mission_extracts_topic_archive(tmp_path):
    mission = tmp_path / "mission-a"
    mission.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("ok", encoding="utf-8")
    archive = mission / "topic.tar"
    with tarfile.open(archive, "w") as stream:
        stream.add(source, arcname="mission-a/metadata/topic.yaml")

    assert materialize_mission(mission) == 1
    assert (mission / "metadata" / "topic.yaml").read_text(encoding="utf-8") == "ok"
    assert not archive.exists()


def test_extract_tars_handles_grandtour_data_prefix(tmp_path):
    from quadwm.data.grandtour import _extract_tars

    cache = tmp_path / "cache"
    mission_data = cache / "mission-a" / "data"
    mission_data.mkdir(parents=True)
    archive = mission_data / "topic.tar"
    source = tmp_path / "source.txt"
    source.write_text("ok", encoding="utf-8")
    stream = tarfile.open(archive, "w")
    stream.add(source, arcname="data/topic.txt")
    stream.close()

    output = tmp_path / "output"
    _extract_tars(cache, output, ["mission-a/data/topic.tar"])

    assert (output / "mission-a" / "data" / "topic.txt").read_text() == "ok"


def test_mission_ready_handles_implicit_v2_group(tmp_path):
    """GrandTour's data root is an implicit Zarr v2 group: topics are
    sub-groups but there is no `.zgroup` marker at `data/`. Zarr 3 refuses to
    open that, so readiness must materialize the marker instead of failing."""
    zarr = pytest.importorskip("zarr")
    from quadwm.data.grandtour import _mission_ready

    mission = tmp_path / "mission-a"
    root = zarr.open_group(store=mission / "data", mode="w", zarr_format=2)
    root.create_group("alphasense_front_center").create_array(
        "timestamp", data=np.array([0.0])
    )
    (mission / "images" / "alphasense_front_center").mkdir(parents=True)
    # Strip the marker to reproduce the published implicit-group layout.
    (mission / "data" / ".zgroup").unlink()

    assert _mission_ready(mission, ["alphasense_front_center"]) is True
    assert (mission / "data" / ".zgroup").exists()


def test_split_mission_names_is_seeded_and_mission_level():
    first = split_mission_names(["c", "a", "b", "d"], seed=12)
    second = split_mission_names(["d", "b", "a", "c"], seed=12)

    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert set(first[0]) | set(first[1]) == {"a", "b", "c", "d"}


def test_sequence_builder_accepts_single_mission_root(tmp_path):
    zarr = pytest.importorskip("zarr")
    from quadwm.data.grandtour import build_sequence_dataset

    mission = tmp_path / "mission-a"
    root = zarr.open_group(store=mission / "data", mode="w")
    timestamps = np.array([0.0, 0.2])
    camera = root.create_group("alphasense_front_center")
    camera.create_array("timestamp", data=timestamps)
    state = root.create_group("anymal_state_state_estimator")
    state.create_array("timestamp", data=timestamps)
    state.create_array("pose_pos", data=np.zeros((2, 3)))
    state.create_array("twist_lin", data=np.zeros((2, 3)))
    state.create_array("twist_ang", data=np.zeros((2, 3)))
    state.create_array("pose_orien", data=np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)))
    state.create_array("joint_positions", data=np.zeros((2, 12)))
    state.create_array("joint_velocities", data=np.zeros((2, 12)))
    for foot in ("LF", "RF", "LH", "RH"):
        state.create_array(f"{foot}_FOOT_contact", data=np.zeros(2))
    actuator = root.create_group("anymal_state_actuator")
    actuator.create_array("timestamp", data=timestamps)
    for index in range(12):
        actuator.create_array(f"{index:02d}_command_position", data=np.zeros(2))

    dataset = build_sequence_dataset(
        {"missions": []},
        observation="rgb_plus_proprioception",
        platform="anymal_d",
        data_root=mission,
        context_steps=1,
        rollout_steps=1,
        tick_hz=5,
        control_hz=50,
        action_frames=1,
        load_images=False,
    )

    assert dataset.readers[0].mission_dir == mission
