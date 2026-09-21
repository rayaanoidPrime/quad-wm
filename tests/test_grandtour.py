import tarfile

import numpy as np
import pytest

from quadwm.data.grandtour import (
    inspect_mission,
    inspect_root,
    materialize_mission,
    split_mission_names,
)


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


def test_track1_dataset_synchronizes_to_depth_clock(tmp_path):
    zarr = pytest.importorskip("zarr")
    from quadwm.data.grandtour import Track1MissionDataset

    mission = tmp_path / "mission-a"
    data = zarr.open_group(store=mission / "data", mode="w")
    images = mission / "images" / "depth_camera_front_upper"
    images.mkdir(parents=True)
    for sequence_id in range(3):
        (images / f"{sequence_id:06d}.png").write_bytes(b"not-a-real-image")

    def array(group, name, values):
        group.create_array(name, data=np.asarray(values), overwrite=True)

    depth = data.create_group("depth_camera_front_upper")
    array(depth, "timestamp", [0.0, 0.1, 0.2])
    array(depth, "sequence_id", [0, 1, 2])

    state = data.create_group("anymal_state_state_estimator")
    array(state, "timestamp", [0.0, 0.1, 0.2])
    array(state, "pose_pos", np.zeros((3, 3)))
    array(state, "twist_lin", np.ones((3, 3)))
    array(state, "twist_ang", np.ones((3, 3)))
    array(state, "pose_orien", np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)))
    array(state, "joint_positions", np.zeros((3, 12)))
    array(state, "joint_velocities", np.zeros((3, 12)))
    for foot in ("LF", "RF", "LH", "RH"):
        array(state, f"{foot}_FOOT_contact", [1, 0, 1])

    actuator = data.create_group("anymal_state_actuator")
    array(actuator, "timestamp", [0.0, 0.1, 0.2])
    for index in range(12):
        array(actuator, f"{index:02d}_command_position", np.full(3, index))

    dataset = Track1MissionDataset(mission)

    assert len(dataset) == 3
    assert dataset.dimensions == {
        "proprio": 33,
        "proprio_history": 1,
        "state": 40,
        "action": 12,
    }
    assert dataset[0]["proprio"].shape == (33,)
    assert dataset[0]["proprio_history"].shape == (1, 33)
    assert dataset[0]["state"].shape == (40,)
    assert dataset[0]["action"].shape == (12,)
    np.testing.assert_allclose(dataset[0]["action"], np.arange(12))


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


def test_split_mission_names_is_seeded_and_mission_level():
    first = split_mission_names(["c", "a", "b", "d"], seed=12)
    second = split_mission_names(["d", "b", "a", "c"], seed=12)

    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert set(first[0]) | set(first[1]) == {"a", "b", "c", "d"}
