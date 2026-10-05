"""CPU end-to-end `quadwm eval` on synthetic GrandTour-format missions, for both model types."""

import json

import numpy as np
import pytest
import torch

zarr = pytest.importorskip("zarr")
imageio = pytest.importorskip("imageio.v2")

from quadwm.data.grandtour import FEET
from quadwm.models import build_model, save_checkpoint

DEPTH, RGB = "depth_camera_front_upper", "alphasense_front_center"


def _write_mission(mission, seconds=30.0, seed=0):
    rng = np.random.default_rng(seed)
    root = zarr.open_group(store=mission / "data", mode="w")
    for topic, offset, ext in ((DEPTH, 0.0, "png"), (RGB, 0.03, "jpeg")):
        times = np.arange(offset, seconds, 0.1)
        root.create_group(topic).create_array("timestamp", data=times)
        folder = mission / "images" / topic
        folder.mkdir(parents=True)
        for index in range(len(times)):
            if ext == "png":
                image = rng.integers(500, 5000, (32, 32)).astype(np.uint16)  # millimeters
            else:
                image = rng.integers(0, 255, (32, 32, 3)).astype(np.uint8)
            imageio.imwrite(folder / f"{index:06d}.{ext}", image)
    times = np.arange(0.0, seconds, 0.02)
    state = root.create_group("anymal_state_state_estimator")
    state.create_array("timestamp", data=times)
    state.create_array("pose_pos", data=np.stack([times, np.zeros_like(times), np.zeros_like(times)], 1))
    state.create_array("twist_lin", data=np.tile([1.0, 0.0, 0.0], (len(times), 1)))
    state.create_array("twist_ang", data=rng.normal(size=(len(times), 3)))
    state.create_array("pose_orien", data=np.tile([0.0, 0.0, 0.0, 1.0], (len(times), 1)))
    state.create_array("joint_positions", data=np.sin(times[:, None] * 7.85 + np.arange(12)))
    state.create_array("joint_velocities", data=np.cos(times[:, None] * 7.85 + np.arange(12)))
    for phase, foot in enumerate(FEET):
        state.create_array(f"{foot}_FOOT_contact", data=(((times + 0.4 * (phase % 2)) % 0.8) < 0.5).astype(float))
    actuator = root.create_group("anymal_state_actuator")
    actuator.create_array("timestamp", data=times)
    for joint in range(12):
        actuator.create_array(f"{joint:02d}_command_position", data=np.sin(times * 7.85 + joint))


class _FakeEncoder(torch.nn.Module):
    """Stands in for V-JEPA 2.1: [B, 3, H, W] -> [B, 4, 8] tokens."""

    def __init__(self, _checkpoint_root):
        super().__init__()
        self.proj = torch.nn.Linear(3, 8)

    @torch.no_grad()
    def forward(self, images):
        return self.proj(images.flatten(2).transpose(1, 2)[:, :4])


def _run(tmp_path, monkeypatch, model_cfg, observation, extra=None):
    from quadwm.evaluation import common, protocol

    monkeypatch.setattr(common, "VJEPA21Encoder", _FakeEncoder)
    data_root = tmp_path / "grandtour"
    missions = {"train": ["m-train"], "eval": ["m-eval"], "probe": ["m-probe"]}
    for seed, name in enumerate(["m-train", "m-eval", "m-probe"]):
        if not (data_root / name).exists():
            _write_mission(data_root / name, seed=seed)
    data_cfg = {"data_root": str(data_root), "observation": observation, "platform": "anymal_d",
                "tick_hz": 5, "control_hz": 50, "action_frames": 10, "max_gap_ms": 50, "depth_size": 16,
                "depth_range": [0.2, 10.0], "normalize": True,
                "normalization": {"proprio_mean": [0.0] * 33, "proprio_std": [1.0] * 33,
                                  "action_mean": [0.0] * 120, "action_std": [1.0] * 120}}
    config = {"name": f"run-{observation}", "seed": 7, "run_root": str(tmp_path / "runs"),
              "checkpoint_root": str(tmp_path / "ckpt"), "data": data_cfg, "model": model_cfg,
              "wandb": {"group": f"group-{observation}"}, **(extra or {})}
    run_root = tmp_path / "runs" / config["name"]
    run_root.mkdir(parents=True)
    (run_root / "splits.json").write_text(json.dumps(missions), encoding="utf-8")
    model = build_model(model_cfg, tmp_path / "ckpt")
    save_checkpoint(run_root / "last.pt", model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
                    epoch=2, config=config)
    eval_config = {
        "name": "e2e", "seed": 1, "horizons": [1, 3], "context_frames": 4,
        "windows": {"anchor_stride_s": 5.0, "anchor_tolerance_s": 0.1, "max_per_mission": 4,
                    "match_observations": ["rgb_plus_proprioception", "depth_plus_proprioception"]},
        "batch_size": 3, "num_workers": 0, "precision": "bf16", "sigma_floor": 1e-3,
        "probe": {"kinds": ["linear", "mlp"], "hidden_dim": 256, "steps": 5, "batch_size": 32,
                  "learning_rate": 1e-3, "weight_decay": 0.0},
        "gait": {"min_speed": 0.2, "min_segment_s": 4.0, "period_range_s": [0.3, 2.0]},
        "compute": {"enabled": True, "batch_size": 2, "warmup": 1, "repeats": 1},
    }
    result = protocol.evaluate_checkpoint(config, eval_config)
    written = json.loads((run_root / "eval" / "last-e2e.json").read_text(encoding="utf-8"))
    return result, written


def _check(result, written, latent_dim):
    assert written["model"]["latent_dim"] == latent_dim
    assert written["data"]["windows"] == {"probe": 4, "eval": 4}
    for kind in ("linear", "mlp"):
        curve = written["probes"][kind]["eps_k"]
        assert set(curve) == {"1", "3"}
        for k in ("1", "3"):
            assert np.isfinite(curve[k]["model"]["all"])
            assert set(curve[k]["per_mission_all"]) == {"m-eval"}
        assert written["probes"][kind]["quality"]["joint_pos"]["r2"] is not None
    assert written["gait_cycle"]["seconds"] == pytest.approx(0.8, abs=0.05)
    assert written["gait_cycle"]["ticks"] == 4
    assert written["compute"]["rollout_throughput_fps"] > 0
    assert "EV1_sim_EV2_EV3_EV4_EV6" in written["not_run"]
    assert result["model"]["epoch"] == 2


def test_eval_end_to_end_lewm(tmp_path, monkeypatch):
    model_cfg = {"type": "lewm", "image_size": 16, "patch_size": 8, "encoder_dim": 16, "encoder_depth": 1,
                 "encoder_heads": 2, "latent_dim": 8, "predictor_dim": 16, "predictor_depth": 1,
                 "predictor_heads": 2, "predictor_window": 2, "proprio_dim": 33, "action_dim": 120,
                 "context_steps": 2, "rollout_steps": 1, "sigreg_slices": 8,
                 "loss_weights": {"state": 1.0}}
    result, written = _run(tmp_path, monkeypatch, model_cfg, "depth_plus_proprioception")
    _check(result, written, latent_dim=8)
    assert written["compute"]["parameters_M"]["training_only"] > 0  # the PSG state head


def test_eval_end_to_end_baseline_ignores_token_cache(tmp_path, monkeypatch):
    model_cfg = {"image_size": 32, "visual_dim": 8, "tokens_per_frame": 4, "proprio_dim": 33,
                 "proprio_embed_dim": 4, "action_dim": 120, "context_steps": 3, "rollout_context": 2,
                 "predictor_depth": 1, "predictor_heads": 2}
    extra = {"cache": {"mode": "auto", "root": str(tmp_path / "cache"), "batch_size": 4}}
    result, written = _run(tmp_path, monkeypatch, model_cfg, "rgb_plus_proprioception", extra)
    _check(result, written, latent_dim=8 + 4)
    assert not (tmp_path / "cache" / "m-eval.json").is_file()  # eval always encodes on the fly
    assert written["compute"]["single_step_latency_ms"] > 0
