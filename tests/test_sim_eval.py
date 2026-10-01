"""Simulated protocol evals: controller, terrain, CEM, episode windows, and the sim-eval pipeline.

The pipeline test drives a tiny numpy simulator through the real
orchestration on CPU; the `sim`-marked test does the same in MuJoCo and is
skipped when MuJoCo or the Menagerie assets are missing (CI).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from quadwm.config import load_config
from quadwm.data.grandtour import STATE_LAYOUT
from quadwm.evaluation.planning import cem, locomotion_cost
from quadwm.models import build_model, save_checkpoint
from quadwm.sim.base import DynamicsSpec, SimState, TerrainSpec
from quadwm.sim.controller import build_controller
from quadwm.sim.episodes import SimWindowDataset, camera_modality, fallen
from quadwm.sim.terrain import terrain_heights

SIM_CONFIG = load_config(Path("configs/sim/mujoco_anymal.yaml"))["sim"]


def _observation(yaw: float = 0.0) -> dict:
    return {"orientation": np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])}


def test_controller_nominal_is_pure_and_act_advances_the_gait_clock():
    controller = build_controller(SIM_CONFIG)
    nominal = controller.nominal(_observation(), ticks=3)
    assert nominal.shape == (3, 120)
    assert np.allclose(controller.nominal(_observation(), ticks=1)[0], nominal[0]), "nominal must not advance time"
    first = controller.act(_observation())
    assert np.allclose(first, nominal[0])
    assert np.allclose(controller.act(_observation()), nominal[1])
    residual = np.full(12, 0.1)
    controller.reset()
    assert np.allclose(controller.act(_observation(), residual), nominal[0] + 0.1)


def test_controller_trots_in_diagonal_pairs_and_steers_toward_heading():
    controller = build_controller(SIM_CONFIG)
    pose = controller.targets(0.1, stride=0.3, turn=0.0).reshape(4, 3)  # legs LF RF LH RH
    default = np.asarray(SIM_CONFIG["default_joint_pos"]).reshape(4, 3)
    hip = pose[:, 1] - default[:, 1]
    assert np.isclose(hip[0], hip[3]) and np.isclose(hip[1], hip[2]) and not np.isclose(hip[0], hip[1])
    # Facing right of the commanded heading: left strides shorten to turn left.
    controller.heading = 0.3
    left = np.abs(controller.nominal(_observation(0.0), 1)[0].reshape(10, 4, 3)[:, 0, 1] - default[0, 1]).max()
    right = np.abs(controller.nominal(_observation(0.0), 1)[0].reshape(10, 4, 3)[:, 1, 1] - default[1, 1]).max()
    assert left < right


def test_unilateral_steps_are_one_sided_and_mirror():
    kwargs = {"length_m": 6.0, "width_m": 2.0, "cell_m": 0.05, "seed": 0}
    heights = terrain_heights(TerrainSpec("unilateral_steps", 0.1), **kwargs)
    mirrored = terrain_heights(TerrainSpec("unilateral_steps", 0.1, mirrored=True), **kwargs)
    rows = heights.shape[0]
    assert heights[: rows // 2].max() == 0.0 and heights[rows // 2 :].max() == pytest.approx(0.1)
    assert np.array_equal(mirrored, heights[::-1])
    assert heights[:, : round(2.0 / 0.05)].max() == 0.0, "start pad stays flat"


def test_cem_finds_the_minimum_of_a_quadratic():
    target = torch.tensor([[0.2, -0.1], [0.05, 0.15]])
    mean = cem(lambda samples: (samples - target).pow(2).sum((1, 2)), (2, 2), population=200, elites=20,
               iterations=15, init_std=0.3, min_std=1e-3, clip=1.0, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(mean, target, atol=0.02)


def test_locomotion_cost_prefers_tracking_and_upright():
    states = torch.zeros(3, 2, 40)
    states[..., STATE_LAYOUT["gravity"].start + 2] = -1.0
    states[0, :, STATE_LAYOUT["lin_vel"].start] = 0.4  # tracks the command
    states[1, :, STATE_LAYOUT["lin_vel"].start] = 0.0
    states[2, :, STATE_LAYOUT["lin_vel"].start] = 0.4
    states[2, :, STATE_LAYOUT["gravity"]] = torch.tensor([0.7, 0.0, -0.7])  # tipping over
    cost = locomotion_cost(states, 0.4, {"velocity": 1.0, "upright": 1.0, "yaw_rate": 0.1})
    assert cost[0] < cost[1] and cost[0] < cost[2]


def _episode(ticks: int, fall_at: int | None = None) -> dict:
    rng = np.random.default_rng(ticks)
    fallen_flags = np.zeros(ticks, dtype=bool)
    if fall_at is not None:
        fallen_flags[fall_at:] = True
    return {
        "pose_pos": np.cumsum(np.tile([0.1, 0.0, 0.0], (ticks, 1)), 0).astype(np.float32),
        "orientation": np.tile([0.0, 0.0, 0.0, 1.0], (ticks, 1)).astype(np.float32),
        "lin_vel": rng.normal(size=(ticks, 3)).astype(np.float32),
        "ang_vel": rng.normal(size=(ticks, 3)).astype(np.float32),
        "gravity": np.tile([0.0, 0.0, -1.0], (ticks, 1)).astype(np.float32),
        "joint_pos": rng.normal(size=(ticks, 12)).astype(np.float32),
        "joint_vel": rng.normal(size=(ticks, 12)).astype(np.float32),
        "contacts": np.ones((ticks, 4), dtype=np.float32),
        "proprio": rng.normal(size=(ticks, 33)).astype(np.float32),
        "depth": rng.uniform(0.5, 5.0, (ticks, 32, 32)).astype(np.float16),
        "rgb": rng.integers(0, 255, (ticks, 32, 32, 3)).astype(np.uint8),
        "actions": rng.normal(size=(ticks - 1, 120)).astype(np.float32),
        "fallen": fallen_flags,
    }


def test_sim_windows_skip_falls_and_match_the_grandtour_layout():
    dataset = SimWindowDataset([_episode(20), _episode(20, fall_at=8)], context_frames=3, steps=2, stride_ticks=2,
                               modality="depth", normalization=None, depth_size=16)
    assert all(start + 5 <= 8 for episode, start in dataset.index if episode == 1)
    item = dataset[0]
    assert item["images"].shape == (5, 2, 16, 16)
    assert item["proprio"].shape == (5, 33) and item["actions"].shape == (4, 120) and item["state"].shape == (5, 40)
    assert np.allclose(item["state"][2, STATE_LAYOUT["base_pos"]], 0.0), "base position anchored at rollout start"
    rgb = SimWindowDataset([_episode(20)], context_frames=3, steps=2, stride_ticks=2, modality="rgb",
                           normalization=None)
    assert rgb[0]["images"].shape == (5, 3, 32, 32)
    assert camera_modality("depth_plus_proprioception") == "depth"
    assert fallen({"pose_pos": np.array([0, 0, 0.1]), "gravity": np.array([0, 0, -1.0])})


class _ToySim:
    """Deterministic numpy stand-in with the Simulator contract; speed follows the hip sweep."""

    instances = 0

    def __init__(self, config):
        _ToySim.instances += 1
        self.modalities = config["camera"]["modalities"]
        self.tick_hz, self.control_hz, self.frames_per_tick = 5.0, 50.0, 10
        self.default = np.asarray(config["default_joint_pos"])
        self.rendering = True

    def reset(self, *, seed, terrain=TerrainSpec(), dynamics=DynamicsSpec()):
        self.state = {"pos": np.array([0.0, 0.0, 0.5]), "vel": 0.0, "joints": self.default.copy(),
                      "mass": dynamics.base_mass_scale, "seed": seed}
        return self._observe()

    def step(self, action):
        frames = np.asarray(action).reshape(10, 12)
        sweep = np.abs(frames[:, 1::3] - self.default[1::3]).mean()
        self.state["vel"] = 0.7 * self.state["vel"] + 0.3 * 1.5 * sweep / self.state["mass"]
        self.state["pos"] = self.state["pos"] + [0.2 * self.state["vel"], 0.0, 0.0]
        self.state["joints"] = frames[-1]
        return self._observe()

    def get_state(self):
        return SimState(np.zeros(1), {"copy": {key: np.copy(value) for key, value in self.state.items()}})

    def set_state(self, state):
        self.state = {key: np.copy(value) for key, value in state.extra["copy"].items()}

    def _observe(self):
        pos, joints = self.state["pos"], np.asarray(self.state["joints"], dtype=np.float32)
        observation = {
            "pose_pos": pos.astype(np.float32), "orientation": np.array([0, 0, 0, 1], np.float32),
            "lin_vel": np.array([self.state["vel"], 0, 0], np.float32), "ang_vel": np.zeros(3, np.float32),
            "gravity": np.array([0, 0, -1], np.float32), "joint_pos": joints, "joint_vel": np.zeros(12, np.float32),
            "contacts": np.ones(4, np.float32), "terrain_height": np.float32(0.0),
        }
        observation["proprio"] = np.concatenate([observation[k] for k in ("lin_vel", "ang_vel", "gravity",
                                                                           "joint_pos", "joint_vel")])
        if self.rendering:
            shade = (pos[0] * 50) % 255
            if "rgb" in self.modalities:
                observation["rgb"] = np.full((32, 32, 3), shade, np.uint8)
            if "depth" in self.modalities:
                observation["depth"] = np.full((32, 32), 1.0 + pos[0] % 3, np.float32)
        return observation


def _tiny_run(tmp_path) -> tuple[dict, Path]:
    model_cfg = {"type": "lewm", "image_size": 16, "patch_size": 8, "encoder_dim": 16, "encoder_depth": 1,
                 "encoder_heads": 2, "latent_dim": 8, "predictor_dim": 16, "predictor_depth": 1,
                 "predictor_heads": 2, "predictor_window": 2, "proprio_dim": 33, "action_dim": 120,
                 "context_steps": 2, "rollout_steps": 1, "sigreg_slices": 8}
    norm = {"proprio_mean": [0.0] * 33, "proprio_std": [1.0] * 33, "action_mean": [0.0] * 120,
            "action_std": [1.0] * 120}
    config = {"name": "sim-run", "seed": 3, "run_root": str(tmp_path / "runs"), "checkpoint_root": str(tmp_path),
              "model": model_cfg, "wandb": {"group": "lewm-sim"},
              "data": {"observation": "depth_plus_proprioception", "normalization": norm, "depth_size": 16}}
    run = tmp_path / "runs" / "sim-run"
    run.mkdir(parents=True)
    torch.manual_seed(0)
    model = build_model(model_cfg, tmp_path)
    for block in model.predictor.blocks:  # zero-init AdaLN would make the model ignore actions
        torch.nn.init.normal_(block.modulation[-1].weight, std=0.5)
    save_checkpoint(run / "last.pt", model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.1), epoch=1,
                    config=config)
    sim_eval = load_config(Path("configs/eval/sim_smoke.yaml"))
    sim_eval["episodes"]["cache_root"] = str(tmp_path / "episodes")
    return config, sim_eval


def test_sim_eval_pipeline_with_toy_simulator(tmp_path, monkeypatch):
    from quadwm.evaluation import sim_protocol
    from quadwm.evaluation.report import build_report

    monkeypatch.setattr(sim_protocol, "build_simulator", _ToySim)
    config, sim_eval = _tiny_run(tmp_path)

    result = sim_protocol.evaluate_in_sim(config, sim_eval)

    written = json.loads((tmp_path / "runs" / "sim-run" / "eval" / "last-sim-smoke.json").read_text("utf-8"))
    assert written["domain"].startswith("sim")
    assert set(written["probes"]) == {"linear", "mlp"}
    asr = written["ev2"]["mlp"]["1"]["0.05"]
    assert asr["true_divergence"] > 0 and asr["model_divergence"] > 0
    assert set(written["ev3"]) == {"1-flat", "5-gaps"}
    assert set(written["ev3"]["1-flat"]["summary"]) == {"planning", "controller"}
    assert written["ev4"]["retention_percent"]["controller"]["nominal"] == pytest.approx(100.0)
    assert written["ev4"]["retention_percent"]["controller"]["mass+30%"] < 100.0, "heavier toy robot is slower"
    assert set(written["ev6"]["ratio"]) == {"planning", "controller"}
    assert result["model"]["group"] == "lewm-sim"
    assert "EV3 planning" in build_report([written])

    # Episodes are cached: a second eval (e.g. the other model) renders nothing new.
    before = _ToySim.instances
    sim_protocol.evaluate_in_sim(config, sim_eval, only=["ev1"])
    assert _ToySim.instances == before


def test_report_pairs_real_and_sim_runs_for_delta_s2r(tmp_path, monkeypatch):
    from quadwm.evaluation import sim_protocol
    from quadwm.evaluation.common import finite
    from quadwm.evaluation.report import build_report

    monkeypatch.setattr(sim_protocol, "build_simulator", _ToySim)
    config, sim_eval = _tiny_run(tmp_path)
    sim = sim_protocol.evaluate_in_sim(config, sim_eval, only=["ev1"])
    real = json.loads(json.dumps(sim))
    real["domain"] = "real (GrandTour)"
    for curve in real["probes"]["mlp"]["eps_k"].values():
        curve["model"]["all"] += 1.0
    text = build_report([json.loads(json.dumps(finite(sim))), real])
    assert "Δ_s2r" in text
    assert "1.000 (n=1)" in text.split("Δ_s2r")[-1]


@pytest.mark.sim
def test_sim_eval_in_mujoco(tmp_path):
    pytest.importorskip("mujoco")
    if not Path(SIM_CONFIG["mjcf"]).is_file():
        pytest.skip(f"Menagerie ANYmal C not found at {SIM_CONFIG['mjcf']}; set QUADWM_SIM_ASSETS")
    from quadwm.evaluation.sim_protocol import evaluate_in_sim

    if os.name != "nt":
        os.environ.setdefault("MUJOCO_GL", "egl")
    config, sim_eval = _tiny_run(tmp_path)
    config["data"]["depth_size"] = 16
    result = evaluate_in_sim(config, sim_eval)
    assert {"ev2", "ev3", "ev4", "ev6"} <= set(result)
    assert result["data"]["windows"]["eval"] > 0


def test_perturbed_replays_restore_rendering_even_on_failure():
    from quadwm.evaluation.sim_protocol import _perturbed_replays

    sim = _ToySim({"camera": {"modalities": ["depth"]}, "default_joint_pos": SIM_CONFIG["default_joint_pos"]})
    cfg = {"command_mps": 0.3, "deltas_rad": [0.05], "repeats": 2}
    replays = _perturbed_replays(sim, build_controller(SIM_CONFIG), cfg, TerrainSpec(), 0, warmup=2, steps=3,
                                 rng=np.random.default_rng(0))
    assert replays.sequences.shape == (3, 3, 120) and replays.states.shape == (3, 4, 40)
    assert replays.deltas == [0.05, 0.05] and len(replays.actions) == 2 and sim.rendering

    def broken(state):
        raise RuntimeError("simulator died")

    sim.set_state = broken
    with pytest.raises(RuntimeError):
        _perturbed_replays(sim, build_controller(SIM_CONFIG), cfg, TerrainSpec(), 0, warmup=2, steps=3,
                           rng=np.random.default_rng(0))
    assert sim.rendering, "a failed replay must not leave rendering off for the next eval"
