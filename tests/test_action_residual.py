"""Joint-residual action input and its statistics (docs/adr/0007)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quadwm.data.grandtour import PROPRIO_LAYOUT, command_residual_stats, normalization_stats
from quadwm.models import LeWorldModel, build_model

JOINTS = PROPRIO_LAYOUT["joint_pos"]
FRAMES = 2  # control frames per tick in the tiny model: action_dim = 2 * 12


def _config(**overrides) -> dict:
    return {"type": "lewm", "image_size": 16, "patch_size": 8, "encoder_dim": 16, "encoder_depth": 1,
            "encoder_heads": 2, "latent_dim": 8, "predictor_dim": 16, "predictor_depth": 1,
            "predictor_heads": 2, "predictor_window": 2, "proprio_dim": 33, "action_dim": 12 * FRAMES,
            "context_steps": 2, "rollout_steps": 2, "sigreg_slices": 8,
            "loss_weights": {"state": 1.0}, "action_input": "joint_residual"} | overrides


def _normalization(rng: np.random.Generator) -> dict:
    return {
        "proprio_mean": rng.normal(size=33).tolist(), "proprio_std": rng.uniform(0.5, 2, 33).tolist(),
        "action_mean": rng.normal(size=12 * FRAMES).tolist(), "action_std": rng.uniform(0.5, 2, 12 * FRAMES).tolist(),
        "residual_mean": rng.normal(size=12 * FRAMES).tolist(),
        "residual_std": rng.uniform(0.01, 0.1, 12 * FRAMES).tolist(),
    }


def _model(normalization: dict | None = None) -> LeWorldModel:
    torch.manual_seed(0)
    model = build_model(_config(), "unused")
    model.set_input_normalization(normalization or _normalization(np.random.default_rng(0)))
    return model


def test_residual_is_raw_command_minus_decoded_joints_standardized():
    norm = _normalization(np.random.default_rng(1))
    model = _model(norm)
    latents, actions = torch.randn(3, 4, 8), torch.randn(3, 4, 12 * FRAMES)
    features = model.predictor_actions(latents, actions)

    stats = {key: np.asarray(value) for key, value in norm.items()}
    joints = model.state_head(latents).detach().numpy()[..., JOINTS]
    joints = joints * stats["proprio_std"][JOINTS] + stats["proprio_mean"][JOINTS]
    commands = actions.numpy() * stats["action_std"] + stats["action_mean"]
    expected = (commands - np.tile(joints, FRAMES) - stats["residual_mean"]) / stats["residual_std"]
    assert features.shape == actions.shape
    np.testing.assert_allclose(features.detach().numpy(), expected, rtol=1e-4, atol=1e-3)


def test_absolute_input_is_unchanged_and_needs_no_statistics():
    model = build_model(_config(action_input="absolute", loss_weights={}), "unused")
    model.set_input_normalization({})  # nothing to load
    actions = torch.randn(2, 3, 12 * FRAMES)
    assert model.predictor_actions(torch.randn(2, 3, 8), actions) is actions
    assert not hasattr(model, "residual_std")


def test_joint_residual_needs_the_state_head_and_residual_statistics():
    with pytest.raises(ValueError, match="state"):
        build_model(_config(loss_weights={}), "unused")
    with pytest.raises(ValueError, match="joint frames"):
        build_model(_config(action_dim=13), "unused")
    norm = _normalization(np.random.default_rng(0))
    del norm["residual_std"]
    with pytest.raises(ValueError, match="residual_std"):
        build_model(_config(), "unused").set_input_normalization(norm)


def test_state_head_counts_as_inference_and_gets_no_gradient_from_prediction():
    model = _model()
    for block in model.predictor.blocks:  # zero-init AdaLN would block every action gradient
        torch.nn.init.normal_(block.modulation[-1].weight, std=0.5)
    assert model.state_head not in model.training_only_modules()
    batch = {"images": torch.rand(2, 4, 2, 16, 16), "proprio": torch.randn(2, 4, 33),
             "actions": torch.randn(2, 3, 12 * FRAMES)}
    losses = model(batch)
    assert {"pred_loss", "rollout_loss", "state_loss"} <= set(losses)
    (losses["pred_loss"] + losses["rollout_loss"]).backward()
    assert all(p.grad is None for p in model.state_head.parameters())
    assert model.predictor.action[0].weight.grad.abs().sum() > 0


def test_normalization_buffers_travel_with_the_checkpoint():
    norm = _normalization(np.random.default_rng(2))
    restored = build_model(_config(), "unused")
    restored.load_state_dict(_model(norm).state_dict())
    np.testing.assert_allclose(restored.residual_std.numpy(), norm["residual_std"], rtol=1e-6)
    np.testing.assert_allclose(restored.joint_mean.numpy(), np.asarray(norm["proprio_mean"])[JOINTS], rtol=1e-6)


def _reader(offset: float) -> SimpleNamespace:
    """Estimator at 100 Hz with joints = t; actuator at 50 Hz commanding joints + ``offset``."""
    proprio_t = np.arange(0.0, 4.0, 0.01)
    actuator_t = np.arange(0.0, 4.0, 0.02)
    joints = np.tile(proprio_t[:, None], (1, 12)).astype(np.float32)
    return SimpleNamespace(
        proprio_timestamps=proprio_t, actuator_timestamps=actuator_t, max_gap_s=0.05,
        proprio={"joint_positions": joints},
        actions=np.tile(actuator_t[:, None] + offset, (1, 12)).astype(np.float32),
        proprio_vectors=np.zeros((len(proprio_t), 33), np.float32),
    )


def test_residual_statistics_follow_the_frame_offset_layout():
    stats = command_residual_stats([_reader(0.5)], action_frames=3, control_hz=50.0, samples_per_mission=64)
    mean = np.asarray(stats["residual_mean"]).reshape(3, 12)
    # command(t + f / 50) - joints(t) = offset + f / 50, the same for every joint.
    np.testing.assert_allclose(mean, np.tile([[0.5], [0.52], [0.54]], (1, 12)), atol=0.011)
    assert len(stats["residual_std"]) == 36
    full = normalization_stats([_reader(0.5)], action_frames=3, control_hz=50.0)
    assert {"residual_mean", "residual_std", "action_mean", "proprio_std"} <= set(full)
    assert "residual_mean" not in normalization_stats([_reader(0.5)], action_frames=3)
