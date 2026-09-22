import torch

from quadwm.models.jepawm import JEPAWorldModel


def test_tiny_jepa_rollout_loss_backpropagates():
    model = JEPAWorldModel(
        visual_dim=8,
        proprio_dim=33,
        proprio_embed_dim=4,
        action_dim=6,
        tokens_per_frame=4,
        predictor_depth=2,
        predictor_heads=2,
        context_steps=2,
        rollout_context=2,
    )
    outputs = model.loss(
        torch.randn(2, 4, 4, 8),
        torch.randn(2, 4, 33),
        torch.randn(2, 3, 6),
    )
    outputs["loss"].backward()

    assert torch.isfinite(outputs["loss"])
    assert outputs["loss_step_1"].ndim == 0


def test_jepa_evaluate_reports_per_step_persistence_and_variance():
    model = JEPAWorldModel(
        visual_dim=8,
        proprio_dim=33,
        proprio_embed_dim=4,
        action_dim=6,
        tokens_per_frame=4,
        predictor_depth=1,
        predictor_heads=2,
        context_steps=2,
        rollout_context=2,
    )

    metrics = model.evaluate(
        torch.randn(2, 4, 4, 8),
        torch.randn(2, 4, 33),
        torch.randn(2, 3, 6),
    )

    for step in (1, 2):
        assert f"step_{step}/visual_mse" in metrics
        assert f"step_{step}/proprio_mse" in metrics
        assert f"step_{step}/persistence_visual_mse" in metrics
        assert f"step_{step}/persistence_proprio_mse" in metrics
    assert metrics["visual_mse"] >= 0
    assert metrics["proprio_variance"] >= 0
