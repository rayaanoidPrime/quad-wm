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
        {"proprio": torch.randn(2, 4, 33), "actions": torch.randn(2, 3, 6)},
        visual_tokens=torch.randn(2, 4, 4, 8),
    )

    for step in (1, 2):
        assert f"step_{step}/visual_mse" in metrics
        assert f"step_{step}/proprio_mse" in metrics
        assert f"step_{step}/persistence_visual_mse" in metrics
        assert f"step_{step}/persistence_proprio_mse" in metrics
    assert metrics["visual_mse"] >= 0
    assert metrics["proprio_variance"] >= 0


def test_rollout_feeds_back_only_the_last_rollout_context_frames():
    torch.manual_seed(0)
    model = JEPAWorldModel(visual_dim=8, proprio_dim=33, proprio_embed_dim=4, action_dim=6, tokens_per_frame=4,
                           predictor_depth=1, predictor_heads=2, context_steps=3, rollout_context=1).eval()
    context, actions = torch.randn(2, 3, 4, 12), torch.randn(2, 5, 6)
    first = model.predict_next(context, actions[:, 2])  # step 1 sees the whole context
    second = model.predict_next(first[:, None], actions[:, 3])  # rollout_context=1: only the prediction
    third = model.predict_next(second[:, None], actions[:, 4])
    predicted = model.rollout(context, actions, 3)
    assert predicted.shape == (2, 3, 4, 12)
    assert torch.allclose(predicted, torch.stack((first, second, third), 1), atol=1e-6)
