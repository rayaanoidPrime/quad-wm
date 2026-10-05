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


def _reference_predictor(predictor, context, actions):
    """The pre-optimization predictor: AdaLN condition per token and a dense frame-causal mask."""
    batch, frames, tokens, dim = context.shape
    frame = torch.arange(frames * tokens) // tokens
    lag = frame[:, None] - frame[None, :]
    mask = torch.full(lag.shape, float("-inf")).masked_fill((lag >= 0) & (lag < predictor.local_window_time), 0.0)
    values = context.reshape(batch, frames * tokens, dim)
    condition = actions.unsqueeze(2).expand(batch, frames, tokens, dim).reshape(batch, frames * tokens, dim)
    for block in predictor.blocks:
        values = block(values, condition, mask[None, None])  # one "frame" per token: per-token modulation
    hidden = predictor.norm(values)
    visual = torch.nn.functional.layer_norm(predictor.visual_head(hidden), (predictor.visual_dim,))
    proprio = torch.nn.functional.layer_norm(predictor.proprio_head(hidden), (predictor.proprio_dim,))
    return torch.cat((visual, proprio), -1).reshape(batch, frames, tokens, -1)[:, -1]


def test_predictor_speedups_match_the_dense_reference():
    """Per-frame modulation + block-sparse attention change speed, not outputs or gradients."""
    torch.manual_seed(0)
    model = JEPAWorldModel(visual_dim=8, proprio_dim=33, proprio_embed_dim=4, action_dim=6, tokens_per_frame=4,
                           predictor_depth=3, predictor_heads=2, context_steps=5, rollout_context=2)
    for block in model.predictor.blocks:  # zero-init modulation would hide a broken broadcast
        torch.nn.init.normal_(block.modulation[-1].weight, std=0.1)
        torch.nn.init.normal_(block.modulation[-1].bias, std=0.1)
    context = torch.randn(3, 5, 4, 12, requires_grad=True)
    actions = torch.randn(3, 5, 12)

    fast = model.predictor(context, actions)
    reference = _reference_predictor(model.predictor, context, actions)
    assert torch.allclose(fast, reference, atol=1e-5)
    fast_grad, = torch.autograd.grad(fast.square().sum(), context)
    reference_grad, = torch.autograd.grad(reference.square().sum(), context)
    assert torch.allclose(fast_grad, reference_grad, atol=1e-4)
