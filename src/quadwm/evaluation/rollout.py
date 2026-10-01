"""Model adapters: one latent vector per frame and open-loop rollouts, for any Track 1 model.

The probe space needs a single vector z per frame. LeWM already has one.
The frozen-encoder baseline has 576 tokens x (768 visual + 16 proprio) per
frame; its probe latent is the token mean of the per-slice LayerNorm space
the predictor is trained in (docs/adr/0005).

Window layout (shared by every model): ``context_frames`` frames, the last
of which is the rollout start t, followed by ``max(horizons)`` future frames.
``actions[:, i]`` moves frame i to i + 1. A model whose own context is
shorter uses only the trailing ``context_steps`` frames of the context.

``encode_frames`` and ``rollout_latents`` are split so planners (EV2/EV3)
encode a context once and roll out many candidate action sequences from it.
"""

from __future__ import annotations

import torch
from torch import Tensor

from ..models import JEPAWorldModel, LeWorldModel


def _pooled(model: JEPAWorldModel, tokens: Tensor) -> Tensor:
    visual, prop = model._normalized_slices(tokens)
    return torch.cat((visual.mean(-2), prop.mean(-2)), dim=-1)


def _check(model) -> None:
    if not isinstance(model, (LeWorldModel, JEPAWorldModel)):
        raise TypeError(f"no protocol adapter for {type(model).__name__}")


def encode_frames(model, batch: dict, visual_tokens: Tensor | None = None) -> Tensor:
    """Per-frame model state: LeWM latents [B, T, D]; baseline observation tokens [B, T, N, D]."""
    _check(model)
    if isinstance(model, LeWorldModel):
        return model.encode(batch["images"], batch["proprio"])
    if visual_tokens is None:
        raise ValueError("the frozen-encoder baseline needs visual tokens")
    return model.encode_observation(visual_tokens, batch["proprio"])


def pool_frames(model, frames: Tensor) -> Tensor:
    """encode_frames output -> probe latents [B, T, D]."""
    return frames if isinstance(model, LeWorldModel) else _pooled(model, frames)


def rollout_latents(model, frames: Tensor, actions: Tensor, context_frames: int, steps: int) -> Tensor:
    """Open-loop from frame ``context_frames - 1``: probe latents [B, steps, D] for t+1..t+steps.

    ``frames`` needs only the first ``context_frames`` frames; ``actions``
    needs ``context_frames - 1 + steps`` entries (frame-aligned).
    """
    _check(model)
    if model.context_steps > context_frames:
        raise ValueError(
            f"model context_steps={model.context_steps} exceeds eval context_frames={context_frames}"
        )
    start = context_frames - model.context_steps
    if isinstance(model, LeWorldModel):
        return model.rollout(frames[:, start:context_frames], actions[:, start:], steps)
    # Same recurrence as JEPAWorldModel.evaluate, extended to `steps`.
    context = frames[:, start:context_frames]
    keep = model.rollout_context - 1
    outputs = []
    for step in range(steps):
        window = context if step == 0 else context[:, context.shape[1] - model.rollout_context :]
        prediction = model.predict_next(window, actions[:, context_frames - 1 + step])
        outputs.append(_pooled(model, prediction))
        previous = context[:, context.shape[1] - keep :] if keep else context[:, :0]
        context = torch.cat((previous, prediction.unsqueeze(1)), dim=1)
    return torch.stack(outputs, dim=1)


@torch.no_grad()
def protocol_latents(model, batch: dict, *, context_frames: int, steps: int,
                     visual_tokens: Tensor | None = None) -> dict[str, Tensor]:
    """encoded [B, T, D] for every frame; predicted [B, steps, D] for frames t+1..t+steps.

    ``steps = 0`` skips the rollout (probe-fit split). ``visual_tokens`` is
    required for the baseline and ignored by LeWM.
    """
    if model.context_steps > context_frames:
        raise ValueError(
            f"model context_steps={model.context_steps} exceeds eval context_frames={context_frames}"
        )
    frames = encode_frames(model, batch, visual_tokens)
    result = {"encoded": pool_frames(model, frames)}
    if steps:
        result["predicted"] = rollout_latents(model, frames, batch["actions"], context_frames, steps)
    return result


def training_only_modules(model) -> list[torch.nn.Module]:
    """Heads that exist only for training losses (EV7 reports them separately)."""
    return [head for head in (getattr(model, "state_head", None), getattr(model, "transition_head", None))
            if head is not None]
