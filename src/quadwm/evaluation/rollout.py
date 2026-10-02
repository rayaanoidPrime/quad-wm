"""Protocol windows -> probe latents, for any model with the Track 1 interface.

Each model owns its encoding, recurrence, and probe latent (``encode_frames``,
``rollout``, ``probe_latent``; docs/adr/0005 for the baseline's pooling).
This module only aligns the shared window to the model:

Window layout (shared by every model): ``context_frames`` frames, the last
of which is the rollout start t, followed by ``max(horizons)`` future frames.
``actions[:, i]`` moves frame i to i + 1. A model whose own context is
shorter uses only the trailing ``context_steps`` frames of the context.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _check_context(model, context_frames: int) -> None:
    if model.context_steps > context_frames:
        raise ValueError(
            f"model context_steps={model.context_steps} exceeds eval context_frames={context_frames}"
        )


def rollout_latents(model, frames: Tensor, actions: Tensor, context_frames: int, steps: int) -> Tensor:
    """Open-loop from frame ``context_frames - 1``: probe latents [B, steps, D] for t+1..t+steps.

    ``frames`` (``encode_frames`` output) needs only the first ``context_frames``
    frames; ``actions`` needs ``context_frames - 1 + steps`` entries (frame-aligned).
    Split from encoding so planners (EV2/EV3) encode a context once and roll
    out many candidate action sequences from it.
    """
    _check_context(model, context_frames)
    start = context_frames - model.context_steps
    return model.probe_latent(model.rollout(frames[:, start:context_frames], actions[:, start:], steps))


@torch.no_grad()
def protocol_latents(model, batch: dict, *, context_frames: int, steps: int,
                     visual_tokens: Tensor | None = None) -> dict[str, Tensor]:
    """encoded [B, T, D] for every frame; predicted [B, steps, D] for frames t+1..t+steps.

    ``steps = 0`` skips the rollout (probe-fit split). ``visual_tokens`` is
    required for the baseline and ignored by LeWM.
    """
    _check_context(model, context_frames)
    frames = model.encode_frames(batch, visual_tokens)
    result = {"encoded": model.probe_latent(frames)}
    if steps:
        result["predicted"] = rollout_latents(model, frames, batch["actions"], context_frames, steps)
    return result
