"""The minimal Track 1 RGB + proprioception JEPA world model."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _tokens(output: object) -> Tensor:
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, Tensor):
        raise TypeError(f"encoder returned {type(output)!r}, expected a tensor")
    if output.ndim == 4:
        output = output.flatten(2).transpose(1, 2)
    if output.ndim != 3:
        raise ValueError(f"expected encoder tokens with shape [B, N, D], got {tuple(output.shape)}")
    return output


class VJEPA21Encoder(nn.Module):
    """Load the official frozen V-JEPA 2.1 ViT-B/16 image backbone."""

    def __init__(self, checkpoint_root: str | Path):
        super().__init__()
        checkpoint_root = Path(checkpoint_root)
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        # The official hub entry constructs the exact architecture and loads
        # this file from Torch's cache.  shared.ensure_checkpoint puts it there.
        torch_checkpoint = checkpoint_root / "torch" / "hub" / "checkpoints"
        torch_checkpoint.mkdir(parents=True, exist_ok=True)
        local_checkpoint = checkpoint_root / "vjepa2_1_vitb_dist_vitG_384.pt"
        cached_checkpoint = torch_checkpoint / local_checkpoint.name
        if not cached_checkpoint.exists():
            try:
                os.link(local_checkpoint, cached_checkpoint)
            except OSError:
                shutil.copy2(local_checkpoint, cached_checkpoint)
        os.environ["TORCH_HOME"] = str(checkpoint_root)
        loaded = torch.hub.load(
            "facebookresearch/vjepa2",
            "vjepa2_1_vit_base_384",
            pretrained=True,
            source="github",
            num_frames=1,
            tubelet_size=1,
        )
        self.encoder = loaded[0] if isinstance(loaded, tuple) else loaded
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 4:
            raise ValueError(f"expected images [B, 3, H, W], got {tuple(images.shape)}")
        # V-JEPA 2.1's image path is a one-frame video: [B, C, T, H, W].
        return _tokens(self.encoder(images.unsqueeze(2)))


class ProprioEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim))

    def forward(self, proprio: Tensor) -> Tensor:
        return self.net(proprio)


class ActionEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim))

    def forward(self, action: Tensor) -> Tensor:
        return self.net(action)


class RotaryAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f"predictor dim {dim} must divide evenly by heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.rotary_dim = self.head_dim - self.head_dim % 2
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        inverse = 1.0 / (
            10000 ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.register_buffer("inverse_frequency", inverse, persistent=False)

    def _rotate(self, values: Tensor) -> Tensor:
        if self.rotary_dim == 0:
            return values
        length = values.shape[-2]
        positions = torch.arange(length, device=values.device, dtype=self.inverse_frequency.dtype)
        angles = torch.einsum("l,d->ld", positions, self.inverse_frequency)
        cos, sin = angles.cos()[None, None], angles.sin()[None, None]
        rotated = values[..., : self.rotary_dim]
        first, second = rotated[..., 0::2], rotated[..., 1::2]
        rotated = torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1)
        rotated = rotated.flatten(-2)
        return torch.cat((rotated, values[..., self.rotary_dim :]), dim=-1)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        batch, length, dim = values.shape
        qkv = self.qkv(values).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = self._rotate(q.transpose(1, 2))
        k = self._rotate(k.transpose(1, 2))
        v = v.transpose(1, 2)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(output.transpose(1, 2).reshape(batch, length, dim))


class AdaLNBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_attention = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = RotaryAttention(dim, heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, values: Tensor, condition: Tensor, mask: Tensor) -> Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.modulation(condition).chunk(6, dim=-1)
        attention_input = self.norm_attention(values) * (1 + scale_a) + shift_a
        values = values + gate_a * self.attention(attention_input, mask)
        mlp_input = self.norm_mlp(values) * (1 + scale_m) + shift_m
        return values + gate_m * self.mlp(mlp_input)


class Predictor(nn.Module):
    def __init__(self, dim: int, heads: int, depth: int, local_window_time: int, tokens_per_frame: int):
        super().__init__()
        self.local_window_time = local_window_time
        self.tokens_per_frame = tokens_per_frame
        self.blocks = nn.ModuleList(AdaLNBlock(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def _mask(self, frames: int, device: torch.device) -> Tensor:
        length = frames * self.tokens_per_frame
        frame = torch.arange(length, device=device) // self.tokens_per_frame
        allowed = (frame[:, None] >= frame[None, :]) & (
            frame[:, None] - frame[None, :] < self.local_window_time
        )
        mask = torch.full((length, length), float("-inf"), device=device)
        return mask.masked_fill(allowed, 0.0)[None, None]

    def forward(self, context: Tensor, actions: Tensor) -> Tensor:
        batch, frames, tokens, dim = context.shape
        if tokens != self.tokens_per_frame:
            raise ValueError(f"expected {self.tokens_per_frame} visual tokens, got {tokens}")
        condition = actions.unsqueeze(2).expand(batch, frames, tokens, dim)
        values = context.reshape(batch, frames * tokens, dim)
        condition = condition.reshape(batch, frames * tokens, dim)
        mask = self._mask(frames, values.device)
        for block in self.blocks:
            values = block(values, condition, mask)
        return self.norm(values).reshape(batch, frames, tokens, dim)[:, -1]


class JEPAWorldModel(nn.Module):
    def __init__(
        self,
        *,
        visual_dim: int = 768,
        proprio_dim: int = 33,
        proprio_embed_dim: int = 16,
        action_dim: int = 120,
        tokens_per_frame: int = 576,
        predictor_depth: int = 12,
        predictor_heads: int = 16,
        context_steps: int = 7,
        rollout_context: int = 3,
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.visual_dim = visual_dim
        self.model_dim = visual_dim + proprio_embed_dim
        self.tokens_per_frame = tokens_per_frame
        self.context_steps = context_steps
        self.rollout_context = rollout_context
        self.visual_encoder = encoder
        self.toy_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=4),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, visual_dim * tokens_per_frame),
        )
        self.proprio_encoder = ProprioEncoder(proprio_dim, proprio_embed_dim)
        self.action_encoder = ActionEncoder(action_dim, self.model_dim)
        self.predictor = Predictor(
            self.model_dim,
            predictor_heads,
            predictor_depth,
            rollout_context,
            tokens_per_frame,
        )

    def encode_visual(self, images: Tensor) -> Tensor:
        if self.visual_encoder is None:
            raise RuntimeError("visual encoder is not loaded; provide cached visual tokens")
        return self.visual_encoder(images)

    def encode_observation(self, visual: Tensor, proprio: Tensor) -> Tensor:
        prop = self.proprio_encoder(proprio).unsqueeze(-2)
        prop = prop.expand(*prop.shape[:-2], visual.shape[-2], prop.shape[-1])
        return torch.cat((visual, prop), dim=-1)

    def predict_next(self, context: Tensor, actions: Tensor) -> Tensor:
        action = self.action_encoder(actions)
        if action.ndim == 2:
            action = action.unsqueeze(1).expand(-1, context.shape[1], -1)
        return self.predictor(context, action)

    def loss(self, visual_tokens: Tensor, proprio: Tensor, actions: Tensor) -> dict[str, Tensor]:
        observations = self.encode_observation(visual_tokens, proprio)
        context_steps = self.context_steps
        if observations.shape[1] <= context_steps or actions.shape[1] < context_steps:
            raise ValueError(
                f"expected observations={context_steps + 1}+ and actions={context_steps}+, "
                f"got {observations.shape[1]} and {actions.shape[1]}"
            )
        context = observations[:, :context_steps]
        rollout_actions = actions[:, context_steps - 1 :]
        losses: dict[str, Tensor] = {}
        rollout_losses = []
        visual_losses = []
        proprio_losses = []
        for step, action in enumerate(rollout_actions.unbind(dim=1)):
            predictor_context = context if step == 0 else context[:, -self.rollout_context :]
            prediction = self.predict_next(predictor_context, action)
            target = observations[:, context_steps + step].detach()
            visual_loss = F.mse_loss(prediction[..., : self.visual_dim], target[..., : self.visual_dim])
            proprio_loss = F.mse_loss(prediction[..., self.visual_dim :], target[..., self.visual_dim :])
            step_loss = visual_loss + proprio_loss
            losses[f"loss_step_{step + 1}"] = step_loss
            rollout_losses.append(step_loss)
            visual_losses.append(visual_loss)
            proprio_losses.append(proprio_loss)
            context = torch.cat((context[:, -self.rollout_context + 1 :], prediction.detach().unsqueeze(1)), dim=1)
        if not rollout_losses:
            raise ValueError("batch contains no rollout actions")
        first = rollout_losses[0]
        later = rollout_losses[1:]
        total = first / (len(rollout_losses) + 1)
        if later:
            total = total + torch.stack(later).sum() / len(rollout_losses)
        losses["loss"] = total
        losses["visual_loss"] = torch.stack(visual_losses).mean()
        losses["proprio_loss"] = torch.stack(proprio_losses).mean()
        return losses

    def training_step(self, batch: dict[str, Tensor], _: float = 0.0) -> dict[str, Tensor]:
        if "depth_t" in batch:
            prediction = self.toy_encoder(batch["depth_t"].float())
            target = self.toy_encoder(batch["depth_t1"].float()).detach()
            loss = F.mse_loss(prediction, target)
            return {"loss": loss, "visual_loss": loss, "proprio_loss": loss.detach() * 0}
        if "visual_tokens" in batch:
            visual_tokens = batch["visual_tokens"]
        else:
            images = batch["images"]
            batch_size, frames = images.shape[:2]
            visual_tokens = self.encode_visual(images.reshape(batch_size * frames, *images.shape[2:]))
            visual_tokens = visual_tokens.reshape(batch_size, frames, *visual_tokens.shape[1:])
        return self.loss(visual_tokens, batch["proprio"], batch["actions"])

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.training_step(batch)


JepaWM = JEPAWorldModel
