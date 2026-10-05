"""The minimal Track 1 RGB + proprioception JEPA world model."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

# Shared with models.shared.ensure_vjepa21_checkpoint, which downloads this file.
VJEPA21_FILENAME = "vjepa2_1_vitb_dist_vitG_384.pt"


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
        checkpoint_root = Path(checkpoint_root) # STORAGE_ROOT/checkpoints
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        # Point torch.hub at our storage *before* resolving its cache dir, then
        # seed the file where torch.hub.load_state_dict_from_url actually looks
        # for it: $TORCH_HOME/hub/checkpoints/<basename>.  Seeding the wrong
        # path makes the official hub entry re-download (or, since upstream
        # vjepa2 main points VJEPA_BASE_URL at localhost, fail with a 404).
        os.environ["TORCH_HOME"] = str(checkpoint_root)
        torch_checkpoint = Path(torch.hub.get_dir()) / "checkpoints" # STORAGE_ROOT/checkpoints/hub/checkpoints
        torch_checkpoint.mkdir(parents=True, exist_ok=True) 
        local_checkpoint = torch_checkpoint / VJEPA21_FILENAME # STORAGE_ROOT/checkpoints/file
        if not local_checkpoint.is_file():
            raise FileNotFoundError(
                f"V-JEPA 2.1 checkpoint not found at {local_checkpoint}; "
                "run `quadwm prepare` first"
            )
        loaded = torch.hub.load(
            "facebookresearch/vjepa2",
            "vjepa2_1_vit_base_384",
            pretrained=True,
            source="github",
            # The published checkpoint carries the tubelet-2 video patch_embed
            # plus the tubelet-1 image patch_embed.  Build the video-temporal
            # stem (num_frames>1) so both keys exist; the hub's
            # img_temporal_dim_size=1 then routes single-frame forward passes
            # through patch_embed_img at inference.
            num_frames=2,
            tubelet_size=2,
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


class EncoderProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, output_dim))

    def forward(self, values: Tensor) -> Tensor:
        return self.net(values)


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

# TODO understand this
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
    def __init__(
        self,
        dim: int,
        heads: int,
        depth: int,
        local_window_time: int,
        tokens_per_frame: int,
        visual_dim: int,
        proprio_dim: int,
    ):
        super().__init__()
        self.local_window_time = local_window_time
        self.tokens_per_frame = tokens_per_frame
        self.visual_dim = visual_dim
        self.proprio_dim = proprio_dim
        self.blocks = nn.ModuleList(AdaLNBlock(dim, heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        # Separate heads so the P proprio dims are not forced to share one
        # LayerNorm with the D visual dims.
        self.visual_head = nn.Linear(dim, visual_dim)
        self.proprio_head = nn.Linear(dim, proprio_dim)

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
        hidden = self.norm(values)
        visual = self.visual_head(hidden)
        proprio = self.proprio_head(hidden)

        visual = F.layer_norm(
            visual, (self.visual_dim,)
        )

        proprio = F.layer_norm(
            proprio, (self.proprio_dim,)
        )
        return torch.cat((visual, proprio), dim=-1).reshape(
            batch, frames, tokens, self.visual_dim + self.proprio_dim
        )[:, -1] # take the last frame [B, 576, 784]


class JEPAWorldModel(nn.Module):
    """Frozen V-JEPA 2.1 tokens + proprio -> next-frame tokens.

    Track 1 model interface (shared with LeWorldModel, used by every eval):
    ``encode_frames``, ``rollout``, ``probe_latent``, ``training_only_modules``,
    ``frame_shape``, ``forward`` (training losses) and ``evaluate`` (E1.1
    metrics), both ``(batch, visual_tokens=None)``, and the attributes ``frozen_visual_encoder``,
    ``image_channels``, ``image_size``, ``context_steps``, ``action_dim``,
    ``proprio_dim``.
    """

    frozen_visual_encoder = True  # images go through V-JEPA (or the token cache) first
    image_channels = 3

    def __init__(
        self,
        *,
        image_size: int = 384,
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
        self.image_size = image_size
        self.action_dim, self.proprio_dim = action_dim, proprio_dim
        self.visual_dim = visual_dim
        self.proprio_embed_dim = proprio_embed_dim
        self.model_dim = visual_dim + proprio_embed_dim
        self.tokens_per_frame = tokens_per_frame
        self.context_steps = context_steps
        self.rollout_context = rollout_context
        self.visual_encoder = encoder
        self.proprio_encoder = EncoderProjection(proprio_dim, proprio_embed_dim) # [B, T, 33] -> [B, T, 16]
        self.action_encoder = EncoderProjection(action_dim, self.model_dim) # [B,T, 120] -> [B,T,784]
        self.predictor = Predictor(
            self.model_dim,
            predictor_heads,
            predictor_depth,
            rollout_context,
            tokens_per_frame,
            visual_dim,
            proprio_embed_dim,
        )

    def encode_visual(self, images: Tensor) -> Tensor:
        if self.visual_encoder is None:
            raise RuntimeError("visual encoder is not loaded; provide cached visual tokens")
        return self.visual_encoder(images)

    def encode_observation(self, visual: Tensor, proprio: Tensor) -> Tensor:
        prop = self.proprio_encoder(proprio) # [B,T,16]
        # Pin the proprio embedding to unit variance (no affine -> no learnable
        # params) so E_prop cannot drift its output scale; otherwise every raw
        # proprio MSE is a moving yardstick and the loss is scale-dominated.
        prop = F.layer_norm(prop, (prop.shape[-1],)).unsqueeze(-2) # [B,T,1,16]
        prop = prop.expand(*prop.shape[:-2], visual.shape[-2], prop.shape[-1]) # [B,T,576,16]
        return torch.cat((visual, prop), dim=-1) # [B, T, 576, 784]

    def normalized_slices(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Per-slice LayerNorm (no affine): the space every loss and metric uses."""
        visual, prop = tokens[..., : self.visual_dim], tokens[..., self.visual_dim :]
        return F.layer_norm(visual, (self.visual_dim,)), F.layer_norm(prop, (self.proprio_embed_dim,))

    @property
    def frame_shape(self) -> tuple[int, ...]:
        return (self.tokens_per_frame, self.model_dim)

    def _visual_tokens(self, batch: dict[str, Tensor], visual_tokens: Tensor | None) -> Tensor:
        """Frozen tokens [B, T, N, Dv]: ``visual_tokens`` if given, else V-JEPA on prepared ``batch["images"]``."""
        if visual_tokens is not None:
            return visual_tokens
        images = batch["images"]  # [B, T, C, H, W], already prepared for V-JEPA
        return self.encode_visual(images.flatten(0, 1)).unflatten(0, images.shape[:2])

    def encode_frames(self, batch: dict[str, Tensor], visual_tokens: Tensor | None = None) -> Tensor:
        """Observation tokens [B, T, N, D] from frozen ``visual_tokens`` (or prepared ``batch["images"]``)."""
        return self.encode_observation(self._visual_tokens(batch, visual_tokens), batch["proprio"])

    def predict_next(self, context: Tensor, actions: Tensor) -> Tensor:
        action = self.action_encoder(actions) # [B, T , 784]
        if action.ndim == 2:
            action = action.unsqueeze(1).expand(-1, context.shape[1], -1)
        return self.predictor(context, action)

    def rollout(self, context: Tensor, actions: Tensor, steps: int, *, detach_feedback: bool = False) -> Tensor:
        """Open-loop: [B, W, N, D] context + frame-aligned actions (``actions[:, i]`` moves frame i to
        i + 1, so W + steps - 1 are needed) -> [B, steps, N, D] predicted frames.

        The first step sees the whole context; later steps see the last
        ``rollout_context`` frames, predictions included. Training detaches
        the fed-back predictions so each step's gradient stays local.
        """
        start = context.shape[1] - 1
        if actions.shape[1] < start + steps:
            raise ValueError(f"need {start + steps} frame-aligned actions, got {actions.shape[1]}")
        keep = self.rollout_context - 1
        window, predictions = context, []
        for step in range(steps):
            prediction = self.predict_next(window, actions[:, start + step])
            predictions.append(prediction)
            fed = prediction.detach() if detach_feedback else prediction
            history = window[:, window.shape[1] - keep :] if keep else window[:, :0]
            window = torch.cat((history, fed.unsqueeze(1)), dim=1)
        return torch.stack(predictions, dim=1)

    def probe_latent(self, frames: Tensor) -> Tensor:
        """[..., N, D] frames -> [..., visual + proprio] token mean in the normalized space (docs/adr/0005)."""
        visual, prop = self.normalized_slices(frames)
        return torch.cat((visual.mean(-2), prop.mean(-2)), dim=-1)

    def training_only_modules(self) -> list[nn.Module]:
        return []

    def loss(self, visual_tokens: Tensor, proprio: Tensor, actions: Tensor) -> dict[str, Tensor]:
        observations = self.encode_observation(visual_tokens, proprio) # [B,T,576, 784]
        context_steps = self.context_steps
        if observations.shape[1] <= context_steps or actions.shape[1] < context_steps:
            raise ValueError(
                f"expected observations={context_steps + 1}+ and actions={context_steps}+, "
                f"got {observations.shape[1]} and {actions.shape[1]}"
            )
        steps = actions.shape[1] - (context_steps - 1)  # one prediction per action after the context
        # [B, steps, 576, 784] from the first 7 frames as context
        predictions = self.rollout(observations[:, :context_steps], actions, steps, detach_feedback=True)
        losses: dict[str, Tensor] = {}
        rollout_losses = []
        visual_losses = []
        proprio_losses = []
        for step in range(steps):
            prediction = predictions[:, step]
            target_visual, target_prop = self.normalized_slices(
                observations[:, context_steps + step].detach()
            )
            visual_loss = F.mse_loss(prediction[..., : self.visual_dim], target_visual)
            proprio_loss = F.mse_loss(prediction[..., self.visual_dim :], target_prop)
            step_loss = visual_loss + proprio_loss
            losses[f"loss_step_{step + 1}"] = step_loss
            rollout_losses.append(step_loss)
            visual_losses.append(visual_loss)
            proprio_losses.append(proprio_loss)
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

    @torch.no_grad()
    def evaluate(self, batch: dict[str, Tensor], visual_tokens: Tensor | None = None) -> dict[str, float]:
        """E1.1 metrics: per-rollout-step error, persistence baseline, proprio variance.

        Persistence predicts the context's last observation for every future step
        (the gate the predictor must beat on held-out missions).
        """
        observations = self.encode_frames(batch, visual_tokens)
        actions = batch["actions"]
        context_steps = self.context_steps
        steps = actions.shape[1] - (context_steps - 1)
        predictions = self.rollout(observations[:, :context_steps], actions, steps)
        persistence = observations[:, context_steps - 1]
        metrics: dict[str, float] = {}
        visual_errors = []
        proprio_errors = []
        for step in range(steps):
            prediction = predictions[:, step]
            prefix = f"step_{step + 1}"
            # Model, target and persistence all go through the same per-slice
            # LayerNorm, so the persistence gate compares like with like.
            pred_visual, pred_prop = self.normalized_slices(prediction)
            target_visual, target_prop = self.normalized_slices(observations[:, context_steps + step])
            persist_visual, persist_prop = self.normalized_slices(persistence)
            visual = F.mse_loss(pred_visual, target_visual)
            prop = F.mse_loss(pred_prop, target_prop)
            metrics[f"{prefix}/visual_mse"] = float(visual)
            metrics[f"{prefix}/proprio_mse"] = float(prop)
            metrics[f"{prefix}/persistence_visual_mse"] = float(F.mse_loss(persist_visual, target_visual))
            metrics[f"{prefix}/persistence_proprio_mse"] = float(F.mse_loss(persist_prop, target_prop))
            visual_errors.append(visual)
            proprio_errors.append(prop)
        metrics["visual_mse"] = float(torch.stack(visual_errors).mean())
        metrics["proprio_mse"] = float(torch.stack(proprio_errors).mean())
        # Collapse signature from recipe §7: the proprio slice's variance should stay > 0.
        metrics["proprio_variance"] = float(
            observations[..., self.visual_dim :].var(dim=(0, 1)).mean()
        )
        return metrics

    def forward(self, batch: dict[str, Tensor], visual_tokens: Tensor | None = None) -> dict[str, Tensor]:
        return self.loss(self._visual_tokens(batch, visual_tokens), batch["proprio"], batch["actions"])
