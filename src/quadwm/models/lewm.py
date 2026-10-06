"""Track 1 from-scratch JEPA world model: depth + proprio, trained end to end.

LeWM-style (arXiv 2603.19312): encoder and predictor train jointly with a
next-latent MSE and SIGReg as the only anti-collapse term (no EMA target,
no stop-gradient). Track 1 recipe additions: symlog depth tokenizer (§2.1)
and optional PSG-JEPA grounding heads (§2.4, training only). See
docs/adr/0004 for where this departs from the recipe.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..data.grandtour import PROPRIO_LAYOUT
from .jepawm import AdaLNBlock

# How the predictor sees actions: normalized absolute joint commands, or commands
# relative to the current joint positions (docs/adr/0007).
ACTION_INPUTS = ("absolute", "joint_residual")


def sigreg(z: Tensor, slices: int = 1024, knots: int = 17, t_max: float = 3.0) -> Tensor:
    """SIGReg (LeJEPA): Epps-Pulley distance of random 1-D projections of z from N(0, 1).

    z: [..., N, D], statistic computed over the N samples of each leading
    index and averaged. Minimized when z is an isotropic standard Gaussian,
    which rules out collapsed (constant or low-rank) latents.
    """
    with torch.autocast(device_type=z.device.type, enabled=False):
        z = z.float()
        directions = F.normalize(torch.randn(z.shape[-1], slices, device=z.device), dim=0)
        t = torch.linspace(0.0, t_max, knots, device=z.device)
        x = (z @ directions).unsqueeze(-1) * t  # [..., N, slices, knots]
        gaussian = torch.exp(-0.5 * t**2)  # characteristic function of N(0, 1)
        error = (x.cos().mean(-3) - gaussian) ** 2 + x.sin().mean(-3) ** 2
        return torch.trapezoid(error * gaussian, t, dim=-1).mean() * z.shape[-2]


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))


class DepthProprioEncoder(nn.Module):
    """[B, 2, S, S] pooled depth (``grandtour.pool_depth``) + [B, 33] proprio -> [B, latent_dim]."""

    def __init__(self, *, image_size, patch_size, dim, depth, heads, proprio_dim, latent_dim):
        super().__init__()
        self.image_size = image_size
        # Two input channels: symlog depth and the fraction of valid pixels,
        # so no-return regions are explicit rather than a sentinel value.
        self.patchify = nn.Conv2d(2, dim, patch_size, patch_size)
        self.proprio = mlp(proprio_dim, dim, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.randn(1, (image_size // patch_size) ** 2 + 2, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.projector = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, latent_dim))

    def forward(self, depth: Tensor, proprio: Tensor) -> Tensor:
        if depth.shape[-1] != self.image_size:
            raise ValueError(f"depth is {depth.shape[-1]}px but model.image_size={self.image_size}; match data.depth_size")
        meters, coverage = depth[:, :1], depth[:, 1:]
        symlog = torch.log1p(meters) * (coverage > 0)  # depth >= 0, so symlog = log1p (recipe §2.1)
        patches = self.patchify(torch.cat((symlog, coverage), dim=1)).flatten(2).transpose(1, 2)
        tokens = torch.cat((self.cls.expand(len(patches), -1, -1), self.proprio(proprio)[:, None], patches), 1)
        return self.projector(self.transformer(tokens + self.position)[:, 0])


class LatentPredictor(nn.Module):
    """Frame-causal AdaLN transformer: latents [B, T, D] + actions [B, T, A] -> next latents."""

    def __init__(self, *, latent_dim, dim, depth, heads, action_dim, window):
        super().__init__()
        self.window = window
        self.lift = nn.Linear(latent_dim, dim)
        self.action = mlp(action_dim, dim, dim)
        self.blocks = nn.ModuleList(AdaLNBlock(dim, heads) for _ in range(depth))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, latent_dim))

    def forward(self, latents: Tensor, actions: Tensor) -> Tensor:
        frames = torch.arange(latents.shape[1], device=latents.device)
        lag = frames[:, None] - frames[None, :]
        mask = torch.zeros(lag.shape, device=latents.device).masked_fill((lag < 0) | (lag >= self.window), float("-inf"))
        values, condition = self.lift(latents), self.action(actions)
        for block in self.blocks:
            values = block(values, condition, mask)
        return self.head(values)  # position t predicts the latent at t + 1


class LeWorldModel(nn.Module):
    """End-to-end depth + proprio JEPA; same Track 1 model interface as JEPAWorldModel."""

    frozen_visual_encoder = False  # takes raw depth; no V-JEPA, no token cache
    image_channels = 2  # pooled depth (m) + valid fraction

    def __init__(
        self,
        *,
        image_size: int = 64,
        patch_size: int = 8,
        encoder_dim: int = 192,
        encoder_depth: int = 12,
        encoder_heads: int = 3,
        latent_dim: int = 192,
        predictor_dim: int = 384,
        predictor_depth: int = 6,
        predictor_heads: int = 6,
        predictor_window: int = 4,
        proprio_dim: int = 33,
        action_dim: int = 120,
        context_steps: int = 4,
        rollout_steps: int = 4,
        loss_weights: dict[str, float] | None = None,
        sigreg_slices: int = 1024,
        transition_horizons: tuple[int, ...] = (1, 4),
        action_input: str = "absolute",
    ):
        super().__init__()
        if action_input not in ACTION_INPUTS:
            raise ValueError(f"action_input={action_input!r}; expected one of {ACTION_INPUTS}")
        self.action_input = action_input
        self.context_steps, self.rollout_steps = context_steps, rollout_steps
        self.image_size, self.latent_dim = image_size, latent_dim
        self.action_dim, self.proprio_dim = action_dim, proprio_dim
        self.weights = {"pred": 1.0, "rollout": 1.0, "sigreg": 0.1, "state": 0.0, "transition": 0.0}
        self.weights |= loss_weights or {}
        self.sigreg_slices, self.transition_horizons = sigreg_slices, tuple(transition_horizons)
        self.encoder = DepthProprioEncoder(
            image_size=image_size, patch_size=patch_size, dim=encoder_dim, depth=encoder_depth,
            heads=encoder_heads, proprio_dim=proprio_dim, latent_dim=latent_dim,
        )
        self.predictor = LatentPredictor(
            latent_dim=latent_dim, dim=predictor_dim, depth=predictor_depth, heads=predictor_heads,
            action_dim=action_dim, window=predictor_window,
        )
        # PSG-JEPA grounding heads: training only, built only when weighted so
        # the base arm has no unused (DDP-breaking) parameters.
        self.joint_slice = joints = PROPRIO_LAYOUT["joint_pos"]
        self.state_head = mlp(latent_dim, 256, proprio_dim) if self.weights["state"] else None
        self.transition_head = (
            mlp(2 * latent_dim, 256, joints.stop - joints.start) if self.weights["transition"] else None
        )
        if action_input == "joint_residual":
            # Joint positions come from the state head, so the residual exists for
            # predicted frames too (open loop never reads future measured state).
            if self.state_head is None:
                raise ValueError("action_input=joint_residual decodes joint positions with the state head; "
                                 "set loss_weights.state > 0")
            joint_count = joints.stop - joints.start
            if action_dim % joint_count:
                raise ValueError(f"action_dim={action_dim} is not a whole number of {joint_count}-joint frames")
            # Input normalization (docs/adr/0002, 0007), set by training from the
            # data statistics and restored with the state dict.
            for name, size in (("action_mean", action_dim), ("action_std", action_dim),
                               ("joint_mean", joint_count), ("joint_std", joint_count),
                               ("residual_mean", action_dim), ("residual_std", action_dim)):
                self.register_buffer(name, torch.zeros(size) if name.endswith("mean") else torch.ones(size))

    def set_input_normalization(self, normalization: dict) -> None:
        """Load the run's data statistics; joint_residual models need them to un-normalize inputs."""
        if self.action_input != "joint_residual":
            return
        missing = {"residual_mean", "residual_std"} - set(normalization)
        if missing:
            raise ValueError(f"normalization lacks {sorted(missing)}; recompute it with control_hz")
        values = {
            "action_mean": normalization["action_mean"], "action_std": normalization["action_std"],
            "joint_mean": normalization["proprio_mean"][self.joint_slice],
            "joint_std": normalization["proprio_std"][self.joint_slice],
            "residual_mean": normalization["residual_mean"], "residual_std": normalization["residual_std"],
        }
        for name, value in values.items():
            getattr(self, name).copy_(torch.as_tensor(value, dtype=torch.float32))

    def predictor_actions(self, latents: Tensor, actions: Tensor) -> Tensor:
        """What the predictor is conditioned on: ``actions`` [B, T, A] (normalized commands) as given, or,
        for ``joint_residual``, each tick's commands minus the joint positions decoded from ``latents``
        [B, T, D] at that tick, standardized by the residual statistics."""
        if self.action_input == "absolute":
            return actions
        with torch.autocast(device_type=latents.device.type, enabled=False):
            # No gradient through the decoded joints: the state head (and the encoder
            # through it) is trained by state_loss alone, so it keeps meaning "joint positions".
            joints = self.state_head(latents.float())[..., self.joint_slice].detach()
            joints = joints * self.joint_std + self.joint_mean
            commands = actions.float() * self.action_std + self.action_mean
            frames = commands.unflatten(-1, (-1, joints.shape[-1])) - joints.unsqueeze(-2)
            residual = (frames.flatten(-2) - self.residual_mean) / self.residual_std
        return residual.to(actions.dtype)

    def encode(self, depth: Tensor, proprio: Tensor) -> Tensor:
        """depth [B, T, 2, S, S], proprio [B, T, P] -> latents [B, T, D]."""
        batch, frames = proprio.shape[:2]
        latents = self.encoder(depth.flatten(0, 1), proprio.flatten(0, 1))
        return latents.view(batch, frames, -1)

    @property
    def frame_shape(self) -> tuple[int, ...]:
        return (self.latent_dim,)

    def encode_frames(self, batch: dict[str, Tensor], visual_tokens: Tensor | None = None) -> Tensor:
        """Latents [B, T, D]; ``visual_tokens`` is ignored (no frozen encoder)."""
        return self.encode(batch["images"], batch["proprio"])

    def probe_latent(self, frames: Tensor) -> Tensor:
        return frames

    def training_only_modules(self) -> list[nn.Module]:
        """PSG grounding heads: they exist only for training losses (EV7 reports them separately).

        A joint_residual model's state head also builds its rollout inputs, so it counts as inference.
        """
        heads = [self.transition_head]
        if self.action_input != "joint_residual":
            heads.append(self.state_head)
        return [head for head in heads if head is not None]

    def rollout(self, context: Tensor, actions: Tensor, steps: int) -> Tensor:
        """Open-loop: [B, W, D] context + actions aligned to frames (actions[:, i] moves
        frame i to i + 1, so W + steps - 1 are needed) -> [B, steps, D] predicted latents."""
        frames = context
        for _ in range(steps):
            start = max(0, frames.shape[1] - self.predictor.window)
            window = frames[:, start:]
            conditioning = self.predictor_actions(window, actions[:, start : frames.shape[1]])
            following = self.predictor(window, conditioning)[:, -1]
            frames = torch.cat((frames, following[:, None]), dim=1)
        return frames[:, context.shape[1] :]

    def forward(self, batch: dict[str, Tensor], visual_tokens: Tensor | None = None) -> dict[str, Tensor]:
        """Training losses; ``visual_tokens`` is ignored (no frozen encoder)."""
        proprio, actions = batch["proprio"], batch["actions"]
        z = self.encode(batch["images"], proprio)
        losses = {
            # Teacher-forced next-latent loss at every position, in parallel.
            "pred_loss": F.mse_loss(
                self.predictor(z[:, :-1], self.predictor_actions(z[:, :-1], actions)), z[:, 1:]
            ),
            # Per-timestep SIGReg over the batch (samples, not frames, are i.i.d.).
            "sigreg_loss": sigreg(z.transpose(0, 1), slices=self.sigreg_slices),
        }
        if self.rollout_steps > 1:
            future = self.rollout(z[:, : self.context_steps], actions, self.rollout_steps)
            losses["rollout_loss"] = F.mse_loss(future, z[:, self.context_steps :])
        if self.state_head is not None:
            losses["state_loss"] = F.mse_loss(self.state_head(z), proprio)
        if self.transition_head is not None:
            joints = proprio[..., self.joint_slice]
            losses["transition_loss"] = torch.stack([
                F.mse_loss(self.transition_head(torch.cat((z[:, :-h], z[:, h:]), -1)), joints[:, h:] - joints[:, :-h])
                for h in self.transition_horizons
            ]).mean()
        losses["loss"] = sum(self.weights[name.removesuffix("_loss")] * value for name, value in losses.items())
        return losses

    @torch.no_grad()
    def evaluate(self, batch: dict[str, Tensor], visual_tokens: Tensor | None = None) -> dict[str, float]:
        """E1.1: per-step open-loop error vs persistence, plus collapse signatures; ``visual_tokens`` is ignored."""
        z = self.encode(batch["images"], batch["proprio"])
        future = self.rollout(z[:, : self.context_steps], batch["actions"], self.rollout_steps)
        targets, last = z[:, self.context_steps :], z[:, self.context_steps - 1 : self.context_steps]
        metrics = {}
        for step in range(self.rollout_steps):
            metrics[f"step_{step + 1}/latent_mse"] = float(F.mse_loss(future[:, step], targets[:, step]))
            metrics[f"step_{step + 1}/persistence_mse"] = float(F.mse_loss(last[:, 0], targets[:, step]))
        metrics["relative_error"] = float((future - targets).norm(dim=-1).mean() / targets.norm(dim=-1).mean())
        flat = z.flatten(0, 1).float()
        metrics["latent_std"] = float(flat.std(0).mean())
        singular = torch.linalg.svdvals(flat - flat.mean(0))
        spectrum = singular / singular.sum()
        metrics["effective_rank"] = float(torch.exp(-(spectrum * spectrum.clamp_min(1e-12).log()).sum()))
        return metrics
