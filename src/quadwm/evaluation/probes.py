"""Frozen-latent state probes, fixed across every model (protocol §1.2-1.3).

Architecture, optimizer, learning rate, and step budget all come from the
shared eval config, never from the model being probed; only the input width
follows the latent size. Fitting never touches the world model: it sees
detached latent tensors.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

PROBE_KINDS = ("linear", "mlp")


def build_probe(kind: str, input_dim: int, output_dim: int, hidden_dim: int = 256) -> nn.Module:
    if kind == "linear":
        return nn.Linear(input_dim, output_dim)
    if kind == "mlp":  # protocol §1.2: 2 hidden layers, width 256, GELU
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
    raise ValueError(f"unknown probe kind {kind!r}; expected one of {PROBE_KINDS}")


class StateProbe(nn.Module):
    """Probe head plus the input/target standardization fitted on the probe split."""

    def __init__(self, head: nn.Module, feature_mean: Tensor, feature_std: Tensor,
                 target_mean: Tensor, target_std: Tensor):
        super().__init__()
        self.head = head
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_std", feature_std)
        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)

    def standardized(self, latents: Tensor) -> Tensor:
        return self.head((latents - self.feature_mean) / self.feature_std)

    @torch.no_grad()
    def forward(self, latents: Tensor, batch_size: int = 65536) -> Tensor:
        """Latents [..., D] -> states [..., S] in raw units."""
        flat = latents.reshape(-1, latents.shape[-1]).to(self.feature_mean.device, torch.float32)
        outputs = torch.cat([self.standardized(chunk) for chunk in flat.split(batch_size)])
        outputs = outputs * self.target_std + self.target_mean
        return outputs.reshape(*latents.shape[:-1], -1).cpu()


def fit_probe(kind: str, latents: Tensor, states: Tensor, config: dict, *, seed: int,
              device: torch.device) -> tuple[StateProbe, dict[str, float]]:
    """Fit one probe on (latent, state) pairs; returns the probe and its final fit loss.

    Fixed step budget with cosine decay and no early stopping: there is no
    third split to select on without touching eval (protocol §1.3 rule 4).
    """
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    features = latents.reshape(-1, latents.shape[-1]).float()
    targets = states.reshape(-1, states.shape[-1]).float()
    feature_mean, feature_std = features.mean(0), features.std(0).clamp_min(1e-6)
    target_mean, target_std = targets.mean(0), targets.std(0).clamp_min(1e-6)
    head = build_probe(kind, features.shape[-1], targets.shape[-1], int(config.get("hidden_dim", 256)))
    probe = StateProbe(head, feature_mean, feature_std, target_mean, target_std).to(device)
    features, targets = features.to(device), ((targets - target_mean) / target_std).to(device)
    steps = int(config["steps"])
    batch_size = min(int(config["batch_size"]), len(features))
    optimizer = torch.optim.AdamW(probe.head.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 0.5 * (1.0 + math.cos(math.pi * step / max(steps, 1)))
    )
    probe.train()
    loss = torch.tensor(float("nan"))
    for _ in range(steps):
        rows = torch.randint(len(features), (batch_size,), generator=generator).to(device)
        loss = F.mse_loss(probe.standardized(features[rows]), targets[rows])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
    probe.eval()
    with torch.no_grad():
        fit_loss = float(sum(
            F.mse_loss(probe.standardized(chunk_f), chunk_t, reduction="sum")
            for chunk_f, chunk_t in zip(features.split(65536), targets.split(65536))
        ) / targets.numel())
    parameters = sum(parameter.numel() for parameter in probe.head.parameters())
    return probe, {"fit_mse_standardized": fit_loss, "last_batch_mse": float(loss), "parameters": parameters}
