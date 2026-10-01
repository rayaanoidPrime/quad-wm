"""EV3 planning: CEM over controller residuals in latent space, scored in probe space.

Every simulated control episode (EV3/EV4/EV6) runs through ``run_episode``,
with or without a planner. Without one it is the scripted controller alone,
the reference that shows whether planning through the world model helps.

Departures from protocol §4.2, recorded in docs/adr/0006:
  - Search space: per-tick 12-d joint residuals on top of the controller's
    targets (H x 12 = 72 dims at H=6), not raw 120-d actions per tick
    (720 dims, beyond anything the CEM references validated).
  - Cost: locomotion has no goal image, so candidates are scored by the
    frozen probe on predicted latents (velocity tracking, uprightness, yaw
    rate) instead of L2 to a goal latent.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor

from ..data.grandtour import STATE_LAYOUT
from ..sim.base import DynamicsSpec, TerrainSpec
from ..sim.episodes import fallen, frame_images
from ..tokens import model_inputs
from .probes import StateProbe
from .rollout import rollout_latents


def cem(cost, shape: tuple[int, ...], *, population: int, elites: int, iterations: int, init_std: float,
        min_std: float, clip: float, mean: Tensor | None = None, generator: torch.Generator | None = None,
        device: torch.device | None = None) -> Tensor:
    """Cross-entropy method: refit a diagonal Gaussian to the lowest-cost elites (protocol §4.2)."""
    device = device or torch.device("cpu")
    mean = torch.zeros(shape, device=device) if mean is None else mean.to(device)
    std = torch.full(shape, init_std, device=device)
    for _ in range(iterations):
        noise = torch.randn((population, *shape), generator=generator).to(device)
        samples = (mean + std * noise).clamp(-clip, clip)
        best = samples[torch.topk(cost(samples), elites, largest=False).indices]
        mean, std = best.mean(0), best.std(0).clamp_min(min_std)
    return mean


def locomotion_cost(states: Tensor, command_mps: float, weights: dict) -> Tensor:
    """[N, H, 40] probe states -> [N] cost: forward-velocity error, tilt, and yaw rate."""
    velocity = states[..., STATE_LAYOUT["lin_vel"].start]
    gravity = states[..., STATE_LAYOUT["gravity"]]
    yaw_rate = states[..., STATE_LAYOUT["ang_vel"].start + 2]
    upright = gravity.new_tensor([0.0, 0.0, -1.0])
    return (
        float(weights.get("velocity", 1.0)) * (velocity - command_mps).pow(2).sum(1)
        + float(weights.get("upright", 1.0)) * (gravity - upright).pow(2).sum(-1).sum(1)
        + float(weights.get("yaw_rate", 0.1)) * yaw_rate.pow(2).sum(1)
    )


@dataclass
class Planner:
    """A frozen world model (with its frozen encoder, if any) + frozen probe, planning controller
    residuals by CEM."""

    model: torch.nn.Module
    probe: StateProbe
    normalization: dict
    modality: str
    depth_size: int
    depth_range: tuple
    config: dict
    device: torch.device
    precision: torch.dtype

    def __post_init__(self):
        stats = {key: torch.as_tensor(np.asarray(value, dtype=np.float32), device=self.device)
                 for key, value in self.normalization.items()}
        self.stats = stats
        self.context_steps = self.model.context_steps
        self.generator = torch.Generator().manual_seed(int(self.config.get("seed", 0)))

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.precision,
                              enabled=self.device.type == "cuda")

    @torch.no_grad()
    def encode(self, observations: list[dict]) -> Tensor:
        """The last ``context_steps`` observations -> ``model.encode_frames`` output with batch 1."""
        frames = [observation[self.modality] for observation in observations]
        images = frame_images(frames, self.modality, self.depth_size, self.depth_range)[None]
        proprio = torch.as_tensor(np.stack([o["proprio"] for o in observations]), device=self.device)[None]
        proprio = (proprio - self.stats["proprio_mean"]) / self.stats["proprio_std"]
        with self._autocast():
            return self.model.encode_frames(*model_inputs(self.model, {"images": images, "proprio": proprio},
                                                          self.device))

    @torch.no_grad()
    def predict_states(self, frames: Tensor, past_actions: np.ndarray, future_actions: Tensor) -> Tensor:
        """future_actions [N, H, 120] raw targets -> probe states [N, H, 40] in raw units."""
        count, horizon = future_actions.shape[:2]
        past = torch.as_tensor(past_actions, dtype=torch.float32, device=self.device)[None].expand(count, -1, -1)
        actions = torch.cat((past, future_actions.float()), dim=1)
        actions = (actions - self.stats["action_mean"]) / self.stats["action_std"]
        with self._autocast():
            latents = rollout_latents(self.model, frames.expand(count, *frames.shape[1:]), actions,
                                      self.context_steps, horizon)
        probe = self.probe
        return probe.standardized(latents.float()) * probe.target_std + probe.target_mean

    def plan(self, observations: list[dict], past_actions: np.ndarray, nominal: np.ndarray,
             command_mps: float, warm_start: Tensor | None) -> Tensor:
        """Best residual sequence [H, 12] for the controller's ``nominal`` [H, 120] continuation."""
        cfg = self.config
        frames = self.encode(observations)
        nominal_t = torch.as_tensor(nominal, dtype=torch.float32, device=self.device)
        frames_per_tick = nominal.shape[1] // 12

        def cost(samples: Tensor) -> Tensor:
            future = nominal_t[None] + samples.repeat(1, 1, frames_per_tick)
            states = self.predict_states(frames, past_actions, future)
            return (locomotion_cost(states, command_mps, cfg.get("cost", {}))
                    + float(cfg.get("residual_weight", 0.1)) * samples.pow(2).sum((1, 2)))

        return cem(cost, nominal.shape[:1] + (12,), population=int(cfg["population"]),
                   elites=int(cfg["elites"]), iterations=int(cfg["iterations"]),
                   init_std=float(cfg["init_std"]), min_std=float(cfg["min_std"]),
                   clip=float(cfg["max_residual_rad"]), mean=warm_start, generator=self.generator,
                   device=self.device)


def run_episode(sim, controller, *, planner: Planner | None, terrain: TerrainSpec, seed: int, ticks: int,
                command_mps: float, warmup_ticks: int, success_m: float, horizon: int,
                dynamics: DynamicsSpec = DynamicsSpec()) -> dict:
    """One closed-loop episode at 5 Hz; replans every tick when ``planner`` is given."""
    observation = sim.reset(seed=seed, terrain=terrain, dynamics=dynamics)
    controller.command_mps = command_mps
    controller.reset(heading=0.0)
    context = planner.context_steps if planner is not None else 1
    history: deque = deque(maxlen=context)
    past_actions: deque = deque(maxlen=max(context - 1, 1))
    start_x = float(observation["pose_pos"][0])
    errors, warm, fell, success, tick = [], None, False, False, 0
    for tick in range(ticks):
        history.append(observation)
        if fallen(observation):
            fell = True
            break
        if float(observation["pose_pos"][0]) - start_x >= success_m:
            success = True
            break
        residual = None
        if planner is not None and tick >= warmup_ticks and len(history) == context:
            past = np.stack(past_actions)[-(context - 1):] if context > 1 else np.zeros((0, controller.frames_per_tick * 12))
            plan = planner.plan(list(history), past, controller.nominal(observation, horizon), command_mps, warm)
            residual = plan[0].float().cpu().numpy()
            warm = torch.cat((plan[1:], torch.zeros_like(plan[:1])))  # shift for the next tick
        action = controller.act(observation, residual)
        past_actions.append(action)
        observation = sim.step(action)
        if tick >= warmup_ticks:
            errors.append(abs(float(observation["lin_vel"][0]) - command_mps))
    else:  # ran out of ticks: score the final state too
        fell = fallen(observation)
        success = not fell and float(observation["pose_pos"][0]) - start_x >= success_m
    return {
        "success": success,
        "fell": fell,
        "progress_m": float(observation["pose_pos"][0]) - start_x,
        "tracking_error_mps": float(np.mean(errors)) if errors else None,
        "ticks": tick + 1,
    }
