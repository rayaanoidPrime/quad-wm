"""`quadwm sim-eval`: the simulated half of the shared protocol (docs/adr/0006).

For one checkpoint, in MuJoCo with the scripted trot controller:
  - EV1-sim: the same probe procedure and ε_k as the real-data eval, on
    cached episodes (probe-fit seeds vs eval seeds). With a real-data eval
    JSON for the same checkpoint, ``quadwm report`` adds Δ_s2r(k).
  - EV2: action-sensitivity ratio ASR(k, |δ|) from simulator replays of
    perturbed action sequences from one saved state.
  - EV3: CEM planning over the terrain suite, next to the controller alone.
  - EV4: retention of EV3 performance under the dynamics-shift grid.
  - EV6: unilateral-terrain success, original vs mirrored.
The models were trained on GrandTour only, so every number here is
zero-shot sim transfer with a stand-in robot (ANYmal C) and a pinhole camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import torch

from ..config import checkpoint_dir, load_config, referenced_path
from ..data.grandtour import state_vectors
from ..sim import DynamicsSpec, TerrainSpec, build_simulator
from ..sim.controller import build_controller
from ..sim.episodes import SimWindowDataset, camera_modality, episode_set, fallen, load_episode
from ..utils import run_metadata
from .common import (
    collect_latents,
    evaluate_probes,
    load_model,
    model_record,
    open_checkpoint,
    precision_dtype,
    write_result,
)
from .planning import EpisodeSpec, Planner, run_episode
from .probes import StateProbe

EVALS = ("ev1", "ev2", "ev3", "ev4", "ev6")
ARMS = ("planning", "controller")


def _mean_std(values: list[float]) -> dict:
    values = [value for value in values if value is not None]
    if not values:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(np.mean(values)), "std": float(np.std(values)), "n": len(values)}


def _summarize(episodes: list[dict]) -> dict:
    return {
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "fall_rate": float(np.mean([e["fell"] for e in episodes])),
        "progress_m": _mean_std([e["progress_m"] for e in episodes]),
        "tracking_error_mps": _mean_std([e["tracking_error_mps"] for e in episodes]),
        "episodes": len(episodes),
    }


@dataclass
class _Arms:
    """Runs the same episodes with the planner and with the controller alone."""

    planner: Planner
    planning_sim: object  # renders the model's camera
    controller_sim: object  # renders nothing
    controller_factory: object
    spec: EpisodeSpec

    def run(self, terrain: TerrainSpec, seeds: list[int], dynamics: DynamicsSpec = DynamicsSpec()) -> dict:
        results = {}
        for arm, sim, planner in (("planning", self.planning_sim, self.planner),
                                  ("controller", self.controller_sim, None)):
            episodes = [run_episode(sim, self.controller_factory(), self.spec, terrain=terrain, seed=int(seed),
                                    planner=planner, dynamics=dynamics)
                        for seed in seeds]
            results[arm] = _summarize(episodes) | {"per_episode": episodes}
        return results


@dataclass
class _Replays:
    history: list[dict]  # observations up to and including the rollout start
    actions: list[np.ndarray]  # actions taken during warmup
    sequences: np.ndarray  # [1 + V, steps, 120]: the nominal continuation, then V perturbed copies
    deltas: list[float]  # perturbation std of each copy
    states: torch.Tensor  # [1 + V, steps + 1, 40] simulated states, same order


def _perturbed_replays(sim, controller, cfg: dict, terrain: TerrainSpec, seed: int, *, warmup: int, steps: int,
                       rng: np.random.Generator) -> _Replays | None:
    """Warm up under the controller, then replay its continuation and perturbed copies of it from one
    saved simulator state. None if the robot fell during warmup."""
    observation = sim.reset(seed=seed, terrain=terrain)
    controller.command_mps = float(cfg["command_mps"])
    controller.reset()
    history, actions = [observation], []
    for _ in range(warmup):
        actions.append(controller.act(observation))
        observation = sim.step(actions[-1])
        history.append(observation)
    if fallen(observation):
        return None
    saved, rendering = sim.get_state(), sim.rendering
    sim.rendering = False  # replays only need states
    try:
        nominal, trajectory = [], [observation]
        for _ in range(steps):
            nominal.append(controller.act(trajectory[-1]))
            trajectory.append(sim.step(nominal[-1]))
        nominal = np.stack(nominal)
        sequences, deltas, states = [nominal], [], [state_vectors(trajectory)]
        for delta in cfg["deltas_rad"]:
            for _ in range(int(cfg["repeats"])):
                perturbed = nominal + rng.normal(0.0, float(delta), nominal.shape)
                sim.set_state(saved)
                states.append(state_vectors([observation] + [sim.step(action) for action in perturbed]))
                sequences.append(perturbed)
                deltas.append(float(delta))
    finally:
        sim.rendering = rendering
    return _Replays(history, actions, np.stack(sequences), deltas, torch.from_numpy(np.stack(states)))


def _asr_table(cells: dict[tuple[int, float], tuple[list[float], list[float]]]) -> dict:
    table = {}
    for (k, delta), (true, model) in cells.items():
        true, model = np.asarray(true), np.asarray(model)
        valid = true > 1e-6
        table.setdefault(str(k), {})[f"{delta:g}"] = {
            "asr_ratio_of_means": float(model.mean() / true.mean()) if valid.any() else None,
            "asr_median": float(np.median(model[valid] / true[valid])) if valid.any() else None,
            "true_divergence": float(true.mean()) if len(true) else None,
            "model_divergence": float(model.mean()) if len(model) else None,
            "samples": int(valid.sum()),
        }
    return table


def _ev2(planner: Planner, probes: dict[str, StateProbe], sigma: torch.Tensor, sim, controller_factory,
         cfg: dict, horizons: list[int]) -> dict:
    """ASR(k, |δ|) = ‖probe(ẑ^A) − probe(ẑ^A')‖ / ‖s^A − s^A'‖, both σ-normalized (protocol §3)."""
    context = planner.context_steps
    rng = np.random.default_rng(int(cfg["seed"]))
    sigma = sigma.cpu()
    records = {kind: {(k, d): ([], []) for k in horizons for d in cfg["deltas_rad"]} for kind in probes}
    skipped = 0
    for (terrain_kind, level), seed in product(cfg["terrains"], cfg["seeds"]):
        replays = _perturbed_replays(sim, controller_factory(), cfg, TerrainSpec(kind=terrain_kind, level=float(level)),
                                     int(seed), warmup=max(int(cfg["warmup_ticks"]), context - 1),
                                     steps=max(horizons), rng=rng)
        if replays is None:
            skipped += 1
            continue
        frames = planner.encode(replays.history[-context:])
        future = torch.as_tensor(replays.sequences, device=planner.device)
        for kind, probe in probes.items():
            predicted = planner.predict_states(frames, replays.actions, future, probe=probe).float().cpu()
            for index, delta in enumerate(replays.deltas, start=1):
                for k in horizons:
                    true = ((replays.states[0, k] - replays.states[index, k]) / sigma).norm()
                    model = ((predicted[0, k - 1] - predicted[index, k - 1]) / sigma).norm()
                    records[kind][(k, delta)][0].append(float(true))
                    records[kind][(k, delta)][1].append(float(model))
    return {"skipped_fallen": skipped} | {kind: _asr_table(cells) for kind, cells in records.items()}


def _ev3(arms: _Arms, cfg: dict) -> dict:
    tiers = {}
    for tier in cfg["tiers"]:
        levels = {f"{level:g}": arms.run(TerrainSpec(kind=tier["kind"], level=float(level)), cfg["seeds"])
                  for level in tier["levels"]}
        summary = {}
        for arm in ARMS:
            summary[arm] = {"fall_rate": float(np.mean([result[arm]["fall_rate"] for result in levels.values()]))}
            if tier["metric"] == "tracking":
                summary[arm]["tracking_error_mps"] = _mean_std([e["tracking_error_mps"] for result in levels.values()
                                                                for e in result[arm]["per_episode"]])
            elif tier["metric"] == "success":
                summary[arm]["success_rate"] = float(np.mean([r[arm]["success_rate"] for r in levels.values()]))
            else:  # max_level: the largest gap/step a majority of seeds still traverses
                passed = [float(level) for level, result in levels.items()
                          if result[arm]["success_rate"] >= float(cfg["max_level_success"])]
                summary[arm]["max_level"] = max(passed) if passed else None
        tiers[tier["name"]] = {"kind": tier["kind"], "metric": tier["metric"], "summary": summary,
                               "levels": levels}
        print(f"stage=sim_eval ev=3 tier={tier['name']} " + " ".join(f"{a}={v}" for a, v in summary.items()),
              flush=True)
    return tiers


def _dynamics_grid(cfg: dict) -> dict[str, DynamicsSpec]:
    grid = {"nominal": DynamicsSpec()}
    for percent in cfg["mass_percent"]:
        grid[f"mass{percent:+g}%"] = DynamicsSpec(base_mass_scale=1 + percent / 100)
    for scale in cfg["friction_scale"]:
        grid[f"friction{scale:g}x"] = DynamicsSpec(friction_scale=float(scale))
    for latency in cfg["latency_ms"]:
        grid[f"latency{latency:g}ms"] = DynamicsSpec(latency_ms=float(latency))
    for compound in cfg["compound"]:
        name = f"compound_mass{compound['mass_percent']:+g}%"
        if name in grid:  # the name carries only the mass shift; a second one would overwrite the first
            raise ValueError(f"ev4.compound conditions need distinct mass_percent; {name} repeats")
        grid[name] = DynamicsSpec(
            base_mass_scale=1 + compound["mass_percent"] / 100, friction_scale=float(compound["friction_scale"]),
            latency_ms=float(compound["latency_ms"]))
    return grid


def _ev4(arms: _Arms, cfg: dict) -> dict:
    kind, level = cfg["terrain"]
    terrain = TerrainSpec(kind=kind, level=float(level))
    conditions = {name: arms.run(terrain, cfg["seeds"], dynamics) for name, dynamics in _dynamics_grid(cfg).items()}
    retention = {}
    for arm in ARMS:
        nominal = conditions["nominal"][arm]["progress_m"]["mean"]
        retention[arm] = {
            name: (100.0 * result[arm]["progress_m"]["mean"] / nominal) if nominal and nominal > 0 else None
            for name, result in conditions.items()
        }
    return {"terrain": cfg["terrain"], "performance": "forward progress (m) in a fixed episode",
            "retention_percent": retention, "conditions": conditions}


def _ev6(arms: _Arms, cfg: dict) -> dict:
    sides = {name: arms.run(TerrainSpec(kind="unilateral_steps", level=float(cfg["level"]), mirrored=mirrored),
                            cfg["seeds"])
             for name, mirrored in (("original", False), ("mirrored", True))}
    ratio = {}
    for arm in ARMS:
        original, mirrored = sides["original"][arm], sides["mirrored"][arm]
        ratio[arm] = {
            "success_ratio": (mirrored["success_rate"] / original["success_rate"]
                              if original["success_rate"] > 0 else None),
            "progress_ratio": (mirrored["progress_m"]["mean"] / original["progress_m"]["mean"]
                               if original["progress_m"]["mean"] else None),
        }
    return {"level": cfg["level"], "ratio": ratio, "sides": sides}


def _rendered(sim_cfg: dict, modalities: list[str]) -> dict:
    return sim_cfg | {"camera": sim_cfg["camera"] | {"modalities": list(modalities)}}


def evaluate_in_sim(config: dict, sim_eval_config: dict, checkpoint: Path | None = None,
                    output: Path | None = None, only: list[str] | None = None) -> dict:
    only = set(only or EVALS)
    if unknown := only - set(EVALS):
        raise ValueError(f"unknown evals {sorted(unknown)}; choose from {EVALS}")
    # Horizons, context, probe procedure, sigma floor: the real-data eval's, so EV1-sim pairs with EV5.
    protocol_cfg = load_config(referenced_path(sim_eval_config, sim_eval_config["protocol_config"]))
    seed = int(sim_eval_config["seed"])
    run_root, checkpoint, saved, device = open_checkpoint(config, checkpoint, seed)
    data_cfg = saved["config"]["data"]
    normalization = data_cfg.get("normalization")
    modality = camera_modality(data_cfg["observation"])
    precision = precision_dtype(protocol_cfg["precision"])
    horizons = sorted(int(k) for k in protocol_cfg["horizons"])
    context_frames = int(protocol_cfg["context_frames"])
    sim_cfg = load_config(referenced_path(sim_eval_config, sim_eval_config["sim_config"]))["sim"]
    controller_cfg = sim_eval_config["controller"]

    def controller_factory():
        return build_controller(sim_cfg, controller_cfg)

    model = load_model(saved, checkpoint_dir(config)).to(device).eval()
    depth_size, depth_range = int(data_cfg.get("depth_size", 64)), tuple(data_cfg.get("depth_range", (0.2, 10.0)))
    print(f"stage=sim_eval status=starting checkpoint={checkpoint} modality={modality} evals={sorted(only)}",
          flush=True)

    # EV1-sim, and the probes every other sim eval scores with.
    episode_cfg = sim_eval_config["episodes"]
    paths = {split: episode_set(lambda: build_simulator(_rendered(sim_cfg, ["rgb", "depth"])), controller_factory,
                                sim_cfg, controller_cfg, episode_cfg, split) for split in ("probe", "eval")}
    datasets = {
        split: SimWindowDataset([load_episode(path) for path in split_paths], context_frames=context_frames,
                                steps=max(horizons), stride_ticks=int(episode_cfg["window_stride_ticks"]),
                                modality=modality, normalization=normalization, depth_size=depth_size,
                                depth_range=depth_range)
        for split, split_paths in paths.items()
    }
    for split, dataset in datasets.items():
        if not len(dataset):
            raise ValueError(f"no fall-free sim windows in the {split} episodes")
        print(f"stage=sim_eval split={split} episodes={len(paths[split])} windows={len(dataset)}", flush=True)
    collected = {
        split: collect_latents(model, dataset, context_frames=context_frames,
                               steps=max(horizons) if split == "eval" else 0,
                               batch_size=int(protocol_cfg["batch_size"]),
                               # episodes are in memory; workers would copy them
                               num_workers=int(sim_eval_config["num_workers"]),
                               device=device, precision=precision)
        for split, dataset in datasets.items()
    }
    probe_eval = evaluate_probes(collected, protocol_cfg, [path.stem for path in paths["eval"]], device)

    planning_cfg = sim_eval_config["planning"]
    planner = Planner(model=model, probe=probe_eval.probes[planning_cfg["probe"]], normalization=normalization,
                      modality=modality, depth_size=depth_size, depth_range=depth_range, config=planning_cfg,
                      device=device, precision=precision)
    result = {
        "protocol": "recipes/shared_evaluation_protocol.md",
        "eval_config": sim_eval_config["name"],
        "probe_eval_config": protocol_cfg["name"],
        "domain": "sim (MuJoCo, stand-in ANYmal C, scripted trot; zero-shot from GrandTour training)",
        "model": model_record(saved, checkpoint, model, int(collected["eval"]["encoded"].shape[-1])),
        "data": {"observation": data_cfg["observation"], "modality": modality, "tick_hz": float(sim_cfg["tick_hz"]),
                 "windows": {split: len(dataset) for split, dataset in datasets.items()},
                 "episodes": {split: [path.name for path in split_paths] for split, split_paths in paths.items()},
                 "context_frames": context_frames},
        "horizons": horizons,
        "evals": sorted(only),
        "probes": probe_eval.results if "ev1" in only else {},
        "not_run": {
            "EV3_learned_policy": "actor-critic trained in imagination is not implemented; only the CEM "
                                  "planning variant runs (docs/adr/0006)",
        },
        "metadata": run_metadata(config) | {"eval_seed": seed, "device": str(device)},
    }
    if "ev2" in only:
        print("stage=sim_eval ev=2 status=starting", flush=True)
        result["ev2"] = _ev2(planner, probe_eval.probes, probe_eval.sigma,
                             build_simulator(_rendered(sim_cfg, [modality])), controller_factory,
                             sim_eval_config["ev2"], horizons)
    if only & {"ev3", "ev4", "ev6"}:
        arms = _Arms(planner, build_simulator(_rendered(sim_cfg, [modality])), build_simulator(_rendered(sim_cfg, [])),
                     controller_factory, EpisodeSpec.from_config(sim_eval_config["control_episode"]))
        for name, evaluate in (("ev3", _ev3), ("ev4", _ev4), ("ev6", _ev6)):
            if name in only:
                print(f"stage=sim_eval ev={name[2:]} status=starting", flush=True)
                result[name] = evaluate(arms, sim_eval_config[name])
    # A subset (one shard of a Slurm array) gets its own file; `quadwm report` merges the shards.
    name = sim_eval_config["name"] if only == set(EVALS) else f"{sim_eval_config['name']}-{'-'.join(sorted(only))}"
    output = write_result(result, output, run_root, checkpoint, name)
    print(f"stage=sim_eval status=complete output={output}", flush=True)
    return result


def collect_episodes(sim_eval_config: dict) -> dict[str, list[Path]]:
    """Render and cache a config's episodes without a model (`quadwm sim-collect`, CPU only).

    The EV1-sim probe/eval episodes by default; a config whose ``episodes`` block names
    ``splits`` and ``modalities`` (the fine-tuning set, docs/adr/0007) renders those instead.
    """
    sim_cfg = load_config(referenced_path(sim_eval_config, sim_eval_config["sim_config"]))["sim"]
    controller_cfg, spec = sim_eval_config["controller"], sim_eval_config["episodes"]
    modalities = spec.get("modalities", ["rgb", "depth"])
    return {split: episode_set(lambda: build_simulator(_rendered(sim_cfg, modalities)),
                               lambda: build_controller(sim_cfg, controller_cfg),
                               sim_cfg, controller_cfg, spec, split)
            for split in spec.get("splits", ("probe", "eval"))}
