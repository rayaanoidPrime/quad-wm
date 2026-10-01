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

import json
from itertools import product
from pathlib import Path

import numpy as np
import torch

from ..config import load_config
from ..data.grandtour import state_vectors
from ..sim import DynamicsSpec, TerrainSpec, build_simulator
from ..sim.controller import build_controller
from ..sim.episodes import SimWindowDataset, camera_modality, episode_set, fallen, load_episode
from ..utils import run_metadata
from .metrics import evaluation_sigma
from .planning import Planner, run_episode
from .protocol import _collect, _finite, _probe_results, load_model, model_record, open_checkpoint

EVALS = ("ev1", "ev2", "ev3", "ev4", "ev6")


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


class _Arms:
    """Runs the same episode with the planner and with the controller alone."""

    def __init__(self, planner: Planner, planner_sim, reference_sim, controller_factory, episode_cfg: dict,
                 horizon: int):
        self.planner, self.sims = planner, {"planning": planner_sim, "controller": reference_sim}
        self.controller_factory, self.cfg, self.horizon = controller_factory, episode_cfg, horizon

    def run(self, terrain: TerrainSpec, seeds: list[int], dynamics: DynamicsSpec = DynamicsSpec()) -> dict:
        results = {}
        for arm, sim in self.sims.items():
            episodes = [
                run_episode(sim, self.controller_factory(), planner=self.planner if arm == "planning" else None,
                            terrain=terrain, seed=int(seed), ticks=int(self.cfg["ticks"]),
                            command_mps=float(self.cfg["command_mps"]),
                            warmup_ticks=int(self.cfg["warmup_ticks"]), success_m=float(self.cfg["success_m"]),
                            horizon=self.horizon, dynamics=dynamics)
                for seed in seeds
            ]
            results[arm] = _summarize(episodes) | {"per_episode": episodes}
        return results


def _ev2(planner: Planner, probes: dict, sigma: torch.Tensor, sim, controller_factory, cfg: dict,
         horizons: list[int]) -> dict:
    """ASR(k, |δ|) = ‖probe(ẑ^A) − probe(ẑ^A')‖ / ‖s^A − s^A'‖, both σ-normalized (protocol §3)."""
    steps = max(horizons)
    context = planner.context_steps
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    sigma = sigma.cpu()
    planning_probe = planner.probe
    records = {kind: {(k, d): ([], []) for k in horizons for d in cfg["deltas_rad"]} for kind in probes}
    skipped = 0
    for (kind_terrain, level), seed in product(cfg["terrains"], cfg["seeds"]):
        controller = controller_factory()
        observation = sim.reset(seed=int(seed), terrain=TerrainSpec(kind=kind_terrain, level=float(level)))
        controller.command_mps = float(cfg["command_mps"])
        controller.reset()
        history, actions = [observation], []
        for _ in range(max(int(cfg["warmup_ticks"]), context - 1)):
            actions.append(controller.act(observation))
            observation = sim.step(actions[-1])
            history.append(observation)
        if fallen(observation):
            skipped += 1
            continue
        start, saved = observation, sim.get_state()
        sim.rendering = False  # replays only need states
        nominal, trajectory = [], [start]
        for _ in range(steps):
            nominal.append(controller.act(trajectory[-1]))
            trajectory.append(sim.step(nominal[-1]))
        nominal = np.stack(nominal)
        variants = []
        for delta in cfg["deltas_rad"]:
            for _ in range(int(cfg["repeats"])):
                perturbed = nominal + rng.normal(0.0, float(delta), nominal.shape)
                sim.set_state(saved)
                replay = [start] + [sim.step(action) for action in perturbed]
                variants.append((float(delta), perturbed, replay))
        sim.rendering = True
        true_states = [torch.from_numpy(state_vectors(trajectory))] + [
            torch.from_numpy(state_vectors(replay)) for *_, replay in variants
        ]
        past = np.stack(actions[len(actions) - (context - 1):]) if context > 1 else np.zeros((0, nominal.shape[1]))
        frames = planner.encode(history[-context:])
        future = torch.as_tensor(np.stack([nominal] + [v[1] for v in variants]), device=planner.device)
        for kind, probe in probes.items():
            planner.probe = probe
            predicted = planner.predict_states(frames, past, future).float().cpu()
            for index, (delta, *_rest) in enumerate(variants, start=1):
                for k in horizons:
                    true = ((true_states[0][k] - true_states[index][k]) / sigma).norm()
                    model = ((predicted[0, k - 1] - predicted[index, k - 1]) / sigma).norm()
                    records[kind][(k, delta)][0].append(float(true))
                    records[kind][(k, delta)][1].append(float(model))
    planner.probe = planning_probe
    result = {"skipped_fallen": skipped}
    for kind, cells in records.items():
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
        result[kind] = table
    return result


def _ev3(arms: _Arms, cfg: dict) -> dict:
    tiers = {}
    for tier in cfg["tiers"]:
        levels = {f"{level:g}": arms.run(TerrainSpec(kind=tier["kind"], level=float(level)), cfg["seeds"])
                  for level in tier["levels"]}
        summary = {}
        for arm in ("planning", "controller"):
            summary[arm] = {"fall_rate": float(np.mean([result[arm]["fall_rate"] for result in levels.values()]))}
            if tier["metric"] == "tracking":
                summary[arm]["tracking_error_mps"] = _mean_std([e["tracking_error_mps"] for result in levels.values()
                                                                for e in result[arm]["per_episode"]])
            elif tier["metric"] == "success":
                summary[arm]["success_rate"] = float(np.mean([r[arm]["success_rate"] for r in levels.values()]))
            else:  # max_level: the largest gap/step a majority of seeds still traverses
                passed = [float(level) for level, result in levels.items()
                          if result[arm]["success_rate"] >= float(cfg.get("max_level_success", 0.5))]
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
    for compound in cfg.get("compound", []):
        grid[f"compound_mass{compound['mass_percent']:+g}%"] = DynamicsSpec(
            base_mass_scale=1 + compound["mass_percent"] / 100, friction_scale=float(compound["friction_scale"]),
            latency_ms=float(compound["latency_ms"]))
    return grid


def _ev4(arms: _Arms, cfg: dict) -> dict:
    kind, level = cfg["terrain"]
    terrain = TerrainSpec(kind=kind, level=float(level))
    conditions = {name: arms.run(terrain, cfg["seeds"], dynamics) for name, dynamics in _dynamics_grid(cfg).items()}
    retention = {}
    for arm in ("planning", "controller"):
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
    for arm in ("planning", "controller"):
        original, mirrored = sides["original"][arm], sides["mirrored"][arm]
        ratio[arm] = {
            "success_ratio": (mirrored["success_rate"] / original["success_rate"]
                              if original["success_rate"] > 0 else None),
            "progress_ratio": (mirrored["progress_m"]["mean"] / original["progress_m"]["mean"]
                               if original["progress_m"]["mean"] else None),
        }
    return {"level": cfg["level"], "ratio": ratio, "sides": sides}


def evaluate_in_sim(config: dict, sim_eval_config: dict, checkpoint: Path | None = None,
                    output: Path | None = None, only: list[str] | None = None) -> dict:
    only = set(only or EVALS)
    if unknown := only - set(EVALS):
        raise ValueError(f"unknown evals {sorted(unknown)}; choose from {EVALS}")
    protocol_cfg = load_config(Path(sim_eval_config["protocol_config"]))
    protocol_cfg["num_workers"] = int(sim_eval_config.get("num_workers", 0))
    seed = int(sim_eval_config.get("seed", 4551))
    run_root, checkpoint, saved, device = open_checkpoint(config, checkpoint, seed)
    trained = saved["config"]
    data_cfg = trained["data"]
    normalization = data_cfg.get("normalization")
    modality = camera_modality(data_cfg["observation"])
    precision = torch.bfloat16 if protocol_cfg.get("precision", "bf16") == "bf16" else torch.float16
    horizons = sorted(int(k) for k in protocol_cfg["horizons"])
    sim_cfg = load_config(Path(sim_eval_config["sim_config"]))["sim"]
    controller_cfg = sim_eval_config.get("controller", {})

    def sim_factory(modalities):
        return build_simulator(sim_cfg | {"camera": sim_cfg["camera"] | {"modalities": list(modalities)}})

    def controller_factory():
        return build_controller(sim_cfg, controller_cfg)

    checkpoint_root = Path(config.get("checkpoint_root", "checkpoints"))
    model = load_model(saved, checkpoint_root).to(device).eval()
    depth_size, depth_range = int(data_cfg.get("depth_size", 64)), tuple(data_cfg.get("depth_range", (0.2, 10.0)))
    print(f"stage=sim_eval status=starting checkpoint={checkpoint} modality={modality} evals={sorted(only)}",
          flush=True)

    # EV1-sim, and the probes every other sim eval scores with.
    episode_cfg = sim_eval_config["episodes"]
    paths = {split: episode_set(lambda: sim_factory(["rgb", "depth"]), controller_factory, sim_cfg, controller_cfg,
                                episode_cfg, split) for split in ("probe", "eval")}
    datasets = {
        split: SimWindowDataset([load_episode(path) for path in split_paths],
                                context_frames=int(protocol_cfg["context_frames"]), steps=max(horizons),
                                stride_ticks=int(episode_cfg["window_stride_ticks"]), modality=modality,
                                normalization=normalization, depth_size=depth_size, depth_range=depth_range)
        for split, split_paths in paths.items()
    }
    for split, dataset in datasets.items():
        if not len(dataset):
            raise ValueError(f"no fall-free sim windows in the {split} episodes")
        print(f"stage=sim_eval split={split} episodes={len(paths[split])} windows={len(dataset)}", flush=True)
    collected = {
        split: _collect(model, dataset, [], use_cache=False, eval_config=protocol_cfg,
                        device=device, precision=precision, rollout=split == "eval")
        for split, dataset in datasets.items()
    }
    names = [path.stem for path in paths["eval"]]
    probes, fitted = {}, {}
    for kind in protocol_cfg["probe"]["kinds"]:
        probes[kind], fitted[kind] = _probe_results(kind, collected["probe"], collected["eval"], protocol_cfg,
                                                    horizons, names, device)
    sigma = evaluation_sigma(collected["eval"]["states"], float(protocol_cfg.get("sigma_floor", 1e-3)))

    planning_cfg = sim_eval_config["planning"]
    planner = Planner(model=model, probe=fitted[planning_cfg.get("probe", "mlp")],
                      normalization=normalization, modality=modality, depth_size=depth_size,
                      depth_range=depth_range, config=planning_cfg, device=device,
                      precision=precision)
    result = {
        "protocol": "recipes/shared_evaluation_protocol.md",
        "eval_config": sim_eval_config.get("name"),
        "probe_eval_config": protocol_cfg.get("name"),
        "domain": "sim (MuJoCo, stand-in ANYmal C, scripted trot; zero-shot from GrandTour training)",
        "model": model_record(saved, checkpoint, model, int(collected["eval"]["encoded"].shape[-1])),
        "data": {"observation": data_cfg["observation"], "modality": modality, "tick_hz": float(sim_cfg["tick_hz"]),
                 "windows": {split: len(dataset) for split, dataset in datasets.items()},
                 "episodes": {split: [path.name for path in split_paths] for split, split_paths in paths.items()},
                 "context_frames": int(protocol_cfg["context_frames"])},
        "horizons": horizons,
        "probes": probes if "ev1" in only else {},
        "not_run": {
            "EV3_learned_policy": "actor-critic trained in imagination is not implemented; only the CEM "
                                  "planning variant runs (docs/adr/0006)",
        },
        "metadata": run_metadata(config) | {"eval_seed": seed, "device": str(device)},
    }
    if "ev2" in only:
        print("stage=sim_eval ev=2 status=starting", flush=True)
        ev2_cfg = sim_eval_config["ev2"]
        ev2_horizons = [k for k in horizons if k <= int(ev2_cfg.get("max_horizon", max(horizons)))]
        result["ev2"] = _ev2(planner, fitted, sigma, sim_factory([modality]), controller_factory, ev2_cfg,
                             ev2_horizons)
    if only & {"ev3", "ev4", "ev6"}:
        arms = _Arms(planner, sim_factory([modality]), sim_factory([]), controller_factory,
                     sim_eval_config["control_episode"], int(planning_cfg["horizon_ticks"]))
        if "ev3" in only:
            result["ev3"] = _ev3(arms, sim_eval_config["ev3"])
        if "ev4" in only:
            print("stage=sim_eval ev=4 status=starting", flush=True)
            result["ev4"] = _ev4(arms, sim_eval_config["ev4"])
        if "ev6" in only:
            print("stage=sim_eval ev=6 status=starting", flush=True)
            result["ev6"] = _ev6(arms, sim_eval_config["ev6"])
    output = Path(output) if output else run_root / "eval" / f"{checkpoint.stem}-{sim_eval_config.get('name', 'sim')}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_finite(result), indent=2) + "\n", encoding="utf-8")
    print(f"stage=sim_eval status=complete output={output}", flush=True)
    return result


def collect_episodes(sim_eval_config: dict) -> dict[str, list[Path]]:
    """Render and cache the EV1-sim episodes without a model (`quadwm sim-collect`, CPU only)."""
    sim_cfg = load_config(Path(sim_eval_config["sim_config"]))["sim"]
    controller_cfg = sim_eval_config.get("controller", {})
    rendered = sim_cfg | {"camera": sim_cfg["camera"] | {"modalities": ["rgb", "depth"]}}
    return {split: episode_set(lambda: build_simulator(rendered), lambda: build_controller(sim_cfg, controller_cfg),
                               sim_cfg, controller_cfg, sim_eval_config["episodes"], split)
            for split in ("probe", "eval")}
