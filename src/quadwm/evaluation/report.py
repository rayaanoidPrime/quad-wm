"""`quadwm report`: aggregate eval JSONs across seeds into the protocol §10 template.

Takes real-data (`quadwm eval`) and simulated (`quadwm sim-eval`) JSONs in
any mix. Runs are grouped by model group (the W&B group, shared by a
recipe's seeds). Every cell is mean ± std with its seed count; pairwise group
comparisons use Mann-Whitney U with Holm correction (protocol §11) and need
n >= 3 per group. Δ_s2r(k) pairs the real and sim JSON of each checkpoint.
Sim evals run as a Slurm array write one file per shard; they are merged
back into one record per checkpoint first (``merge_sim_shards``).
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from ..data.grandtour import STATE_LAYOUT
from .metrics import holm, mann_whitney

COMPARABILITY_KEYS = {
    "real": (("eval_config",), ("horizons",), ("data", "eval_missions"), ("data", "probe_missions"),
             ("data", "context_frames"), ("data", "match_observations")),
    "sim": (("eval_config",), ("probe_eval_config",), ("horizons",), ("data", "episodes"),
            ("data", "context_frames")),
}


def _get(record: dict, path: tuple[str, ...]):
    for key in path:
        record = record.get(key) if isinstance(record, dict) else None
    return record


def _cell(values: list[float | None], digits: int = 3) -> str:
    values = [value for value in values if value is not None]
    if not values:
        return "–"
    if len(values) == 1:
        return f"{values[0]:.{digits}f} (n=1)"
    return f"{np.mean(values):.{digits}f} ± {np.std(values, ddof=1):.{digits}f} (n={len(values)})"


def _p(value: float | None) -> str:
    return "n/a (n<3)" if value is None else f"{value:.4f}"


def _domain(run: dict) -> str:
    return "sim" if str(run.get("domain", "real")).startswith("sim") else "real"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(r) + " |" for r in rows]


def comparability_problems(runs: list[dict]) -> list[str]:
    """Settings that must match for runs to share a table (protocol §1.3 rules 1, 2, 5)."""
    problems = []
    for domain in ("real", "sim"):
        members = [run for run in runs if _domain(run) == domain]
        for path in COMPARABILITY_KEYS[domain]:
            seen = {json.dumps(_get(run, path), sort_keys=True) for run in members}
            if len(seen) > 1:
                problems.append(f"{domain}: {'.'.join(path)} differs across runs: {sorted(seen)}")
    return problems


SIM_SHARD_KEYS = ("ev2", "ev3", "ev4", "ev6")


def merge_sim_shards(runs: list[dict]) -> list[dict]:
    """One sim record per (checkpoint, sim eval config), from shards run with ``--only`` subsets.

    Shards of a checkpoint must have identical windows and episodes, and each
    eval may come from only one shard; anything else is an error, not a guess.
    """
    merged: dict[tuple, dict] = {}
    result = []
    for run in runs:
        if _domain(run) != "sim":
            result.append(run)
            continue
        key = (run["model"]["checkpoint"], run["eval_config"])
        if key not in merged:
            merged[key] = dict(run)
            result.append(merged[key])
            continue
        base = merged[key]
        if run["data"] != base["data"] or run["horizons"] != base["horizons"]:
            raise ValueError(f"sim shards of {key[0]} ({key[1]}) used different episodes, windows or horizons")
        duplicates = [name for name in SIM_SHARD_KEYS if name in run and name in base]
        if run.get("probes") and base.get("probes"):
            duplicates.append("ev1")
        if duplicates:
            raise ValueError(f"{key[0]} ({key[1]}): {duplicates} appear in more than one sim eval file")
        base.update({name: run[name] for name in SIM_SHARD_KEYS if name in run})
        if run.get("probes"):
            base["probes"] = run["probes"]
        base["evals"] = sorted(set(base.get("evals", [])) | set(run.get("evals", [])))
    return result


def _grouped(runs: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for run in runs:
        groups[run["model"]["group"]].append(run)
    return dict(groups)


def _probe_sections(groups: dict[str, list[dict]], label: str) -> list[str]:
    first = next(iter(groups.values()))[0]
    horizons, kinds = first["horizons"], list(first["probes"])
    columns = [f"k={k}" for k in horizons]
    lines = []
    for kind in kinds:
        rows = [[name, reference, *[_cell([m["probes"][kind]["eps_k"][str(k)][reference]["all"] for m in members])
                                    for k in horizons]]
                for name, members in groups.items() for reference in ("model", "persistence", "encoded_floor")]
        lines += ["", f"## {label} ε_k, {kind} probe (σ-normalized, all 40 dims)", "",
                  *_table(["group", "reference", *columns], rows)]
        rows = [[name, component, *[_cell([m["probes"][kind]["eps_k"][str(k)]["model"][component] for m in members])
                                    for k in horizons]]
                for name, members in groups.items() for component in [*STATE_LAYOUT, "all_excl_base_pos"]]
        lines += ["", f"### Per component, {kind} probe, model rollout", "",
                  *_table(["group", "component", *columns], rows)]
        components = [*STATE_LAYOUT, "all"]
        rows = [[name, metric, *[_cell([m["probes"][kind]["quality"][c][metric] for m in members]) for c in components]]
                for name, members in groups.items() for metric in ("r2", "pearson")]
        lines += ["", f"### Probe quality on encoded latents, {kind} probe", "",
                  *_table(["group", "metric", *components], rows)]
    names = list(groups)
    if len(names) > 1:
        tests = [(kind, a, b, k) for kind in kinds for a, b in combinations(names, 2) for k in horizons]
        p_values = [mann_whitney([m["probes"][kind]["eps_k"][str(k)]["model"]["all"] for m in groups[a]],
                                 [m["probes"][kind]["eps_k"][str(k)]["model"]["all"] for m in groups[b]])
                    for kind, a, b, k in tests]
        rows = [[kind, a, b, str(k), _p(p), _p(adjusted)]
                for (kind, a, b, k), p, adjusted in zip(tests, p_values, holm(p_values))]
        lines += ["", f"### {label} seed-level comparisons (Mann-Whitney U, two-sided, Holm-corrected over this table)",
                  "", *_table(["probe", "A", "B", "k", "p", "p (Holm)"], rows)]
    return lines


def _delta_s2r(real: list[dict], sim: list[dict]) -> list[str]:
    """Δ_s2r(k) = ε_k(real) − ε_k(sim), per checkpoint, then aggregated per group (protocol §6)."""
    by_checkpoint = {run["model"]["checkpoint"]: run for run in sim if run.get("probes")}
    pairs = [(run, by_checkpoint[run["model"]["checkpoint"]]) for run in real
             if run["model"]["checkpoint"] in by_checkpoint]
    if not pairs:
        return []
    horizons = [k for k in pairs[0][0]["horizons"] if k in pairs[0][1]["horizons"]]
    groups: dict[str, list] = defaultdict(list)
    for pair in pairs:
        groups[pair[0]["model"]["group"]].append(pair)
    rows = [[name, kind, *[_cell([r["probes"][kind]["eps_k"][str(k)]["model"]["all"]
                                  - s["probes"][kind]["eps_k"][str(k)]["model"]["all"] for r, s in members])
                           for k in horizons]]
            for name, members in groups.items() for kind in members[0][0]["probes"] if kind in members[0][1]["probes"]]
    return ["", "## Δ_s2r(k) = ε_k(real) − ε_k(sim), paired by checkpoint", "",
            ("> Sim is a stand-in ANYmal C with a pinhole camera and a scripted controller (docs/adr/0006); "
             "Δ_s2r mixes the sim-to-real gap with that robot/camera mismatch."), "",
            *_table(["group", "probe", *[f"k={k}" for k in horizons]], rows)]


def _sim_sections(groups: dict[str, list[dict]]) -> list[str]:
    lines = []
    with_ev2 = {name: [m for m in members if "ev2" in m] for name, members in groups.items()}
    if any(with_ev2.values()):
        first = next(m for members in with_ev2.values() for m in members)
        kinds = [kind for kind in first["ev2"] if kind != "skipped_fallen"]
        horizons = list(first["ev2"][kinds[0]])
        for kind in kinds:
            deltas = list(first["ev2"][kind][horizons[0]])
            rows = [[name, delta, *[_cell([m["ev2"][kind][k][delta]["asr_ratio_of_means"] for m in members])
                                    for k in horizons]]
                    for name, members in with_ev2.items() if members for delta in deltas]
            lines += ["", f"## EV2 ASR(k, δ), {kind} probe (ratio of mean divergences; ideal ≈ 1)", "",
                      *_table(["group", "δ (rad)", *[f"k={k}" for k in horizons]], rows)]
    with_ev3 = {name: [m for m in members if "ev3" in m] for name, members in groups.items()}
    if any(with_ev3.values()):
        first = next(m for members in with_ev3.values() for m in members)
        rows = []
        for tier, spec in first["ev3"].items():
            metric_key = {"tracking": "tracking_error_mps", "success": "success_rate", "max_level": "max_level"}[spec["metric"]]
            for name, members in with_ev3.items():
                for arm in ("planning", "controller"):
                    values = [_get(m["ev3"][tier]["summary"][arm], (metric_key, "mean"))
                              if spec["metric"] == "tracking" else m["ev3"][tier]["summary"][arm][metric_key]
                              for m in members]
                    falls = [m["ev3"][tier]["summary"][arm].get("fall_rate") for m in members]
                    rows.append([tier, name, arm, metric_key, _cell(values), _cell(falls)])
        lines += ["", "## EV3 planning (CEM) vs controller alone, per terrain tier", "",
                  *_table(["tier", "group", "arm", "metric", "value", "fall rate"], rows)]
    with_ev4 = {name: [m for m in members if "ev4" in m] for name, members in groups.items()}
    if any(with_ev4.values()):
        first = next(m for members in with_ev4.values() for m in members)
        conditions = list(first["ev4"]["retention_percent"]["planning"])
        rows = [[name, arm, *[_cell([m["ev4"]["retention_percent"][arm][c] for m in members], 1) for c in conditions]]
                for name, members in with_ev4.items() if members for arm in ("planning", "controller")]
        lines += ["", f"## EV4 retention (% of nominal forward progress) on {first['ev4']['terrain']}", "",
                  *_table(["group", "arm", *conditions], rows)]
    with_ev6 = {name: [m for m in members if "ev6" in m] for name, members in groups.items()}
    if any(with_ev6.values()):
        rows = [[name, arm, _cell([m["ev6"]["ratio"][arm]["success_ratio"] for m in members]),
                 _cell([m["ev6"]["ratio"][arm]["progress_ratio"] for m in members])]
                for name, members in with_ev6.items() if members for arm in ("planning", "controller")]
        lines += ["", "## EV6 mirrored / original (unilateral steps)", "",
                  *_table(["group", "arm", "success ratio", "progress ratio"], rows)]
    return lines


def _compute_section(groups: dict[str, list[dict]]) -> list[str]:
    rows = []
    for name, members in groups.items():
        compute = [m.get("compute") or {} for m in members]
        cells = [_cell([_get(c, path) for c in compute]) for path in (
            ("parameters_M", "inference_total"), ("parameters_M", "training_only"),
            ("parameters_M", "frozen_encoder"), ("rollout_throughput_fps",), ("single_step_latency_ms",),
            ("training", "gpu_hours_approx"),
        )]
        rows.append([name, *cells, ", ".join(sorted({str(c.get("device")) for c in compute}))])
    return ["", "## EV7 compute and latency", "",
            *_table(["group", "inference params (M)", "training-only (M)", "frozen encoder (M)", "rollout FPS",
                     "single-step ms", "training GPU-h (approx)", "device"], rows)]


def build_report(runs: list[dict]) -> str:
    runs = merge_sim_shards(runs)
    real = [run for run in runs if _domain(run) == "real"]
    sim = [run for run in runs if _domain(run) == "sim"]
    lines = ["# Shared-protocol report", ""]
    problems = comparability_problems(runs)
    if problems:
        lines += ["> **Not comparable** — fix before reading any table:", ""]
        lines += [f"> - {problem}" for problem in problems] + [""]
    groups = _grouped(runs)
    small = [name for name, members in _grouped(real or sim).items() if len(members) < 3]
    if small:
        lines += [(f"> Fewer than 3 seeds (protocol §11 minimum): {', '.join(small)}. "
                   "No significance tests for these groups."), ""]
    rows = [[name, str(len(members)), str(sorted({str(m['model']['seed']) for m in members})),
             members[0]["model"]["type"], ", ".join(sorted({_domain(m) for m in members}))]
            for name, members in groups.items()]
    lines += _table(["group", "eval files", "seeds", "type", "domains"], rows)

    if real:
        real_groups = _grouped(real)
        lines += ["", "# Real data (GrandTour)", *_probe_sections(real_groups, "EV5 (EV1 metric on real data)"),
                  *_compute_section(real_groups)]
        gait = [run["gait_cycle"]["seconds"] for run in real if run.get("gait_cycle", {}).get("seconds")]
        if gait:
            tick_hz = real[0]["data"]["tick_hz"]
            lines += ["", (f"Measured gait cycle: {np.median(gait):.2f} s ≈ {round(np.median(gait) * tick_hz)} "
                           f"ticks at {tick_hz:g} Hz (protocol k=12 assumes one gait cycle; change horizons in "
                           "configs/eval only via a versioned, ADR-recorded edit).")]
    if sim:
        sim_groups = _grouped(sim)
        lines += ["", "# Simulation (MuJoCo, stand-in ANYmal C, zero-shot)"]
        if any(run.get("probes") for run in sim):
            lines += _probe_sections({n: m for n, m in sim_groups.items() if m[0].get("probes")}, "EV1-sim")
        lines += _sim_sections(sim_groups)
    if real and sim:
        lines += _delta_s2r(real, sim)
    not_run = {}
    for run in runs:
        not_run |= run.get("not_run", {})
    lines += ["", "## Not run", ""] + [f"- **{name}**: {reason}" for name, reason in not_run.items()]
    return "\n".join(lines) + "\n"


def report(paths: list[Path], output: Path | None = None) -> str:
    runs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    if not runs:
        raise ValueError("no eval JSON files given")
    text = build_report(runs)
    if output is not None:
        Path(output).write_text(text, encoding="utf-8")
    return text
