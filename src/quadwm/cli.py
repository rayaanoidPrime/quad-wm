"""Command-line entry point for training, end-to-end smoke, and data prep."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from quadwm.training import prepare, train as train_world_model

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="quadwm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser(
        "smoke", help="run the end-to-end pipeline on the tiny smoke config"
    )
    smoke_parser.add_argument("--config", type=Path, default=Path("configs/jepa-wm/baseline_smoke.yaml"))
    train_parser = subparsers.add_parser("train", help="train the JEPA world model")
    train_parser.add_argument("--config", type=Path, default=Path("configs/jepa-wm/baseline.yaml"))
    prepare_parser = subparsers.add_parser("prepare", help="download the local encoder checkpoint and data")
    prepare_parser.add_argument("--config", type=Path, default=Path("configs/jepa-wm/baseline.yaml"))
    eval_parser = subparsers.add_parser(
        "eval", help="shared-protocol eval of a trained checkpoint (probes, EV5 eps_k, EV7)"
    )
    eval_parser.add_argument("--config", type=Path, required=True, help="the run's training config")
    eval_parser.add_argument("--eval-config", type=Path, default=Path("configs/eval/protocol.yaml"))
    eval_parser.add_argument("--checkpoint", type=Path, help="default: <run_root>/<name>/last.pt")
    eval_parser.add_argument("--output", type=Path, help="default: <run_root>/<name>/eval/<ckpt>-<eval name>.json")
    sim_eval_parser = subparsers.add_parser(
        "sim-eval", help="simulated protocol evals of a checkpoint: EV1-sim, EV2, EV3, EV4, EV6"
    )
    sim_eval_parser.add_argument("--config", type=Path, required=True, help="the run's training config")
    sim_eval_parser.add_argument("--sim-eval-config", type=Path, default=Path("configs/eval/sim.yaml"))
    sim_eval_parser.add_argument("--checkpoint", type=Path, help="default: <run_root>/<name>/last.pt")
    sim_eval_parser.add_argument("--output", type=Path, help="default: <run_root>/<name>/eval/<ckpt>-<sim name>.json")
    sim_eval_parser.add_argument("--only", help="comma-separated subset of ev1,ev2,ev3,ev4,ev6 (probes always fit)")
    collect_parser = subparsers.add_parser("sim-collect", help="render and cache the EV1-sim episodes (CPU)")
    collect_parser.add_argument("--sim-eval-config", type=Path, default=Path("configs/eval/sim.yaml"))
    report_parser = subparsers.add_parser("report", help="aggregate eval JSONs across seeds and models")
    report_parser.add_argument("evals", type=Path, nargs="+")
    report_parser.add_argument("--output", type=Path, help="write markdown here as well as stdout")
    sim_parser = subparsers.add_parser("sim-smoke", help="E1.0 simulator gate: stand, render, measure FPS")
    sim_parser.add_argument("--config", type=Path, default=Path("configs/sim/mujoco_anymal.yaml"))
    args = parser.parse_args()
    if args.command == "sim-smoke":
        from .sim import smoke

        smoke(load_config(args.config))
        return
    if args.command == "eval":
        from .evaluation.protocol import evaluate_checkpoint

        evaluate_checkpoint(load_config(args.config), load_config(args.eval_config), args.checkpoint, args.output)
        return
    if args.command == "sim-eval":
        from .evaluation.sim_protocol import evaluate_in_sim

        only = args.only.split(",") if args.only else None
        evaluate_in_sim(load_config(args.config), load_config(args.sim_eval_config), args.checkpoint, args.output,
                        only)
        return
    if args.command == "sim-collect":
        from .evaluation.sim_protocol import collect_episodes

        for split, paths in collect_episodes(load_config(args.sim_eval_config)).items():
            print(f"stage=sim_collect split={split} episodes={len(paths)} folder={paths[0].parent}", flush=True)
        return
    if args.command == "report":
        from .evaluation import report

        text = report(args.evals, args.output)
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")  # ε, σ, ± on consoles that default to cp1252
        print(text, end="")
        return
    if args.command in ("smoke", "train"):
        # smoke is the same pipeline as train, just the tiny smoke config -- it
        # exercises download/data/cache/encoder/train/eval before a full run,
        # then the protocol eval on the checkpoint it just wrote.
        config = load_config(args.config)
        train_world_model(config)
        if args.command == "smoke":
            from .evaluation.protocol import evaluate_checkpoint

            evaluate_checkpoint(config, load_config(Path("configs/eval/smoke.yaml")))
    elif args.command == "prepare":
        prepare(load_config(args.config))
