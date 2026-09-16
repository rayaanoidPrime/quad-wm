"""Small command-line entry point used by local and Slurm smoke tests."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path

from .config import load_config


def _metadata(config: dict | None = None) -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "config_path": config["config_path"]
    }


def smoke(
    config: dict | None = None,
) -> None:
    steps = int(config.get("steps", 20))
    output_dir = Path(config.get("run_root", "runs")) / "smoke"
    use_wandb = bool(config.get("wandb", {}).get("enabled", False))

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    run_metadata = _metadata(config)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    wandb_run = None
    if use_wandb:
        wandb_config = config.get("wandb", {})
        try:
            import wandb

            wandb_run = wandb.init(
                project=wandb_config.get("project", "quad-wm"),
                config={"kind": "infrastructure_smoke", "steps": steps, **run_metadata},
                name=wandb_config.get("name", "qwad-wm-smoke"),
            )
        except ImportError as exc:
            raise SystemExit("WANDB logging requested, but wandb is not installed") from exc

    with metrics_path.open("w", encoding="utf-8") as stream:
        for step in range(1, steps + 1):
            metric = {"step": step, "loss": 1.0 / step, "timestamp": time.time()}
            stream.write(json.dumps(metric) + "\n")
            stream.flush()
            if wandb_run is not None:
                wandb_run.log({"loss": metric["loss"], "step": step})

    if wandb_run is not None:
        wandb_run.finish()

    print(f"wrote {steps} metrics to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="quadwm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke", help="run the CPU-safe infrastructure smoke test")
    smoke_parser.add_argument("--config", type=Path, default=Path("configs/jepa-wm/smoke.yaml"))
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "smoke":
        smoke(config)
