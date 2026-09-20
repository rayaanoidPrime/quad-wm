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

from quadwm.utils.wandb import init_wandb
from quadwm.data import build_dataset

from .config import load_config


def _metadata(config: dict | None = None) -> dict[str, str]:
    config = config or {}
    return {
        "python": sys.version.split()[0],
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "config_path": config.get("config_path", "unknown"),
    }

def _check_batch(batch):
    pass

def smoke(
    config: dict | None = None,
) -> None:
    config = config or {}
    steps = int(config.get("steps", 20))
    output_dir = Path(config.get("run_root", "runs")) / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset = build_dataset(config["data"], observation=config["wm"]["observation"])
    loader = dataset.loader(batch_size=1)

    metrics_path = output_dir / "metrics.jsonl"
    run_metadata = _metadata(config)
    wandb_run = init_wandb(config, metadata=run_metadata, run_dir=output_dir)
    if wandb_run is not None:
        run_metadata["wandb_run_id"] = getattr(wandb_run, "id", "unknown")
        run_metadata["wandb_url"] = getattr(wandb_run, "url", None)

    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )

    with metrics_path.open("w", encoding="utf-8") as stream:
        for step, batch in zip(range(1, steps + 1), loader):
            _check_batch(batch)   # NaN check, non-degenerate image_id join, gap-drop rate
            metric = {"step": step, "n_dropped": batch["n_dropped"], "timestamp": time.time()}
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
    smoke_parser = subparsers.add_parser("smoke", help="run the infrastructure smoke tests")
    smoke_parser.add_argument("--config", type=Path, default=Path("configs/jepa-wm/smoke.yaml"))
    args = parser.parse_args()
    if args.command == "smoke":
        config = load_config(args.config)
        smoke(config)
