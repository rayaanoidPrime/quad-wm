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

import torch

from quadwm.data import build_dataset, fetch_missions, verify_joint_order_consistency
from quadwm.models import build_model
from quadwm.utils import init_wandb

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

# should this just be a test? TODO
def _check_batch(batch: dict) -> None:
    for key, value in batch.items():
        if torch.is_tensor(value) and (torch.isnan(value).any() or torch.isinf(value).any()):
            raise ValueError(f"Batch field {key!r} contains NaN/Inf")
    # image_id join sanity: t and t+horizon must actually differ -- a bug in
    # the image_id/sequence_id join, or a horizon that rounds to zero frames,
    # would otherwise silently repeat one frame and train on a trivial pair.
    if torch.allclose(batch["depth_t"], batch["depth_t1"]):
        raise ValueError(
            "depth_t == depth_t1 for this batch -- check the image_id join and horizon_steps, "
            "these should never be identical"
        )


def smoke(config: dict | None = None) -> None:
    config = config or {}
    steps = int(config.get("steps", 20))
    output_dir = Path(config.get("run_root", "runs")) / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]
    wm_cfg = config["wm"]
    data_root = config["data_root"]

    fetch_missions(data_cfg["missions"], data_root)

    dataset = build_dataset(
        data_cfg,
        observation=wm_cfg["observation"],
        platform=wm_cfg["platform"],
        data_root=data_root,
        horizon=wm_cfg.get("horizon_steps", 4),
    )
    print(
        f"dataset: kept {dataset.stats['kept']}/{dataset.stats['total']} pairs "
        f"(drop_rate={dataset.stats['drop_rate']:.2%})"
    )
    max_drop_rate = data_cfg.get("max_drop_rate", 0.3)
    if dataset.stats["drop_rate"] > max_drop_rate:
        raise ValueError(
            f"drop_rate {dataset.stats['drop_rate']:.2%} exceeds max_drop_rate "
            f"{max_drop_rate:.2%} -- check sync settings before trusting this data"
        )
    if len(dataset) == 0:
        raise ValueError("dataset has zero usable pairs -- nothing to smoke-test")

    for reader in dataset.readers:
        if not verify_joint_order_consistency(reader.root):
            raise ValueError(
                f"joint order mismatch between anymal_state_actuator and "
                f"anymal_state_state_estimator in {reader.mission_dir}"
            )

    loader = dataset.loader(batch_size=int(config.get("batch_size", 4)))

    model = build_model(wm_cfg, checkpoint_root=config.get("checkpoint_root", "checkpoints"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 3e-4)))

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
            _check_batch(batch)

            ema_progress = step / steps
            optimizer.zero_grad()
            losses = model.training_step(batch, ema_progress)
            losses["loss"].backward()
            optimizer.step()

            metric = {"step": step, "timestamp": time.time(), **{k: v.item() for k, v in losses.items()}}
            stream.write(json.dumps(metric) + "\n")
            stream.flush()
            if wandb_run is not None:
                wandb_run.log(metric)

    torch.save({"model": model.state_dict(), "step": steps}, output_dir / "checkpoint.pt")

    if wandb_run is not None:
        wandb_run.finish()

    print(f"wrote {steps} steps to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="quadwm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser("smoke", help="run the infrastructure smoke tests")
    smoke_parser.add_argument("--config", type=Path, default=Path("configs/jepa-wm/smoke.yaml"))
    args = parser.parse_args()
    if args.command == "smoke":
        config = load_config(args.config)
        smoke(config)