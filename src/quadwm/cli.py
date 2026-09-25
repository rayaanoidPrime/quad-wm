"""Command-line entry point for training, end-to-end smoke, and data prep."""

from __future__ import annotations

import argparse
from pathlib import Path

from quadwm.data import fetch_missions
from quadwm.models import ensure_vjepa21_checkpoint
from quadwm.training import train as train_world_model

from .config import load_config


# download encoder pretrained checkpoints,
# fetch data
def prepare(config: dict) -> None:
    print("stage=prepare status=starting", flush=True)
    checkpoint = ensure_vjepa21_checkpoint(config.get("checkpoint_root", "checkpoints"))
    data_cfg = config.get("data", {})
    if data_cfg.get("download", False):
        data_root = data_cfg.get("data_root", config.get("data_root", "data/grandtour"))
        print(
            f"stage=data status=ensuring root={data_root} "
            f"missions={data_cfg.get('missions', 'all')}",
            flush=True,
        )
        fetch_missions(
            data_cfg.get("missions"),
            data_root,
            data_cfg.get("download_topics"),
        )
        print("stage=data status=ready", flush=True)
    print(f"stage=checkpoint status=ready path={checkpoint}", flush=True)
    print("stage=prepare status=complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(prog="quadwm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke_parser = subparsers.add_parser(
        "smoke", help="run the end-to-end pipeline on the tiny smoke config"
    )
    smoke_parser.add_argument("--config", type=Path, default=Path("configs/jepa-wm/baseline_smoke.yaml"))
    train_parser = subparsers.add_parser("train", help="train the Track 1 JEPA world model")
    train_parser.add_argument("--config", type=Path, default=Path("configs/jepa-wm/baseline.yaml"))
    prepare_parser = subparsers.add_parser("prepare", help="download the local encoder checkpoint and data")
    prepare_parser.add_argument("--config", type=Path, default=Path("configs/jepa-wm/baseline.yaml"))
    args = parser.parse_args()
    if args.command in ("smoke", "train"):
        # smoke is the same pipeline as train, just the tiny smoke config -- it
        # exercises download/data/cache/encoder/train/eval before a full run.
        train_world_model(load_config(args.config))
    elif args.command == "prepare":
        prepare(load_config(args.config))
