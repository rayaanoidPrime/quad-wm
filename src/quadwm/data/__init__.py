"""Canonical trajectory schemas and dataset adapters."""

from .grandtour import (
  GrandTourDataset,
  GrandTourPairDataset,
  Track1MissionDataset,
  build_dataset,
  build_sequence_dataset,
  fetch_missions,
  inspect_mission,
  inspect_root,
  materialize_mission,
  split_mission_names,
  verify_joint_order_consistency,
)

__all__ = [
    "GrandTourDataset",
    "GrandTourPairDataset",
    "Track1MissionDataset",
    "build_dataset",
    "build_sequence_dataset",
    "fetch_missions",
    "inspect_mission",
    "inspect_root",
    "materialize_mission",
    "split_mission_names",
    "verify_joint_order_consistency",
]
