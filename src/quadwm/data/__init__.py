"""Canonical trajectory schemas and dataset adapters."""

from .grandtour import (
  GrandTourDataset,
  GrandTourPairDataset,
  build_dataset,
  fetch_missions,
  verify_joint_order_consistency
)

__all__ = [
    "GrandTourDataset",
    "GrandTourPairDataset",
    "build_dataset",
    "fetch_missions",
    "verify_joint_order_consistency",
]
