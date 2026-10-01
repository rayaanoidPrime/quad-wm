"""Frozen probes, collapse checks, and shared evaluation metrics.

``protocol`` (GPU, real data) is imported lazily by the CLI; the pure metric,
probe, rollout, and report modules have no data or device requirements.
"""

from .metrics import (
    evaluation_sigma,
    gait_cycle_seconds,
    holm,
    mann_whitney,
    r2_and_pearson,
    rollout_errors,
    select_anchor_windows,
)
from .probes import PROBE_KINDS, StateProbe, build_probe, fit_probe
from .report import build_report, comparability_problems, report
from .rollout import protocol_latents

__all__ = [
    "PROBE_KINDS",
    "StateProbe",
    "build_probe",
    "build_report",
    "comparability_problems",
    "evaluation_sigma",
    "fit_probe",
    "gait_cycle_seconds",
    "holm",
    "mann_whitney",
    "protocol_latents",
    "r2_and_pearson",
    "report",
    "rollout_errors",
    "select_anchor_windows",
]
