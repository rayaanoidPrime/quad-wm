"""Pure shared-protocol metrics: no model, data, or device dependencies.

Every function here is deterministic on CPU so the metric definitions are
unit-tested independently of the GPU pipeline (AGENTS.md).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from ..data.grandtour import STATE_LAYOUT


def evaluation_sigma(states: Tensor, floor: float) -> Tensor:
    """Per-component std of the 40-d state over the evaluation set (protocol §2).

    ``floor`` keeps a component that never varies on the eval set (e.g. a
    foot that is always in contact) from dividing by zero.
    """
    return states.reshape(-1, states.shape[-1]).float().std(dim=0).clamp_min(floor)


def rollout_errors(predicted: Tensor, target: Tensor, sigma: Tensor) -> dict[str, Tensor]:
    """Per-window normalized L2 errors, protocol §2 ε_k before averaging.

    predicted/target: [N, 40] probe outputs and logged states at one horizon.
    Returns [N] errors for the full state, each STATE_LAYOUT component, and
    the full state without base position (docs/adr/0005).
    """
    error = (predicted.float() - target.float()) / sigma
    errors = {"all": error.norm(dim=-1)}
    for name, part in STATE_LAYOUT.items():
        errors[name] = error[:, part].norm(dim=-1)
    keep = torch.ones(error.shape[-1], dtype=torch.bool)
    keep[STATE_LAYOUT["base_pos"]] = False
    errors["all_excl_base_pos"] = error[:, keep].norm(dim=-1)
    return errors


def r2_and_pearson(predicted: Tensor, target: Tensor) -> dict[str, dict[str, float]]:
    """Probe quality per STATE_LAYOUT component: mean over its dims of R² and Pearson r.

    Dims with no variance on the evaluation set are left out of the mean
    (their R² and r are undefined); a component with none left reports NaN.
    """
    predicted, target = predicted.double(), target.double()
    centered_target = target - target.mean(0)
    centered_predicted = predicted - predicted.mean(0)
    total = (centered_target**2).sum(0)
    r2 = 1.0 - ((predicted - target) ** 2).sum(0) / total
    covariance = (centered_target * centered_predicted).sum(0)
    pearson = covariance / (total.sqrt() * (centered_predicted**2).sum(0).sqrt()).clamp_min(1e-12)
    defined = total > 1e-12
    result = {}
    for name, part in [*STATE_LAYOUT.items(), ("all", slice(0, target.shape[-1]))]:
        mask = defined[part]
        result[name] = {
            "r2": float(r2[part][mask].mean()) if mask.any() else float("nan"),
            "pearson": float(pearson[part][mask].mean()) if mask.any() else float("nan"),
        }
    return result


def select_anchor_windows(
    own_anchors: np.ndarray,
    other_anchors: list[np.ndarray],
    *,
    stride_s: float,
    tolerance_s: float,
    max_windows: int,
) -> list[int]:
    """Indices into ``own_anchors`` for one mission's protocol windows.

    Anchors are rollout-start times. A fixed time grid every ``stride_s``
    seconds keeps a grid point only if every observation (own and others) has
    a valid window within ``tolerance_s`` of it, so models that see different
    cameras are scored on the same moments (protocol §1.3 rule 5). The kept
    points are then thinned evenly to at most ``max_windows``.
    """
    sets = [np.sort(np.asarray(own_anchors, dtype=np.float64))] + [
        np.sort(np.asarray(anchors, dtype=np.float64)) for anchors in other_anchors
    ]
    if any(len(anchors) == 0 for anchors in sets):
        return []
    order = np.argsort(np.asarray(own_anchors, dtype=np.float64), kind="stable")
    start = max(anchors[0] for anchors in sets)
    stop = min(anchors[-1] for anchors in sets)
    if stop < start:
        return []
    grid = np.arange(start, stop + 1e-9, stride_s)
    chosen: list[int] = []
    for point in grid:
        nearest = []
        for anchors in sets:
            after = min(int(np.searchsorted(anchors, point)), len(anchors) - 1)
            best = min((max(after - 1, 0), after), key=lambda i: abs(anchors[i] - point))
            nearest.append(best if abs(anchors[best] - point) <= tolerance_s else None)
        if all(index is not None for index in nearest):
            own = int(order[nearest[0]])
            if not chosen or chosen[-1] != own:
                chosen.append(own)
    if max_windows and len(chosen) > max_windows:
        keep = np.unique(np.linspace(0, len(chosen) - 1, max_windows).round().astype(int))
        chosen = [chosen[i] for i in keep]
    return chosen


def gait_cycle_seconds(
    times: np.ndarray,
    contact: np.ndarray,
    speed: np.ndarray,
    *,
    min_speed: float = 0.2,
    min_segment_s: float = 4.0,
    rate_hz: float = 50.0,
    period_range_s: tuple[float, float] = (0.3, 2.0),
    max_gap_s: float = 0.1,
) -> list[float]:
    """Gait-cycle period of one foot's contact signal, per moving segment.

    Protocol §2 says to measure the ANYmal D gait cycle rather than copy
    k = 12. For each stretch of at least ``min_segment_s`` where planar speed
    exceeds ``min_speed``, resample the binary contact to ``rate_hz`` and take
    the strongest autocorrelation peak inside ``period_range_s``.
    """
    times = np.asarray(times, dtype=np.float64)
    moving = np.asarray(speed) > min_speed
    breaks = np.flatnonzero(~moving[1:] | ~moving[:-1] | (np.diff(times) > max_gap_s)) + 1
    periods = []
    for segment in np.split(np.arange(len(times)), breaks):
        segment = segment[moving[segment]]
        if len(segment) < 2 or times[segment[-1]] - times[segment[0]] < min_segment_s:
            continue
        grid = np.arange(times[segment[0]], times[segment[-1]], 1.0 / rate_hz)
        nearest = np.clip(np.searchsorted(times[segment], grid), 0, len(segment) - 1)
        signal = np.asarray(contact, dtype=np.float64)[segment][nearest]
        signal = signal - signal.mean()
        if not np.any(signal):
            continue
        correlation = np.correlate(signal, signal, mode="full")[len(signal) - 1 :]
        correlation = correlation / correlation[0]
        low, high = (round(bound * rate_hz) for bound in period_range_s)
        window = correlation[low : min(high + 1, len(correlation))]
        if len(window) < 3:
            continue
        # Interior local maxima only, so the decaying edge of the lag-0 peak never wins.
        peaks = np.flatnonzero((window[1:-1] > window[:-2]) & (window[1:-1] >= window[2:])) + 1
        if len(peaks) and window[peaks].max() > 0:
            periods.append(float((low + peaks[np.argmax(window[peaks])]) / rate_hz))
    return periods


def mann_whitney(first: list[float], second: list[float]) -> float | None:
    """Two-sided Mann-Whitney U p-value (protocol §11), None when n < 3 in either group."""
    if len(first) < 3 or len(second) < 3:
        return None
    from scipy.stats import mannwhitneyu

    return float(mannwhitneyu(first, second, alternative="two-sided").pvalue)


def holm(p_values: list[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjusted p-values (protocol §11 multiple comparisons)."""
    present = sorted((p, i) for i, p in enumerate(p_values) if p is not None)
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    for rank, (p, index) in enumerate(present):
        running = max(running, min(1.0, (len(present) - rank) * p))
        adjusted[index] = running
    return adjusted
