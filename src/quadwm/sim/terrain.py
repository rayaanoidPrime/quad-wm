"""Heightfields for the shared terrain suite; pure numpy so it runs anywhere."""

from __future__ import annotations

import numpy as np

from .base import TerrainSpec


def terrain_heights(
    spec: TerrainSpec, *, length_m: float, width_m: float, cell_m: float, seed: int
) -> np.ndarray:
    """[rows(y), cols(x)] heights in meters; the robot starts at x=-length/2 and walks +x.

    The first 2 m are always flat so every episode starts from the same
    footing regardless of tier.
    """
    rng = np.random.default_rng(seed)
    x = np.arange(round(length_m / cell_m)) * cell_m  # distance from the start edge
    rows = round(width_m / cell_m)
    course = np.clip(x - 2.0, 0.0, None)  # 0 on the start pad
    if spec.kind == "flat":
        profile = np.zeros_like(x)
    elif spec.kind == "slope":
        profile = course * np.tan(np.radians(spec.level))
    elif spec.kind == "stairs":  # up for the first half, back down after
        tread = 0.3
        up = np.floor(course / tread) * spec.level
        profile = np.minimum(up, up[::-1])
    elif spec.kind == "gaps":  # 1.0 m platforms separated by pits of width `level`
        period = 1.0 + spec.level
        profile = np.where((course > 0) & (course % period > 1.0), -1.0, 0.0)
    elif spec.kind == "steps":  # a single climb onto a raised platform
        profile = np.where(course > 1.0, spec.level, 0.0)
    elif spec.kind == "rough":
        profile = np.zeros_like(x)
    else:
        raise ValueError(f"unknown terrain kind {spec.kind!r}")
    heights = np.tile(profile, (rows, 1))
    if spec.kind == "rough":
        heights += rng.uniform(-spec.level, spec.level, heights.shape) * (course > 0)
    if spec.mirrored:
        heights = heights[::-1]
    return heights
