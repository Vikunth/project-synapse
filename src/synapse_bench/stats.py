"""Small dependency-free statistics helpers."""

from __future__ import annotations

import math
import statistics


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percent / 100
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def latency_summary(
    values: list[float], *, trim_extremes: bool = False
) -> dict[str, float | int | None]:
    working = sorted(values)
    if trim_extremes and len(working) >= 5:
        working = working[1:-1]
    if not working:
        return {
            "successful_trials": 0,
            "median_s": None,
            "mean_s": None,
            "stddev_s": None,
            "p95_s": None,
        }
    return {
        "successful_trials": len(values),
        "included_trials": len(working),
        "median_s": statistics.median(working),
        "mean_s": statistics.fmean(working),
        "stddev_s": statistics.pstdev(working),
        "p95_s": percentile(working, 95),
    }
