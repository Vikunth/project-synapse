"""Dependency-free paired analysis for NativeProbeArtifact 1.0."""

from __future__ import annotations

import random
import statistics

from synapse_bench.stats import percentile


def prompt_counts_match(left: int | None, right: int | None) -> bool:
    if left is None or right is None:
        return False
    return abs(left - right) <= max(2, round(max(left, right) * 0.01))


def bootstrap_median(
    values: list[float], seed: int, iterations: int = 10_000
) -> dict[str, float | int | None]:
    if not values:
        return {
            "seed": seed,
            "iterations": iterations,
            "valid_units": 0,
            "estimate": None,
            "lower": None,
            "upper": None,
        }
    rng = random.Random(seed)
    estimates = [statistics.median(rng.choices(values, k=len(values))) for _ in range(iterations)]
    return {
        "seed": seed,
        "iterations": iterations,
        "valid_units": len(values),
        "estimate": statistics.median(values),
        "lower": percentile(estimates, 2.5),
        "upper": percentile(estimates, 97.5),
    }


def paired_finding(
    differences: list[float],
    relative: list[float],
    *,
    min_units: int,
    min_favors: int,
    absolute_gate: float,
    relative_gate: float,
    seed: int,
) -> dict[str, object]:
    boot = bootstrap_median(differences, seed)
    estimate = statistics.median(differences) if differences else None
    relative_estimate = statistics.median(relative) if relative else None
    favors = sum(value > 0 for value in differences)
    finding = bool(
        len(differences) >= min_units
        and favors >= min_favors
        and estimate is not None
        and estimate >= absolute_gate
        and relative_estimate is not None
        and relative_estimate >= relative_gate
        and isinstance(boot["lower"], float)
        and boot["lower"] > 0
    )
    return {
        "finding": finding,
        "valid_units": len(differences),
        "favors": favors,
        "median_absolute_effect_s": estimate,
        "median_relative_effect": relative_estimate,
        "bootstrap": boot,
    }


def saturation_finding(
    throughput_c8: list[float],
    throughput_c16: list[float],
    p95_c8: list[float],
    p95_c16: list[float],
) -> dict[str, object]:
    complete = min(len(throughput_c8), len(throughput_c16), len(p95_c8), len(p95_c16))
    throughput_ratio = (
        statistics.median(throughput_c16) / statistics.median(throughput_c8)
        if complete and statistics.median(throughput_c8)
        else None
    )
    latency_ratio = (
        statistics.median(p95_c16) / statistics.median(p95_c8)
        if complete and statistics.median(p95_c8)
        else None
    )
    return {
        "finding": bool(
            complete >= 10
            and throughput_ratio is not None
            and throughput_ratio <= 1.05
            and latency_ratio is not None
            and latency_ratio >= 1.5
        ),
        "complete_blocks": complete,
        "throughput_ratio": throughput_ratio,
        "latency_ratio": latency_ratio,
    }
