"""Render deterministic, dependency-free SVG figures from the Phase 0 baseline."""

# SVG markup is kept as literal strings; wrapping it would obscure the generated structure.
# ruff: noqa: E501

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "results" / "baseline.json"
FIGURES = ROOT / "docs" / "figures"
WIDTH = 1040
HEIGHT = 680


def percentile(values: list[float], percent: float) -> float:
    """Return a linearly interpolated percentile, matching the benchmark runner."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = (len(ordered) - 1) * percent / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def scenario(data: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in data["scenarios"] if item["name"] == name)


def b2_points(data: dict[str, Any]) -> list[tuple[int, float, float, int]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for trial in scenario(data, "B2_prefix_scaling")["trials"]:
        if trial["status"] == "success":
            groups.setdefault(int(trial["prompt_eval_count"]), []).append(trial)
    points = []
    for tokens, trials in sorted(groups.items()):
        ordered = sorted(trials, key=lambda trial: int(trial["trial"]))
        if len(ordered) < 2:
            raise ValueError(f"B2 group {tokens} needs a first and a later trial")
        first = float(ordered[0]["ttft_s"])
        later = statistics.median(float(trial["ttft_s"]) for trial in ordered[1:])
        points.append((tokens, first, later, len(ordered) - 1))
    return points


def b4_points(data: dict[str, Any]) -> list[tuple[int, float, float, int]]:
    result = scenario(data, "B4_concurrency")
    groups: dict[int, list[dict[str, Any]]] = {}
    for trial in result["trials"]:
        if trial["status"] == "success":
            groups.setdefault(int(trial["concurrency"]), []).append(trial)
    points = []
    for concurrency, trials in sorted(groups.items()):
        # Exact aggregate throughput depends on batch wall time, which is retained
        # in the scenario summary; p95 is reproducible from the raw trial latencies.
        throughput = float(result["summary"][f"c{concurrency}_tokens_per_second"])
        p95 = percentile([float(trial["total_s"]) for trial in trials], 95)
        points.append((concurrency, throughput, p95, len(trials)))
    return points


def svg_start(title: str, description: str) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        f'  <title id="title">{title}</title>',
        f'  <desc id="desc">{description}</desc>',
        '  <rect width="1040" height="680" fill="#ffffff"/>',
        "  <style>text{font-family:Arial,sans-serif;fill:#172033}.title{font-size:26px;font-weight:700}.subtitle{font-size:14px;fill:#526079}.axis{stroke:#526079;stroke-width:1}.grid{stroke:#d9dee8;stroke-width:1}.tick{font-size:13px}.label{font-size:14px;font-weight:600}.value{font-size:12px;font-weight:700}.note{font-size:13px;fill:#526079}</style>",
    ]


def render_b2(points: list[tuple[int, float, float, int]]) -> str:
    lines = svg_start(
        "B2 time to first token: first request versus repeats",
        "Grouped bars compare first-request TTFT with the median TTFT of four later requests for each actual prompt evaluation count.",
    )
    lines += [
        '  <text x="70" y="48" class="title">B2: first request vs repeat TTFT</text>',
        '  <text x="70" y="76" class="subtitle">Grouped by actual prompt_eval_count, not requested prefix target</text>',
    ]
    left, top, plot_w, plot_h = 90, 120, 870, 390
    maximum = 10.0
    for tick in range(0, 11, 2):
        y = top + plot_h - tick / maximum * plot_h
        lines.append(
            f'  <line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>'
        )
        lines.append(
            f'  <text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" class="tick">{tick}</text>'
        )
    lines += [
        f'  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
        f'  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
        '  <text x="24" y="315" transform="rotate(-90 24 315)" class="label">TTFT (seconds)</text>',
        '  <rect x="650" y="91" width="14" height="14" fill="#2457c5"/><text x="672" y="103" class="tick">First request</text>',
        '  <rect x="790" y="91" width="14" height="14" fill="#16a085"/><text x="812" y="103" class="tick">Later median</text>',
    ]
    group_w = plot_w / len(points)
    bar_w = 58
    for index, (tokens, first, later, later_n) in enumerate(points):
        center = left + group_w * (index + 0.5)
        for offset, value, color in ((-bar_w, first, "#2457c5"), (6, later, "#16a085")):
            height = value / maximum * plot_h
            x = center + offset
            y = top + plot_h - height
            lines.append(
                f'  <rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 6}" height="{height:.1f}" rx="3" fill="{color}"/>'
            )
            lines.append(
                f'  <text x="{x + (bar_w - 6) / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="value">{value:.3f}s</text>'
            )
        lines.append(
            f'  <text x="{center:.1f}" y="535" text-anchor="middle" class="label">{tokens} tokens</text>'
        )
        lines.append(
            f'  <text x="{center:.1f}" y="555" text-anchor="middle" class="tick">first n=1; later n={later_n}</text>'
        )
    lines += [
        '  <text x="525" y="590" text-anchor="middle" class="label">Actual prompt evaluation count</text>',
        '  <text x="70" y="625" class="note">Methodology: “first” is trial 0 within each measured token group; “later” is the median of trials 1–4.</text>',
        '  <text x="70" y="649" class="note">Caveat: the run was already warm; this comparison does not isolate KV-prefix reuse from model residency or OS caching.</text>',
        "</svg>",
    ]
    return "\n".join(lines) + "\n"


def render_b4(points: list[tuple[int, float, float, int]]) -> str:
    lines = svg_start(
        "B4 concurrency saturation",
        "A dual-axis line chart shows aggregate generated tokens per second and p95 request latency at concurrency 1, 4, 8, and 16.",
    )
    lines += [
        '  <text x="70" y="48" class="title">B4: concurrency saturation</text>',
        '  <text x="70" y="76" class="subtitle">Aggregate throughput plateaus while tail latency continues to rise</text>',
    ]
    left, top, plot_w, plot_h = 95, 125, 835, 375
    throughput_max, latency_max = 5.0, 100.0
    for tick in range(0, 6):
        y = top + plot_h - tick / throughput_max * plot_h
        lines.append(
            f'  <line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" class="grid"/>'
        )
        lines.append(
            f'  <text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" class="tick">{tick}</text>'
        )
        lines.append(
            f'  <text x="{left + plot_w + 14}" y="{y + 5:.1f}" class="tick">{tick * 20}</text>'
        )
    lines += [
        f'  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>',
        f'  <line x1="{left + plot_w}" y1="{top}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
        f'  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>',
        '  <text x="24" y="320" transform="rotate(-90 24 320)" class="label">Aggregate tokens/second</text>',
        '  <text x="1010" y="320" transform="rotate(90 1010 320)" class="label">p95 request latency (seconds)</text>',
        '  <line x1="620" y1="98" x2="650" y2="98" stroke="#2457c5" stroke-width="4"/><text x="660" y="103" class="tick">Throughput</text>',
        '  <line x1="770" y1="98" x2="800" y2="98" stroke="#d1495b" stroke-width="4"/><text x="810" y="103" class="tick">p95 latency</text>',
    ]
    step = plot_w / (len(points) - 1)
    coordinates = []
    for index, (_, throughput, p95, _) in enumerate(points):
        x = left + index * step
        y_throughput = top + plot_h - throughput / throughput_max * plot_h
        y_latency = top + plot_h - p95 / latency_max * plot_h
        coordinates.append((x, y_throughput, y_latency))
    lines.append(
        '  <polyline points="'
        + " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in coordinates)
        + '" fill="none" stroke="#2457c5" stroke-width="4"/>'
    )
    lines.append(
        '  <polyline points="'
        + " ".join(f"{x:.1f},{y:.1f}" for x, _, y in coordinates)
        + '" fill="none" stroke="#d1495b" stroke-width="4"/>'
    )
    for (concurrency, throughput, p95, count), (x, y_throughput, y_latency) in zip(
        points, coordinates, strict=True
    ):
        lines += [
            f'  <circle cx="{x:.1f}" cy="{y_throughput:.1f}" r="6" fill="#2457c5"/>',
            f'  <text x="{x:.1f}" y="{y_throughput - 12:.1f}" text-anchor="middle" class="value">{throughput:.2f} tok/s</text>',
            f'  <circle cx="{x:.1f}" cy="{y_latency:.1f}" r="6" fill="#d1495b"/>',
            f'  <text x="{x:.1f}" y="{y_latency + 22:.1f}" text-anchor="middle" class="value">{p95:.2f}s</text>',
            f'  <text x="{x:.1f}" y="527" text-anchor="middle" class="label">c{concurrency}</text>',
            f'  <text x="{x:.1f}" y="548" text-anchor="middle" class="tick">n={count}</text>',
        ]
    lines += [
        '  <text x="512" y="580" text-anchor="middle" class="label">Requested concurrency</text>',
        '  <text x="70" y="618" class="note">Methodology: throughput is generated tokens divided by batch wall time; p95 uses linear interpolation of total request time.</text>',
        '  <text x="70" y="642" class="note">Caveat: each level is one batch with unequal sample counts; results describe this machine and model, not universal capacity.</text>',
        "</svg>",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    FIGURES.mkdir(parents=True, exist_ok=True)
    (FIGURES / "b2-first-vs-repeat.svg").write_text(
        render_b2(b2_points(data)), encoding="utf-8", newline="\n"
    )
    (FIGURES / "b4-saturation.svg").write_text(
        render_b4(b4_points(data)), encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
