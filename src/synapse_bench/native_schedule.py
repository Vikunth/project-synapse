"""Domain-separated deterministic schedules and synthetic namespaces."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from synapse_bench.native_models import ProbeSection, ProbeUnit

_WORDS = (
    "amber",
    "bridge",
    "cache",
    "delta",
    "ember",
    "forest",
    "gateway",
    "harbor",
    "index",
    "kernel",
    "local",
    "model",
    "native",
    "orbit",
    "prefix",
    "queue",
)


def domain_seed(seed: int, domain: str) -> int:
    digest = hashlib.sha256(f"native-probe-1.0:{seed}:{domain}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def balanced_orders(count: int, seed: int, labels: tuple[str, str] = ("A", "B")) -> list[list[str]]:
    if count % 2:
        raise ValueError("balanced schedules require an even count")
    orders = [[labels[0], labels[1]] for _ in range(count // 2)]
    orders += [[labels[1], labels[0]] for _ in range(count // 2)]
    random.Random(seed).shuffle(orders)
    return orders


def build_schedule(seed: int, profile: str, probes: set[str]) -> dict[str, Any]:
    prefix_n = 2 if profile == "smoke" else 30
    dual_n = 1 if profile == "smoke" else 30
    concurrency_rounds = 1 if profile == "smoke" else 10
    prefix_orders = (
        balanced_orders(prefix_n, domain_seed(seed, "prefix"), ("reuse", "control"))
        if "prefix-reuse" in probes
        else []
    )
    if "dual-residency" in probes:
        dual_orders = (
            [["solo", "dual"]]
            if dual_n == 1
            else balanced_orders(dual_n, domain_seed(seed, "dual"), ("solo", "dual"))
        )
    else:
        dual_orders = []
    levels = [1, 4] if profile == "smoke" else [1, 4, 8, 16]
    rng = random.Random(domain_seed(seed, "concurrency"))
    concurrency: list[list[int]] = []
    if "concurrency" in probes:
        for _ in range(concurrency_rounds):
            block = levels.copy()
            rng.shuffle(block)
            concurrency.append(block)
    return {"prefix": prefix_orders, "dual": dual_orders, "concurrency": concurrency}


def schedule_sha256(schedule: dict[str, Any]) -> str:
    wire = json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(wire).hexdigest()


def section_from_orders(prefix: str, orders: list[list[str | int]]) -> ProbeSection:
    return ProbeSection(
        planned_count=len(orders),
        units=[
            ProbeUnit(unit_id=f"{prefix}-{index:03d}", order=[str(v) for v in order])
            for index, order in enumerate(orders, 1)
        ],
    )


def synthetic_namespace(seed: int, domain: str, case_id: str, token_target: int = 512) -> str:
    """Return a unique matched-size namespace; callers must never persist it."""
    rng = random.Random(domain_seed(seed, f"{domain}:{case_id}"))
    marker = hashlib.sha256(f"{seed}:{domain}:{case_id}".encode()).hexdigest()[:20]
    words = [marker]
    while len(words) < max(2, round(token_target / 1.3)):
        words.append(rng.choice(_WORDS))
    return " ".join(words)
