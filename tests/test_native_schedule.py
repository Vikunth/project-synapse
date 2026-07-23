import json

from synapse_bench.native_schedule import (
    balanced_orders,
    build_schedule,
    domain_seed,
    schedule_sha256,
    synthetic_namespace,
)


def test_full_schedules_are_balanced_and_concurrency_blocks_are_permutations() -> None:
    schedule = build_schedule(42, "full", {"prefix-reuse", "dual-residency", "concurrency"})

    assert schedule["prefix"].count(["reuse", "control"]) == 15
    assert schedule["prefix"].count(["control", "reuse"]) == 15
    assert schedule["dual"].count(["solo", "dual"]) == 15
    assert schedule["dual"].count(["dual", "solo"]) == 15
    assert len(schedule["concurrency"]) == 10
    assert all(sorted(block) == [1, 4, 8, 16] for block in schedule["concurrency"])


def test_domain_separation_keeps_other_probe_schedule_stable() -> None:
    all_probes = build_schedule(9, "full", {"prefix-reuse", "dual-residency", "concurrency"})
    prefix_only = build_schedule(9, "full", {"prefix-reuse"})

    assert all_probes["prefix"] == prefix_only["prefix"]
    assert domain_seed(9, "prefix") != domain_seed(9, "dual")


def test_schedule_replay_and_hash_are_stable() -> None:
    left = build_schedule(123, "smoke", {"prefix-reuse", "concurrency"})
    right = build_schedule(123, "smoke", {"prefix-reuse", "concurrency"})

    assert left == right
    assert schedule_sha256(left) == schedule_sha256(json.loads(json.dumps(right)))


def test_smoke_dual_order_is_seeded_and_can_use_either_order() -> None:
    schedules = [build_schedule(seed, "smoke", {"dual-residency"})["dual"] for seed in range(20)]

    assert all(schedule in [[["solo", "dual"]], [["dual", "solo"]]] for schedule in schedules)
    assert len({tuple(schedule[0]) for schedule in schedules}) == 2


def test_synthetic_namespaces_are_unique_matched_length_and_differ_at_start() -> None:
    values = [synthetic_namespace(42, "case", str(index), 128) for index in range(20)]

    assert len(set(values)) == len(values)
    assert len({len(value.split()) for value in values}) == 1
    assert len({value.split()[0] for value in values}) == len(values)


def test_balanced_order_rejects_odd_count() -> None:
    try:
        balanced_orders(3, 1)
    except ValueError as error:
        assert str(error) == "balanced schedules require an even count"
    else:
        raise AssertionError("odd schedule was accepted")
