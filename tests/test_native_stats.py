from synapse_bench.native_stats import (
    bootstrap_median,
    paired_finding,
    prompt_counts_match,
    saturation_finding,
)


def test_prompt_count_matching_uses_two_token_or_one_percent_tolerance() -> None:
    assert prompt_counts_match(100, 102)
    assert not prompt_counts_match(100, 103)
    assert prompt_counts_match(1000, 1010)
    assert not prompt_counts_match(None, 100)


def test_bootstrap_is_deterministic_and_empty_is_explicit() -> None:
    assert bootstrap_median([1.0, 2.0, 3.0], 99, 100) == bootstrap_median([1.0, 2.0, 3.0], 99, 100)
    assert bootstrap_median([], 99)["valid_units"] == 0


def test_prefix_finding_threshold_boundaries() -> None:
    result = paired_finding(
        [0.15] * 30,
        [0.20] * 30,
        min_units=27,
        min_favors=24,
        absolute_gate=0.15,
        relative_gate=0.20,
        seed=1,
    )

    assert result["finding"] is True
    below = paired_finding(
        [0.149] * 30,
        [0.20] * 30,
        min_units=27,
        min_favors=24,
        absolute_gate=0.15,
        relative_gate=0.20,
        seed=1,
    )
    assert below["finding"] is False


def test_partial_pairs_remain_counted_as_insufficient() -> None:
    result = paired_finding(
        [2.0] * 26,
        [2.0] * 26,
        min_units=27,
        min_favors=24,
        absolute_gate=1.0,
        relative_gate=1.0,
        seed=1,
    )
    assert result["valid_units"] == 26
    assert result["finding"] is False


def test_saturation_requires_ten_complete_blocks_and_both_boundaries() -> None:
    result = saturation_finding([10.0] * 10, [10.5] * 10, [2.0] * 10, [3.0] * 10)
    assert result["finding"] is True
    assert saturation_finding([10.0] * 9, [10.0] * 9, [2.0] * 9, [4.0] * 9)["finding"] is False
    assert saturation_finding([10.0] * 10, [10.6] * 10, [2.0] * 10, [3.0] * 10)["finding"] is False
