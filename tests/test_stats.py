import pytest

from synapse_bench.stats import latency_summary, percentile


def test_percentile_interpolates() -> None:
    assert percentile([1.0, 2.0, 3.0, 4.0], 95) == pytest.approx(3.85)


def test_latency_summary_trims_one_value_from_each_end() -> None:
    summary = latency_summary([100.0, 2.0, 3.0, 4.0, 0.1], trim_extremes=True)

    assert summary["successful_trials"] == 5
    assert summary["included_trials"] == 3
    assert summary["median_s"] == 3.0


def test_latency_summary_handles_empty_input() -> None:
    assert latency_summary([])["median_s"] is None
