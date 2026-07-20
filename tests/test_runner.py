from typing import Any

import pytest

from synapse_bench.config import BenchmarkSettings
from synapse_bench.models import RunArtifact, ScenarioResult, Status, TrialResult
from synapse_bench.runner import (
    _add_b1_baseline_comparison,
    _aggregate_run_status,
    _finish_scenario,
    _status,
    benchmark_b0,
    benchmark_b1,
    benchmark_b2,
    benchmark_b3,
    benchmark_b4,
)


def _trial(status: Status, total: float | None = 1.0) -> TrialResult:
    return TrialResult(
        scenario="test",
        trial=0,
        status=status,
        model="model",
        ttft_s=total / 2 if total is not None else None,
        total_s=total,
        prompt_sha256="0" * 64,
    )


def test_status_is_partial_when_minimum_succeeds_with_failures() -> None:
    assert _status([_trial(Status.SUCCESS), _trial(Status.FAILED)], minimum_successes=1) == (
        Status.PARTIAL
    )


def test_finish_scenario_fails_when_evidence_is_insufficient() -> None:
    scenario = ScenarioResult(
        name="test", status=Status.PARTIAL, trials=[_trial(Status.SUCCESS), _trial(Status.FAILED)]
    )

    finished = _finish_scenario(scenario, minimum_successes=2)

    assert finished.status == Status.FAILED
    assert finished.finished_at is not None


class FakeClient:
    def __init__(self, *, evict_on_switch: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.preloads: list[str] = []
        self.resident: set[str] = set()
        self.evict_on_switch = evict_on_switch

    async def unload(self, model: str) -> bool:
        self.resident.discard(model)
        return True

    async def preload(self, model: str) -> bool:
        self.preloads.append(model)
        self.resident.add(model)
        return True

    async def installed_models(self) -> set[str]:
        return {"primary", "secondary"}

    async def resident_models(self) -> set[str]:
        return set(self.resident)

    async def generate(self, **kwargs: Any) -> TrialResult:
        self.calls.append(kwargs)
        model = str(kwargs["model"])
        if self.evict_on_switch:
            self.resident.difference_update({"primary", "secondary"} - {model})
        self.resident.add(model)
        return TrialResult(
            scenario=str(kwargs["scenario"]),
            trial=int(kwargs["trial"]),
            status=Status.SUCCESS,
            model=model,
            prefix_tokens_target=kwargs.get("prefix_tokens_target"),
            concurrency=kwargs.get("concurrency"),
            ttft_s=1.0,
            total_s=2.0,
            prompt_sha256="0" * 64,
        )


def _settings(profile: str = "full") -> BenchmarkSettings:
    return BenchmarkSettings(
        ollama_host="http://127.0.0.1:11434",
        primary_model="primary",
        secondary_model="secondary",
        profile=profile,
    )


@pytest.mark.asyncio
async def test_b0_full_uses_seven_512_token_runtime_cold_trials() -> None:
    client = FakeClient()

    result = await benchmark_b0(client, _settings(), lambda _result: None)  # type: ignore[arg-type]

    assert len(result.trials) == 7
    assert {trial.prefix_tokens_target for trial in result.trials} == {512}
    assert result.summary["ttft_median_s"] == 1.0
    assert result.summary["total_median_s"] == 2.0


@pytest.mark.asyncio
async def test_b1_full_uses_five_suffixes() -> None:
    client = FakeClient()

    result = await benchmark_b1(client, _settings(), lambda _result: None)  # type: ignore[arg-type]

    assert len(result.trials) == 5
    assert client.preloads == ["primary"]
    assert [trial.trial for trial in result.trials] == [0, 1, 2, 3, 4]
    assert result.summary["ttft_median_s"] == 1.0


@pytest.mark.asyncio
async def test_b2_full_uses_five_trials_at_all_four_targets() -> None:
    client = FakeClient()

    result = await benchmark_b2(client, _settings(), lambda _result: None)  # type: ignore[arg-type]

    assert len(result.trials) == 20
    for target in (128, 256, 512, 1024):
        assert sum(trial.prefix_tokens_target == target for trial in result.trials) == 5
        assert result.summary[f"{target}_ttft_median_s"] == 1.0
        assert result.summary[f"{target}_total_median_s"] == 2.0


@pytest.mark.asyncio
async def test_b3_full_alternates_ten_requests() -> None:
    client = FakeClient(evict_on_switch=True)

    result = await benchmark_b3(client, _settings(), lambda _result: None)  # type: ignore[arg-type]

    alternating = [trial for trial in result.trials if trial.measurement_kind == "standard"]
    recoveries = [
        trial for trial in result.trials if trial.measurement_kind == "evicted_runtime_cold"
    ]
    assert len(alternating) == 10
    assert [trial.model for trial in alternating] == ["primary", "secondary"] * 5
    assert {trial.model for trial in recoveries} == {"primary", "secondary"}
    assert all(trial.resident_models_before is not None for trial in alternating)
    assert any(trial.evicted_models for trial in alternating)


@pytest.mark.asyncio
async def test_b4_full_runs_all_bounded_concurrency_levels() -> None:
    client = FakeClient()

    result = await benchmark_b4(client, _settings(), lambda _result: None)  # type: ignore[arg-type]

    assert len(result.trials) == 29
    assert [
        sum(trial.concurrency == level for trial in result.trials) for level in (1, 4, 8, 16)
    ] == [
        1,
        4,
        8,
        16,
    ]
    assert result.summary["c16_successful"] == 16


def test_b1_comparison_uses_b0_median_ttft_without_creating_gate() -> None:
    b0 = ScenarioResult(
        name="B0_runtime_cold", status=Status.SUCCESS, summary={"ttft_median_s": 10.0}
    )
    b1 = ScenarioResult(
        name="B1_stable_prefix", status=Status.SUCCESS, summary={"ttft_median_s": 5.0}
    )
    artifact = RunArtifact(run_id="test", environment={}, configuration={}, scenarios=[b0, b1])

    _add_b1_baseline_comparison(artifact)

    assert b1.summary["baseline_b0_ttft_median_s"] == 10.0
    assert b1.summary["ttft_reduction_vs_b0_percent"] == 50.0
    assert "descriptive evidence, not a pass/fail gate" in b1.notes[0]
    assert b1.status == Status.SUCCESS


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ({Status.SUCCESS}, Status.SUCCESS),
        ({Status.INFEASIBLE}, Status.INFEASIBLE),
        ({Status.FAILED}, Status.FAILED),
        ({Status.SUCCESS, Status.INFEASIBLE}, Status.PARTIAL),
        ({Status.SUCCESS, Status.FAILED}, Status.PARTIAL),
        ({Status.FAILED, Status.INFEASIBLE}, Status.FAILED),
        ({Status.PARTIAL, Status.INFEASIBLE}, Status.PARTIAL),
        (set(), Status.FAILED),
    ],
)
def test_aggregate_run_status_precedence(statuses: set[Status], expected: Status) -> None:
    assert _aggregate_run_status(statuses) == expected


@pytest.mark.asyncio
async def test_b0_records_unload_lifecycle_timeout_as_setup_failure() -> None:
    class TimeoutClient(FakeClient):
        async def unload(self, model: str) -> bool:
            raise TimeoutError(f"unload timed out for {model}")

    result = await benchmark_b0(
        TimeoutClient(),
        _settings("quick"),
        lambda _result: None,  # type: ignore[arg-type]
    )

    assert len(result.trials) == 3
    assert {trial.error_type for trial in result.trials} == {"TimeoutError"}
