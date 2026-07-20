from synapse_bench.models import ScenarioResult, Status, TrialResult
from synapse_bench.runner import _finish_scenario, _status


def _trial(status: Status, total: float | None = 1.0) -> TrialResult:
    return TrialResult(
        scenario="test",
        trial=0,
        status=status,
        model="model",
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
