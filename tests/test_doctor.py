from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import synapse_bench.doctor_io as doctor_io
from synapse_bench.cli import app
from synapse_bench.doctor import analyze_artifact
from synapse_bench.doctor_io import (
    DoctorInputError,
    load_artifact,
    render_json,
    render_markdown,
    write_reports,
)
from synapse_bench.doctor_models import Source
from synapse_bench.models import RunArtifact, ScenarioResult, Status, TrialResult

BASELINE = Path(__file__).parents[1] / "benchmarks" / "results" / "baseline.json"


def _source() -> Source:
    return Source(
        run_id="test-run",
        artifact_schema_version="1.0",
        artifact_sha256="0" * 64,
        artifact_name="test.json",
    )


def _trial(
    scenario: str,
    trial: int,
    *,
    ttft: float = 1.0,
    prefix: int | None = None,
    concurrency: int | None = None,
    total: float = 1.0,
    throughput: float = 1.0,
    status: Status = Status.SUCCESS,
) -> TrialResult:
    return TrialResult(
        scenario=scenario,
        trial=trial,
        status=status,
        model="local-model",
        ttft_s=ttft,
        total_s=total,
        tokens_per_second=throughput,
        prefix_tokens_target=prefix,
        concurrency=concurrency,
        prompt_sha256="0" * 64,
    )


def _artifact(scenarios: list[ScenarioResult]) -> RunArtifact:
    return RunArtifact(
        run_id="test-run",
        status=Status.SUCCESS,
        environment={},
        configuration={},
        scenarios=scenarios,
    )


def _baseline_document() -> dict[str, Any]:
    value: dict[str, Any] = json.loads(BASELINE.read_text(encoding="utf-8"))
    return value


def test_baseline_has_expected_diagnoses_and_closed_evidence_gate() -> None:
    artifact, source = load_artifact(BASELINE)

    report = analyze_artifact(artifact, source)
    ids = {item.id for item in report.diagnoses}

    assert "SYN-DOC-RESIDENCY-001" in ids
    assert {f"SYN-DOC-PREFIX-TAIL-{size:04d}" for size in (256, 512, 1024)} <= ids
    assert "SYN-DOC-CONCURRENCY-KNEE-001" in ids
    assert "SYN-DOC-CONCURRENCY-DOMINATED-0016" in ids
    assert "SYN-DOC-CONCURRENCY-DOMINATED-0008" not in ids
    assert "SYN-DOC-EVICTION-001" in ids
    assert report.summary.proxy_gate == "not_evaluable"
    assert report.status == "limited"
    concurrency = [item for item in report.diagnoses if item.category == "concurrency"]
    assert concurrency
    assert all(item.confidence != "high" for item in concurrency)
    assert any("unverified producer summary" in item for item in report.limitations)


def test_residency_threshold_is_inclusive() -> None:
    cold_trials = [_trial("B0_runtime_cold", index, ttft=4.0) for index in range(5)]
    warm_trials = [_trial("B1_stable_prefix", index, ttft=2.8) for index in range(4)]
    artifact = _artifact(
        [
            ScenarioResult(
                name="B0_runtime_cold",
                status=Status.SUCCESS,
                trials=cold_trials,
                summary={"ttft_median_s": 4.0, "ttft_successful_trials": 5},
            ),
            ScenarioResult(
                name="B1_stable_prefix",
                status=Status.SUCCESS,
                trials=warm_trials,
                summary={"ttft_median_s": 2.8, "ttft_successful_trials": 4},
            ),
        ]
    )

    report = analyze_artifact(artifact, _source())
    diagnosis = next(item for item in report.diagnoses if item.id == "SYN-DOC-RESIDENCY-001")

    assert diagnosis.title == "Controlled residency experiment is warranted"


def test_tail_threshold_is_inclusive() -> None:
    # Linear p95 for these four values is exactly 3.0 and median is exactly 1.0.
    values = [1.0, 1.0, 1.0, 3.3529411764705883]
    trials = [
        _trial("B2_prefix_scaling", i, ttft=value, prefix=256) for i, value in enumerate(values)
    ]
    artifact = _artifact(
        [
            ScenarioResult(
                name="B2_prefix_scaling",
                status=Status.SUCCESS,
                trials=trials,
                summary={
                    "256_ttft_median_s": 1.0,
                    "256_ttft_p95_s": 3.0,
                    "256_ttft_successful_trials": 4,
                },
            )
        ]
    )

    report = analyze_artifact(artifact, _source())

    assert any(item.id == "SYN-DOC-PREFIX-TAIL-0256" for item in report.diagnoses)


def test_summary_contradiction_is_limited_warning() -> None:
    trials = [_trial("B0_runtime_cold", index, ttft=4.0) for index in range(5)]
    artifact = _artifact(
        [
            ScenarioResult(
                name="B0_runtime_cold",
                status=Status.SUCCESS,
                trials=trials,
                summary={"ttft_median_s": 99.0, "ttft_successful_trials": 5},
            ),
            ScenarioResult(
                name="B1_stable_prefix",
                status=Status.SUCCESS,
                trials=[_trial("B1_stable_prefix", index, ttft=2.0) for index in range(4)],
                summary={"ttft_median_s": 2.0, "ttft_successful_trials": 4},
            ),
        ]
    )

    report = analyze_artifact(artifact, _source())

    assert report.status == "limited"
    limitation = next(
        item for item in report.diagnoses if item.id == "SYN-DOC-ARTIFACT-LIMITED-001"
    )
    assert limitation.severity == "warning"
    residency = next(item for item in report.diagnoses if item.id == "SYN-DOC-RESIDENCY-001")
    assert residency.evidence.value == 50.0


def test_malicious_b4_throughput_summary_never_becomes_trusted(tmp_path: Path) -> None:
    path = tmp_path / "malicious-throughput.json"
    document = _baseline_document()
    b4 = next(item for item in document["scenarios"] if item["name"] == "B4_concurrency")
    b4["summary"]["c1_tokens_per_second"] = 999_999.0
    path.write_text(json.dumps(document), encoding="utf-8")

    artifact, source = load_artifact(path)
    report = analyze_artifact(artifact, source)
    concurrency = [item for item in report.diagnoses if item.category == "concurrency"]

    assert report.status == "limited"
    assert concurrency
    assert all(item.confidence in {"low", "medium"} for item in concurrency)
    assert all(
        any("unverified producer summary" in limitation for limitation in item.limitations)
        for item in concurrency
    )


def test_duplicate_and_mismatched_trial_identities_are_excluded() -> None:
    cold = [_trial("B0_runtime_cold", index, ttft=4.0) for index in range(5)]
    cold[1] = cold[1].model_copy(update={"trial": 0})
    cold[2] = cold[2].model_copy(update={"scenario": "wrong-container"})
    warm = [_trial("B1_stable_prefix", index, ttft=2.0) for index in range(4)]
    artifact = _artifact(
        [
            ScenarioResult(name="B0_runtime_cold", status=Status.SUCCESS, trials=cold),
            ScenarioResult(name="B1_stable_prefix", status=Status.SUCCESS, trials=warm),
        ]
    )

    report = analyze_artifact(artifact, _source())

    assert report.status == "limited"
    assert any("trial identities" in item for item in report.limitations)
    assert not any(item.id == "SYN-DOC-RESIDENCY-001" for item in report.diagnoses)


def test_failure_rate_at_twenty_percent_is_critical() -> None:
    trials = [_trial("custom", index) for index in range(4)]
    trials.append(_trial("custom", 4, status=Status.FAILED))
    artifact = _artifact([ScenarioResult(name="custom", status=Status.PARTIAL, trials=trials)])

    report = analyze_artifact(artifact, _source())
    failure = next(item for item in report.diagnoses if item.id == "SYN-DOC-FAILURES-001")

    assert failure.severity == "critical"
    assert failure.evidence.value == 20.0


def test_input_rejects_prompt_payload_without_echoing_it(tmp_path: Path) -> None:
    path = tmp_path / "private.json"
    document = _baseline_document()
    document["prompt"] = "TOP SECRET PROMPT"
    path.write_text(json.dumps(document), encoding="utf-8")

    result = CliRunner().invoke(app, ["doctor", str(path)])

    assert result.exit_code == 2
    assert "TOP SECRET PROMPT" not in result.output
    assert "prompt-bearing field" in result.output


@pytest.mark.parametrize("payload", ["not json", "[]", '{"schema_version":"2.0"}'])
def test_input_rejects_malformed_or_wrong_schema(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(DoctorInputError):
        load_artifact(path)


def test_input_accepts_v1_extra_keys(tmp_path: Path) -> None:
    path = tmp_path / "extra.json"
    document = _baseline_document()
    document["future_metadata"] = {"safe": True}
    path.write_text(json.dumps(document), encoding="utf-8")

    artifact, _ = load_artifact(path)

    assert artifact.run_id == document["run_id"]


def test_input_rejects_negative_consumed_metric(tmp_path: Path) -> None:
    path = tmp_path / "negative.json"
    document = _baseline_document()
    document["scenarios"][0]["trials"][0]["ttft_s"] = -1.0
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(DoctorInputError, match="invalid consumed numeric metric"):
        load_artifact(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_input_rejects_nonstandard_json_constants_anywhere(tmp_path: Path, constant: str) -> None:
    path = tmp_path / "constant.json"
    path.write_text(
        '{"schema_version":"1.0","run_id":"x","environment":{},'
        f'"configuration":{{}},"scenarios":[],"extra":{constant}}}',
        encoding="utf-8",
    )

    with pytest.raises(DoctorInputError, match="valid UTF-8 JSON"):
        load_artifact(path)


def test_input_rejects_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "invalid-utf8.json"
    path.write_bytes(b'{"schema_version":"1.0","extra":"\xff"}')

    with pytest.raises(DoctorInputError, match="valid UTF-8 JSON"):
        load_artifact(path)


def test_input_rejects_more_than_one_hundred_scenarios(tmp_path: Path) -> None:
    path = tmp_path / "too-many.json"
    document = {
        "schema_version": "1.0",
        "run_id": "bounded-run",
        "status": "success",
        "environment": {},
        "configuration": {},
        "scenarios": [
            {"name": f"scenario-{index}", "status": "success", "trials": []} for index in range(101)
        ],
    }
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(DoctorInputError, match="100 scenarios"):
        load_artifact(path)


def test_reports_never_expose_host_or_raw_errors(tmp_path: Path) -> None:
    path = tmp_path / "private-errors.json"
    document = _baseline_document()
    secret = "PRIVATE-ERROR-CONTENT"
    document["configuration"]["ollama_host"] = "http://192.0.2.99:11434"
    document["scenarios"][0]["trials"][0]["error_message"] = secret
    path.write_text(json.dumps(document), encoding="utf-8")
    artifact, source = load_artifact(path)

    report = analyze_artifact(artifact, source)
    output = render_json(report) + render_markdown(report)

    assert secret not in output
    assert "192.0.2.99" not in output
    assert str(path.resolve()) not in output


def test_doctor_does_not_use_network_environment_or_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("external state access is forbidden")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(os, "getenv", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    artifact, source = load_artifact(BASELINE)

    report = analyze_artifact(artifact, source)

    assert report.source.run_id == artifact.run_id


def test_output_collision_creates_neither_report(tmp_path: Path) -> None:
    artifact, source = load_artifact(BASELINE)
    report = analyze_artifact(artifact, source)
    existing = tmp_path / f"{artifact.run_id}.doctor.v1.md"
    existing.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_reports(report, tmp_path)

    assert existing.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / f"{artifact.run_id}.doctor.v1.json").exists()


def test_second_atomic_publish_failure_rolls_back_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, source = load_artifact(BASELINE)
    report = analyze_artifact(artifact, source)
    real_link = os.link
    calls = 0

    def fail_second(source_path: Path, destination_path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second publish failure")
        real_link(source_path, destination_path)

    monkeypatch.setattr(os, "link", fail_second)

    with pytest.raises(OSError):
        write_reports(report, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_second_temporary_creation_failure_cleans_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, source = load_artifact(BASELINE)
    report = analyze_artifact(artifact, source)
    real_temporary = doctor_io._temporary
    calls = 0

    def fail_second(path: Path, content: str) -> Path:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated temporary creation failure")
        return real_temporary(path, content)

    monkeypatch.setattr(doctor_io, "_temporary", fail_second)

    with pytest.raises(OSError):
        write_reports(report, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_cli_stdout_writes_nothing_and_strict_reports_findings(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["doctor", str(BASELINE)])
    strict = CliRunner().invoke(app, ["doctor", str(BASELINE), "--strict"])

    assert result.exit_code == 0
    assert result.output.startswith("# Synapse Doctor Report")
    assert strict.exit_code == 3
    assert list(tmp_path.iterdir()) == []


def test_cli_output_dir_creates_both_and_refuses_overwrite(tmp_path: Path) -> None:
    first = CliRunner().invoke(app, ["doctor", str(BASELINE), "--output-dir", str(tmp_path)])
    second = CliRunner().invoke(app, ["doctor", str(BASELINE), "--output-dir", str(tmp_path)])

    assert first.exit_code == 0
    assert len(list(tmp_path.glob("*.doctor.v1.*"))) == 2
    assert second.exit_code == 1
    assert "refusing to overwrite" in second.output


def test_markdown_escapes_untrusted_artifact_name(tmp_path: Path) -> None:
    path = tmp_path / "artifact_`&raw.json"
    path.write_text(BASELINE.read_text(encoding="utf-8"), encoding="utf-8")
    artifact, source = load_artifact(path)

    output = render_markdown(analyze_artifact(artifact, source))

    assert "artifact_`&raw.json" not in output
    assert "artifact_&#96;&amp;raw.json" in output


def test_markdown_normalizes_controls_and_renders_contract_fields() -> None:
    artifact, source = load_artifact(BASELINE)
    source = source.model_copy(update={"artifact_name": "evil\r\nheading\x00.json"})
    report = analyze_artifact(artifact, source)

    output = render_markdown(report)

    assert "evil\r\nheading" not in output
    assert "evil heading .json" in output
    assert "Rationale:" in output
    assert "Diagnoses:" in output
    assert "Ollama controls:" in output
    assert "Limitations:" in output
    assert "Evidence: `" in output


def test_missing_metadata_rejects_null_and_unknown_placeholders() -> None:
    artifact, source = load_artifact(BASELINE)
    artifact = artifact.model_copy(
        update={
            "environment": {"ram_bytes": None, "gpu": "unknown"},
            "configuration": {"num_ctx": "", "keep_alive": "none"},
        }
    )

    report = analyze_artifact(artifact, source)
    recommendation = next(
        item for item in report.recommendations if item.id == "SYN-REC-ARTIFACT-RERUN-001"
    )

    assert "RAM capacity" in " ".join(report.limitations)
    assert recommendation.title == "Capture richer benchmark metadata"
    assert "current runner alone cannot resolve" in recommendation.action
    assert "remain limited for B4" in recommendation.verification
    assert "future schema and analyzer" in recommendation.verification
    assert "recomputable batch elapsed time and token totals" in recommendation.verification
    assert all(
        "reports complete" not in item.verification.casefold()
        and "strict success" not in item.verification.casefold()
        for item in report.recommendations
    )
    assert any(
        "10 milliseconds median" in item for item in report.acceptance.proxy_gate_requirements
    )
