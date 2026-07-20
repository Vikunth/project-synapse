"""Bounded input and atomic, non-overwriting report output."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from synapse_bench.doctor import contains_private_payload
from synapse_bench.doctor_models import DoctorReport, Source
from synapse_bench.models import RunArtifact

MAX_ARTIFACT_BYTES = 10 * 1024 * 1024
MAX_SCENARIOS = 100
MAX_TRIALS = 10_000
SAFE_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class DoctorInputError(ValueError):
    """Safe, non-payload-bearing input failure."""


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-standard numeric constant")


def _validate_consumed_numbers(artifact: RunArtifact) -> None:
    values: list[float | int | None] = []
    for scenario in artifact.scenarios:
        values.extend(
            value for value in scenario.summary.values() if isinstance(value, (int, float))
        )
        for trial in scenario.trials:
            values.extend(
                [
                    trial.prefix_tokens_target,
                    trial.concurrency,
                    trial.ttft_s,
                    trial.total_s,
                    trial.tokens_per_second,
                ]
            )
    for value in values:
        if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value < 0):
            raise DoctorInputError("artifact contains an invalid consumed numeric metric")


def load_artifact(path: Path) -> tuple[RunArtifact, Source]:
    """Read one bounded V1 artifact without following symlinks."""
    if path.is_symlink():
        raise DoctorInputError("artifact must not be a symbolic link")
    try:
        stat = path.stat()
    except OSError as error:
        raise DoctorInputError("artifact cannot be read") from error
    if not path.is_file():
        raise DoctorInputError("artifact must be a regular file")
    if stat.st_size > MAX_ARTIFACT_BYTES:
        raise DoctorInputError(f"artifact exceeds {MAX_ARTIFACT_BYTES} bytes")
    try:
        raw = path.read_bytes()
        decoded = raw.decode("utf-8", errors="strict")
        document: Any = json.loads(decoded, parse_constant=_reject_json_constant)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise DoctorInputError("artifact must be valid UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise DoctorInputError("artifact root must be an object")
    if document.get("schema_version") != "1.0":
        raise DoctorInputError("artifact schema_version must be 1.0")
    if contains_private_payload(document):
        raise DoctorInputError("artifact contains a prohibited prompt-bearing field")
    try:
        artifact = RunArtifact.model_validate(document)
    except ValidationError as error:
        raise DoctorInputError("artifact does not match RunArtifact schema version 1.0") from error
    if not SAFE_RUN_ID.fullmatch(artifact.run_id):
        raise DoctorInputError("artifact run_id is not safe for report filenames")
    if len(artifact.scenarios) > MAX_SCENARIOS:
        raise DoctorInputError(f"artifact exceeds {MAX_SCENARIOS} scenarios")
    if sum(len(scenario.trials) for scenario in artifact.scenarios) > MAX_TRIALS:
        raise DoctorInputError(f"artifact exceeds {MAX_TRIALS} trials")
    _validate_consumed_numbers(artifact)
    source = Source(
        run_id=artifact.run_id,
        artifact_schema_version=artifact.schema_version,
        artifact_sha256=hashlib.sha256(raw).hexdigest(),
        artifact_name=path.name,
    )
    return artifact, source


def render_json(report: DoctorReport) -> str:
    return json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def _text(value: object) -> str:
    normalized = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in str(value)
    )
    normalized = " ".join(normalized.split())
    return html.escape(normalized, quote=True).replace("`", "&#96;")


def render_markdown(report: DoctorReport) -> str:
    """Render deterministic Markdown while escaping all artifact-derived text."""
    lines = [
        "# Synapse Doctor Report",
        "",
        f"- Run: `{_text(report.source.run_id)}`",
        f"- Artifact: `{_text(report.source.artifact_name)}`",
        f"- Analysis status: `{report.status}`",
        f"- Proxy gate: `{report.summary.proxy_gate}`",
        "",
        _text(report.summary.headline),
        "",
        "## Diagnoses",
        "",
    ]
    if not report.diagnoses:
        lines.extend(["No diagnostic rule matched.", ""])
    for diagnosis in report.diagnoses:
        lines.extend(
            [
                f"### {diagnosis.id}: {_text(diagnosis.title)}",
                "",
                f"Severity: `{diagnosis.severity}`; Confidence: `{diagnosis.confidence}`",
                "",
                _text(diagnosis.finding),
                "",
                (
                    f"Evidence: `{_text(diagnosis.evidence.scenario)}` / "
                    f"`{_text(diagnosis.evidence.metric)}` = "
                    f"`{_text(diagnosis.evidence.value)}` "
                    f"`{_text(diagnosis.evidence.unit)}` "
                    f"(n={diagnosis.evidence.sample_count})."
                ),
                "",
                "Limitations: "
                + (
                    "; ".join(_text(item) for item in diagnosis.limitations)
                    if diagnosis.limitations
                    else "None."
                ),
                "",
                "Recommendations: "
                + (
                    ", ".join(f"`{_text(item)}`" for item in diagnosis.recommendation_ids)
                    if diagnosis.recommendation_ids
                    else "None."
                ),
                "",
            ]
        )
    lines.extend(["## Recommendations", ""])
    for recommendation in report.recommendations:
        lines.extend(
            [
                f"### {recommendation.id}: {_text(recommendation.title)}",
                "",
                f"Priority: `{recommendation.priority}`; Type: `{recommendation.type}`",
                "",
                f"Rationale: {_text(recommendation.rationale)}",
                "",
                _text(recommendation.action),
                "",
                f"Verification: {_text(recommendation.verification)}",
                "",
                f"Risk: {_text(recommendation.risk)}",
                "",
                "Diagnoses: "
                + (
                    ", ".join(f"`{_text(item)}`" for item in recommendation.diagnosis_ids)
                    if recommendation.diagnosis_ids
                    else "None."
                ),
                "",
            ]
        )
        if recommendation.ollama_controls:
            lines.extend(["Ollama controls:", ""])
            for control in recommendation.ollama_controls:
                lines.append(
                    "- "
                    f"`{_text(control.name)}`; scope={_text(control.scope)}; "
                    f"current={_text(control.current_value)}; "
                    f"proposed={_text(control.proposed_value)}; "
                    f"docs={_text(control.docs_url)}"
                )
            lines.append("")
    lines.extend(["## Limitations", ""])
    if report.limitations:
        lines.extend(f"- {_text(item)}" for item in report.limitations)
    else:
        lines.append("- None recorded.")
    lines.extend(["", "## Proxy Gate Requirements", ""])
    lines.extend(f"- {_text(item)}" for item in report.acceptance.proxy_gate_requirements)
    return "\n".join(lines).rstrip() + "\n"


def _temporary(path: Path, content: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def write_reports(report: DoctorReport, output_dir: Path) -> tuple[Path, Path]:
    """Stage both reports, publish each exclusively, and roll back caught failures."""
    if not output_dir.is_dir():
        raise DoctorInputError("output directory must already exist")
    json_path = output_dir / f"{report.source.run_id}.doctor.v1.json"
    markdown_path = output_dir / f"{report.source.run_id}.doctor.v1.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("refusing to overwrite an existing doctor report")
    temporary_json = _temporary(json_path, render_json(report))
    try:
        temporary_markdown = _temporary(markdown_path, render_markdown(report))
    except BaseException:
        temporary_json.unlink(missing_ok=True)
        raise
    created: list[Path] = []
    try:
        os.link(temporary_json, json_path)
        created.append(json_path)
        os.link(temporary_markdown, markdown_path)
        created.append(markdown_path)
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    finally:
        temporary_json.unlink(missing_ok=True)
        temporary_markdown.unlink(missing_ok=True)
    return json_path, markdown_path
