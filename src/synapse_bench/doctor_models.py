"""Versioned, prompt-free output models for offline benchmark diagnosis."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Severity = Literal["info", "warning", "critical"]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: str
    metric: str
    value: float | int | str | bool
    unit: str
    sample_count: int


class Diagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    severity: Severity
    category: str
    title: str
    finding: str
    confidence: Literal["low", "medium", "high"]
    evidence: Evidence
    limitations: list[str] = Field(default_factory=list)
    recommendation_ids: list[str] = Field(default_factory=list)


class OllamaControl(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    scope: str
    current_value: Literal["unknown"] = "unknown"
    proposed_value: None = None
    docs_url: None = None


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    priority: Literal["now", "next", "defer"]
    type: Literal["observe", "experiment", "configure", "do_not_change"]
    title: str
    rationale: str
    ollama_controls: list[OllamaControl] = Field(default_factory=list)
    action: str
    verification: str
    risk: str
    diagnosis_ids: list[str] = Field(default_factory=list)


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    artifact_schema_version: str
    artifact_sha256: str
    artifact_name: str


class ReportSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    headline: str
    diagnosis_counts: dict[Severity, int]
    proxy_gate: Literal["not_evaluable", "closed"]


class Acceptance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_complete: bool
    proxy_gate_requirements: list[str]


class DoctorReport(BaseModel):
    """Stable V1 output contract for ``synapse doctor``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    kind: Literal["synapse.doctor.report"] = "synapse.doctor.report"
    analyzer_version: str
    source: Source
    status: Literal["complete", "limited"]
    summary: ReportSummary
    diagnoses: list[Diagnosis]
    recommendations: list[Recommendation]
    limitations: list[str]
    acceptance: Acceptance
