"""Versioned, prompt-free models for the native crossover probe."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class HostMemory(BaseModel):
    total_bytes: int
    available_bytes: int
    pagefile_total_bytes: int
    pagefile_used_bytes: int
    source: Literal["windows_cim", "wsl_proc"]


class ResidentModel(BaseModel):
    name: str
    digest: str | None = None
    size: int | None = None
    size_vram: int | None = None
    context_length: int | None = None
    expires_at: str | None = None


class Observation(BaseModel):
    case_id: str
    status: Literal["success", "failed"]
    ttft_s: float | None = None
    total_s: float | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    load_duration_s: float | None = None
    prompt_eval_duration_s: float | None = None
    eval_duration_s: float | None = None
    error_code: str | None = None
    telemetry_before: HostMemory | None = None
    telemetry_after: HostMemory | None = None
    residents_before: list[ResidentModel] = Field(default_factory=list)
    residents_after: list[ResidentModel] = Field(default_factory=list)


class BatchObservation(BaseModel):
    concurrency: int
    elapsed_s: float
    generated_tokens: int
    request_case_ids: list[str]
    telemetry_before: HostMemory | None = None
    telemetry_after: HostMemory | None = None
    residents_before: list[ResidentModel] = Field(default_factory=list)
    residents_after: list[ResidentModel] = Field(default_factory=list)


class Attempt(BaseModel):
    number: int
    order: list[str]
    state: Literal[
        "pending",
        "resetting",
        "prepared",
        "warming",
        "measuring",
        "observed",
        "cleaned",
        "failed",
        "interrupted",
    ] = "pending"
    observations: list[Observation] = Field(default_factory=list)
    batches: list[BatchObservation] = Field(default_factory=list)
    telemetry_before: HostMemory | None = None
    telemetry_after: HostMemory | None = None
    residents_before: list[ResidentModel] = Field(default_factory=list)
    residents_after: list[ResidentModel] = Field(default_factory=list)
    elapsed_s: float | None = None
    generated_tokens: int | None = None
    error_code: str | None = None


class ProbeUnit(BaseModel):
    unit_id: str
    order: list[str]
    attempts: list[Attempt] = Field(default_factory=list)


class ProbeSection(BaseModel):
    planned_count: int
    completed_count: int = 0
    units: list[ProbeUnit] = Field(default_factory=list)
    analysis: dict[str, Any] = Field(default_factory=dict)


class NativeProbeArtifact(BaseModel):
    artifact_type: Literal["native_probe"] = "native_probe"
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    seed: int
    schedule_sha256: str
    status: Literal[
        "planned",
        "preflight",
        "running",
        "cleanup",
        "completed",
        "interrupted",
        "cleanup_failed",
        "infeasible",
        "failed",
    ] = "planned"
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    environment: dict[str, Any] = Field(default_factory=dict)
    configuration: dict[str, Any]
    preflight: dict[str, Any] = Field(default_factory=dict)
    prefix_reuse: ProbeSection
    dual_residency: ProbeSection
    concurrency: ProbeSection
    analysis: dict[str, Any] = Field(default_factory=dict)
    cleanup: dict[str, Any] = Field(default_factory=dict)
