"""Versioned, prompt-free models for the native crossover probe."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HostMemory(StrictModel):
    total_bytes: int
    available_bytes: int
    pagefile_total_bytes: int
    pagefile_used_bytes: int
    source: Literal["windows_cim", "wsl_proc"]


class ResidentModel(StrictModel):
    name: str
    digest: str | None = None
    size: int | None = None
    size_vram: int | None = None
    context_length: int | None = None
    expires_at: str | None = None


class Observation(StrictModel):
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


class BatchObservation(StrictModel):
    concurrency: int
    elapsed_s: float
    generated_tokens: int
    request_case_ids: list[str]
    telemetry_before: HostMemory | None = None
    telemetry_after: HostMemory | None = None
    residents_before: list[ResidentModel] = Field(default_factory=list)
    residents_after: list[ResidentModel] = Field(default_factory=list)


class Attempt(StrictModel):
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
    warmups: list[Observation] = Field(default_factory=list)
    batches: list[BatchObservation] = Field(default_factory=list)
    telemetry_before: HostMemory | None = None
    telemetry_after: HostMemory | None = None
    residents_before: list[ResidentModel] = Field(default_factory=list)
    residents_after: list[ResidentModel] = Field(default_factory=list)
    elapsed_s: float | None = None
    generated_tokens: int | None = None
    error_code: str | None = None


class ProbeUnit(StrictModel):
    unit_id: str
    order: list[str]
    attempts: list[Attempt] = Field(default_factory=list)


class ProbeSection(StrictModel):
    planned_count: int
    completed_count: int = 0
    units: list[ProbeUnit] = Field(default_factory=list)
    analysis: dict[str, Any] = Field(default_factory=dict)
    execution_status: Literal[
        "planned", "completed", "partial", "failed", "infeasible", "skipped"
    ] = "planned"
    execution_reason: str | None = None


class NativeProbeConfiguration(StrictModel):
    profile: Literal["smoke", "full"]
    probes: list[Literal["prefix-reuse", "dual-residency", "concurrency"]] = Field(min_length=1)
    primary_model: str = Field(min_length=1)
    secondary_model: str = Field(min_length=1)
    ollama_host: str = Field(min_length=1)
    num_ctx: Literal[1024] = 1024
    num_predict: Literal[24] = 24
    request_timeout_s: float = Field(gt=0, le=1800)
    unit_timeout_s: float = Field(default=300.0, gt=0, le=300)
    overall_timeout_s: float = Field(gt=0, le=14400)
    acknowledge_full_load: bool
    schedule: dict[str, list[list[str | int]]]


class NativeProbeArtifact(StrictModel):
    artifact_type: Literal["native_probe"] = "native_probe"
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    seed: int
    schedule_sha256: str
    configuration_sha256: str
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
        "partial",
    ] = "planned"
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    environment: dict[str, Any] = Field(default_factory=dict)
    configuration: NativeProbeConfiguration
    preflight: dict[str, Any] = Field(default_factory=dict)
    prefix_reuse: ProbeSection
    dual_residency: ProbeSection
    concurrency: ProbeSection
    analysis: dict[str, Any] = Field(default_factory=dict)
    cleanup: dict[str, Any] = Field(default_factory=dict)
    owned_models: list[str] = Field(default_factory=list)
    execution_summary: dict[str, Any] = Field(default_factory=dict)
