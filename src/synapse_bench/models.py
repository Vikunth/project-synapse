"""Versioned result models; prompt content is deliberately absent."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Status(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    INFEASIBLE = "infeasible"


class TrialResult(BaseModel):
    scenario: str
    trial: int
    status: Status
    model: str
    prefix_tokens_target: int | None = None
    concurrency: int | None = None
    ttft_s: float | None = None
    total_s: float | None = None
    load_duration_s: float | None = None
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    tokens_per_second: float | None = None
    prompt_sha256: str
    error_type: str | None = None
    error_message: str | None = None


class ScenarioResult(BaseModel):
    name: str
    status: Status
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    summary: dict[str, float | int | str | None] = Field(default_factory=dict)
    trials: list[TrialResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RunArtifact(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    status: Status = Status.PARTIAL
    environment: dict[str, Any]
    configuration: dict[str, Any]
    scenarios: list[ScenarioResult] = Field(default_factory=list)
