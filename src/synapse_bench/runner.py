"""Scenario orchestration with incremental result checkpoints."""

from __future__ import annotations

import asyncio
import os
import platform
import time
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from synapse_bench.client import OllamaClient
from synapse_bench.config import BenchmarkSettings, is_wsl2
from synapse_bench.models import RunArtifact, ScenarioResult, Status, TrialResult
from synapse_bench.prompts import prompt_for, stable_prefix
from synapse_bench.stats import latency_summary, percentile
from synapse_bench.storage import write_artifact_atomic

Progress = Callable[[ScenarioResult], None]


def environment_metadata() -> dict[str, str | int | bool]:
    """Collect non-secret, comparison-relevant runtime metadata."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count() or 0,
        "is_wsl2": is_wsl2(),
    }


def _status(trials: list[TrialResult], *, minimum_successes: int = 1) -> Status:
    successful = sum(trial.status == Status.SUCCESS for trial in trials)
    if successful >= minimum_successes:
        return Status.SUCCESS if successful == len(trials) else Status.PARTIAL
    return Status.FAILED


def _finish_scenario(
    scenario: ScenarioResult,
    *,
    trim_extremes: bool = False,
    minimum_successes: int = 1,
) -> ScenarioResult:
    durations = [
        trial.total_s
        for trial in scenario.trials
        if trial.status == Status.SUCCESS and trial.total_s is not None
    ]
    scenario.status = _status(scenario.trials, minimum_successes=minimum_successes)
    scenario.summary.update(latency_summary(durations, trim_extremes=trim_extremes))
    scenario.finished_at = datetime.now(UTC)
    return scenario


async def benchmark_b0(
    client: OllamaClient, settings: BenchmarkSettings, progress: Progress
) -> ScenarioResult:
    """B0: runtime-cold loads, explicitly not reboot/OS-page-cache cold."""
    result = ScenarioResult(name="B0_runtime_cold", status=Status.PARTIAL)
    result.notes.append(
        "Runtime-cold: Ollama residency is cleared; operating-system page cache is not."
    )
    prefix = stable_prefix(128, settings.seed)
    trial_count = 3 if settings.profile == "quick" else 7
    minimum_successes = 2 if settings.profile == "quick" else 5
    for index in range(trial_count):
        try:
            unloaded = await client.unload(settings.primary_model)
        except Exception as error:  # scenario boundary must survive upstream failures
            result.trials.append(
                _setup_failure(
                    "B0_runtime_cold",
                    index,
                    settings.primary_model,
                    prompt_for(prefix, index),
                    error,
                )
            )
            progress(result)
            continue
        if not unloaded:
            result.trials.append(
                _setup_failure(
                    "B0_runtime_cold",
                    index,
                    settings.primary_model,
                    prompt_for(prefix, index),
                    RuntimeError("model remained resident after unload timeout"),
                )
            )
        else:
            result.trials.append(
                await client.generate(
                    scenario="B0_runtime_cold",
                    trial=index,
                    model=settings.primary_model,
                    prompt=prompt_for(prefix, index),
                    prefix_tokens_target=128,
                )
            )
        progress(result)
    return _finish_scenario(
        result,
        trim_extremes=settings.profile == "full",
        minimum_successes=minimum_successes,
    )


async def benchmark_b1(
    client: OllamaClient, settings: BenchmarkSettings, progress: Progress
) -> ScenarioResult:
    """B1: repeated stable prefix with unique suffixes."""
    result = ScenarioResult(name="B1_stable_prefix", status=Status.PARTIAL)
    prefix = stable_prefix(512, settings.seed)
    trial_count = 3 if settings.profile == "quick" else 7
    for index in range(trial_count):
        result.trials.append(
            await client.generate(
                scenario="B1_stable_prefix",
                trial=index,
                model=settings.primary_model,
                prompt=prompt_for(prefix, index),
                prefix_tokens_target=512,
            )
        )
        progress(result)
    return _finish_scenario(result, minimum_successes=2 if settings.profile == "quick" else 5)


async def benchmark_b2(
    client: OllamaClient, settings: BenchmarkSettings, progress: Progress
) -> ScenarioResult:
    """B2: prompt-prefix length sensitivity."""
    result = ScenarioResult(name="B2_prefix_scaling", status=Status.PARTIAL)
    targets = (128, 512) if settings.profile == "quick" else (128, 256, 512, 1024)
    trials_per_target = 2 if settings.profile == "quick" else 5
    for target in targets:
        prefix = stable_prefix(target, settings.seed)
        for index in range(trials_per_target):
            result.trials.append(
                await client.generate(
                    scenario="B2_prefix_scaling",
                    trial=index,
                    model=settings.primary_model,
                    prompt=prompt_for(prefix, index),
                    prefix_tokens_target=target,
                )
            )
            progress(result)
    minimum = 3 if settings.profile == "quick" else 16
    result = _finish_scenario(result, minimum_successes=minimum)
    for target in targets:
        values = [
            trial.total_s
            for trial in result.trials
            if trial.prefix_tokens_target == target
            and trial.status == Status.SUCCESS
            and trial.total_s is not None
        ]
        target_summary = latency_summary(values)
        for key, value in target_summary.items():
            result.summary[f"{target}_{key}"] = value
    return result


async def benchmark_b3(
    client: OllamaClient, settings: BenchmarkSettings, progress: Progress
) -> ScenarioResult:
    """B3: alternating model residency stress."""
    result = ScenarioResult(name="B3_model_residency", status=Status.PARTIAL)
    try:
        installed = await client.installed_models()
    except Exception as error:
        result.status = Status.FAILED
        result.notes.append(f"Could not list installed models: {type(error).__name__}: {error}")
        result.finished_at = datetime.now(UTC)
        return result
    missing = {settings.primary_model, settings.secondary_model} - installed
    if missing:
        result.status = Status.INFEASIBLE
        result.notes.append(f"Required local model(s) not installed: {', '.join(sorted(missing))}")
        result.finished_at = datetime.now(UTC)
        return result
    prefix = stable_prefix(128, settings.seed)
    repeats = 1 if settings.profile == "quick" else 3
    models = (settings.primary_model, settings.secondary_model) * repeats
    for index, model in enumerate(models):
        result.trials.append(
            await client.generate(
                scenario="B3_model_residency",
                trial=index,
                model=model,
                prompt=prompt_for(prefix, index),
                prefix_tokens_target=128,
            )
        )
        progress(result)
    result = _finish_scenario(result, minimum_successes=1 if settings.profile == "quick" else 4)
    if result.status == Status.FAILED:
        result.status = Status.INFEASIBLE
        result.notes.append("Alternating-model load was not feasible within configured bounds.")
    return result


async def benchmark_b4(
    client: OllamaClient, settings: BenchmarkSettings, progress: Progress
) -> ScenarioResult:
    """B4: bounded concurrency and aggregate throughput."""
    result = ScenarioResult(name="B4_concurrency", status=Status.PARTIAL)
    prefix = stable_prefix(128, settings.seed)
    trial_offset = 0
    levels = (1, 4) if settings.profile == "quick" else (1, 4, 8, 16)
    for concurrency in levels:
        started = time.perf_counter()
        tasks = [
            client.generate(
                scenario="B4_concurrency",
                trial=trial_offset + index,
                model=settings.primary_model,
                prompt=prompt_for(prefix, trial_offset + index),
                prefix_tokens_target=128,
                concurrency=concurrency,
            )
            for index in range(concurrency)
        ]
        batch = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - started
        result.trials.extend(batch)
        progress(result)
        successful = [trial for trial in batch if trial.status == Status.SUCCESS]
        totals = [trial.total_s for trial in successful if trial.total_s is not None]
        tokens = sum(trial.eval_count or 0 for trial in successful)
        result.summary[f"c{concurrency}_successful"] = len(successful)
        result.summary[f"c{concurrency}_p95_s"] = percentile(totals, 95)
        result.summary[f"c{concurrency}_tokens_per_second"] = tokens / elapsed if elapsed else None
        trial_offset += concurrency
    return _finish_scenario(result, minimum_successes=4 if settings.profile == "quick" else 23)


SCENARIOS: dict[
    str,
    Callable[[OllamaClient, BenchmarkSettings, Progress], Awaitable[ScenarioResult]],
] = {
    "b0": benchmark_b0,
    "b1": benchmark_b1,
    "b2": benchmark_b2,
    "b3": benchmark_b3,
    "b4": benchmark_b4,
}


async def run_benchmarks(settings: BenchmarkSettings, selected: list[str]) -> Path:
    """Run selected scenarios and save an artifact after every progress event."""
    settings.validated_url()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid.uuid4().hex[:8]}"
    artifact = RunArtifact(
        run_id=run_id,
        environment=environment_metadata(),
        configuration={
            "ollama_host": settings.validated_url(),
            "primary_model": settings.primary_model,
            "secondary_model": settings.secondary_model,
            "connect_timeout_s": settings.connect_timeout_s,
            "response_timeout_s": settings.response_timeout_s,
            "seed": settings.seed,
            "profile": settings.profile,
            "prompt_persistence": False,
        },
    )
    output_path = settings.output_dir / f"{run_id}.json"

    def checkpoint(active: ScenarioResult | None = None) -> None:
        if active is not None:
            for index, existing in enumerate(artifact.scenarios):
                if existing.name == active.name:
                    artifact.scenarios[index] = active
                    break
            else:
                artifact.scenarios.append(active)
        write_artifact_atomic(output_path, artifact)

    checkpoint()
    async with OllamaClient(settings) as client:
        for name in selected:
            scenario = await SCENARIOS[name](client, settings, checkpoint)
            checkpoint(scenario)
    statuses = {scenario.status for scenario in artifact.scenarios}
    artifact.status = (
        Status.SUCCESS
        if statuses == {Status.SUCCESS}
        else Status.FAILED
        if statuses == {Status.FAILED}
        else Status.PARTIAL
    )
    artifact.finished_at = datetime.now(UTC)
    checkpoint()
    return output_path


def _setup_failure(
    scenario: str, trial: int, model: str, prompt: str, error: Exception
) -> TrialResult:
    from synapse_bench.prompts import prompt_fingerprint

    return TrialResult(
        scenario=scenario,
        trial=trial,
        status=Status.FAILED,
        model=model,
        prompt_sha256=prompt_fingerprint(prompt),
        error_type=type(error).__name__,
        error_message=str(error)[:500],
    )
