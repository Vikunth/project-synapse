"""Crash-resilient execution of NativeProbeArtifact 1.0."""

from __future__ import annotations

import asyncio
import os
import statistics
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path

from synapse_bench.config import BenchmarkSettings, is_wsl2
from synapse_bench.native_client import NativeOllamaClient
from synapse_bench.native_models import (
    Attempt,
    BatchObservation,
    HostMemory,
    NativeProbeArtifact,
    Observation,
    ProbeSection,
    ProbeUnit,
)
from synapse_bench.native_schedule import (
    build_schedule,
    domain_seed,
    schedule_sha256,
    section_from_orders,
    synthetic_namespace,
)
from synapse_bench.native_stats import paired_finding, prompt_counts_match, saturation_finding
from synapse_bench.stats import percentile
from synapse_bench.storage import write_model_atomic
from synapse_bench.telemetry import windows_host_memory

PROBES = {"prefix-reuse", "dual-residency", "concurrency"}


def new_artifact(
    settings: BenchmarkSettings, *, profile: str, probes: set[str], seed: int
) -> NativeProbeArtifact:
    schedule = build_schedule(seed, profile, probes)
    configuration = {
        "profile": profile,
        "probes": sorted(probes),
        "primary_model": settings.primary_model,
        "secondary_model": settings.secondary_model,
        "ollama_host": settings.validated_url(),
        "num_ctx": 1024,
        "num_predict": 24,
        "request_timeout_s": settings.response_timeout_s,
        "unit_timeout_s": 300.0,
        "overall_timeout_s": 1200.0 if profile == "smoke" else 14400.0,
        "schedule": schedule,
    }
    return NativeProbeArtifact(
        run_id=f"native-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
        seed=seed,
        schedule_sha256=schedule_sha256(schedule),
        configuration=configuration,
        prefix_reuse=section_from_orders("prefix", schedule["prefix"]),
        dual_residency=section_from_orders("dual", schedule["dual"]),
        concurrency=section_from_orders("concurrency", schedule["concurrency"]),
    )


async def run_native_probe(
    settings: BenchmarkSettings,
    *,
    profile: str,
    probes: set[str],
    seed: int,
    output_dir: Path,
    acknowledge_full_load: bool = False,
) -> tuple[Path, int]:
    if not probes or not probes <= PROBES:
        raise ValueError("unknown_or_empty_probe_selection")
    if profile == "full" and not acknowledge_full_load:
        raise ValueError("full_profile_requires_acknowledge_full_load")
    artifact = new_artifact(settings, profile=profile, probes=probes, seed=seed)
    path = output_dir / f"{artifact.run_id}.json"
    return await _execute(settings, artifact, path)


async def resume_native_probe(path: Path) -> tuple[Path, int]:
    payload = await asyncio.to_thread(path.read_text, encoding="utf-8")
    artifact = NativeProbeArtifact.model_validate_json(payload)
    schedule = artifact.configuration.get("schedule")
    if not isinstance(schedule, dict) or schedule_sha256(schedule) != artifact.schedule_sha256:
        raise ValueError("artifact_schedule_mismatch")
    _validate_resume_configuration(artifact, schedule)
    if artifact.status == "completed":
        return path, 0
    settings = BenchmarkSettings(
        ollama_host=str(artifact.configuration["ollama_host"]),
        primary_model=str(artifact.configuration["primary_model"]),
        secondary_model=str(artifact.configuration["secondary_model"]),
        response_timeout_s=float(artifact.configuration["request_timeout_s"]),
    )
    for section in (artifact.prefix_reuse, artifact.dual_residency, artifact.concurrency):
        for unit in section.units:
            if unit.attempts and unit.attempts[-1].state not in {
                "cleaned",
                "failed",
                "interrupted",
            }:
                unit.attempts[-1].state = "interrupted"
    artifact.status = "interrupted"
    write_model_atomic(path, artifact)
    return await _execute(settings, artifact, path)


async def _execute(
    settings: BenchmarkSettings, artifact: NativeProbeArtifact, path: Path
) -> tuple[Path, int]:
    lock_path = path.parent / ".native-probe.lock"
    with exclusive_lock(lock_path):
        write_model_atomic(path, artifact)
        try:
            async with asyncio.timeout(float(artifact.configuration["overall_timeout_s"])):
                async with NativeOllamaClient(settings) as client:
                    return path, await _run_with_client(client, artifact, path)
        except (TimeoutError, asyncio.CancelledError):
            artifact.status = "interrupted"
            artifact.finished_at = datetime.now(UTC)
            artifact.cleanup = {"status": "attempted_after_interrupt"}
            write_model_atomic(path, artifact)
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            return path, 3


async def _run_with_client(
    client: NativeOllamaClient, artifact: NativeProbeArtifact, path: Path
) -> int:
    artifact.status = "preflight"
    write_model_atomic(path, artifact)
    installed = await asyncio.wait_for(client.installed(), 20)
    required = {
        str(artifact.configuration["primary_model"]),
        str(artifact.configuration["secondary_model"]),
    }
    if not required <= installed.keys():
        return _finish(artifact, path, "infeasible", 3, "required_model_missing")
    stored_digests = artifact.preflight.get("model_digests")
    if stored_digests and stored_digests != installed:
        return _finish(artifact, path, "infeasible", 3, "model_digest_changed")
    residents = await client.residents()
    if {row.name for row in residents} - required:
        return _finish(artifact, path, "infeasible", 3, "unknown_resident_model")
    if residents:
        return _finish(artifact, path, "infeasible", 3, "models_already_resident")
    host = None
    host_error = None
    cleanup_failures: list[str] = []
    try:
        host = await windows_host_memory()
    except (OSError, ValueError):
        host_error = "windows_host_telemetry_unavailable"
    artifact.preflight = {
        "model_digests": installed,
        "windows_host_memory": host.model_dump() if host else None,
        "windows_host_memory_error": host_error,
        "wsl2": is_wsl2(),
    }
    if artifact.configuration["profile"] == "full" and not artifact.preflight["wsl2"]:
        for section in (artifact.dual_residency, artifact.concurrency):
            if section.planned_count:
                section.analysis = {
                    "finding": False,
                    "status": "infeasible",
                    "reason": "full_load_requires_wsl2",
                }
    full_dual = (
        artifact.configuration["profile"] == "full" and artifact.dual_residency.planned_count > 0
    )
    if full_dual and host is None:
        artifact.dual_residency.analysis = {
            "finding": False,
            "status": "infeasible",
            "reason": host_error,
        }
    if (
        host
        and full_dual
        and not _memory_preflight_ok(
            host.total_bytes,
            host.available_bytes,
            host.pagefile_total_bytes - host.pagefile_used_bytes,
        )
    ):
        artifact.dual_residency.analysis = {
            "finding": False,
            "status": "infeasible",
            "reason": "insufficient_host_memory",
        }
    artifact.status = "running"
    write_model_atomic(path, artifact)
    owned: set[str] = set()
    try:
        await _run_prefix(client, artifact, path, owned)
        await _run_concurrency(client, artifact, path, owned)
        await _run_dual(client, artifact, path, owned)
        _analyze(artifact)
    finally:
        artifact.status = "cleanup"
        write_model_atomic(path, artifact)
        for model in sorted(owned):
            try:
                await asyncio.shield(asyncio.wait_for(client.set_loaded(model, False), 30))
            except (OSError, TimeoutError, ValueError):
                cleanup_failures.append(model)
        if cleanup_failures:
            artifact.cleanup = {"status": "failed", "models": cleanup_failures}
            artifact.status = "cleanup_failed"
            artifact.finished_at = datetime.now(UTC)
            write_model_atomic(path, artifact)
    if cleanup_failures:
        return 4
    artifact.cleanup = {"status": "success", "models_unloaded": sorted(owned)}
    artifact.status = "completed"
    artifact.finished_at = datetime.now(UTC)
    write_model_atomic(path, artifact)
    return 0


async def _run_prefix(
    client: NativeOllamaClient, artifact: NativeProbeArtifact, path: Path, owned: set[str]
) -> None:
    model = str(artifact.configuration["primary_model"])
    consecutive_failures = 0
    for unit in artifact.prefix_reuse.units:
        if _complete(unit):
            continue
        attempt = _new_attempt(unit)
        try:
            async with asyncio.timeout(float(artifact.configuration["unit_timeout_s"])):
                for arm in unit.order:
                    await _transition(attempt, "resetting", artifact, path)
                    await client.set_loaded(model, False)
                    await client.set_loaded(model, True)
                    owned.add(model)
                    await _transition(attempt, "warming", artifact, path)
                    if arm == "reuse":
                        warm_prefix = synthetic_namespace(
                            artifact.seed, "prefix-reuse", unit.unit_id
                        )
                        measured_prefix = warm_prefix
                    else:
                        warm_prefix = synthetic_namespace(
                            artifact.seed, "prefix-control-warm", unit.unit_id
                        )
                        measured_prefix = synthetic_namespace(
                            artifact.seed, "prefix-control-measure", unit.unit_id
                        )
                    warm = await client.generate(
                        model,
                        warm_prefix + "\nSynthetic suffix zero.",
                        f"{unit.unit_id}-{arm}-warm",
                        num_ctx=1024,
                        num_predict=24,
                    )
                    if warm.status != "success":
                        raise RuntimeError(warm.error_code or "warmup_failed")
                    await _transition(attempt, "measuring", artifact, path)
                    residents_before = await client.residents()
                    observation = await client.generate(
                        model,
                        measured_prefix + "\nSynthetic suffix one.",
                        f"{unit.unit_id}-{arm}",
                        num_ctx=1024,
                        num_predict=24,
                    )
                    observation.residents_before = residents_before
                    observation.residents_after = await client.residents()
                    attempt.observations.append(observation)
                    await _transition(attempt, "observed", artifact, path)
                    if observation.status != "success":
                        raise RuntimeError(observation.error_code or "measurement_failed")
                await client.set_loaded(model, False)
                await _transition(attempt, "cleaned", artifact, path)
        except (OSError, TimeoutError, RuntimeError, ValueError) as error:
            attempt.state = "failed"
            attempt.error_code = _bounded_code(error)
            write_model_atomic(path, artifact)
        _update_completed(artifact.prefix_reuse)
        consecutive_failures = consecutive_failures + 1 if attempt.state == "failed" else 0
        if consecutive_failures >= 3:
            artifact.prefix_reuse.analysis = {
                "finding": False,
                "status": "limited",
                "reason": "three_consecutive_unit_failures",
            }
            break


async def _run_dual(
    client: NativeOllamaClient, artifact: NativeProbeArtifact, path: Path, owned: set[str]
) -> None:
    if artifact.dual_residency.analysis.get("status") == "infeasible":
        return
    primary, secondary = (
        str(artifact.configuration["primary_model"]),
        str(artifact.configuration["secondary_model"]),
    )
    consecutive_failures = 0
    for unit in artifact.dual_residency.units:
        if _complete(unit):
            continue
        attempt = _new_attempt(unit)
        try:
            async with asyncio.timeout(float(artifact.configuration["unit_timeout_s"])):
                attempt.telemetry_before = await windows_host_memory()
                _require_safe_host(attempt.telemetry_before)
                for arm in unit.order:
                    await _transition(attempt, "resetting", artifact, path)
                    await client.set_loaded(primary, False)
                    await client.set_loaded(secondary, False)
                    await client.set_loaded(primary, True)
                    owned.add(primary)
                    if arm == "dual":
                        await client.set_loaded(secondary, True)
                        owned.add(secondary)
                    attempt.residents_before = await client.residents()
                    expected = {primary, secondary} if arm == "dual" else {primary}
                    if {row.name for row in attempt.residents_before} != expected:
                        raise RuntimeError("residency_mismatch")
                    await _transition(attempt, "measuring", artifact, path)
                    telemetry_before = await windows_host_memory()
                    _require_safe_host(telemetry_before)
                    prompt = synthetic_namespace(artifact.seed, f"dual-{arm}", unit.unit_id)
                    observation = await client.generate(
                        primary,
                        prompt + "\nSynthetic measurement.",
                        f"{unit.unit_id}-{arm}",
                        num_ctx=1024,
                        num_predict=24,
                    )
                    observation.telemetry_before = telemetry_before
                    observation.telemetry_after = await windows_host_memory()
                    _require_safe_host(observation.telemetry_after, telemetry_before)
                    observation.residents_before = attempt.residents_before
                    observation.residents_after = await client.residents()
                    attempt.observations.append(observation)
                    await _transition(attempt, "observed", artifact, path)
                    if attempt.observations[-1].status != "success":
                        raise RuntimeError(
                            attempt.observations[-1].error_code or "measurement_failed"
                        )
                attempt.telemetry_after = await windows_host_memory()
                _require_safe_host(attempt.telemetry_after, attempt.telemetry_before)
                await client.set_loaded(primary, False)
                await client.set_loaded(secondary, False)
                await _transition(attempt, "cleaned", artifact, path)
        except (OSError, TimeoutError, RuntimeError, ValueError) as error:
            attempt.state = "failed"
            attempt.error_code = _bounded_code(error)
            write_model_atomic(path, artifact)
        _update_completed(artifact.dual_residency)
        consecutive_failures = consecutive_failures + 1 if attempt.state == "failed" else 0
        if consecutive_failures >= 3:
            artifact.dual_residency.analysis = {
                "finding": False,
                "status": "limited",
                "reason": "three_consecutive_unit_failures",
            }
            break


async def _run_concurrency(
    client: NativeOllamaClient, artifact: NativeProbeArtifact, path: Path, owned: set[str]
) -> None:
    if artifact.concurrency.analysis.get("status") == "infeasible":
        return
    model = str(artifact.configuration["primary_model"])
    consecutive_failures = 0
    for unit in artifact.concurrency.units:
        if _complete(unit):
            continue
        attempt = _new_attempt(unit)
        try:
            async with asyncio.timeout(float(artifact.configuration["unit_timeout_s"])):
                try:
                    attempt.telemetry_before = await windows_host_memory()
                    _require_safe_host(attempt.telemetry_before)
                except (OSError, ValueError):
                    attempt.telemetry_before = None
                await client.set_loaded(model, False)
                await client.set_loaded(model, True)
                owned.add(model)
                for level_text in unit.order:
                    level = int(level_text)
                    await _transition(attempt, "measuring", artifact, path)
                    residents_before = await client.residents()
                    try:
                        batch_telemetry_before = await windows_host_memory()
                        _require_safe_host(batch_telemetry_before)
                    except (OSError, ValueError):
                        batch_telemetry_before = None
                    started = time.perf_counter()
                    observations = await asyncio.gather(
                        *(
                            client.generate(
                                model,
                                synthetic_namespace(
                                    artifact.seed,
                                    f"concurrency-{level}",
                                    f"{unit.unit_id}-{index}",
                                    128,
                                )
                                + "\nSynthetic concurrent measurement.",
                                f"{unit.unit_id}-c{level}-{index}",
                                num_ctx=1024,
                                num_predict=24,
                            )
                            for index in range(level)
                        )
                    )
                    elapsed = time.perf_counter() - started
                    attempt.observations.extend(observations)
                    attempt.elapsed_s = elapsed
                    attempt.generated_tokens = sum(row.eval_count or 0 for row in observations)
                    residents_after = await client.residents()
                    try:
                        batch_telemetry_after = await windows_host_memory()
                        _require_safe_host(batch_telemetry_after, batch_telemetry_before)
                    except (OSError, ValueError):
                        batch_telemetry_after = None
                    attempt.batches.append(
                        BatchObservation(
                            concurrency=level,
                            elapsed_s=elapsed,
                            generated_tokens=sum(row.eval_count or 0 for row in observations),
                            request_case_ids=[row.case_id for row in observations],
                            telemetry_before=batch_telemetry_before,
                            telemetry_after=batch_telemetry_after,
                            residents_before=residents_before,
                            residents_after=residents_after,
                        )
                    )
                    await _transition(attempt, "observed", artifact, path)
                    if any(row.status != "success" for row in observations):
                        raise RuntimeError("partial_concurrency_batch")
                await client.set_loaded(model, False)
                try:
                    attempt.telemetry_after = await windows_host_memory()
                    _require_safe_host(attempt.telemetry_after, attempt.telemetry_before)
                except (OSError, ValueError):
                    attempt.telemetry_after = None
                await _transition(attempt, "cleaned", artifact, path)
        except (OSError, TimeoutError, RuntimeError, ValueError) as error:
            attempt.state = "failed"
            attempt.error_code = _bounded_code(error)
            write_model_atomic(path, artifact)
        _update_completed(artifact.concurrency)
        consecutive_failures = consecutive_failures + 1 if attempt.state == "failed" else 0
        if consecutive_failures >= 3:
            artifact.concurrency.analysis = {
                "finding": False,
                "status": "limited",
                "reason": "three_consecutive_unit_failures",
            }
            break


def _analyze(artifact: NativeProbeArtifact) -> None:
    if artifact.configuration["profile"] == "smoke":
        artifact.analysis = {"finding": None, "status": "mechanics_only"}
        return
    prefix_abs: list[float] = []
    prefix_rel: list[float] = []
    effects_by_order: dict[str, list[float]] = {"reuse": [], "control": []}
    control_ttft: list[float] = []
    for unit in artifact.prefix_reuse.units:
        rows = _latest_success(unit)
        by_arm = {row.case_id.rsplit("-", 1)[-1]: row for row in rows}
        reuse, control = by_arm.get("reuse"), by_arm.get("control")
        if (
            reuse
            and control
            and reuse.ttft_s is not None
            and control.ttft_s
            and prompt_counts_match(reuse.prompt_eval_count, control.prompt_eval_count)
        ):
            delta = control.ttft_s - reuse.ttft_s
            prefix_abs.append(delta)
            prefix_rel.append(delta / control.ttft_s)
            effects_by_order[unit.order[0]].append(delta)
            control_ttft.append(control.ttft_s)
    artifact.prefix_reuse.analysis = paired_finding(
        prefix_abs,
        prefix_rel,
        min_units=27,
        min_favors=24,
        absolute_gate=0.150,
        relative_gate=0.20,
        seed=domain_seed(artifact.seed, "bootstrap-prefix"),
    )
    carryover_difference = None
    if all(effects_by_order.values()):
        carryover_difference = abs(
            statistics.median(effects_by_order["reuse"])
            - statistics.median(effects_by_order["control"])
        )
    carryover_limited = bool(
        carryover_difference is not None
        and (
            carryover_difference >= 0.150
            or bool(control_ttft)
            and carryover_difference >= 0.20 * statistics.median(control_ttft)
        )
    )
    artifact.prefix_reuse.analysis["order_carryover_difference_s"] = carryover_difference
    artifact.prefix_reuse.analysis["carryover_limited"] = carryover_limited
    if carryover_limited:
        artifact.prefix_reuse.analysis.update(
            {"finding": False, "status": "limited", "reason": "order_carryover"}
        )
    dual_abs: list[float] = []
    dual_rel: list[float] = []
    for unit in artifact.dual_residency.units:
        by_arm = {row.case_id.rsplit("-", 1)[-1]: row for row in _latest_success(unit)}
        solo, dual = by_arm.get("solo"), by_arm.get("dual")
        if (
            solo
            and dual
            and solo.ttft_s
            and dual.ttft_s is not None
            and prompt_counts_match(solo.prompt_eval_count, dual.prompt_eval_count)
        ):
            delta = dual.ttft_s - solo.ttft_s
            dual_abs.append(delta)
            dual_rel.append(delta / solo.ttft_s)
    if (
        artifact.dual_residency.planned_count
        and artifact.dual_residency.analysis.get("status") != "infeasible"
    ):
        dual_analysis = paired_finding(
            dual_abs,
            dual_rel,
            min_units=27,
            min_favors=24,
            absolute_gate=1.0,
            relative_gate=1.0,
            seed=domain_seed(artifact.seed, "bootstrap-dual"),
        )
        memory_evidence = _dual_memory_evidence(artifact.dual_residency)
        dual_analysis["memory_evidence"] = memory_evidence
        dual_analysis["finding"] = bool(
            dual_analysis["finding"] and memory_evidence["pressure_observed"]
        )
        artifact.dual_residency.analysis = dual_analysis
    # Each current attempt stores a complete randomized block; reconstruct level batches by case id.
    throughputs: dict[int, list[float]] = {8: [], 16: []}
    p95s: dict[int, list[float]] = {8: [], 16: []}
    for unit in artifact.concurrency.units:
        rows = _latest_success(unit)
        attempt = unit.attempts[-1] if _complete(unit) else None
        for level in (8, 16):
            batch = [row for row in rows if f"-c{level}-" in row.case_id]
            totals = [row.total_s for row in batch if row.total_s is not None]
            stored_batch = (
                next((item for item in attempt.batches if item.concurrency == level), None)
                if attempt
                else None
            )
            elapsed = stored_batch.elapsed_s if stored_batch else 0.0
            tokens = stored_batch.generated_tokens if stored_batch else 0
            if len(batch) == level and elapsed > 0:
                throughputs[level].append(tokens / elapsed)
                p95s[level].append(percentile(totals, 95) or 0.0)
    artifact.concurrency.analysis = saturation_finding(
        throughputs[8], throughputs[16], p95s[8], p95s[16]
    )
    artifact.analysis = {
        "prefix_reuse": artifact.prefix_reuse.analysis.get("finding"),
        "dual_residency": artifact.dual_residency.analysis.get("finding"),
        "concurrency": artifact.concurrency.analysis.get("finding"),
    }


async def _transition(
    attempt: Attempt, state: str, artifact: NativeProbeArtifact, path: Path
) -> None:
    attempt.state = state  # type: ignore[assignment]
    write_model_atomic(path, artifact)
    await asyncio.sleep(0)


def _new_attempt(unit: ProbeUnit) -> Attempt:
    attempt = Attempt(number=len(unit.attempts) + 1, order=unit.order.copy())
    unit.attempts.append(attempt)
    return attempt


def _complete(unit: ProbeUnit) -> bool:
    return bool(unit.attempts and unit.attempts[-1].state == "cleaned")


def _latest_success(unit: ProbeUnit) -> list[Observation]:
    return unit.attempts[-1].observations if _complete(unit) else []


def _update_completed(section: ProbeSection) -> None:
    section.completed_count = sum(_complete(unit) for unit in section.units)


def _finish(artifact: NativeProbeArtifact, path: Path, status: str, code: int, reason: str) -> int:
    artifact.status = status  # type: ignore[assignment]
    artifact.finished_at = datetime.now(UTC)
    artifact.analysis = {"finding": False, "reason": reason}
    write_model_atomic(path, artifact)
    return code


def _memory_preflight_ok(total: int, available: int, pagefile_free: int) -> bool:
    gib = 1024**3
    return (
        total >= 7 * gib and available >= 2 * gib and (pagefile_free >= gib or available >= 3 * gib)
    )


def _dual_memory_evidence(section: ProbeSection) -> dict[str, float | bool | int | None]:
    available_declines: list[float] = []
    pagefile_growth: list[int] = []
    for unit in section.units:
        if not _complete(unit):
            continue
        observations = {
            row.case_id.rsplit("-", 1)[-1]: row for row in unit.attempts[-1].observations
        }
        solo, dual = observations.get("solo"), observations.get("dual")
        if solo is None or dual is None:
            continue
        before, after = solo.telemetry_before, dual.telemetry_before
        if before is None or after is None or before.available_bytes <= 0:
            continue
        available_declines.append(
            (before.available_bytes - after.available_bytes) / before.available_bytes
        )
        pagefile_growth.append(after.pagefile_used_bytes - before.pagefile_used_bytes)
    median_decline = statistics.median(available_declines) if available_declines else None
    median_pagefile_growth = statistics.median(pagefile_growth) if pagefile_growth else None
    return {
        "valid_units": len(available_declines),
        "median_available_decline": median_decline,
        "median_pagefile_growth_bytes": median_pagefile_growth,
        "pressure_observed": bool(
            median_decline is not None
            and median_decline >= 0.15
            or median_pagefile_growth is not None
            and median_pagefile_growth > 0
        ),
    }


def _require_safe_host(memory: HostMemory | None, baseline: HostMemory | None = None) -> None:
    if memory is None:
        raise RuntimeError("windows_host_telemetry_unavailable")
    floor = max(round(memory.total_bytes * 0.10), 768 * 1024**2)
    if memory.available_bytes < floor:
        raise RuntimeError("host_memory_abort")
    if baseline and memory.pagefile_used_bytes - baseline.pagefile_used_bytes >= 512 * 1024**2:
        raise RuntimeError("pagefile_growth_abort")


def _bounded_code(error: Exception) -> str:
    value = str(error)
    allow = {
        "warmup_failed",
        "residency_mismatch",
        "partial_concurrency_batch",
        "windows_telemetry_timeout",
        "windows_telemetry_failed",
    }
    return (
        value
        if value in allow
        else ("timeout" if isinstance(error, TimeoutError) else "unit_failed")
    )


def _validate_resume_configuration(
    artifact: NativeProbeArtifact, schedule: dict[str, object]
) -> None:
    configuration = artifact.configuration
    probes = configuration.get("probes")
    if (
        configuration.get("profile") not in {"smoke", "full"}
        or not isinstance(probes, list)
        or not probes
        or not all(isinstance(item, str) for item in probes)
        or not set(probes) <= PROBES
    ):
        raise ValueError("invalid_resume_configuration")
    for key in ("primary_model", "secondary_model", "ollama_host"):
        if not isinstance(configuration.get(key), str):
            raise ValueError("invalid_resume_configuration")
    for prefix, section in (
        ("prefix", artifact.prefix_reuse),
        ("dual", artifact.dual_residency),
        ("concurrency", artifact.concurrency),
    ):
        orders = schedule.get(prefix)
        if not isinstance(orders, list):
            raise ValueError("invalid_resume_configuration")
        expected = section_from_orders(prefix, orders)
        actual_units = [(unit.unit_id, unit.order) for unit in section.units]
        expected_units = [(unit.unit_id, unit.order) for unit in expected.units]
        if actual_units != expected_units:
            raise ValueError("artifact_units_mismatch")


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError("native_probe_already_running") from None
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        with suppress(OSError):
            os.close(descriptor)
        path.unlink(missing_ok=True)
