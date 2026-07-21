"""Crash-resilient execution of NativeProbeArtifact 1.0."""

from __future__ import annotations

import asyncio
import getpass
import hashlib
import json
import os
import statistics
import tempfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import httpx

from synapse_bench.config import BenchmarkSettings, is_wsl2
from synapse_bench.native_client import NativeOllamaClient, safe_error_code
from synapse_bench.native_models import (
    Attempt,
    BatchObservation,
    HostMemory,
    NativeProbeArtifact,
    NativeProbeConfiguration,
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
from synapse_bench.native_stats import (
    bootstrap_median,
    paired_finding,
    prompt_counts_match,
    saturation_finding,
)
from synapse_bench.stats import percentile
from synapse_bench.storage import write_model_atomic
from synapse_bench.telemetry import windows_host_memory

PROBES = {"prefix-reuse", "dual-residency", "concurrency"}
SAFETY_CODES = {
    "host_memory_abort",
    "pagefile_growth_abort",
    "oom",
    "http_503",
    "http_500",
    "http_502",
    "http_504",
    "residency_mismatch",
}


class ProbeAbort(RuntimeError):
    """A bounded terminal probe failure that still requires cleanup."""


def new_artifact(
    settings: BenchmarkSettings,
    *,
    profile: Literal["smoke", "full"],
    probes: set[str],
    seed: int,
    acknowledge_full_load: bool = False,
) -> NativeProbeArtifact:
    schedule = build_schedule(seed, profile, probes)
    configuration = NativeProbeConfiguration(
        profile=profile,
        probes=cast(list[Literal["prefix-reuse", "dual-residency", "concurrency"]], sorted(probes)),
        primary_model=settings.primary_model,
        secondary_model=settings.secondary_model,
        ollama_host=settings.validated_url(),
        request_timeout_s=settings.response_timeout_s,
        overall_timeout_s=1200.0 if profile == "smoke" else 14400.0,
        acknowledge_full_load=acknowledge_full_load,
        schedule=schedule,
    )
    return NativeProbeArtifact(
        run_id=f"native-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}",
        seed=seed,
        schedule_sha256=schedule_sha256(schedule),
        configuration_sha256=_configuration_sha256(configuration),
        configuration=configuration,
        prefix_reuse=section_from_orders("prefix", schedule["prefix"]),
        dual_residency=section_from_orders("dual", schedule["dual"]),
        concurrency=section_from_orders("concurrency", schedule["concurrency"]),
    )


async def run_native_probe(
    settings: BenchmarkSettings,
    *,
    profile: Literal["smoke", "full"],
    probes: set[str],
    seed: int,
    output_dir: Path,
    acknowledge_full_load: bool = False,
) -> tuple[Path, int]:
    if not probes or not probes <= PROBES:
        raise ValueError("unknown_or_empty_probe_selection")
    if profile == "full" and not acknowledge_full_load:
        raise ValueError("full_profile_requires_acknowledge_full_load")
    artifact = new_artifact(
        settings,
        profile=profile,
        probes=probes,
        seed=seed,
        acknowledge_full_load=acknowledge_full_load,
    )
    path = output_dir / f"{artifact.run_id}.json"
    return await _execute(settings, artifact, path)


async def resume_native_probe(path: Path) -> tuple[Path, int]:
    payload = await asyncio.to_thread(path.read_text, encoding="utf-8")
    artifact = NativeProbeArtifact.model_validate_json(payload)
    schedule = artifact.configuration.schedule
    if schedule_sha256(schedule) != artifact.schedule_sha256:
        raise ValueError("artifact_schedule_mismatch")
    if _configuration_sha256(artifact.configuration) != artifact.configuration_sha256:
        raise ValueError("artifact_configuration_mismatch")
    _validate_resume_configuration(artifact, schedule)
    if (
        artifact.configuration.profile == "full"
        and not artifact.configuration.acknowledge_full_load
    ):
        raise ValueError("full_profile_requires_acknowledge_full_load")
    if artifact.status == "completed":
        return path, 0
    settings = BenchmarkSettings(
        ollama_host=artifact.configuration.ollama_host,
        primary_model=artifact.configuration.primary_model,
        secondary_model=artifact.configuration.secondary_model,
        response_timeout_s=artifact.configuration.request_timeout_s,
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
    lock_path = global_native_probe_lock()
    with exclusive_lock(lock_path):
        write_model_atomic(path, artifact)
        try:
            async with NativeOllamaClient(settings) as client:
                return path, await _run_with_client(client, artifact, path)
        except (TimeoutError, asyncio.CancelledError):
            artifact.status = "interrupted"
            artifact.finished_at = datetime.now(UTC)
            if not artifact.cleanup:
                artifact.cleanup = {"status": "attempted_after_interrupt"}
            write_model_atomic(path, artifact)
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            return path, 3


async def _run_with_client(
    client: NativeOllamaClient, artifact: NativeProbeArtifact, path: Path
) -> int:
    resuming = bool(artifact.owned_models) or any(
        unit.attempts
        for section in (artifact.prefix_reuse, artifact.dual_residency, artifact.concurrency)
        for unit in section.units
    )
    owned = set(artifact.owned_models)
    cleanup_failures: list[str] = []
    deferred_cancellation: asyncio.CancelledError | None = None
    terminal_reason: str | None = None
    artifact.status = "preflight"
    write_model_atomic(path, artifact)
    deadline = asyncio.get_running_loop().time() + artifact.configuration.overall_timeout_s
    try:
        await _probe_body(client, artifact, path, owned, resuming, deadline)
    except asyncio.CancelledError as cancellation:
        deferred_cancellation = cancellation
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            current.uncancel()
    except ProbeAbort as error:
        terminal_reason = str(error)
    except TimeoutError:
        terminal_reason = "overall_timeout"
    except (httpx.HTTPError, OSError, RuntimeError, ValueError) as error:
        terminal_reason = _bounded_code(error)
    finally:
        artifact.status = "cleanup"
        write_model_atomic(path, artifact)
        cleanup_failures, cleanup_cancellation = await _cleanup_all_bounded(client, owned)
        deferred_cancellation = deferred_cancellation or cleanup_cancellation
        if cleanup_failures:
            artifact.cleanup = {"status": "failed", "models": cleanup_failures}
            artifact.status = "cleanup_failed"
            artifact.finished_at = datetime.now(UTC)
            write_model_atomic(path, artifact)
        else:
            artifact.cleanup = {"status": "success", "models_unloaded": sorted(owned)}
            artifact.owned_models = []
            write_model_atomic(path, artifact)
    if cleanup_failures:
        return 4
    if deferred_cancellation is not None:
        artifact.status = "interrupted"
        artifact.finished_at = datetime.now(UTC)
        artifact.execution_summary = {"result": "interrupted"}
        write_model_atomic(path, artifact)
        raise deferred_cancellation
    _finalize_section_statuses(artifact, terminal_reason)
    incomplete = [
        section
        for section in (artifact.prefix_reuse, artifact.dual_residency, artifact.concurrency)
        if section.planned_count and section.execution_status != "completed"
    ]
    infeasible_reasons = {
        "required_model_missing",
        "model_digest_changed",
        "unknown_resident_model",
        "models_already_resident",
    }
    if terminal_reason:
        if terminal_reason == "overall_timeout":
            artifact.status = "interrupted"
            result = "interrupted"
        else:
            artifact.status = "infeasible" if terminal_reason in infeasible_reasons else "failed"
            result = "aborted"
        artifact.execution_summary = {"result": result, "reason": terminal_reason}
        artifact.analysis = {"finding": False, "status": "aborted", "reason": terminal_reason}
        exit_code = 3
    elif incomplete:
        artifact.status = "partial"
        artifact.execution_summary = {
            "result": "partial",
            "sections": [section.execution_status for section in incomplete],
        }
        exit_code = 3
    else:
        artifact.status = "completed"
        artifact.execution_summary = {"result": "completed"}
        exit_code = 0
    artifact.finished_at = datetime.now(UTC)
    write_model_atomic(path, artifact)
    return exit_code


async def _probe_body(
    client: NativeOllamaClient,
    artifact: NativeProbeArtifact,
    path: Path,
    owned: set[str],
    resuming: bool,
    deadline: float,
) -> None:
    installed = await asyncio.wait_for(client.installed(), min(20.0, _remaining(deadline)))
    required = {artifact.configuration.primary_model}
    if "dual-residency" in artifact.configuration.probes:
        required.add(artifact.configuration.secondary_model)
    if not required <= installed.keys():
        raise ProbeAbort("required_model_missing")
    stored_digests = artifact.preflight.get("model_digests")
    if isinstance(stored_digests, dict) and any(
        stored_digests.get(model) != installed.get(model) for model in required
    ):
        raise ProbeAbort("model_digest_changed")
    residents = await client.residents()
    resident_names = {row.name for row in residents}
    if resident_names - required:
        raise ProbeAbort("unknown_resident_model")
    if resident_names:
        if not resuming or not resident_names <= owned:
            raise ProbeAbort("models_already_resident")
        for model in sorted(resident_names):
            await asyncio.wait_for(client.set_loaded(model, False), 30)
    host = None
    host_error = None
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
    _mark_infeasible_sections(artifact, host, host_error)
    artifact.status = "running"
    write_model_atomic(path, artifact)
    await _run_prefix(client, artifact, path, owned, deadline)
    await _run_concurrency(client, artifact, path, owned, deadline)
    await _run_dual(client, artifact, path, owned, deadline)
    _finalize_section_statuses(artifact, None)
    _analyze(artifact)


async def _run_prefix(
    client: NativeOllamaClient,
    artifact: NativeProbeArtifact,
    path: Path,
    owned: set[str],
    deadline: float,
) -> None:
    model = artifact.configuration.primary_model
    consecutive_failures = 0
    for unit in artifact.prefix_reuse.units:
        if _complete(unit):
            continue
        attempt = _new_attempt(unit)
        try:
            async with asyncio.timeout(
                min(artifact.configuration.unit_timeout_s, _remaining(deadline))
            ):
                for arm in unit.order:
                    await _transition(attempt, "resetting", artifact, path)
                    await client.set_loaded(model, False)
                    await _load_owned(client, model, artifact, path, owned)
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
                    attempt.warmups.append(warm)
                    write_model_atomic(path, artifact)
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
                    if {row.name for row in observation.residents_after} != {model}:
                        raise RuntimeError("residency_mismatch")
                    attempt.observations.append(observation)
                    await _transition(attempt, "observed", artifact, path)
                    if observation.status != "success":
                        raise RuntimeError(observation.error_code or "measurement_failed")
                await client.set_loaded(model, False)
                await _transition(attempt, "cleaned", artifact, path)
        except ProbeAbort:
            raise
        except (httpx.HTTPError, OSError, TimeoutError, RuntimeError, ValueError) as error:
            if _deadline_expired(deadline):
                raise ProbeAbort("overall_timeout") from error
            attempt.state = "failed"
            attempt.error_code = _bounded_code(error)
            write_model_atomic(path, artifact)
            consecutive_failures = _failure_streak_or_abort(
                attempt.error_code, consecutive_failures
            )
        _update_completed(artifact.prefix_reuse)
        if attempt.state != "failed":
            consecutive_failures = 0


async def _run_dual(
    client: NativeOllamaClient,
    artifact: NativeProbeArtifact,
    path: Path,
    owned: set[str],
    deadline: float,
) -> None:
    if artifact.dual_residency.execution_status == "infeasible":
        return
    primary, secondary = (
        artifact.configuration.primary_model,
        artifact.configuration.secondary_model,
    )
    consecutive_failures = 0
    for unit in artifact.dual_residency.units:
        if _complete(unit):
            continue
        attempt = _new_attempt(unit)
        try:
            async with asyncio.timeout(
                min(artifact.configuration.unit_timeout_s, _remaining(deadline))
            ):
                attempt.telemetry_before = await windows_host_memory()
                _require_safe_host(attempt.telemetry_before)
                for arm in unit.order:
                    await _transition(attempt, "resetting", artifact, path)
                    await client.set_loaded(primary, False)
                    await client.set_loaded(secondary, False)
                    await _load_owned(client, primary, artifact, path, owned)
                    if arm == "dual":
                        await _load_owned(client, secondary, artifact, path, owned)
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
                    if {row.name for row in observation.residents_after} != expected:
                        raise RuntimeError("residency_mismatch")
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
        except ProbeAbort:
            raise
        except (httpx.HTTPError, OSError, TimeoutError, RuntimeError, ValueError) as error:
            if _deadline_expired(deadline):
                raise ProbeAbort("overall_timeout") from error
            attempt.state = "failed"
            attempt.error_code = _bounded_code(error)
            write_model_atomic(path, artifact)
            consecutive_failures = _failure_streak_or_abort(
                attempt.error_code, consecutive_failures
            )
        _update_completed(artifact.dual_residency)
        if attempt.state != "failed":
            consecutive_failures = 0


async def _run_concurrency(
    client: NativeOllamaClient,
    artifact: NativeProbeArtifact,
    path: Path,
    owned: set[str],
    deadline: float,
) -> None:
    if artifact.concurrency.execution_status == "infeasible":
        return
    model = artifact.configuration.primary_model
    consecutive_failures = 0
    for unit in artifact.concurrency.units:
        if _complete(unit):
            continue
        attempt = _new_attempt(unit)
        try:
            async with asyncio.timeout(
                min(artifact.configuration.unit_timeout_s, _remaining(deadline))
            ):
                try:
                    attempt.telemetry_before = await windows_host_memory()
                    _require_safe_host(attempt.telemetry_before)
                except (OSError, ValueError):
                    attempt.telemetry_before = None
                await client.set_loaded(model, False)
                await _load_owned(client, model, artifact, path, owned)
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
                    if {row.name for row in residents_after} != {model}:
                        raise RuntimeError("residency_mismatch")
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
                        error_codes = {row.error_code for row in observations if row.error_code}
                        safety = error_codes & SAFETY_CODES
                        if safety:
                            raise ProbeAbort(sorted(safety)[0])
                        if error_codes == {"timeout"}:
                            raise RuntimeError("timeout")
                        raise RuntimeError("partial_concurrency_batch")
                await client.set_loaded(model, False)
                try:
                    attempt.telemetry_after = await windows_host_memory()
                    _require_safe_host(attempt.telemetry_after, attempt.telemetry_before)
                except (OSError, ValueError):
                    attempt.telemetry_after = None
                await _transition(attempt, "cleaned", artifact, path)
        except ProbeAbort:
            raise
        except (httpx.HTTPError, OSError, TimeoutError, RuntimeError, ValueError) as error:
            if _deadline_expired(deadline):
                raise ProbeAbort("overall_timeout") from error
            attempt.state = "failed"
            attempt.error_code = _bounded_code(error)
            write_model_atomic(path, artifact)
            consecutive_failures = _failure_streak_or_abort(
                attempt.error_code, consecutive_failures
            )
        _update_completed(artifact.concurrency)
        if attempt.state != "failed":
            consecutive_failures = 0


def _analyze(artifact: NativeProbeArtifact) -> None:
    if artifact.configuration.profile == "smoke":
        proven = all(
            section.execution_status == "completed"
            for section in (artifact.prefix_reuse, artifact.dual_residency, artifact.concurrency)
            if section.planned_count
        )
        artifact.analysis = {
            "finding": None,
            "status": "mechanics_only" if proven else "mechanics_unproven",
        }
        return
    prefix_abs: list[float] = []
    prefix_rel: list[float] = []
    effects_by_order: dict[str, list[float]] = {"reuse": [], "control": []}
    control_ttft: list[float] = []
    for unit in artifact.prefix_reuse.units:
        rows = _latest_success(unit)
        warmups = (
            {
                row.case_id.removesuffix("-warm").rsplit("-", 1)[-1]: row
                for row in unit.attempts[-1].warmups
            }
            if _complete(unit)
            else {}
        )
        by_arm = {row.case_id.rsplit("-", 1)[-1]: row for row in rows}
        reuse, control = by_arm.get("reuse"), by_arm.get("control")
        reuse_warm, control_warm = warmups.get("reuse"), warmups.get("control")
        if (
            reuse
            and control
            and reuse.ttft_s is not None
            and control.ttft_s
            and prompt_counts_match(reuse.prompt_eval_count, control.prompt_eval_count)
            and prompt_counts_match(
                reuse_warm.prompt_eval_count if reuse_warm else None,
                reuse.prompt_eval_count,
            )
            and prompt_counts_match(
                control_warm.prompt_eval_count if control_warm else None,
                control.prompt_eval_count,
            )
        ):
            delta = control.ttft_s - reuse.ttft_s
            prefix_abs.append(delta)
            prefix_rel.append(delta / control.ttft_s)
            effects_by_order[unit.order[0]].append(delta)
            control_ttft.append(control.ttft_s)
    if artifact.prefix_reuse.execution_status != "completed":
        artifact.prefix_reuse.analysis.update({"finding": False, "status": "limited"})
    else:
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
        and artifact.dual_residency.execution_status == "completed"
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
    if artifact.concurrency.execution_status == "completed":
        artifact.concurrency.analysis = saturation_finding(
            throughputs[8], throughputs[16], p95s[8], p95s[16]
        )
        artifact.concurrency.analysis["bootstrap"] = _concurrency_bootstrap(
            throughputs[8], throughputs[16], p95s[8], p95s[16], artifact.seed
        )
    elif artifact.concurrency.planned_count:
        artifact.concurrency.analysis.update({"finding": False, "status": "limited"})
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
        before, after = solo.telemetry_after, dual.telemetry_after
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


def _concurrency_bootstrap(
    throughput_c8: list[float],
    throughput_c16: list[float],
    p95_c8: list[float],
    p95_c16: list[float],
    seed: int,
) -> dict[str, object]:
    throughput_ratios = [
        right / left for left, right in zip(throughput_c8, throughput_c16, strict=True) if left > 0
    ]
    latency_ratios = [right / left for left, right in zip(p95_c8, p95_c16, strict=True) if left > 0]
    return {
        "throughput_ratio": bootstrap_median(
            throughput_ratios, domain_seed(seed, "bootstrap-concurrency-throughput")
        ),
        "latency_ratio": bootstrap_median(
            latency_ratios, domain_seed(seed, "bootstrap-concurrency-latency")
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
    if isinstance(error, (httpx.HTTPError, TimeoutError)):
        return safe_error_code(error)
    value = str(error)
    if value in SAFETY_CODES:
        return value
    allow = {
        "warmup_failed",
        "residency_mismatch",
        "partial_concurrency_batch",
        "windows_telemetry_timeout",
        "windows_telemetry_failed",
        "host_memory_abort",
        "pagefile_growth_abort",
        "oom",
        "http_503",
        "timeout",
        "measurement_failed",
        "required_model_missing",
        "model_digest_changed",
        "unknown_resident_model",
        "models_already_resident",
    }
    return (
        value
        if value in allow
        else ("timeout" if isinstance(error, TimeoutError) else "unit_failed")
    )


def _failure_streak_or_abort(code: str | None, timeout_streak: int) -> int:
    bounded = code or "unit_failed"
    if bounded in SAFETY_CODES:
        raise ProbeAbort(bounded)
    if bounded == "timeout":
        timeout_streak += 1
        if timeout_streak >= 2:
            raise ProbeAbort("two_consecutive_timeouts")
        return timeout_streak
    return 0


def _remaining(deadline: float) -> float:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise ProbeAbort("overall_timeout")
    return remaining


def _deadline_expired(deadline: float) -> bool:
    return asyncio.get_running_loop().time() >= deadline


async def _load_owned(
    client: NativeOllamaClient,
    model: str,
    artifact: NativeProbeArtifact,
    path: Path,
    owned: set[str],
) -> None:
    owned.add(model)
    artifact.owned_models = sorted(owned)
    write_model_atomic(path, artifact)
    await client.set_loaded(model, True)


async def _cleanup_all_bounded(
    client: NativeOllamaClient, owned: set[str]
) -> tuple[list[str], asyncio.CancelledError | None]:
    async def cleanup_all() -> list[str]:
        failures: list[str] = []
        for model in sorted(owned):
            try:
                await asyncio.wait_for(client.set_loaded(model, False), 30)
            except (httpx.HTTPError, OSError, TimeoutError, RuntimeError, ValueError):
                failures.append(model)
        return failures

    return await cleanup_all(), None


def _mark_infeasible_sections(
    artifact: NativeProbeArtifact, host: HostMemory | None, host_error: str | None
) -> None:
    if artifact.configuration.profile == "full" and not artifact.preflight["wsl2"]:
        for section in (artifact.dual_residency, artifact.concurrency):
            if section.planned_count:
                section.execution_status = "infeasible"
                section.execution_reason = "full_load_requires_wsl2"
                section.analysis = {"finding": False, "status": "infeasible"}
    full_dual = (
        artifact.configuration.profile == "full" and artifact.dual_residency.planned_count > 0
    )
    if full_dual and host is None:
        artifact.dual_residency.execution_status = "infeasible"
        artifact.dual_residency.execution_reason = host_error
        artifact.dual_residency.analysis = {"finding": False, "status": "infeasible"}
    elif (
        host
        and full_dual
        and not _memory_preflight_ok(
            host.total_bytes,
            host.available_bytes,
            host.pagefile_total_bytes - host.pagefile_used_bytes,
        )
    ):
        artifact.dual_residency.execution_status = "infeasible"
        artifact.dual_residency.execution_reason = "insufficient_host_memory"
        artifact.dual_residency.analysis = {"finding": False, "status": "infeasible"}


def _finalize_section_statuses(artifact: NativeProbeArtifact, terminal_reason: str | None) -> None:
    for section in (artifact.prefix_reuse, artifact.dual_residency, artifact.concurrency):
        if section.execution_status == "infeasible":
            continue
        failures = sum(
            bool(unit.attempts and unit.attempts[-1].state == "failed") for unit in section.units
        )
        if terminal_reason and section.completed_count < section.planned_count:
            section.execution_status = "failed"
            section.execution_reason = terminal_reason
        elif failures or section.completed_count < section.planned_count:
            section.execution_status = "partial" if section.completed_count else "failed"
            section.execution_reason = "unit_failure"
        else:
            section.execution_status = "completed"


def _validate_resume_configuration(
    artifact: NativeProbeArtifact, schedule: dict[str, list[list[str | int]]]
) -> None:
    configuration = artifact.configuration
    if not configuration.probes or not set(configuration.probes) <= PROBES:
        raise ValueError("invalid_resume_configuration")
    expected_schedule = build_schedule(
        artifact.seed, configuration.profile, set(configuration.probes)
    )
    if schedule != expected_schedule or set(schedule) != {"prefix", "dual", "concurrency"}:
        raise ValueError("invalid_resume_schedule_semantics")
    for prefix, section in (
        ("prefix", artifact.prefix_reuse),
        ("dual", artifact.dual_residency),
        ("concurrency", artifact.concurrency),
    ):
        orders = schedule.get(prefix)
        if orders is None:
            raise ValueError("invalid_resume_configuration")
        expected = section_from_orders(prefix, orders)
        actual_units = [(unit.unit_id, unit.order) for unit in section.units]
        expected_units = [(unit.unit_id, unit.order) for unit in expected.units]
        if actual_units != expected_units:
            raise ValueError("artifact_units_mismatch")


def _configuration_sha256(configuration: NativeProbeConfiguration) -> str:
    payload = json.dumps(
        configuration.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def global_native_probe_lock() -> Path:
    """Return one fail-closed per-user lock path, independent of artifact output."""
    user_key = hashlib.sha256(getpass.getuser().encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"synapse-native-probe-{user_key}.lock"


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError(
            f"native_probe_already_running_or_stale; inspect PID and remove manually: {path}"
        ) from None
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        with suppress(OSError):
            os.close(descriptor)
        path.unlink(missing_ok=True)
