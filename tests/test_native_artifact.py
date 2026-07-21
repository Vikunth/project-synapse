import asyncio
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from synapse_bench.config import BenchmarkSettings
from synapse_bench.native_client import safe_error_code
from synapse_bench.native_models import (
    Attempt,
    HostMemory,
    NativeProbeArtifact,
    Observation,
    ResidentModel,
)
from synapse_bench.native_runner import (
    _complete,
    _concurrency_bootstrap,
    _dual_memory_evidence,
    _execute,
    _memory_preflight_ok,
    _run_with_client,
    exclusive_lock,
    new_artifact,
    resume_native_probe,
    run_native_probe,
)
from synapse_bench.storage import write_model_atomic


def settings() -> BenchmarkSettings:
    return BenchmarkSettings(
        ollama_host="http://127.0.0.1:11434",
        primary_model="primary",
        secondary_model="secondary",
    )


def test_artifact_contains_schedule_but_no_prompt_or_response_content(tmp_path: Path) -> None:
    artifact = new_artifact(
        settings(), profile="smoke", probes={"prefix-reuse", "dual-residency"}, seed=42
    )
    path = tmp_path / "artifact.json"
    write_model_atomic(path, artifact)
    payload = path.read_text(encoding="utf-8")

    assert artifact.artifact_type == "native_probe"
    assert artifact.schema_version == "1.0"
    assert '"schedule_sha256"' in payload
    assert '"prompt"' not in payload.lower()
    assert '"response"' not in payload.lower()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.asyncio
async def test_full_requires_explicit_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="acknowledge"):
        await run_native_probe(
            settings(),
            profile="full",
            probes={"prefix-reuse"},
            seed=42,
            output_dir=tmp_path,
        )


@pytest.mark.asyncio
async def test_resume_rejects_schedule_tampering_before_network(tmp_path: Path) -> None:
    artifact = new_artifact(settings(), profile="smoke", probes={"prefix-reuse"}, seed=42)
    artifact.configuration.schedule["prefix"].reverse()
    path = tmp_path / "artifact.json"
    write_model_atomic(path, artifact)

    with pytest.raises(ValueError, match="artifact_schedule_mismatch"):
        await resume_native_probe(path)


@pytest.mark.asyncio
async def test_resume_rejects_unit_tampering_before_network(tmp_path: Path) -> None:
    artifact = new_artifact(settings(), profile="smoke", probes={"prefix-reuse"}, seed=42)
    artifact.prefix_reuse.units[0].order.reverse()
    path = tmp_path / "units.json"
    write_model_atomic(path, artifact)

    with pytest.raises(ValueError, match="artifact_units_mismatch"):
        await resume_native_probe(path)


@pytest.mark.asyncio
async def test_resume_rejects_profile_timeout_or_ack_tampering(tmp_path: Path) -> None:
    artifact = new_artifact(settings(), profile="smoke", probes={"prefix-reuse"}, seed=42)
    artifact.configuration.profile = "full"
    artifact.configuration.overall_timeout_s = 100
    path = tmp_path / "configuration.json"
    write_model_atomic(path, artifact)

    with pytest.raises(ValueError, match="artifact_configuration_mismatch"):
        await resume_native_probe(path)


def test_artifact_models_forbid_unknown_fields() -> None:
    artifact = new_artifact(settings(), profile="smoke", probes={"prefix-reuse"}, seed=42)
    payload = artifact.model_dump(mode="json")
    payload["unexpected"] = "rejected"

    with pytest.raises(ValidationError, match="extra_forbidden"):
        NativeProbeArtifact.model_validate(payload)

    payload = artifact.model_dump(mode="json")
    payload["configuration"]["unexpected"] = True
    with pytest.raises(ValidationError, match="extra_forbidden"):
        NativeProbeArtifact.model_validate(payload)


def test_completed_unit_is_not_eligible_for_reexecution() -> None:
    artifact = new_artifact(settings(), profile="smoke", probes={"prefix-reuse"}, seed=42)
    unit = artifact.prefix_reuse.units[0]
    unit.attempts.append(Attempt(number=1, order=unit.order, state="cleaned"))
    assert _complete(unit)


def test_lock_prevents_second_writer_and_is_removed(tmp_path: Path) -> None:
    path = tmp_path / "probe.lock"
    with (
        exclusive_lock(path),
        pytest.raises(ValueError, match="already_running"),
        exclusive_lock(path),
    ):
        pass
    assert not path.exists()


def test_memory_preflight_boundaries() -> None:
    gib = 1024**3
    assert _memory_preflight_ok(7 * gib, 2 * gib, gib)
    assert _memory_preflight_ok(7 * gib, 3 * gib, 0)
    assert not _memory_preflight_ok(7 * gib, 2 * gib - 1, gib)
    assert not _memory_preflight_ok(7 * gib, 2 * gib, gib - 1)


def test_dual_memory_evidence_requires_host_pressure() -> None:
    artifact = new_artifact(settings(), profile="smoke", probes={"dual-residency"}, seed=42)
    section = artifact.dual_residency
    unit = section.units[0]
    before = HostMemory(
        total_bytes=8_000,
        available_bytes=4_000,
        pagefile_total_bytes=4_000,
        pagefile_used_bytes=100,
        source="windows_cim",
    )
    after = before.model_copy(update={"available_bytes": 3_400})
    unit.attempts.append(
        Attempt(
            number=1,
            order=unit.order,
            state="cleaned",
            observations=[
                Observation(case_id="dual-001-solo", status="success", telemetry_after=before),
                Observation(case_id="dual-001-dual", status="success", telemetry_after=after),
            ],
        )
    )

    evidence = _dual_memory_evidence(section)

    assert evidence["median_available_decline"] == pytest.approx(0.15)
    assert evidence["pressure_observed"] is True


def test_concurrency_bootstrap_is_deterministic_and_paired() -> None:
    first = _concurrency_bootstrap([10.0] * 10, [10.5] * 10, [2.0] * 10, [3.0] * 10, seed=42)
    second = _concurrency_bootstrap([10.0] * 10, [10.5] * 10, [2.0] * 10, [3.0] * 10, seed=42)

    assert first == second
    assert first["throughput_ratio"]["estimate"] == pytest.approx(1.05)
    assert first["latency_ratio"]["estimate"] == pytest.approx(1.5)


def test_error_codes_never_include_raw_error_text() -> None:
    secret = "private prompt text"
    assert secret not in safe_error_code(RuntimeError(secret))


class FakeClient:
    def __init__(self, *, resident: list[str] | None = None) -> None:
        self.loaded = set(resident or [])
        self.unload_calls: list[str] = []

    async def installed(self) -> dict[str, str]:
        return {"primary": "digest-1", "secondary": "digest-2"}

    async def residents(self) -> list[ResidentModel]:
        return [
            ResidentModel(name=name, digest=f"digest-{index}")
            for index, name in enumerate(sorted(self.loaded), 1)
        ]

    async def set_loaded(self, model: str, loaded: bool, *, num_ctx: int = 1024) -> None:
        del num_ctx
        if loaded:
            self.loaded.add(model)
        else:
            self.unload_calls.append(model)
            self.loaded.discard(model)

    async def generate(
        self, model: str, prompt: str, case_id: str, *, num_ctx: int, num_predict: int
    ) -> Observation:
        assert model in self.loaded
        assert prompt
        del num_ctx, num_predict
        return Observation(
            case_id=case_id,
            status="success",
            ttft_s=0.1,
            total_s=0.2,
            prompt_eval_count=400,
            eval_count=24,
        )


class FailingGenerateClient(FakeClient):
    def __init__(self, code: str) -> None:
        super().__init__()
        self.code = code

    async def generate(
        self, model: str, prompt: str, case_id: str, *, num_ctx: int, num_predict: int
    ) -> Observation:
        del model, prompt, num_ctx, num_predict
        return Observation(case_id=case_id, status="failed", error_code=self.code)


class PrimaryOnlyClient(FakeClient):
    async def installed(self) -> dict[str, str]:
        return {"primary": "digest-1"}


class AmbiguousLoadClient(FakeClient):
    async def set_loaded(self, model: str, loaded: bool, *, num_ctx: int = 1024) -> None:
        await super().set_loaded(model, loaded, num_ctx=num_ctx)
        if loaded:
            raise TimeoutError("residency_transition_timeout")


class CleanupHttpFailureClient(FakeClient):
    async def set_loaded(self, model: str, loaded: bool, *, num_ctx: int = 1024) -> None:
        if not loaded:
            request = httpx.Request("POST", "http://127.0.0.1/api/generate")
            raise httpx.ConnectError("bounded cleanup failure", request=request)
        await super().set_loaded(model, loaded, num_ctx=num_ctx)


class BlockingGenerateClient(FakeClient):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()

    async def generate(
        self, model: str, prompt: str, case_id: str, *, num_ctx: int, num_predict: int
    ) -> Observation:
        del model, prompt, case_id, num_ctx, num_predict
        self.started.set()
        await asyncio.sleep(0.2)
        return Observation(
            case_id="blocking",
            status="success",
            ttft_s=0.1,
            total_s=0.2,
            prompt_eval_count=400,
            eval_count=24,
        )


@pytest.mark.asyncio
async def test_smoke_run_completes_all_units_and_never_makes_finding(
    tmp_path: Path, monkeypatch
) -> None:
    memory = HostMemory(
        total_bytes=8 * 1024**3,
        available_bytes=3 * 1024**3,
        pagefile_total_bytes=4 * 1024**3,
        pagefile_used_bytes=0,
        source="windows_cim",
    )

    async def fake_memory() -> HostMemory:
        return memory

    monkeypatch.setattr("synapse_bench.native_runner.windows_host_memory", fake_memory)
    artifact = new_artifact(
        settings(),
        profile="smoke",
        probes={"prefix-reuse", "dual-residency", "concurrency"},
        seed=42,
    )
    path = tmp_path / "smoke.json"

    code = await _run_with_client(FakeClient(), artifact, path)  # type: ignore[arg-type]

    assert code == 0
    assert artifact.status == "completed"
    assert artifact.analysis == {"finding": None, "status": "mechanics_only"}
    assert artifact.prefix_reuse.completed_count == 2
    assert artifact.dual_residency.completed_count == 1
    assert artifact.concurrency.completed_count == 1
    assert len(artifact.prefix_reuse.units[0].attempts[-1].warmups) == 2


@pytest.mark.asyncio
async def test_preflight_refuses_to_mutate_already_resident_model(tmp_path: Path) -> None:
    artifact = new_artifact(settings(), profile="smoke", probes={"prefix-reuse"}, seed=42)
    path = tmp_path / "refused.json"
    client = FakeClient(resident=["primary"])

    code = await _run_with_client(client, artifact, path)  # type: ignore[arg-type]

    assert code == 3
    assert artifact.status == "infeasible"
    assert artifact.analysis["reason"] == "models_already_resident"
    assert client.loaded == {"primary"}


@pytest.mark.asyncio
async def test_ordinary_smoke_unit_failures_are_nonzero_and_mechanics_unproven(
    tmp_path: Path, monkeypatch
) -> None:
    async def no_memory():
        raise ValueError("unavailable")

    monkeypatch.setattr("synapse_bench.native_runner.windows_host_memory", no_memory)
    artifact = new_artifact(settings(), profile="smoke", probes={"prefix-reuse"}, seed=42)

    code = await _run_with_client(
        FailingGenerateClient("invalid_response"), artifact, tmp_path / "fail.json"
    )  # type: ignore[arg-type]

    assert code == 3
    assert artifact.status == "partial"
    assert artifact.analysis["status"] == "mechanics_unproven"
    assert artifact.prefix_reuse.completed_count == 0


@pytest.mark.asyncio
async def test_two_consecutive_timeouts_abort_section(tmp_path: Path, monkeypatch) -> None:
    async def no_memory():
        raise ValueError("unavailable")

    monkeypatch.setattr("synapse_bench.native_runner.windows_host_memory", no_memory)
    artifact = new_artifact(settings(), profile="smoke", probes={"prefix-reuse"}, seed=42)

    code = await _run_with_client(
        FailingGenerateClient("timeout"), artifact, tmp_path / "timeout.json"
    )  # type: ignore[arg-type]

    assert code == 3
    assert artifact.execution_summary["reason"] == "two_consecutive_timeouts"
    assert sum(bool(unit.attempts) for unit in artifact.prefix_reuse.units) == 2


@pytest.mark.asyncio
async def test_upstream_5xx_aborts_immediately(tmp_path: Path, monkeypatch) -> None:
    async def no_memory():
        raise ValueError("unavailable")

    monkeypatch.setattr("synapse_bench.native_runner.windows_host_memory", no_memory)
    artifact = new_artifact(settings(), profile="smoke", probes={"prefix-reuse"}, seed=42)

    code = await _run_with_client(
        FailingGenerateClient("http_500"), artifact, tmp_path / "oom.json"
    )  # type: ignore[arg-type]

    assert code == 3
    assert artifact.execution_summary["reason"] == "http_500"
    assert sum(bool(unit.attempts) for unit in artifact.prefix_reuse.units) == 1


@pytest.mark.asyncio
async def test_ambiguous_load_is_cleanup_owned(tmp_path: Path, monkeypatch) -> None:
    async def no_memory():
        raise ValueError("unavailable")

    monkeypatch.setattr("synapse_bench.native_runner.windows_host_memory", no_memory)
    artifact = new_artifact(settings(), profile="smoke", probes={"prefix-reuse"}, seed=42)
    client = AmbiguousLoadClient()

    code = await _run_with_client(client, artifact, tmp_path / "ambiguous.json")  # type: ignore[arg-type]

    assert code == 3
    assert client.loaded == set()
    assert artifact.cleanup["status"] == "success"


@pytest.mark.asyncio
async def test_cleanup_http_failure_is_preserved_as_exit_four(tmp_path: Path) -> None:
    artifact = new_artifact(settings(), profile="smoke", probes={"prefix-reuse"}, seed=42)
    artifact.owned_models = ["primary"]
    client = CleanupHttpFailureClient(resident=["primary"])

    code = await _run_with_client(client, artifact, tmp_path / "cleanup.json")  # type: ignore[arg-type]

    assert code == 4
    assert artifact.status == "cleanup_failed"
    assert artifact.cleanup == {"status": "failed", "models": ["primary"]}


@pytest.mark.asyncio
async def test_resume_marks_running_attempt_interrupted_before_execution(
    tmp_path: Path, monkeypatch
) -> None:
    artifact = new_artifact(settings(), profile="smoke", probes={"prefix-reuse"}, seed=42)
    unit = artifact.prefix_reuse.units[0]
    unit.attempts.append(Attempt(number=1, order=unit.order, state="measuring"))
    path = tmp_path / "resume.json"
    write_model_atomic(path, artifact)
    observed_state = None

    async def fake_execute(_settings, resumed, resumed_path):
        nonlocal observed_state
        observed_state = resumed.prefix_reuse.units[0].attempts[-1].state
        return resumed_path, 3

    monkeypatch.setattr("synapse_bench.native_runner._execute", fake_execute)

    await resume_native_probe(path)

    assert observed_state == "interrupted"


@pytest.mark.asyncio
async def test_resume_resets_run_owned_resident_model_and_continues(
    tmp_path: Path, monkeypatch
) -> None:
    memory = HostMemory(
        total_bytes=8 * 1024**3,
        available_bytes=3 * 1024**3,
        pagefile_total_bytes=4 * 1024**3,
        pagefile_used_bytes=0,
        source="windows_cim",
    )

    async def fake_memory() -> HostMemory:
        return memory

    monkeypatch.setattr("synapse_bench.native_runner.windows_host_memory", fake_memory)
    artifact = new_artifact(settings(), profile="smoke", probes={"prefix-reuse"}, seed=42)
    artifact.owned_models = ["primary"]
    client = FakeClient(resident=["primary"])

    code = await _run_with_client(client, artifact, tmp_path / "resume-owned.json")  # type: ignore[arg-type]

    assert code == 0
    assert artifact.status == "completed"
    assert client.loaded == set()


@pytest.mark.asyncio
async def test_prefix_only_does_not_require_secondary_model(tmp_path: Path, monkeypatch) -> None:
    async def no_memory():
        raise ValueError("unavailable")

    monkeypatch.setattr("synapse_bench.native_runner.windows_host_memory", no_memory)
    artifact = new_artifact(settings(), profile="smoke", probes={"prefix-reuse"}, seed=42)

    code = await _run_with_client(PrimaryOnlyClient(), artifact, tmp_path / "primary-only.json")  # type: ignore[arg-type]

    assert code == 0
    assert artifact.status == "completed"


@pytest.mark.asyncio
async def test_dual_low_memory_aborts_immediately_without_retry(
    tmp_path: Path, monkeypatch
) -> None:
    safe = HostMemory(
        total_bytes=8 * 1024**3,
        available_bytes=3 * 1024**3,
        pagefile_total_bytes=4 * 1024**3,
        pagefile_used_bytes=0,
        source="windows_cim",
    )
    low = safe.model_copy(update={"available_bytes": 512 * 1024**2})
    readings = iter([safe, low])

    async def fake_memory() -> HostMemory:
        return next(readings)

    monkeypatch.setattr("synapse_bench.native_runner.windows_host_memory", fake_memory)
    artifact = new_artifact(settings(), profile="smoke", probes={"dual-residency"}, seed=42)

    code = await _run_with_client(FakeClient(), artifact, tmp_path / "low-memory.json")  # type: ignore[arg-type]

    assert code == 3
    assert artifact.execution_summary["reason"] == "host_memory_abort"
    assert len(artifact.dual_residency.units[0].attempts) == 1


@pytest.mark.asyncio
async def test_cancellation_waits_for_owned_model_cleanup(tmp_path: Path, monkeypatch) -> None:
    async def no_memory():
        raise ValueError("unavailable")

    monkeypatch.setattr("synapse_bench.native_runner.windows_host_memory", no_memory)
    artifact = new_artifact(settings(), profile="smoke", probes={"prefix-reuse"}, seed=42)
    artifact.owned_models = ["primary", "secondary"]
    client = BlockingGenerateClient()
    task = asyncio.create_task(
        _run_with_client(client, artifact, tmp_path / "cancel.json")  # type: ignore[arg-type]
    )
    await client.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert client.loaded == set()
    assert {"primary", "secondary"} <= set(client.unload_calls)
    assert artifact.cleanup["status"] == "success"


@pytest.mark.asyncio
async def test_overall_timeout_preserves_verified_cleanup_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    async def no_memory():
        raise ValueError("unavailable")

    monkeypatch.setattr("synapse_bench.native_runner.windows_host_memory", no_memory)
    artifact = new_artifact(settings(), profile="smoke", probes={"prefix-reuse"}, seed=42)
    artifact.configuration.overall_timeout_s = 0.01
    artifact.owned_models = ["primary", "secondary"]
    client = BlockingGenerateClient()

    class Context:
        async def __aenter__(self):
            return client

        async def __aexit__(self, *_args):
            return None

    monkeypatch.setattr(
        "synapse_bench.native_runner.NativeOllamaClient", lambda _settings: Context()
    )

    _path, code = await _execute(settings(), artifact, tmp_path / "overall-timeout.json")

    assert code == 3
    assert artifact.status == "interrupted"
    assert artifact.cleanup["status"] == "success"
    assert {"primary", "secondary"} <= set(client.unload_calls)
