from pathlib import Path

import pytest

from synapse_bench.config import BenchmarkSettings
from synapse_bench.native_client import safe_error_code
from synapse_bench.native_models import Attempt, HostMemory, Observation, ResidentModel
from synapse_bench.native_runner import (
    _complete,
    _dual_memory_evidence,
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
    artifact.configuration["schedule"]["prefix"].reverse()
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
                Observation(case_id="dual-001-solo", status="success", telemetry_before=before),
                Observation(case_id="dual-001-dual", status="success", telemetry_before=after),
            ],
        )
    )

    evidence = _dual_memory_evidence(section)

    assert evidence["median_available_decline"] == pytest.approx(0.15)
    assert evidence["pressure_observed"] is True


def test_error_codes_never_include_raw_error_text() -> None:
    secret = "private prompt text"
    assert secret not in safe_error_code(RuntimeError(secret))


class FakeClient:
    def __init__(self, *, resident: list[str] | None = None) -> None:
        self.loaded = set(resident or [])

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
    assert not FakeClient().loaded


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
