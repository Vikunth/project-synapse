"""Bounded private-host Ollama operations for native probes."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from synapse_bench.config import BenchmarkSettings
from synapse_bench.native_models import Observation, ResidentModel


class NativeOllamaClient:
    def __init__(
        self, settings: BenchmarkSettings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.settings = settings
        self.http = httpx.AsyncClient(
            base_url=settings.validated_url(),
            timeout=httpx.Timeout(settings.response_timeout_s, connect=settings.connect_timeout_s),
            transport=transport,
        )

    async def __aenter__(self) -> NativeOllamaClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.http.aclose()

    async def installed(self) -> dict[str, str | None]:
        response = await self.http.get("/api/tags")
        response.raise_for_status()
        return {
            str(row["name"]): str(row.get("digest")) if row.get("digest") else None
            for row in response.json().get("models", [])
            if isinstance(row, dict) and row.get("name")
        }

    async def residents(self) -> list[ResidentModel]:
        response = await self.http.get("/api/ps")
        response.raise_for_status()
        rows = response.json().get("models", [])
        if not isinstance(rows, list):
            raise ValueError("invalid_residency_schema")
        result: list[ResidentModel] = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("name"), str):
                continue
            raw_details = row.get("details")
            details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
            result.append(
                ResidentModel(
                    name=row["name"],
                    digest=_text(row.get("digest")),
                    size=_integer(row.get("size")),
                    size_vram=_integer(row.get("size_vram")),
                    context_length=_integer(
                        row.get("context_length") or details.get("context_length")
                    ),
                    expires_at=_text(row.get("expires_at")),
                )
            )
        return result

    async def set_loaded(self, model: str, loaded: bool, *, num_ctx: int = 1024) -> None:
        payload: dict[str, Any] = {"model": model, "keep_alive": "5m" if loaded else 0}
        if loaded:
            payload.update({"prompt": "", "stream": False, "options": {"num_ctx": num_ctx}})
        response = await self.http.post("/api/generate", json=payload)
        response.raise_for_status()
        deadline = asyncio.get_running_loop().time() + self.settings.unload_timeout_s
        while True:
            names = {row.name for row in await self.residents()}
            if (model in names) is loaded:
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("residency_transition_timeout")
            await asyncio.sleep(self.settings.poll_interval_s)

    async def generate(
        self, model: str, prompt: str, case_id: str, *, num_ctx: int, num_predict: int
    ) -> Observation:
        started = time.perf_counter()
        first: float | None = None
        final: dict[str, Any] = {}
        try:
            async with asyncio.timeout(self.settings.response_timeout_s):
                async with self.http.stream(
                    "POST",
                    "/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": True,
                        "keep_alive": "5m",
                        "options": {
                            "temperature": 0,
                            "num_ctx": num_ctx,
                            "num_predict": num_predict,
                        },
                    },
                ) as response:
                    response.raise_for_status()
                    async for item in _json_lines(response.aiter_lines()):
                        if item.get("error"):
                            raise RuntimeError("upstream_error")
                        if item.get("response") and first is None:
                            first = time.perf_counter()
                        if item.get("done"):
                            final = item
            if first is None:
                raise RuntimeError("no_generated_token")
            return Observation(
                case_id=case_id,
                status="success",
                ttft_s=first - started,
                total_s=time.perf_counter() - started,
                prompt_eval_count=_integer(final.get("prompt_eval_count")),
                eval_count=_integer(final.get("eval_count")),
                load_duration_s=_seconds(final.get("load_duration")),
                prompt_eval_duration_s=_seconds(final.get("prompt_eval_duration")),
                eval_duration_s=_seconds(final.get("eval_duration")),
            )
        except (httpx.HTTPError, TimeoutError, RuntimeError, ValueError) as error:
            return Observation(
                case_id=case_id,
                status="failed",
                total_s=time.perf_counter() - started,
                error_code=safe_error_code(error),
            )


async def _json_lines(lines: AsyncIterator[str]) -> AsyncIterator[dict[str, Any]]:
    async for line in lines:
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            yield value


def safe_error_code(error: Exception) -> str:
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, httpx.HTTPStatusError):
        return (
            f"http_{error.response.status_code}"
            if error.response.status_code in {400, 404, 429, 500, 502, 503, 504}
            else "http_error"
        )
    if isinstance(error, httpx.HTTPError):
        return "transport_error"
    return (
        str(error)
        if str(error) in {"upstream_error", "no_generated_token", "residency_transition_timeout"}
        else "invalid_response"
    )


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _seconds(value: object) -> float | None:
    return (
        value / 1_000_000_000
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else None
    )
