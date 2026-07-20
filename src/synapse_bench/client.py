"""Bounded asynchronous Ollama API client."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from synapse_bench.config import BenchmarkSettings
from synapse_bench.models import Status, TrialResult
from synapse_bench.prompts import prompt_fingerprint


class OllamaClient:
    def __init__(
        self,
        settings: BenchmarkSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        timeout = httpx.Timeout(
            connect=settings.connect_timeout_s,
            read=settings.response_timeout_s,
            write=settings.connect_timeout_s,
            pool=settings.connect_timeout_s,
        )
        self._settings = settings
        self._client = httpx.AsyncClient(
            base_url=settings.validated_url(), timeout=timeout, transport=transport
        )

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self._client.aclose()

    async def installed_models(self) -> set[str]:
        response = await self._client.get("/api/tags")
        response.raise_for_status()
        data = response.json()
        return {str(model["name"]) for model in data.get("models", []) if "name" in model}

    async def resident_models(self) -> set[str]:
        response = await self._client.get("/api/ps")
        response.raise_for_status()
        data = response.json()
        return {str(model["name"]) for model in data.get("models", []) if "name" in model}

    async def unload(self, model: str) -> bool:
        async with asyncio.timeout(self._settings.unload_timeout_s):
            response = await self._client.post(
                "/api/generate", json={"model": model, "keep_alive": 0}
            )
            response.raise_for_status()
            return await self._poll_residency(model, expected=False)

    async def preload(self, model: str) -> bool:
        """Load a model and confirm residency without counting setup latency."""
        async with asyncio.timeout(self._settings.response_timeout_s):
            response = await self._client.post(
                "/api/generate",
                json={"model": model, "prompt": "", "stream": False, "keep_alive": "5m"},
            )
            response.raise_for_status()
        async with asyncio.timeout(self._settings.unload_timeout_s):
            return await self._poll_residency(model, expected=True)

    async def _poll_residency(self, model: str, *, expected: bool) -> bool:
        while True:
            resident = model in await self.resident_models()
            if resident is expected:
                return True
            await asyncio.sleep(self._settings.poll_interval_s)

    async def generate(
        self,
        *,
        scenario: str,
        trial: int,
        model: str,
        prompt: str,
        prefix_tokens_target: int | None = None,
        concurrency: int | None = None,
    ) -> TrialResult:
        fingerprint = prompt_fingerprint(prompt)
        started = time.perf_counter()
        first_token_at: float | None = None
        final: dict[str, Any] = {}
        try:
            async with asyncio.timeout(self._settings.response_timeout_s):
                async with self._client.stream(
                    "POST",
                    "/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": True,
                        "keep_alive": "5m",
                        "options": {"temperature": 0, "num_predict": 24},
                    },
                ) as response:
                    response.raise_for_status()
                    async for item in _json_lines(response.aiter_lines()):
                        if item.get("error"):
                            raise RuntimeError(str(item["error"]))
                        if item.get("response") and first_token_at is None:
                            first_token_at = time.perf_counter()
                        if item.get("done"):
                            final = item
            finished = time.perf_counter()
            if first_token_at is None:
                raise RuntimeError("stream completed without a generated token")
            eval_count = _optional_int(final.get("eval_count"))
            eval_duration = _nanoseconds_to_seconds(final.get("eval_duration"))
            throughput = eval_count / eval_duration if eval_count and eval_duration else None
            return TrialResult(
                scenario=scenario,
                trial=trial,
                status=Status.SUCCESS,
                model=model,
                prefix_tokens_target=prefix_tokens_target,
                concurrency=concurrency,
                ttft_s=first_token_at - started,
                total_s=finished - started,
                load_duration_s=_nanoseconds_to_seconds(final.get("load_duration")),
                prompt_eval_count=_optional_int(final.get("prompt_eval_count")),
                eval_count=eval_count,
                tokens_per_second=throughput,
                prompt_sha256=fingerprint,
            )
        except (httpx.HTTPError, TimeoutError, RuntimeError, ValueError) as error:
            return TrialResult(
                scenario=scenario,
                trial=trial,
                status=Status.FAILED,
                model=model,
                prefix_tokens_target=prefix_tokens_target,
                concurrency=concurrency,
                total_s=time.perf_counter() - started,
                prompt_sha256=fingerprint,
                error_type=type(error).__name__,
                error_message=_safe_error_message(error, prompt),
            )


async def _json_lines(lines: AsyncIterator[str]) -> AsyncIterator[dict[str, Any]]:
    async for line in lines:
        if line.strip():
            value = json.loads(line)
            if isinstance(value, dict):
                yield value


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _nanoseconds_to_seconds(value: object) -> float | None:
    return value / 1_000_000_000 if isinstance(value, int | float) else None


def _safe_error_message(error: Exception, prompt: str) -> str:
    """Bound error evidence while removing any echoed prompt body."""
    message = str(error)
    if prompt:
        message = message.replace(prompt, "<prompt-redacted>")
    return message[:500]
