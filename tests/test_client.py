import asyncio
import json

import httpx
import pytest

from synapse_bench.client import OllamaClient
from synapse_bench.config import BenchmarkSettings
from synapse_bench.models import Status


def _settings() -> BenchmarkSettings:
    return BenchmarkSettings(ollama_host="http://127.0.0.1:11434")


class SlowTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        return httpx.Response(200, request=request, text="")


class SlowResidencyTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            return httpx.Response(200, request=request, json={"done": True})
        await asyncio.sleep(0.05)
        return httpx.Response(200, request=request, json={"models": [{"name": "test"}]})


@pytest.mark.asyncio
async def test_generate_uses_first_nonempty_token_and_official_timings() -> None:
    body = "\n".join(
        [
            json.dumps({"model": "test", "response": "", "done": False}),
            json.dumps({"model": "test", "response": "hello", "done": False}),
            json.dumps(
                {
                    "model": "test",
                    "response": "",
                    "done": True,
                    "load_duration": 2_000_000_000,
                    "prompt_eval_count": 10,
                    "eval_count": 4,
                    "eval_duration": 1_000_000_000,
                }
            ),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        return httpx.Response(200, text=body)

    async with OllamaClient(_settings(), transport=httpx.MockTransport(handler)) as client:
        result = await client.generate(scenario="test", trial=0, model="test", prompt="synthetic")

    assert result.status == Status.SUCCESS
    assert result.ttft_s is not None
    assert result.load_duration_s == 2.0
    assert result.prompt_eval_count == 10
    assert result.tokens_per_second == 4.0
    assert "synthetic" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_generate_records_http_error_instead_of_raising() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "busy"})

    async with OllamaClient(_settings(), transport=httpx.MockTransport(handler)) as client:
        result = await client.generate(scenario="test", trial=0, model="test", prompt="synthetic")

    assert result.status == Status.FAILED
    assert result.error_type == "HTTPStatusError"
    assert result.prompt_sha256


@pytest.mark.asyncio
async def test_generate_redacts_prompt_echoed_by_upstream_error() -> None:
    prompt = "synthetic private prompt"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=json.dumps({"error": f"invalid: {prompt}"}))

    async with OllamaClient(_settings(), transport=httpx.MockTransport(handler)) as client:
        result = await client.generate(scenario="test", trial=0, model="test", prompt=prompt)

    assert result.status == Status.FAILED
    assert prompt not in (result.error_message or "")
    assert "<prompt-redacted>" in (result.error_message or "")


@pytest.mark.asyncio
async def test_generate_records_timeout_instead_of_hanging() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("bounded timeout", request=request)

    async with OllamaClient(_settings(), transport=httpx.MockTransport(handler)) as client:
        result = await client.generate(scenario="test", trial=0, model="test", prompt="synthetic")

    assert result.status == Status.FAILED
    assert result.error_type == "ReadTimeout"


@pytest.mark.asyncio
async def test_generate_outer_wall_clock_timeout_is_structured() -> None:
    settings = BenchmarkSettings(ollama_host="http://127.0.0.1:11434", response_timeout_s=0.01)
    async with OllamaClient(settings, transport=SlowTransport()) as client:
        result = await client.generate(scenario="test", trial=0, model="test", prompt="synthetic")

    assert result.status == Status.FAILED
    assert result.error_type == "TimeoutError"


@pytest.mark.asyncio
async def test_unload_outer_timeout_covers_request_and_residency_poll() -> None:
    settings = BenchmarkSettings(
        ollama_host="http://127.0.0.1:11434",
        response_timeout_s=1,
        unload_timeout_s=0.01,
        poll_interval_s=0.001,
    )
    async with OllamaClient(settings, transport=SlowResidencyTransport()) as client:
        with pytest.raises(TimeoutError):
            await client.unload("test")


@pytest.mark.asyncio
async def test_installed_and_resident_models_parse_ollama_contract() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "a:latest"}]})
        return httpx.Response(200, json={"models": [{"name": "b:latest"}]})

    async with OllamaClient(_settings(), transport=httpx.MockTransport(handler)) as client:
        assert await client.installed_models() == {"a:latest"}
        assert await client.resident_models() == {"b:latest"}
