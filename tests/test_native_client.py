import json

import httpx
import pytest

from synapse_bench.config import BenchmarkSettings
from synapse_bench.native_client import NativeOllamaClient, safe_error_code


def settings() -> BenchmarkSettings:
    return BenchmarkSettings(ollama_host="http://127.0.0.1:11434")


@pytest.mark.asyncio
async def test_residents_parses_documented_fields_and_tolerates_schema_drift() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "qwen:3b",
                        "digest": "abc",
                        "size": 10,
                        "size_vram": 0,
                        "context_length": 1024,
                        "expires_at": "later",
                        "future_field": {"ignored": True},
                    },
                    {"malformed": True},
                ]
            },
        )

    async with NativeOllamaClient(settings(), transport=httpx.MockTransport(handler)) as client:
        residents = await client.residents()

    assert len(residents) == 1
    assert residents[0].model_dump() == {
        "name": "qwen:3b",
        "digest": "abc",
        "size": 10,
        "size_vram": 0,
        "context_length": 1024,
        "expires_at": "later",
    }


@pytest.mark.asyncio
async def test_partial_stream_failure_returns_only_allowlisted_error_code() -> None:
    secret = "synthetic prompt must not persist"

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(json.dumps({"response": "x"}) + "\nnot-json\n").encode(),
        )

    async with NativeOllamaClient(settings(), transport=httpx.MockTransport(handler)) as client:
        result = await client.generate("model", secret, "opaque-1", num_ctx=1024, num_predict=24)

    assert result.status == "failed"
    assert result.error_code == "invalid_response"
    assert secret not in result.model_dump_json()


def test_http_error_code_does_not_include_response_body() -> None:
    request = httpx.Request("POST", "http://127.0.0.1/api/generate")
    response = httpx.Response(503, request=request, text="secret upstream body")
    error = httpx.HTTPStatusError("secret upstream body", request=request, response=response)

    assert safe_error_code(error) == "http_503"
