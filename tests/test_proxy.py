from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from synapse.app import create_app


class MockStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self.status_code = 200
        self.lines = lines

    async def aiter_lines(self) -> AsyncGenerator[str, None]:
        for line in self.lines:
            yield line

    async def aread(self) -> bytes:
        return b""

    async def aclose(self) -> None:
        pass


@asynccontextmanager
async def mock_stream_context(lines: list[str]) -> AsyncGenerator[MockStreamResponse, None]:
    yield MockStreamResponse(lines)


@pytest.fixture
def client() -> AsyncGenerator[TestClient, None]:
    with patch("synapse.app.resolve_ollama_url", return_value="http://localhost:11434"):
        app = create_app()
        with TestClient(app) as c:
            yield c


def test_api_chat_streams(client: TestClient) -> None:
    lines = [json.dumps({"message": {"content": "hello"}, "done": False})]

    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    mock_http_client.stream.return_value = mock_stream_context(lines)
    client.app.state.http_client = mock_http_client

    response = client.post("/api/chat", json={"model": "test", "messages": []})
    assert response.status_code == 200
    assert b'{"message": {"content": "hello"}, "done": false}' in response.content


def test_api_generate_streams(client: TestClient) -> None:
    lines = [json.dumps({"response": "hello", "done": False})]

    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    mock_http_client.stream.return_value = mock_stream_context(lines)
    client.app.state.http_client = mock_http_client

    response = client.post("/api/generate", json={"model": "test", "prompt": "test"})
    assert response.status_code == 200
    assert b'{"response": "hello", "done": false}' in response.content


def test_api_ps_forwards(client: TestClient) -> None:
    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    mock_response = httpx.Response(200, json={"models": []})
    mock_http_client.get.return_value = mock_response
    client.app.state.http_client = mock_http_client

    response = client.get("/api/ps")
    assert response.status_code == 200
    assert response.json() == {"models": []}


def test_api_ps_upstream_unreachable(client: TestClient) -> None:
    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    mock_http_client.get.side_effect = httpx.ConnectError("Connection refused")
    client.app.state.http_client = mock_http_client

    response = client.get("/api/ps")
    assert response.status_code == 502


def test_v1_non_streaming(client: TestClient) -> None:
    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    ollama_response = {
        "model": "test",
        "message": {"role": "assistant", "content": "hello"},
        "done": True,
        "prompt_eval_count": 10,
        "eval_count": 5,
    }
    mock_response = httpx.Response(200, json=ollama_response)
    mock_http_client.post.return_value = mock_response
    client.app.state.http_client = mock_http_client

    response = client.post(
        "/v1/chat/completions", json={"model": "test", "messages": [], "stream": False}
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert data["choices"][0]["message"]["content"] == "hello"


def test_v1_streaming(client: TestClient) -> None:
    lines = [
        json.dumps(
            {"model": "test", "message": {"role": "assistant", "content": "hello"}, "done": False}
        )
    ]

    mock_http_client = AsyncMock(spec=httpx.AsyncClient)
    mock_http_client.stream.return_value = mock_stream_context(lines)
    client.app.state.http_client = mock_http_client

    response = client.post(
        "/v1/chat/completions", json={"model": "test", "messages": [], "stream": True}
    )
    assert response.status_code == 200
    assert b"data: " in response.content


def test_invalid_json_returns_400(client: TestClient) -> None:
    response = client.post("/api/chat", content="invalid json")
    assert response.status_code == 400


def test_payload_too_large_returns_413(client: TestClient) -> None:
    client.app.state.max_request_bytes = 10
    response = client.post("/api/chat", content='{"model": "test", "messages": []}')
    assert response.status_code == 413
