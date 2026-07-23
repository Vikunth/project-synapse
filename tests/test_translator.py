from __future__ import annotations

import json
from typing import Any

from synapse.translator import (
    generate_request_id,
    translate_ollama_chunk_to_openai_sse,
    translate_ollama_response_to_openai,
    translate_openai_to_ollama,
)


def test_generate_request_id_format() -> None:
    req_id = generate_request_id()
    assert req_id.startswith("chatcmpl-")
    assert len(req_id) == 21


def test_generate_request_id_unique() -> None:
    req_id1 = generate_request_id()
    req_id2 = generate_request_id()
    assert req_id1 != req_id2


def test_translate_basic_request() -> None:
    body = {"model": "test", "messages": [{"role": "user", "content": "hi"}]}
    result = translate_openai_to_ollama(body)
    assert result["model"] == "test"
    assert result["messages"] == [{"role": "user", "content": "hi"}]
    assert result["stream"] is True


def test_translate_max_tokens() -> None:
    body: dict[str, Any] = {"model": "test", "messages": [], "max_tokens": 100}
    result = translate_openai_to_ollama(body)
    assert "options" in result
    assert result["options"].get("num_predict") == 100
    assert "max_tokens" not in result


def test_translate_stop_sequences() -> None:
    body: dict[str, Any] = {"model": "test", "messages": [], "stop": ["STOP", "END"]}
    result = translate_openai_to_ollama(body)
    assert "options" in result
    assert result["options"].get("stop") == ["STOP", "END"]


def test_translate_temperature() -> None:
    body: dict[str, Any] = {"model": "test", "messages": [], "temperature": 0.7}
    result = translate_openai_to_ollama(body)
    assert "options" in result
    assert result["options"].get("temperature") == 0.7


def test_translate_unknown_fields_preserved() -> None:
    body = {"model": "test", "messages": [], "custom_field": "value"}
    result = translate_openai_to_ollama(body)
    assert result.get("custom_field") == "value"


def test_translate_existing_options_merged() -> None:
    body: dict[str, Any] = {
        "model": "test",
        "messages": [],
        "max_tokens": 100,
        "options": {"num_ctx": 4096},
    }
    result = translate_openai_to_ollama(body)
    assert result["options"].get("num_ctx") == 4096
    assert result["options"].get("num_predict") == 100


def test_translate_mapped_values_override_options() -> None:
    body: dict[str, Any] = {
        "model": "test",
        "messages": [],
        "max_tokens": 100,
        "options": {"num_predict": 50},
    }
    result = translate_openai_to_ollama(body)
    assert result["options"].get("num_predict") == 100


def test_translate_stream_false() -> None:
    body = {"model": "test", "messages": [], "stream": False}
    result = translate_openai_to_ollama(body)
    assert result["stream"] is False


def test_translate_chunk_to_sse() -> None:
    chunk = {"model": "test", "message": {"role": "assistant", "content": "hello"}, "done": False}
    req_id = "test-id"
    model = "test"
    sse = translate_ollama_chunk_to_openai_sse(chunk, model, req_id)
    assert sse.startswith("data: ")
    data = json.loads(sse[6:])
    assert data["choices"][0]["delta"]["content"] == "hello"
    assert data["choices"][0]["finish_reason"] is None


def test_translate_chunk_done() -> None:
    chunk = {"model": "test", "done": True, "done_reason": "stop"}
    req_id = "test-id"
    model = "test"
    sse = translate_ollama_chunk_to_openai_sse(chunk, model, req_id)
    data = json.loads(sse[6:])
    assert data["choices"][0]["finish_reason"] == "stop"


def test_translate_chunk_with_usage() -> None:
    chunk = {"model": "test", "done": True, "prompt_eval_count": 10, "eval_count": 5}
    req_id = "test-id"
    model = "test"
    sse = translate_ollama_chunk_to_openai_sse(chunk, model, req_id)
    data = json.loads(sse[6:])
    assert "usage" in data
    assert data["usage"]["prompt_tokens"] == 10
    assert data["usage"]["completion_tokens"] == 5
    assert data["usage"]["total_tokens"] == 15


def test_translate_full_response() -> None:
    response = {
        "model": "test",
        "message": {"role": "assistant", "content": "hello world"},
        "done": True,
        "prompt_eval_count": 10,
        "eval_count": 5,
    }
    req_id = "test-id"
    result = translate_ollama_response_to_openai(response, req_id)
    assert result["choices"][0]["message"]["content"] == "hello world"
    assert result["usage"]["prompt_tokens"] == 10
    assert result["usage"]["completion_tokens"] == 5
