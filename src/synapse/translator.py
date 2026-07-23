from __future__ import annotations

import secrets
import time
from typing import Any


def generate_request_id() -> str:
    """Generate a unique request ID in the format chatcmpl-<hex>."""
    return f"chatcmpl-{secrets.token_hex(6)}"


def translate_openai_to_ollama(body: dict[str, Any]) -> dict[str, Any]:
    """Translate an OpenAI request payload to an Ollama request payload."""
    KNOWN_OPENAI_FIELDS = {
        "max_tokens",
        "stop",
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "n",
        "logprobs",
        "top_logprobs",
        "response_format",
        "seed",
        "user",
        "tools",
        "tool_choice",
    }

    ollama_body: dict[str, Any] = {
        "model": body.get("model"),
        "messages": body.get("messages", []),
        "stream": body.get("stream", True),
    }

    options: dict[str, Any] = body.get("options", {}).copy()

    if "max_tokens" in body:
        options["num_predict"] = body["max_tokens"]
    if "temperature" in body:
        options["temperature"] = body["temperature"]
    if "top_p" in body:
        options["top_p"] = body["top_p"]
    if "frequency_penalty" in body:
        options["frequency_penalty"] = body["frequency_penalty"]
    if "presence_penalty" in body:
        options["presence_penalty"] = body["presence_penalty"]
    if "stop" in body:
        options["stop"] = body["stop"]

    if options:
        ollama_body["options"] = options

    for key, value in body.items():
        if key not in KNOWN_OPENAI_FIELDS and key not in {"model", "messages", "stream", "options"}:
            ollama_body[key] = value

    return ollama_body
