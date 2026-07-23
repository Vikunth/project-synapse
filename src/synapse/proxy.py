from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from synapse.errors import (
    synapse_error_response,
    upstream_timeout_response,
    upstream_unreachable_response,
)
from synapse.translator import (
    generate_request_id,
    translate_ollama_chunk_to_openai_sse,
    translate_ollama_response_to_openai,
    translate_openai_to_ollama,
)

router = APIRouter(tags=["proxy"])


def log_disconnect(request: Request) -> None:
    """Background task to log client disconnects."""
    if hasattr(request.state, "disconnected") and request.state.disconnected:
        # Simplistic logging, ideally use a configured logger
        print(f"Request {request.url} disconnected prematurely.")


async def check_request_size(request: Request) -> Response | None:
    """Check if the request exceeds the maximum allowed size."""
    max_bytes = getattr(request.app.state, "max_request_bytes", None)
    if max_bytes is not None:
        content_length_str = request.headers.get("content-length")
        if content_length_str and int(content_length_str) > max_bytes:
            return JSONResponse({"error": "Payload Too Large"}, status_code=413)
    return None


@router.post("/api/chat")
async def proxy_api_chat(request: Request, background_tasks: BackgroundTasks) -> Response:
    """Transparent proxy for Ollama native /api/chat."""
    background_tasks.add_task(log_disconnect, request)
    size_err = await check_request_size(request)
    if size_err:
        return size_err

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    return await stream_to_upstream(request, "/api/chat", body, media_type="application/x-ndjson")


@router.post("/api/generate")
async def proxy_api_generate(request: Request, background_tasks: BackgroundTasks) -> Response:
    """Transparent proxy for Ollama native /api/generate."""
    background_tasks.add_task(log_disconnect, request)
    size_err = await check_request_size(request)
    if size_err:
        return size_err

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    return await stream_to_upstream(
        request,
        "/api/generate",
        body,
        media_type="application/x-ndjson",
    )


@router.get("/api/ps")
async def proxy_api_ps(request: Request) -> Response:
    """Transparent proxy for Ollama native /api/ps."""
    client: httpx.AsyncClient = request.app.state.http_client
    ollama_url: str = request.app.state.ollama_url

    try:
        upstream_resp = await client.get(f"{ollama_url.rstrip('/')}/api/ps")
        return JSONResponse(upstream_resp.json(), status_code=upstream_resp.status_code)
    except httpx.ConnectError:
        return upstream_unreachable_response()
    except httpx.TimeoutException:
        return upstream_timeout_response()
    except Exception as e:
        return synapse_error_response(502, str(e))


@router.get("/api/tags")
async def proxy_api_tags(request: Request) -> Response:
    """Transparent proxy for Ollama native /api/tags."""
    client: httpx.AsyncClient = request.app.state.http_client
    ollama_url: str = request.app.state.ollama_url

    try:
        upstream_resp = await client.get(f"{ollama_url.rstrip('/')}/api/tags")
        return JSONResponse(upstream_resp.json(), status_code=upstream_resp.status_code)
    except httpx.ConnectError:
        return upstream_unreachable_response()
    except httpx.TimeoutException:
        return upstream_timeout_response()
    except Exception as e:
        return synapse_error_response(502, str(e))


@router.get("/v1/models")
async def proxy_v1_models(request: Request) -> Response:
    """Transparent proxy for Ollama native /v1/models."""
    client: httpx.AsyncClient = request.app.state.http_client
    ollama_url: str = request.app.state.ollama_url

    try:
        upstream_resp = await client.get(f"{ollama_url.rstrip('/')}/v1/models")
        return JSONResponse(upstream_resp.json(), status_code=upstream_resp.status_code)
    except httpx.ConnectError:
        return upstream_unreachable_response()
    except httpx.TimeoutException:
        return upstream_timeout_response()
    except Exception as e:
        return synapse_error_response(502, str(e))


@router.post("/v1/chat/completions")
async def proxy_v1_chat_completions(
    request: Request,
    background_tasks: BackgroundTasks,
) -> Response:
    """Translate and proxy OpenAI /v1/chat/completions to Ollama /api/chat."""
    background_tasks.add_task(log_disconnect, request)
    size_err = await check_request_size(request)
    if size_err:
        return size_err

    try:
        body = await request.json()
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    ollama_body = translate_openai_to_ollama(body)
    stream = ollama_body.get("stream", True)

    client: httpx.AsyncClient = request.app.state.http_client
    ollama_url: str = request.app.state.ollama_url
    target_url = f"{ollama_url.rstrip('/')}/api/chat"

    if not stream:
        try:
            upstream_resp = await client.post(target_url, json=ollama_body)
            if upstream_resp.status_code != 200:
                # Forward error
                return JSONResponse(upstream_resp.json(), status_code=upstream_resp.status_code)

            response_json = upstream_resp.json()
            if "error" in response_json:
                return JSONResponse(response_json, status_code=500)

            request_id = generate_request_id()
            openai_resp = translate_ollama_response_to_openai(response_json, request_id)
            return JSONResponse(openai_resp)
        except httpx.ConnectError:
            return upstream_unreachable_response()
        except httpx.TimeoutException:
            return upstream_timeout_response()
        except Exception as e:
            return synapse_error_response(502, str(e))

    # Streaming mode
    request_id = generate_request_id()
    model = body.get("model", "unknown")

    async def openai_stream_generator() -> AsyncGenerator[str, None]:
        try:
            async with client.stream("POST", target_url, json=ollama_body) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield f"data: {error_text.decode('utf-8')}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                async for line in response.aiter_lines():
                    if await request.is_disconnected():
                        request.state.disconnected = True
                        break
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        if "error" in chunk:
                            yield f"data: {json.dumps(chunk)}\n\n"
                            break
                        sse_chunk = translate_ollama_chunk_to_openai_sse(chunk, model, request_id)
                        yield sse_chunk
                    except json.JSONDecodeError:
                        continue
                if not getattr(request.state, "disconnected", False):
                    yield "data: [DONE]\n\n"
        except httpx.ConnectError:
            yield f"data: {json.dumps({'error': 'Upstream unreachable'})}\n\n"
        except httpx.TimeoutException:
            yield f"data: {json.dumps({'error': 'Upstream timeout'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(openai_stream_generator(), media_type="text/event-stream")


async def stream_to_upstream(
    request: Request,
    path: str,
    body: dict[str, Any],
    media_type: str,
) -> Response:
    """Helper to stream natively to Ollama without translation."""
    client: httpx.AsyncClient = request.app.state.http_client
    ollama_url: str = request.app.state.ollama_url
    target_url = f"{ollama_url.rstrip('/')}{path}"

    async def native_stream_generator() -> AsyncGenerator[str, None]:
        try:
            async with client.stream("POST", target_url, json=body) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield error_text.decode("utf-8")
                    return

                async for line in response.aiter_lines():
                    if await request.is_disconnected():
                        request.state.disconnected = True
                        break
                    if not line:
                        continue
                    yield line + "\n"
        except httpx.ConnectError:
            yield json.dumps({"error": "Upstream unreachable"}) + "\n"
        except httpx.TimeoutException:
            yield json.dumps({"error": "Upstream timeout"}) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(native_stream_generator(), media_type=media_type)
