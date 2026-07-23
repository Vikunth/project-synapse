from __future__ import annotations

import httpx
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from synapse import __version__
from synapse.config import resolve_ollama_url

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> JSONResponse:
    """
    Perform a health check on the proxy and its upstream Ollama instance.

    Returns:
        JSONResponse: Health status indicating if upstream is reachable.
    """
    url = resolve_ollama_url()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(url)
            if response.is_success:
                return JSONResponse(
                    status_code=200,
                    content={"status": "ok", "ollama": "reachable", "version": __version__},
                )
            return JSONResponse(
                status_code=503,
                content={"status": "degraded", "ollama": "unreachable", "version": __version__},
            )
    except httpx.RequestError:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "ollama": "unreachable", "version": __version__},
        )


@router.get("/api/v1/status")
async def api_status() -> JSONResponse:
    """
    Get detailed API status, including proxy version and upstream URL.

    Returns:
        JSONResponse: Status info including proxy version and upstream reachability.
    """
    url = resolve_ollama_url()
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.get(url)
            reachable = True
    except httpx.RequestError:
        reachable = False

    return JSONResponse(
        status_code=200,
        content={
            "proxy_version": __version__,
            "upstream_url": url,
            "upstream_reachable": reachable,
        },
    )
