from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from synapse import __version__
from synapse.config import get_config, resolve_ollama_url
from synapse.health import router as health_router
from synapse.proxy import router as proxy_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage the shared httpx client and resolved upstream URL."""
    config = get_config()
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(config.upstream_timeout, connect=10.0),
        follow_redirects=True,
    )
    app.state.ollama_url = resolve_ollama_url()
    app.state.max_request_bytes = config.max_request_bytes

    yield

    await app.state.http_client.aclose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Synapse Proxy",
        version=__version__,
        description="Transparent async proxy for Ollama",
        lifespan=lifespan,
    )

    app.include_router(health_router)
    app.include_router(proxy_router)

    @app.get("/")
    async def root() -> dict[str, str]:
        """Root endpoint returning basic proxy info."""
        return {"message": "Synapse Proxy is running", "version": __version__}

    return app
