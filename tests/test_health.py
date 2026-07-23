from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from synapse.app import create_app


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_health_reachable(client: TestClient) -> None:
    mock_response = httpx.Response(200)
    with patch("synapse.health.httpx.AsyncClient") as mock_cls:
        mock_instance = mock_cls.return_value
        mock_instance.__aenter__.return_value = mock_instance
        mock_instance.get = AsyncMock(return_value=mock_response)

        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "ollama": "reachable", "version": "0.1.0"}


def test_health_unreachable(client: TestClient) -> None:
    with patch("synapse.health.httpx.AsyncClient") as mock_cls:
        mock_instance = mock_cls.return_value
        mock_instance.__aenter__.return_value = mock_instance
        mock_instance.get = AsyncMock(side_effect=httpx.ConnectError("Unreachable"))

        response = client.get("/health")
        assert response.status_code == 503
        assert response.json() == {
            "status": "degraded",
            "ollama": "unreachable",
            "version": "0.1.0",
        }


def test_api_v1_status(client: TestClient) -> None:
    mock_response = httpx.Response(200)
    with patch("synapse.health.httpx.AsyncClient") as mock_cls:
        mock_instance = mock_cls.return_value
        mock_instance.__aenter__.return_value = mock_instance
        mock_instance.get = AsyncMock(return_value=mock_response)

        response = client.get("/api/v1/status")
        assert response.status_code == 200
        data = response.json()
        assert "proxy_version" in data
        assert "upstream_url" in data
        assert data.get("upstream_reachable") is True
