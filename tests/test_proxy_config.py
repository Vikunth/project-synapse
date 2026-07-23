from __future__ import annotations

from unittest.mock import patch

from synapse.config import SynapseConfig, get_config, resolve_ollama_url


def test_default_config() -> None:
    config = SynapseConfig()
    assert config.host == "127.0.0.1"
    assert config.port == 11435
    assert config.upstream_timeout == 120.0
    assert config.max_request_bytes == 50 * 1024 * 1024


def test_config_from_env(monkeypatch) -> None:
    monkeypatch.setenv("SYNAPSE_HOST", "0.0.0.0")
    monkeypatch.setenv("SYNAPSE_PORT", "8080")
    config = SynapseConfig()
    assert config.host == "0.0.0.0"
    assert config.port == 8080


def test_resolve_ollama_url_from_env_with_scheme(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://192.168.1.10:11434")
    assert resolve_ollama_url() == "http://192.168.1.10:11434"


def test_resolve_ollama_url_from_env_without_scheme(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "192.168.1.10:11434")
    assert resolve_ollama_url() == "http://192.168.1.10:11434"


def test_resolve_ollama_url_default_not_wsl2(monkeypatch) -> None:
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    with patch("platform.system", return_value="Windows"):
        assert resolve_ollama_url() == "http://127.0.0.1:11434"


def test_resolve_ollama_url_adds_port(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "http://myhost")
    assert resolve_ollama_url() == "http://myhost:11434"


def test_get_config_returns_same_instance() -> None:
    get_config.cache_clear()
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2
