from pathlib import Path

import pytest

from synapse_bench.config import discover_wsl_host, validate_local_url


def test_discover_wsl_host_reads_little_endian_gateway(tmp_path: Path) -> None:
    route = tmp_path / "route"
    route.write_text(
        "Iface Destination Gateway Flags RefCnt Use Metric Mask MTU Window IRTT\n"
        "eth0 00000000 01E01BAC 0003 0 0 0 00000000 0 0 0\n",
        encoding="utf-8",
    )

    assert discover_wsl_host(route) == "172.27.224.1"


def test_discover_wsl_host_returns_none_for_bad_route(tmp_path: Path) -> None:
    route = tmp_path / "route"
    route.write_text("Iface Destination Gateway\neth0 00000000 not-hex\n", encoding="utf-8")

    assert discover_wsl_host(route) is None


@pytest.mark.parametrize(
    "url",
    ["http://127.0.0.1:11434", "http://localhost:11434", "http://172.27.224.1:11434"],
)
def test_validate_local_url_accepts_local_and_private_hosts(url: str) -> None:
    assert validate_local_url(url, allow_remote=False) == url


def test_validate_local_url_rejects_public_host_without_opt_in() -> None:
    with pytest.raises(ValueError, match="public Ollama hosts"):
        validate_local_url("https://8.8.8.8:11434", allow_remote=False)


def test_validate_local_url_allows_remote_opt_in() -> None:
    assert validate_local_url("https://ollama.example.com/", allow_remote=True) == (
        "https://ollama.example.com"
    )


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("http://user:secret@127.0.0.1:11434", "userinfo"),
        ("http://127.0.0.1:11434?token=secret", "query or fragment"),
        ("http://127.0.0.1:11434#secret", "query or fragment"),
    ],
)
def test_validate_local_url_rejects_secret_bearing_components(url: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_local_url(url, allow_remote=False)
