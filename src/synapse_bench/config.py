"""Runtime configuration and WSL host discovery."""

from __future__ import annotations

import ipaddress
import platform
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def is_wsl2() -> bool:
    """Return whether the process is running under WSL2."""
    if platform.system() != "Linux":
        return False
    release = platform.release().lower()
    version_path = Path("/proc/version")
    version = version_path.read_text(encoding="utf-8").lower() if version_path.exists() else ""
    return "microsoft" in release or "microsoft" in version


def discover_wsl_host(route_path: Path = Path("/proc/net/route")) -> str | None:
    """Read the default IPv4 gateway from Linux's route table."""
    if not route_path.exists():
        return None
    for line in route_path.read_text(encoding="utf-8").splitlines()[1:]:
        fields = line.split()
        if len(fields) < 3 or fields[1] != "00000000":
            continue
        try:
            raw = bytes.fromhex(fields[2])
        except ValueError:
            continue
        if len(raw) == 4:
            return str(ipaddress.IPv4Address(raw[::-1]))
    return None


def default_ollama_url() -> str:
    host = discover_wsl_host() if is_wsl2() else None
    return f"http://{host or '127.0.0.1'}:11434"


def validate_local_url(value: str, allow_remote: bool) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ValueError("OLLAMA_HOST must be an absolute HTTP(S) URL")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        if parsed.hostname != "localhost" and not allow_remote:
            raise ValueError("remote Ollama hosts require SYNAPSE_ALLOW_REMOTE=true") from None
    else:
        if not (address.is_loopback or address.is_private) and not allow_remote:
            raise ValueError("public Ollama hosts require SYNAPSE_ALLOW_REMOTE=true")
    return value.rstrip("/")


class BenchmarkSettings(BaseSettings):
    """Validated benchmark settings loaded from SYNAPSE_* variables."""

    model_config = SettingsConfigDict(env_prefix="SYNAPSE_", extra="ignore", populate_by_name=True)

    ollama_host: str = Field(default_factory=default_ollama_url, validation_alias="OLLAMA_HOST")
    allow_remote: bool = False
    primary_model: str = "qwen2.5:3b"
    secondary_model: str = "qwen2.5-coder:7b"
    connect_timeout_s: float = Field(default=3.0, gt=0, le=30)
    response_timeout_s: float = Field(default=180.0, gt=0, le=1800)
    unload_timeout_s: float = Field(default=20.0, gt=0, le=120)
    poll_interval_s: float = Field(default=0.25, gt=0, le=5)
    seed: int = 42
    output_dir: Path = Path("benchmarks/results/runs")
    profile: Literal["quick", "full"] = "full"

    @field_validator("ollama_host")
    @classmethod
    def normalize_url(cls, value: str) -> str:
        # Remote opt-in is applied after model construction by validated_url().
        return value.rstrip("/")

    def validated_url(self) -> str:
        return validate_local_url(self.ollama_host, self.allow_remote)
