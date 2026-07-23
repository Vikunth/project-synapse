from __future__ import annotations

import os
import platform
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class SynapseConfig(BaseSettings):
    """Configuration settings for the Synapse proxy."""

    model_config = SettingsConfigDict(env_prefix="SYNAPSE_")

    host: str = "127.0.0.1"
    port: int = 11435
    upstream_timeout: float = 3600.0
    max_request_bytes: int = 50 * 1024 * 1024


def resolve_ollama_url() -> str:
    """
    Resolve the Ollama upstream URL.

    Reads OLLAMA_HOST env var. Performs WSL2 detection if needed.
    Always ensures the URL has the correct scheme and port.

    Returns:
        str: The resolved Ollama URL.
    """
    ollama_host = os.environ.get("OLLAMA_HOST")
    if ollama_host:
        if ollama_host.startswith("http://") or ollama_host.startswith("https://"):
            url = ollama_host
        else:
            url = f"http://{ollama_host}"

        # 0.0.0.0 is used for binding, but is unroutable for clients on Windows.
        # Replace it with 127.0.0.1 so the proxy can actually connect to Ollama.
        url = url.replace("0.0.0.0", "127.0.0.1")
    else:
        # Check for WSL2
        is_wsl2 = platform.system() == "Linux" and "microsoft" in platform.release().lower()
        if is_wsl2:
            try:
                with open("/proc/net/route") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3 and parts[1] == "00000000":
                            # Parse hex-encoded gateway
                            gateway_hex = parts[2]
                            gateway = (
                                f"{int(gateway_hex[6:8], 16)}."
                                f"{int(gateway_hex[4:6], 16)}."
                                f"{int(gateway_hex[2:4], 16)}."
                                f"{int(gateway_hex[0:2], 16)}"
                            )
                            url = f"http://{gateway}:11434"
                            break
                    else:
                        url = "http://127.0.0.1:11434"
            except Exception:
                url = "http://127.0.0.1:11434"
        else:
            url = "http://127.0.0.1:11434"

    # Add port if missing
    scheme_end = url.find("://")
    if scheme_end != -1:
        domain_part = url[scheme_end + 3 :]
        if ":" not in domain_part and "/" not in domain_part:
            url = f"{url}:11434"
        elif ":" not in domain_part and "/" in domain_part:
            slash_idx = domain_part.find("/")
            url = f"{url[: scheme_end + 3 + slash_idx]}:11434{url[scheme_end + 3 + slash_idx :]}"

    return url


@lru_cache(maxsize=1)
def get_config() -> SynapseConfig:
    """
    Get the cached Synapse configuration singleton.

    Returns:
        SynapseConfig: The current configuration.
    """
    return SynapseConfig()