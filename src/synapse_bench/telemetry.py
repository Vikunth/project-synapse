"""Read-only host telemetry. Windows CIM is authoritative for WSL-host gates."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from synapse_bench.native_models import HostMemory

_POWERSHELL = (
    "Get-CimInstance Win32_OperatingSystem | "
    "Select-Object TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json -Compress; "
    "Get-CimInstance Win32_PageFileUsage | "
    "Select-Object AllocatedBaseSize,CurrentUsage | ConvertTo-Json -Compress"
)


def parse_windows_memory(output: str) -> HostMemory:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 2:
        raise ValueError("unexpected_windows_telemetry")
    os_data, page_data = (json.loads(line) for line in lines)
    if isinstance(page_data, list):
        page_rows = page_data
    elif isinstance(page_data, dict):
        page_rows = [page_data]
    else:
        raise ValueError("invalid_pagefile_telemetry")
    total_kib = _positive_int(os_data, "TotalVisibleMemorySize")
    free_kib = _positive_int(os_data, "FreePhysicalMemory")
    allocated_mib = sum(_nonnegative_int(row, "AllocatedBaseSize") for row in page_rows)
    used_mib = sum(_nonnegative_int(row, "CurrentUsage") for row in page_rows)
    if free_kib > total_kib or used_mib > allocated_mib:
        raise ValueError("inconsistent_windows_telemetry")
    return HostMemory(
        total_bytes=total_kib * 1024,
        available_bytes=free_kib * 1024,
        pagefile_total_bytes=allocated_mib * 1024**2,
        pagefile_used_bytes=used_mib * 1024**2,
        source="windows_cim",
    )


async def windows_host_memory(timeout_s: float = 10.0) -> HostMemory:
    process = await asyncio.create_subprocess_exec(
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        _POWERSHELL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout_s)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise ValueError("windows_telemetry_timeout") from None
    if process.returncode != 0:
        raise ValueError("windows_telemetry_failed")
    return parse_windows_memory(stdout.decode("utf-8", errors="strict"))


def read_wsl_memory(path: Path = Path("/proc/meminfo")) -> HostMemory:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, separator, tail = line.partition(":")
        if separator and tail.strip().endswith("kB"):
            try:
                values[name] = int(tail.split()[0]) * 1024
            except (ValueError, IndexError):
                continue
    required = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    if not required <= values.keys():
        raise ValueError("malformed_proc_meminfo")
    return HostMemory(
        total_bytes=values["MemTotal"],
        available_bytes=values["MemAvailable"],
        pagefile_total_bytes=values["SwapTotal"],
        pagefile_used_bytes=values["SwapTotal"] - values["SwapFree"],
        source="wsl_proc",
    )


def _positive_int(data: object, key: str) -> int:
    value = _nonnegative_int(data, key)
    if value <= 0:
        raise ValueError("invalid_windows_telemetry")
    return value


def _nonnegative_int(data: object, key: str) -> int:
    if not isinstance(data, dict) or isinstance(data.get(key), bool):
        raise ValueError("invalid_windows_telemetry")
    try:
        value = int(data[key])
    except (KeyError, TypeError, ValueError):
        raise ValueError("invalid_windows_telemetry") from None
    if value < 0:
        raise ValueError("invalid_windows_telemetry")
    return value
