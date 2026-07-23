import asyncio
from pathlib import Path

import pytest

from synapse_bench.telemetry import (
    _POWERSHELL,
    parse_windows_memory,
    read_wsl_memory,
    windows_host_memory,
)


def test_parse_windows_memory_supports_single_pagefile() -> None:
    output = (
        '{"TotalVisibleMemorySize":8388608,"FreePhysicalMemory":3145728}\n'
        '{"AllocatedBaseSize":4096,"CurrentUsage":512}\n'
    )
    result = parse_windows_memory(output)

    assert result.total_bytes == 8 * 1024**3
    assert result.available_bytes == 3 * 1024**3
    assert result.pagefile_used_bytes == 512 * 1024**2
    assert result.source == "windows_cim"


def test_parse_windows_memory_sums_pagefiles_and_rejects_inconsistent_data() -> None:
    valid = (
        '{"TotalVisibleMemorySize":8388608,"FreePhysicalMemory":3145728}\n'
        '[{"AllocatedBaseSize":100,"CurrentUsage":20},'
        '{"AllocatedBaseSize":200,"CurrentUsage":30}]\n'
    )
    assert parse_windows_memory(valid).pagefile_used_bytes == 50 * 1024**2
    with pytest.raises(ValueError):
        parse_windows_memory(
            '{"TotalVisibleMemorySize":1,"FreePhysicalMemory":2}\n'
            '{"AllocatedBaseSize":1,"CurrentUsage":0}\n'
        )


def test_read_wsl_memory_is_supplemental_and_malformed_input_fails(tmp_path: Path) -> None:
    path = tmp_path / "meminfo"
    path.write_text(
        "MemTotal: 8000000 kB\nMemAvailable: 2000000 kB\n"
        "SwapTotal: 1000000 kB\nSwapFree: 750000 kB\n",
        encoding="utf-8",
    )
    assert read_wsl_memory(path).source == "wsl_proc"
    path.write_text("MemTotal: broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed_proc_meminfo"):
        read_wsl_memory(path)


@pytest.mark.asyncio
async def test_powershell_invocation_is_fixed_noninteractive_and_has_no_shell(monkeypatch) -> None:
    captured: tuple[object, ...] = ()

    class Process:
        returncode = 0

        async def communicate(self):
            return (
                b'{"TotalVisibleMemorySize":8388608,"FreePhysicalMemory":3145728}\n'
                b'{"AllocatedBaseSize":4096,"CurrentUsage":512}\n',
                b"",
            )

    async def fake_exec(*args, **kwargs):
        nonlocal captured
        captured = args
        assert kwargs["stderr"] is asyncio.subprocess.DEVNULL
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    await windows_host_memory()

    assert captured == (
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        _POWERSHELL,
    )
    assert all("user" not in str(argument).lower() for argument in captured)
