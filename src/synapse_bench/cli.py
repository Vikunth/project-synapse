"""Command-line entry point."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Literal

import typer

from synapse_bench.config import BenchmarkSettings
from synapse_bench.runner import SCENARIOS, run_benchmarks

app = typer.Typer(no_args_is_help=True, help="Run bounded local Ollama benchmarks.")


@app.callback()
def main() -> None:
    """Measure local Ollama behavior without persisting prompt content."""


@app.command()
def run(
    scenarios: Annotated[
        list[str] | None,
        typer.Option("--scenario", "-s", help="B0-B4; repeat to select several."),
    ] = None,
    output_dir: Annotated[
        Path | None, typer.Option(help="Artifact directory (prompt text is never saved).")
    ] = None,
    profile: Annotated[
        Literal["quick", "full"],
        typer.Option(help="Quick smoke validation or full measurement profile."),
    ] = "full",
) -> None:
    """Run all scenarios, or the selected B0-B4 scenarios."""
    selected = [value.lower() for value in scenarios] if scenarios else list(SCENARIOS)
    unknown = sorted(set(selected) - SCENARIOS.keys())
    if unknown:
        raise typer.BadParameter(f"unknown scenario(s): {', '.join(unknown)}")
    settings = BenchmarkSettings()
    settings.profile = profile
    if output_dir is not None:
        settings.output_dir = output_dir
    try:
        result_path = asyncio.run(run_benchmarks(settings, selected))
    except (OSError, ValueError) as error:
        typer.echo(f"Benchmark setup failed: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(str(result_path))


if __name__ == "__main__":
    app()
