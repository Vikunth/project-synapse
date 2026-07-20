"""Command-line entry point."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Literal

import typer

from synapse_bench.config import BenchmarkSettings
from synapse_bench.native_runner import PROBES, resume_native_probe, run_native_probe
from synapse_bench.runner import SCENARIOS, run_benchmarks

app = typer.Typer(no_args_is_help=True, help="Run bounded local Ollama benchmarks.")
native_probe_app = typer.Typer(
    no_args_is_help=True, help="Run prompt-free native crossover probes."
)
app.add_typer(native_probe_app, name="native-probe")


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


@native_probe_app.command("run")
def native_probe_run(
    profile: Annotated[
        Literal["smoke", "full"],
        typer.Option(help="Mechanics smoke test or evidence-producing full probe."),
    ] = "smoke",
    probes: Annotated[
        list[str] | None,
        typer.Option(
            "--probe", help="Repeat to select prefix-reuse, dual-residency, or concurrency."
        ),
    ] = None,
    seed: Annotated[int, typer.Option(help="Deterministic schedule seed.")] = 42,
    output_dir: Annotated[Path, typer.Option(help="Prompt-free artifact directory.")] = Path(
        "benchmarks/results/native-probes"
    ),
    acknowledge_full_load: Annotated[
        bool,
        typer.Option(
            "--acknowledge-full-load",
            help="Required acknowledgement for the bounded full load test.",
        ),
    ] = False,
) -> None:
    """Start a new NativeProbeArtifact 1.0 run."""
    selected = set(probes or PROBES)
    try:
        path, code = asyncio.run(
            run_native_probe(
                BenchmarkSettings(),
                profile=profile,
                probes=selected,
                seed=seed,
                output_dir=output_dir,
                acknowledge_full_load=acknowledge_full_load,
            )
        )
    except (OSError, ValueError) as error:
        typer.echo(f"Native probe setup failed: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(str(path))
    if code:
        raise typer.Exit(code=code)


@native_probe_app.command("resume")
def native_probe_resume(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Resume exactly the configuration and schedule stored in PATH."""
    try:
        result_path, code = asyncio.run(resume_native_probe(path))
    except (OSError, ValueError) as error:
        typer.echo(f"Native probe resume failed: {error}", err=True)
        raise typer.Exit(code=2) from error
    typer.echo(str(result_path))
    if code:
        raise typer.Exit(code=code)


if __name__ == "__main__":
    app()
