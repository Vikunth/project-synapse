"""Command-line entry point."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated, Literal

import typer

from synapse_bench.config import BenchmarkSettings
from synapse_bench.doctor import analyze_artifact
from synapse_bench.doctor_io import (
    DoctorInputError,
    load_artifact,
    render_json,
    render_markdown,
    write_reports,
)
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


@app.command()
def doctor(
    artifact: Annotated[Path, typer.Argument(help="RunArtifact 1.0 JSON to analyze offline.")],
    output_format: Annotated[
        Literal["json", "markdown"],
        typer.Option("--format", help="Stdout format when no output path is selected."),
    ] = "markdown",
    output_dir: Annotated[
        Path | None, typer.Option(help="Existing directory for both versioned report files.")
    ] = None,
    strict: Annotated[
        bool,
        typer.Option(help="Exit 3 when analysis is limited or has warnings/critical findings."),
    ] = False,
 ) -> None:
    """Diagnose a saved benchmark without contacting Ollama or inspecting the host."""
    try:
        if output_dir is not None and output_format != "markdown":
            raise DoctorInputError("--format applies only to stdout mode")
        run_artifact, source = load_artifact(artifact)
        report = analyze_artifact(run_artifact, source)
        if output_dir is not None:
            json_path, markdown_path = write_reports(report, output_dir)
            typer.echo(str(json_path))
            typer.echo(str(markdown_path))
        else:
            typer.echo(
                render_json(report) if output_format == "json" else render_markdown(report),
                nl=False,
            )
        if strict and (
            report.status == "limited"
            or any(item.severity in {"warning", "critical"} for item in report.diagnoses)
        ):
            raise typer.Exit(code=3) from None
    except DoctorInputError as error:
        typer.echo(f"Doctor input error: {error}", err=True)
        raise typer.Exit(code=2) from error
    except FileExistsError as error:
        typer.echo(f"Doctor output error: {error}", err=True)
        raise typer.Exit(code=1) from error
    except OSError as error:
        typer.echo("Doctor internal/output error: report creation failed", err=True)
        raise typer.Exit(code=1) from error


if __name__ == "__main__":
    app()
