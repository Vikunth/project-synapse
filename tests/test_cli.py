from typer.testing import CliRunner

from synapse_bench.cli import app


def test_cli_exposes_documented_run_command() -> None:
    result = CliRunner().invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    assert "--profile" in result.stdout
    assert "--scenario" in result.stdout
