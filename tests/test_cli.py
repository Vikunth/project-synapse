from typer.main import get_command
from typer.testing import CliRunner

from synapse_bench.cli import app


def test_cli_exposes_documented_run_command() -> None:
    result = CliRunner().invoke(app, ["run", "--help"])

    assert result.exit_code == 0
    command = get_command(app)
    run_command = command.commands["run"]
    parameter_names = {parameter.name for parameter in run_command.params}
    assert {"profile", "scenarios"} <= parameter_names
