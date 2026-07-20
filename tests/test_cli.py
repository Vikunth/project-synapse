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


def test_cli_exposes_doctor_command() -> None:
    result = CliRunner().invoke(app, ["doctor", "--help"])

    assert result.exit_code == 0
    command = get_command(app)
    doctor_command = command.commands["doctor"]
    parameter_names = {parameter.name for parameter in doctor_command.params}
    assert {"artifact", "output_format", "output_dir", "strict"} <= parameter_names
