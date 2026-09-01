from pathlib import Path

from click.testing import CliRunner

from tracing_math.experiment3.cli import main


def test_extended_help_lists_restartable_stages() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "fit-transition-probe" in result.output
    assert "extract-boundary-controls" in result.output
    assert "run-counterfactual-patching" in result.output


def test_extended_config_validation_command() -> None:
    config = Path("configs/experiment3.yaml").resolve()
    result = CliRunner().invoke(main, ["--config", str(config), "validate-config"])

    assert result.exit_code == 0
    assert "Valid extended follow-up" in result.output

