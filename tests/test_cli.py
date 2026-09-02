"""Tests for the unified Click interface."""

from pathlib import Path

from click.testing import CliRunner

from tracing_math.cli import main


def test_help_lists_pipeline_stages() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "extract-activations" in result.output
    assert "run-interventions" in result.output
    assert "fit-conditional" in result.output
    assert "run-counterfactuals" in result.output
    assert "fit-contextual-baseline" in result.output
    assert "run-all" in result.output


def test_validate_config_command() -> None:
    config = Path("configs/project.yaml").resolve()

    result = CliRunner().invoke(main, ["--config", str(config), "validate-config"])

    assert result.exit_code == 0
    assert "Valid: Qwen/ProcessBench" in result.output


def test_missing_config_reports_click_error() -> None:
    result = CliRunner().invoke(main, ["--config", "missing.yaml", "validate-config"])

    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_fit_probes_rejects_invalid_worker_count() -> None:
    config = Path("configs/project.yaml").resolve()

    result = CliRunner().invoke(main, ["--config", str(config), "fit-probes", "--workers", "0"])

    assert result.exit_code == 2
    assert "must be positive or -1" in result.output
