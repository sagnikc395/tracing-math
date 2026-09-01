"""Tests for the CPU follow-up Click interface."""

from click.testing import CliRunner

from tracing_math.followup.cli import main


def test_help_describes_followup_without_loading_analysis() -> None:
    result = CliRunner().invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "CPU-only follow-up" in result.output
    assert "--config" in result.output


def test_missing_config_reports_click_error() -> None:
    result = CliRunner().invoke(main, ["--config", "missing.yaml"])

    assert result.exit_code == 2
    assert "does not exist" in result.output
