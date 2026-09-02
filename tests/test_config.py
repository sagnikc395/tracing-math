"""Tests for unified project configuration parsing."""

from pathlib import Path

import pytest
import yaml

from tracing_math.config import ProjectConfig


def test_default_config_is_valid() -> None:
    config = ProjectConfig.from_yaml("configs/project.yaml")
    assert config.model.name == "Qwen/Qwen2.5-Math-1.5B-Instruct"
    assert config.data.splits == ("gsm8k", "math", "olympiadbench", "omnimath")
    assert 0.0 in config.intervention.alphas
    assert config.probe.primary_family == "l2"
    assert {family.name for family in config.probe.families} == {"l2"}
    assert config.probe.diagnostic_targets == ()
    assert config.probe.pca_dimensions == ()
    assert config.analysis.exploratory_bootstrap_samples == 2000
    assert config.analysis.localization_tolerances == (0, 1, 2)
    assert config.analysis.workers == 4
    assert config.extraction.batch_size == 1
    assert config.intervention.batch_size == 1


def test_config_uses_safe_defaults_when_optional_sections_are_missing(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/project.yaml").read_text())
    raw.pop("analysis")
    raw["probe"].pop("primary_family")
    raw["probe"].pop("families")
    raw["probe"].pop("diagnostic_targets")
    path = tmp_path / "minimal.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = ProjectConfig.from_yaml(path)
    assert [family.name for family in config.probe.families] == ["l2"]
    assert config.analysis.calibration_bins == 10
    assert config.analysis.contextual_encoder_name == "sentence-transformers/all-MiniLM-L6-v2"


@pytest.mark.parametrize("workers", [0, -2])
def test_invalid_worker_count_is_rejected(tmp_path: Path, workers: int) -> None:
    raw = yaml.safe_load(Path("configs/project.yaml").read_text())
    raw["analysis"]["workers"] = workers
    path = tmp_path / "invalid-workers.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="analysis.workers"):
        ProjectConfig.from_yaml(path)


def test_all_cpu_worker_setting_is_valid(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/project.yaml").read_text())
    raw["analysis"]["workers"] = -1
    path = tmp_path / "all-workers.yaml"
    path.write_text(yaml.safe_dump(raw))

    assert ProjectConfig.from_yaml(path).analysis.workers == -1
