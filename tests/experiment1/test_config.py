"""Tests for Experiment 1 configuration parsing."""

from pathlib import Path

import pytest
import yaml

from tracing_math.experiment1.config import ExperimentConfig


def test_default_config_is_valid() -> None:
    config = ExperimentConfig.from_yaml("configs/experiment1.yaml")
    assert config.model.name == "Qwen/Qwen2.5-Math-1.5B-Instruct"
    assert config.data.splits == ("gsm8k", "math", "olympiadbench", "omnimath")
    assert 0.0 in config.intervention.alphas
    assert config.probe.primary_family == "l2"
    assert {family.name for family in config.probe.families} == {"l2"}
    assert config.probe.diagnostic_targets == ()
    assert config.probe.pca_dimensions == ()
    assert config.analysis.exploratory_bootstrap_samples == 0
    assert config.analysis.localization_tolerances == (0, 1, 2)
    assert config.analysis.workers == 4
    assert config.extraction.batch_size == 1
    assert config.intervention.batch_size == 1


def test_legacy_config_uses_confirmatory_defaults(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/experiment1.yaml").read_text())
    raw.pop("analysis")
    raw["probe"].pop("primary_family")
    raw["probe"].pop("families")
    raw["probe"].pop("diagnostic_targets")
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = ExperimentConfig.from_yaml(path)
    assert [family.name for family in config.probe.families] == ["l2"]
    assert config.analysis.calibration_bins == 10


@pytest.mark.parametrize("workers", [0, -2])
def test_invalid_worker_count_is_rejected(tmp_path: Path, workers: int) -> None:
    raw = yaml.safe_load(Path("configs/experiment1.yaml").read_text())
    raw["analysis"]["workers"] = workers
    path = tmp_path / "invalid-workers.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="analysis.workers"):
        ExperimentConfig.from_yaml(path)


def test_all_cpu_worker_setting_is_valid(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/experiment1.yaml").read_text())
    raw["analysis"]["workers"] = -1
    path = tmp_path / "all-workers.yaml"
    path.write_text(yaml.safe_dump(raw))

    assert ExperimentConfig.from_yaml(path).analysis.workers == -1
