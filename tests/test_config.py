from pathlib import Path

import yaml

from causal_circuits.config import ExperimentConfig


def test_default_config_is_valid() -> None:
    config = ExperimentConfig.from_yaml("configs/experiment.yaml")
    assert config.model.name == "Qwen/Qwen2.5-Math-1.5B-Instruct"
    assert config.data.splits == ("gsm8k", "math", "olympiadbench", "omnimath")
    assert 0.0 in config.intervention.alphas
    assert config.probe.primary_family == "l2"
    assert {family.name for family in config.probe.families} == {"l2"}
    assert config.probe.diagnostic_targets == ()
    assert config.probe.pca_dimensions == ()
    assert config.analysis.exploratory_bootstrap_samples == 0
    assert config.analysis.localization_tolerances == (0, 1, 2)


def test_legacy_config_uses_confirmatory_defaults(tmp_path: Path) -> None:
    raw = yaml.safe_load(Path("configs/experiment.yaml").read_text())
    raw.pop("analysis")
    raw["probe"].pop("primary_family")
    raw["probe"].pop("families")
    raw["probe"].pop("diagnostic_targets")
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = ExperimentConfig.from_yaml(path)
    assert [family.name for family in config.probe.families] == ["l2"]
    assert config.analysis.calibration_bins == 10
