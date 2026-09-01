"""Tests for CPU follow-up configuration parsing."""

from dataclasses import replace

import pytest

from tracing_math.followup.config import FollowupConfig


def test_default_followup_config_is_valid() -> None:
    config = FollowupConfig.from_yaml("configs/experiment2.yaml")

    assert config.permutation_samples == 5_000
    assert config.bootstrap_samples == 2_000
    assert config.experiment1_dir.name == "qwen2.5-math-1.5b"


def test_followup_config_rejects_invalid_resampling_count() -> None:
    config = FollowupConfig.from_yaml("configs/experiment2.yaml")

    with pytest.raises(ValueError, match="resampling counts"):
        replace(config, permutation_samples=0).validate()
