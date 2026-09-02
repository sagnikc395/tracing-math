"""Typed configuration for the CPU-only follow-up."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FollowupConfig:
    """Input paths and frozen resampling settings for the follow-up."""

    experiment1_dir: Path
    data_path: Path
    output_dir: Path
    seed: int = 42
    permutation_samples: int = 5_000
    bootstrap_samples: int = 2_000
    confidence_level: float = 0.95
    subgroup_min_traces: int = 20
    audit_examples_per_category: int = 12
    control_c_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)

    @classmethod
    def from_yaml(cls, path: str | Path) -> FollowupConfig:
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError("CPU follow-up configuration must be a YAML mapping")
        config = cls(
            experiment1_dir=Path(raw["experiment1_dir"]),
            data_path=Path(raw["data_path"]),
            output_dir=Path(raw["output_dir"]),
            seed=int(raw.get("seed", 42)),
            permutation_samples=int(raw.get("permutation_samples", 5_000)),
            bootstrap_samples=int(raw.get("bootstrap_samples", 2_000)),
            confidence_level=float(raw.get("confidence_level", 0.95)),
            subgroup_min_traces=int(raw.get("subgroup_min_traces", 20)),
            audit_examples_per_category=int(
                raw.get("audit_examples_per_category", 12)
            ),
            control_c_values=tuple(
                map(float, raw.get("control_c_values", cls.control_c_values))
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.permutation_samples < 1 or self.bootstrap_samples < 1:
            raise ValueError("resampling counts must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one")
        if self.subgroup_min_traces < 1 or self.audit_examples_per_category < 1:
            raise ValueError("trace and audit sample minima must be positive")
        if not self.control_c_values or any(value <= 0 for value in self.control_c_values):
            raise ValueError("control_c_values must be positive")

