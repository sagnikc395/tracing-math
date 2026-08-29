"""Typed experiment configuration and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    device: str = "auto"
    batch_size: int = 16


@dataclass(frozen=True)
class DataConfig:
    assay: str
    path: Path
    sequence_column: str = "mutated_sequence"
    mutation_column: str = "mutation"
    fitness_column: str = "DMS_score"
    directionality: int = 1


@dataclass(frozen=True)
class ScoringConfig:
    strategy: str
    output_path: Path


@dataclass(frozen=True)
class CircuitConfig:
    backend: str
    checkpoint: Path
    attribution: str
    top_k_fractions: tuple[float, ...]
    random_trials: int


@dataclass(frozen=True)
class AnalysisConfig:
    residue_aggregation: str
    intolerant_fraction: float
    bootstrap_samples: int
    output_dir: Path


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    model: ModelConfig
    data: DataConfig
    scoring: ScoringConfig
    circuit: CircuitConfig
    analysis: AnalysisConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a mapping in {config_path}")
        config = _build_config(raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.model.batch_size < 1:
            raise ValueError("model.batch_size must be positive")
        if self.data.directionality not in {-1, 1}:
            raise ValueError("data.directionality must be either -1 or 1")
        if self.scoring.strategy != "masked_marginal":
            raise ValueError("Only masked_marginal scoring is currently supported")
        if self.circuit.attribution != "activation_x_gradient":
            raise ValueError("Only activation_x_gradient attribution is currently supported")
        if not self.circuit.top_k_fractions:
            raise ValueError("circuit.top_k_fractions cannot be empty")
        if any(not 0 < fraction <= 1 for fraction in self.circuit.top_k_fractions):
            raise ValueError("Every top-k fraction must lie in (0, 1]")
        if not 0 < self.analysis.intolerant_fraction < 1:
            raise ValueError("analysis.intolerant_fraction must lie in (0, 1)")


def _require(raw: dict[str, Any], key: str) -> Any:
    if key not in raw:
        raise ValueError(f"Missing required configuration key: {key}")
    return raw[key]


def _build_config(raw: dict[str, Any]) -> ExperimentConfig:
    model = _require(raw, "model")
    data = _require(raw, "data")
    scoring = _require(raw, "scoring")
    circuit = _require(raw, "circuit")
    analysis = _require(raw, "analysis")
    return ExperimentConfig(
        seed=int(raw.get("seed", 42)),
        model=ModelConfig(**model),
        data=DataConfig(**{**data, "path": Path(data["path"])}),
        scoring=ScoringConfig(
            strategy=scoring["strategy"], output_path=Path(scoring["output_path"])
        ),
        circuit=CircuitConfig(
            backend=circuit["backend"],
            checkpoint=Path(circuit["checkpoint"]),
            attribution=circuit["attribution"],
            top_k_fractions=tuple(float(x) for x in circuit["top_k_fractions"]),
            random_trials=int(circuit["random_trials"]),
        ),
        analysis=AnalysisConfig(
            residue_aggregation=analysis["residue_aggregation"],
            intolerant_fraction=float(analysis["intolerant_fraction"]),
            bootstrap_samples=int(analysis["bootstrap_samples"]),
            output_dir=Path(analysis["output_dir"]),
        ),
    )
