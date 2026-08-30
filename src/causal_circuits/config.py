"""Typed configuration for the mathematical error-tracing experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    device: str = "auto"
    dtype: str = "float16"
    max_length: int = 2048


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    splits: tuple[str, ...]
    output_path: Path
    max_examples_per_split: int | None = None


@dataclass(frozen=True)
class ExtractionConfig:
    output_dir: Path
    save_every: int = 100


@dataclass(frozen=True)
class ProbeConfig:
    target: str
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    c_values: tuple[float, ...]
    max_iter: int
    bootstrap_samples: int
    pca_dimensions: tuple[int, ...]


@dataclass(frozen=True)
class InterventionConfig:
    alphas: tuple[float, ...]
    examples_per_class: int
    random_directions: int
    correct_answer: str
    incorrect_answer: str


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    model: ModelConfig
    data: DataConfig
    extraction: ExtractionConfig
    probe: ProbeConfig
    intervention: InterventionConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path)
        raw = yaml.safe_load(config_path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a mapping in {config_path}")
        config = _build_config(raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.model.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("model.dtype must be float16, bfloat16, or float32")
        if self.model.max_length < 128:
            raise ValueError("model.max_length must be at least 128")
        if not self.data.splits:
            raise ValueError("data.splits cannot be empty")
        if self.data.max_examples_per_split is not None and self.data.max_examples_per_split < 1:
            raise ValueError("data.max_examples_per_split must be positive or null")
        if self.extraction.save_every < 1:
            raise ValueError("extraction.save_every must be positive")
        if self.probe.target not in {"invalid_so_far", "error_onset"}:
            raise ValueError("probe.target must be invalid_so_far or error_onset")
        fractions = (
            self.probe.train_fraction,
            self.probe.validation_fraction,
            self.probe.test_fraction,
        )
        if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-8:
            raise ValueError("probe split fractions must be positive and sum to one")
        if not self.probe.c_values or any(value <= 0 for value in self.probe.c_values):
            raise ValueError("probe.c_values must contain positive values")
        if self.probe.max_iter < 1 or self.probe.bootstrap_samples < 0:
            raise ValueError("probe iteration counts cannot be negative")
        if any(value < 1 for value in self.probe.pca_dimensions):
            raise ValueError("probe.pca_dimensions must be positive")
        if not self.intervention.alphas or 0.0 not in self.intervention.alphas:
            raise ValueError("intervention.alphas must include 0")
        if self.intervention.examples_per_class < 1:
            raise ValueError("intervention.examples_per_class must be positive")
        if self.intervention.random_directions < 0:
            raise ValueError("intervention.random_directions cannot be negative")


def _build_config(raw: dict[str, Any]) -> ExperimentConfig:
    for key in ("model", "data", "extraction", "probe", "intervention"):
        if key not in raw:
            raise ValueError(f"Missing required configuration key: {key}")
    data = raw["data"]
    extraction = raw["extraction"]
    probe = raw["probe"]
    intervention = raw["intervention"]
    return ExperimentConfig(
        seed=int(raw.get("seed", 42)),
        model=ModelConfig(**raw["model"]),
        data=DataConfig(
            dataset=str(data["dataset"]),
            splits=tuple(map(str, data["splits"])),
            output_path=Path(data["output_path"]),
            max_examples_per_split=(
                None
                if data.get("max_examples_per_split") is None
                else int(data["max_examples_per_split"])
            ),
        ),
        extraction=ExtractionConfig(
            output_dir=Path(extraction["output_dir"]),
            save_every=int(extraction.get("save_every", 100)),
        ),
        probe=ProbeConfig(
            target=str(probe["target"]),
            train_fraction=float(probe["train_fraction"]),
            validation_fraction=float(probe["validation_fraction"]),
            test_fraction=float(probe["test_fraction"]),
            c_values=tuple(map(float, probe["c_values"])),
            max_iter=int(probe.get("max_iter", 2000)),
            bootstrap_samples=int(probe.get("bootstrap_samples", 1000)),
            pca_dimensions=tuple(map(int, probe.get("pca_dimensions", []))),
        ),
        intervention=InterventionConfig(
            alphas=tuple(map(float, intervention["alphas"])),
            examples_per_class=int(intervention["examples_per_class"]),
            random_directions=int(intervention["random_directions"]),
            correct_answer=str(intervention.get("correct_answer", " CORRECT")),
            incorrect_answer=str(intervention.get("incorrect_answer", " INCORRECT")),
        ),
    )
