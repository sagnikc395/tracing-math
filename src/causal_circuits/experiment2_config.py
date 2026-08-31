"""Typed configuration for the Experiment 2 robustness and causal-validation study."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from causal_circuits.config import ModelConfig


@dataclass(frozen=True)
class Experiment2DataConfig:
    path: Path
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    test_fraction: float = 0.2


@dataclass(frozen=True)
class Experiment2AnalysisConfig:
    c_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    max_iter: int = 2000
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_points: int = 181
    subgroup_min_traces: int = 20
    pca_dimensions: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    generator_min_test_traces: int = 20


@dataclass(frozen=True)
class SemanticExtractionConfig:
    save_every: int = 100
    batch_size: int = 1


@dataclass(frozen=True)
class VerdictAuditConfig:
    examples_per_class_validation: int = 64
    examples_per_class_test: int = 128
    batch_size: int = 1
    labels: tuple[str, str] = ("A", "B")


@dataclass(frozen=True)
class CausalValidationConfig:
    examples_per_class: int = 32
    alphas: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    batch_size: int = 1


@dataclass(frozen=True)
class Experiment2Config:
    seed: int
    experiment1_dir: Path
    output_dir: Path
    model: ModelConfig
    data: Experiment2DataConfig
    analysis: Experiment2AnalysisConfig = Experiment2AnalysisConfig()
    semantic_extraction: SemanticExtractionConfig = SemanticExtractionConfig()
    verdict: VerdictAuditConfig = VerdictAuditConfig()
    causal: CausalValidationConfig = CausalValidationConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> Experiment2Config:
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError("Experiment 2 configuration must be a YAML mapping")
        config = _build_config(raw)
        config.validate()
        return config

    def with_paths(
        self,
        *,
        experiment1_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        data_path: str | Path | None = None,
    ) -> Experiment2Config:
        """Return a copy with optional runtime path overrides."""
        return replace(
            self,
            experiment1_dir=(
                self.experiment1_dir
                if experiment1_dir is None
                else Path(experiment1_dir)
            ),
            output_dir=self.output_dir if output_dir is None else Path(output_dir),
            data=replace(
                self.data,
                path=self.data.path if data_path is None else Path(data_path),
            ),
        )

    def validate(self) -> None:
        if self.model.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("model.dtype must be float16, bfloat16, or float32")
        if self.model.max_length < 128:
            raise ValueError("model.max_length must be at least 128")
        fractions = (
            self.data.train_fraction,
            self.data.validation_fraction,
            self.data.test_fraction,
        )
        if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1) > 1e-8:
            raise ValueError("data split fractions must be positive and sum to one")
        if not self.analysis.c_values or any(value <= 0 for value in self.analysis.c_values):
            raise ValueError("analysis.c_values must contain positive values")
        if self.analysis.max_iter < 1 or self.analysis.bootstrap_samples < 1:
            raise ValueError("analysis iteration and bootstrap counts must be positive")
        if not 0 < self.analysis.confidence_level < 1:
            raise ValueError("analysis.confidence_level must be between zero and one")
        if not 0 <= self.analysis.threshold_min < self.analysis.threshold_max <= 1:
            raise ValueError("analysis threshold bounds must satisfy 0 <= min < max <= 1")
        if self.analysis.threshold_points < 2:
            raise ValueError("analysis.threshold_points must be at least two")
        if self.analysis.subgroup_min_traces < 1:
            raise ValueError("analysis.subgroup_min_traces must be positive")
        if self.analysis.generator_min_test_traces < 1:
            raise ValueError("analysis.generator_min_test_traces must be positive")
        if any(value < 1 for value in self.analysis.pca_dimensions):
            raise ValueError("analysis.pca_dimensions must be positive")
        if self.semantic_extraction.save_every < 1:
            raise ValueError("semantic_extraction.save_every must be positive")
        if self.semantic_extraction.batch_size < 1:
            raise ValueError("semantic_extraction.batch_size must be positive")
        if self.verdict.examples_per_class_validation < 1:
            raise ValueError("verdict validation sample size must be positive")
        if self.verdict.examples_per_class_test < 1 or self.verdict.batch_size < 1:
            raise ValueError("verdict test sample size and batch size must be positive")
        if len(self.verdict.labels) != 2 or self.verdict.labels[0] == self.verdict.labels[1]:
            raise ValueError("verdict.labels must contain two distinct labels")
        if any(not label.strip() for label in self.verdict.labels):
            raise ValueError("verdict labels cannot be empty")
        if self.causal.examples_per_class < 1 or self.causal.batch_size < 1:
            raise ValueError("causal sample size and batch size must be positive")
        if not self.causal.alphas or 0.0 not in self.causal.alphas:
            raise ValueError("causal.alphas must include zero")


def _build_config(raw: dict[str, Any]) -> Experiment2Config:
    for key in ("experiment1_dir", "output_dir", "model", "data"):
        if key not in raw:
            raise ValueError(f"Missing Experiment 2 configuration key: {key}")
    data = raw["data"]
    analysis = raw.get("analysis", {})
    semantic = raw.get("semantic_extraction", {})
    verdict = raw.get("verdict", {})
    causal = raw.get("causal", {})
    labels = tuple(map(str, verdict.get("labels", ["A", "B"])))
    return Experiment2Config(
        seed=int(raw.get("seed", 42)),
        experiment1_dir=Path(raw["experiment1_dir"]),
        output_dir=Path(raw["output_dir"]),
        model=ModelConfig(**raw["model"]),
        data=Experiment2DataConfig(
            path=Path(data["path"]),
            train_fraction=float(data.get("train_fraction", 0.6)),
            validation_fraction=float(data.get("validation_fraction", 0.2)),
            test_fraction=float(data.get("test_fraction", 0.2)),
        ),
        analysis=Experiment2AnalysisConfig(
            c_values=tuple(map(float, analysis.get("c_values", [0.01, 0.1, 1, 10]))),
            max_iter=int(analysis.get("max_iter", 2000)),
            bootstrap_samples=int(analysis.get("bootstrap_samples", 1000)),
            confidence_level=float(analysis.get("confidence_level", 0.95)),
            threshold_min=float(analysis.get("threshold_min", 0.05)),
            threshold_max=float(analysis.get("threshold_max", 0.95)),
            threshold_points=int(analysis.get("threshold_points", 181)),
            subgroup_min_traces=int(analysis.get("subgroup_min_traces", 20)),
            pca_dimensions=tuple(
                map(int, analysis.get("pca_dimensions", [1, 2, 4, 8, 16, 32, 64, 128]))
            ),
            generator_min_test_traces=int(
                analysis.get("generator_min_test_traces", 20)
            ),
        ),
        semantic_extraction=SemanticExtractionConfig(
            save_every=int(semantic.get("save_every", 100)),
            batch_size=int(semantic.get("batch_size", 1)),
        ),
        verdict=VerdictAuditConfig(
            examples_per_class_validation=int(
                verdict.get("examples_per_class_validation", 64)
            ),
            examples_per_class_test=int(verdict.get("examples_per_class_test", 128)),
            batch_size=int(verdict.get("batch_size", 1)),
            labels=(labels[0], labels[1]),
        ),
        causal=CausalValidationConfig(
            examples_per_class=int(causal.get("examples_per_class", 32)),
            alphas=tuple(map(float, causal.get("alphas", [-2, -1, 0, 1, 2]))),
            batch_size=int(causal.get("batch_size", 1)),
        ),
    )
