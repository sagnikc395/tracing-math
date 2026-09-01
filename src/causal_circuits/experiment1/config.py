"""Typed configuration for Experiment 1."""

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
    batch_size: int = 1


@dataclass(frozen=True)
class ProbeFamilyConfig:
    name: str
    penalty: str
    l1_ratios: tuple[float, ...] = ()


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
    primary_family: str = "l2"
    families: tuple[ProbeFamilyConfig, ...] = (ProbeFamilyConfig(name="l2", penalty="l2"),)
    diagnostic_targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class InterventionConfig:
    alphas: tuple[float, ...]
    examples_per_class: int
    random_directions: int
    correct_answer: str
    incorrect_answer: str
    batch_size: int = 1


@dataclass(frozen=True)
class AnalysisConfig:
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_points: int = 181
    localization_tolerances: tuple[int, ...] = (0, 1, 2)
    calibration_bins: int = 10
    confidence_level: float = 0.95
    subgroup_min_traces: int = 20
    exploratory_bootstrap_samples: int = 250


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    model: ModelConfig
    data: DataConfig
    extraction: ExtractionConfig
    probe: ProbeConfig
    intervention: InterventionConfig
    analysis: AnalysisConfig = AnalysisConfig()

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
        if self.extraction.batch_size < 1:
            raise ValueError("extraction.batch_size must be positive")
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
        family_names = [family.name for family in self.probe.families]
        if not family_names or len(family_names) != len(set(family_names)):
            raise ValueError("probe.families must have unique names")
        if self.probe.primary_family not in family_names:
            raise ValueError("probe.primary_family must name a configured family")
        primary = next(
            family for family in self.probe.families if family.name == self.probe.primary_family
        )
        if primary.penalty != "l2":
            raise ValueError("the confirmatory primary probe family must use the l2 penalty")
        for family in self.probe.families:
            if family.penalty not in {"l1", "l2", "elasticnet"}:
                raise ValueError("probe family penalties must be l1, l2, or elasticnet")
            if family.penalty == "elasticnet":
                if not family.l1_ratios or any(not 0 < value < 1 for value in family.l1_ratios):
                    raise ValueError(
                        "elasticnet families require l1_ratios strictly between 0 and 1"
                    )
            elif family.l1_ratios:
                raise ValueError("l1_ratios are only valid for elasticnet probe families")
        allowed_targets = {"invalid_so_far", "error_onset"}
        if any(target not in allowed_targets for target in self.probe.diagnostic_targets):
            raise ValueError("probe diagnostic targets must be invalid_so_far or error_onset")
        if self.probe.target in self.probe.diagnostic_targets:
            raise ValueError("the primary target cannot also be a diagnostic target")
        if not self.intervention.alphas or 0.0 not in self.intervention.alphas:
            raise ValueError("intervention.alphas must include 0")
        if self.intervention.examples_per_class < 1:
            raise ValueError("intervention.examples_per_class must be positive")
        if self.intervention.random_directions < 0:
            raise ValueError("intervention.random_directions cannot be negative")
        if self.intervention.batch_size < 1:
            raise ValueError("intervention.batch_size must be positive")
        if not 0 <= self.analysis.threshold_min < self.analysis.threshold_max <= 1:
            raise ValueError("analysis threshold bounds must satisfy 0 <= min < max <= 1")
        if self.analysis.threshold_points < 2:
            raise ValueError("analysis.threshold_points must be at least 2")
        if not self.analysis.localization_tolerances or any(
            value < 0 for value in self.analysis.localization_tolerances
        ):
            raise ValueError("analysis.localization_tolerances must be non-negative")
        if self.analysis.calibration_bins < 2:
            raise ValueError("analysis.calibration_bins must be at least 2")
        if not 0 < self.analysis.confidence_level < 1:
            raise ValueError("analysis.confidence_level must be between 0 and 1")
        if self.analysis.subgroup_min_traces < 1:
            raise ValueError("analysis.subgroup_min_traces must be positive")
        if self.analysis.exploratory_bootstrap_samples < 0:
            raise ValueError("analysis.exploratory_bootstrap_samples cannot be negative")


def _build_config(raw: dict[str, Any]) -> ExperimentConfig:
    for key in ("model", "data", "extraction", "probe", "intervention"):
        if key not in raw:
            raise ValueError(f"Missing required configuration key: {key}")
    data = raw["data"]
    extraction = raw["extraction"]
    probe = raw["probe"]
    intervention = raw["intervention"]
    analysis = raw.get("analysis", {})
    family_rows = probe.get(
        "families",
        [{"name": str(probe.get("primary_family", "l2")), "penalty": "l2"}],
    )
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
            batch_size=int(extraction.get("batch_size", 1)),
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
            primary_family=str(probe.get("primary_family", "l2")),
            families=tuple(
                ProbeFamilyConfig(
                    name=str(family["name"]),
                    penalty=str(family["penalty"]),
                    l1_ratios=tuple(map(float, family.get("l1_ratios", []))),
                )
                for family in family_rows
            ),
            diagnostic_targets=tuple(map(str, probe.get("diagnostic_targets", []))),
        ),
        intervention=InterventionConfig(
            alphas=tuple(map(float, intervention["alphas"])),
            examples_per_class=int(intervention["examples_per_class"]),
            random_directions=int(intervention["random_directions"]),
            correct_answer=str(intervention.get("correct_answer", " CORRECT")),
            incorrect_answer=str(intervention.get("incorrect_answer", " INCORRECT")),
            batch_size=int(intervention.get("batch_size", 1)),
        ),
        analysis=AnalysisConfig(
            threshold_min=float(analysis.get("threshold_min", 0.05)),
            threshold_max=float(analysis.get("threshold_max", 0.95)),
            threshold_points=int(analysis.get("threshold_points", 181)),
            localization_tolerances=tuple(
                map(int, analysis.get("localization_tolerances", [0, 1, 2]))
            ),
            calibration_bins=int(analysis.get("calibration_bins", 10)),
            confidence_level=float(analysis.get("confidence_level", 0.95)),
            subgroup_min_traces=int(analysis.get("subgroup_min_traces", 20)),
            exploratory_bootstrap_samples=int(analysis.get("exploratory_bootstrap_samples", 250)),
        ),
    )
