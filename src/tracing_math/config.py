"""Validated configuration for the tracing pipeline."""

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
    workers: int = 1
    permutation_samples: int = 5_000
    audit_examples_per_category: int = 12
    control_c_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    conditional_practical_margin: float = 0.02
    conditional_tfidf_min_df: int = 2
    conditional_tfidf_max_features: int = 20_000
    contextual_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    contextual_encoder_revision: str | None = None
    contextual_batch_size: int = 16
    contextual_max_length: int = 512
    transition_bootstrap_samples: int = 1_000
    transition_max_iter: int = 2_000
    boundary_save_every: int = 100
    boundary_batch_size: int = 1
    patching_batch_size: int = 1
    counterfactual_template_size: int = 160


@dataclass(frozen=True)
class ArtifactConfig:
    analysis_dir: Path
    counterfactual_pairs_path: Path


@dataclass(frozen=True)
class ProjectConfig:
    """Single configuration for extraction, prediction, analysis, and reporting."""

    seed: int
    model: ModelConfig
    data: DataConfig
    extraction: ExtractionConfig
    probe: ProbeConfig
    intervention: InterventionConfig
    artifacts: ArtifactConfig
    analysis: AnalysisConfig = AnalysisConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> ProjectConfig:
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
            raise ValueError("the primary probe family must use the l2 penalty")
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
        if any(
            target not in {"invalid_so_far", "error_onset"}
            for target in self.probe.diagnostic_targets
        ):
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
        if self.intervention.correct_answer == self.intervention.incorrect_answer:
            raise ValueError("intervention verdict answers must differ")
        if (
            self.intervention.correct_answer.strip().casefold(),
            self.intervention.incorrect_answer.strip().casefold(),
        ) != ("no", "yes"):
            raise ValueError("the fixed verdict readout requires No and Yes answers")
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
        if self.analysis.workers == 0 or self.analysis.workers < -1:
            raise ValueError("analysis.workers must be positive or -1 for all available CPUs")
        if self.analysis.permutation_samples < 1:
            raise ValueError("analysis.permutation_samples must be positive")
        if self.analysis.audit_examples_per_category < 1:
            raise ValueError("analysis.audit_examples_per_category must be positive")
        if not self.analysis.control_c_values or any(
            value <= 0 for value in self.analysis.control_c_values
        ):
            raise ValueError("analysis.control_c_values must be positive")
        if self.analysis.transition_bootstrap_samples < 1:
            raise ValueError("analysis.transition_bootstrap_samples must be positive")
        if self.analysis.transition_max_iter < 1:
            raise ValueError("analysis.transition_max_iter must be positive")
        if (
            min(
                self.analysis.conditional_tfidf_min_df,
                self.analysis.conditional_tfidf_max_features,
                self.analysis.boundary_save_every,
                self.analysis.boundary_batch_size,
                self.analysis.patching_batch_size,
                self.analysis.counterfactual_template_size,
                self.analysis.contextual_batch_size,
                self.analysis.contextual_max_length,
            )
            < 1
        ):
            raise ValueError("analysis batch, shard, and template sizes must be positive")
        if not 0 <= self.analysis.conditional_practical_margin <= 1:
            raise ValueError("analysis.conditional_practical_margin must be in [0, 1]")
        if not self.analysis.contextual_encoder_name.strip():
            raise ValueError("analysis.contextual_encoder_name cannot be empty")


def _build_config(raw: dict[str, Any]) -> ProjectConfig:
    for key in ("model", "data", "extraction", "probe", "intervention"):
        if key not in raw:
            raise ValueError(f"Missing required configuration key: {key}")
    data = raw["data"]
    extraction = raw["extraction"]
    probe = raw["probe"]
    intervention = raw["intervention"]
    analysis = raw.get("analysis", {})
    artifacts = raw.get("artifacts", {})
    family_rows = probe.get(
        "families",
        [{"name": str(probe.get("primary_family", "l2")), "penalty": "l2"}],
    )
    return ProjectConfig(
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
            correct_answer=str(intervention.get("correct_answer", " No")),
            incorrect_answer=str(intervention.get("incorrect_answer", " Yes")),
            batch_size=int(intervention.get("batch_size", 1)),
        ),
        artifacts=ArtifactConfig(
            analysis_dir=Path(artifacts.get("analysis_dir", "artifacts/analysis")),
            counterfactual_pairs_path=Path(
                artifacts.get(
                    "counterfactual_pairs_path", "data/processed/counterfactual_pairs.jsonl"
                )
            ),
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
            workers=int(analysis.get("workers", 1)),
            permutation_samples=int(analysis.get("permutation_samples", 5_000)),
            audit_examples_per_category=int(analysis.get("audit_examples_per_category", 12)),
            control_c_values=tuple(
                map(float, analysis.get("control_c_values", [0.01, 0.1, 1.0, 10.0]))
            ),
            conditional_practical_margin=float(analysis.get("conditional_practical_margin", 0.02)),
            conditional_tfidf_min_df=int(analysis.get("conditional_tfidf_min_df", 2)),
            conditional_tfidf_max_features=int(
                analysis.get("conditional_tfidf_max_features", 20_000)
            ),
            contextual_encoder_name=str(
                analysis.get("contextual_encoder_name", "sentence-transformers/all-MiniLM-L6-v2")
            ),
            contextual_encoder_revision=(
                None
                if analysis.get("contextual_encoder_revision") is None
                else str(analysis["contextual_encoder_revision"])
            ),
            contextual_batch_size=int(analysis.get("contextual_batch_size", 16)),
            contextual_max_length=int(analysis.get("contextual_max_length", 512)),
            transition_bootstrap_samples=int(analysis.get("transition_bootstrap_samples", 1_000)),
            transition_max_iter=int(analysis.get("transition_max_iter", 2_000)),
            boundary_save_every=int(analysis.get("boundary_save_every", 100)),
            boundary_batch_size=int(analysis.get("boundary_batch_size", 1)),
            patching_batch_size=int(analysis.get("patching_batch_size", 1)),
            counterfactual_template_size=int(analysis.get("counterfactual_template_size", 160)),
        ),
    )
