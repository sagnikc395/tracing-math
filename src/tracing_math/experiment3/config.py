"""Typed configuration for the exploratory workshop follow-ups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExtendedFollowupConfig:
    experiment1_dir: Path
    data_path: Path
    output_dir: Path
    counterfactual_pairs_path: Path
    model_name: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    device: str = "auto"
    dtype: str = "float16"
    max_length: int = 2048
    seed: int = 42
    c_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    max_iter: int = 2000
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95
    extraction_save_every: int = 100
    extraction_batch_size: int = 1
    patching_batch_size: int = 1
    counterfactual_template_size: int = 160

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExtendedFollowupConfig:
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError("Extended follow-up configuration must be a YAML mapping")
        model = raw.get("model", {})
        transition = raw.get("transition_probe", {})
        extraction = raw.get("boundary_control", {})
        patching = raw.get("counterfactual_patching", {})
        config = cls(
            experiment1_dir=Path(raw["experiment1_dir"]),
            data_path=Path(raw["data_path"]),
            output_dir=Path(raw["output_dir"]),
            counterfactual_pairs_path=Path(patching["pairs_path"]),
            model_name=str(model.get("name", cls.model_name)),
            device=str(model.get("device", cls.device)),
            dtype=str(model.get("dtype", cls.dtype)),
            max_length=int(model.get("max_length", cls.max_length)),
            seed=int(raw.get("seed", cls.seed)),
            c_values=tuple(map(float, transition.get("c_values", cls.c_values))),
            max_iter=int(transition.get("max_iter", cls.max_iter)),
            bootstrap_samples=int(raw.get("bootstrap_samples", cls.bootstrap_samples)),
            confidence_level=float(raw.get("confidence_level", cls.confidence_level)),
            extraction_save_every=int(
                extraction.get("save_every", cls.extraction_save_every)
            ),
            extraction_batch_size=int(
                extraction.get("batch_size", cls.extraction_batch_size)
            ),
            patching_batch_size=int(patching.get("batch_size", cls.patching_batch_size)),
            counterfactual_template_size=int(
                patching.get("template_size", cls.counterfactual_template_size)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("model.dtype must be float16, bfloat16, or float32")
        if self.max_length < 128:
            raise ValueError("model.max_length must be at least 128")
        if not self.c_values or any(value <= 0 for value in self.c_values):
            raise ValueError("transition_probe.c_values must be positive")
        if self.max_iter < 1 or self.bootstrap_samples < 1:
            raise ValueError("iteration and bootstrap counts must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one")
        if min(
            self.extraction_save_every,
            self.extraction_batch_size,
            self.patching_batch_size,
            self.counterfactual_template_size,
        ) < 1:
            raise ValueError("batch, shard, and template sizes must be positive")

