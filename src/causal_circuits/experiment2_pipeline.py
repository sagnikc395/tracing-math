"""Stage orchestration and figures for Experiment 2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from causal_circuits.experiment2_analysis import run_marker_robustness
from causal_circuits.experiment2_causal import audit_verdict_readout, run_causal_validation
from causal_circuits.experiment2_config import Experiment2Config
from causal_circuits.experiment2_runtime import atomic_write_json, atomic_write_text, run_stage
from causal_circuits.experiment2_semantic import (
    extract_semantic_activation_shards,
    fit_semantic_boundary_probes,
)


def validate_experiment2_inputs(config: Experiment2Config) -> dict[str, object]:
    missing = []
    required = [
        config.data.path,
        config.experiment1_dir / "activation_shards",
        config.experiment1_dir / "probes" / "directions.npz",
    ]
    for path in required:
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Experiment 2 requires the full Experiment 1 cache. Missing: " + ", ".join(missing)
        )
    shard_count = len(list((config.experiment1_dir / "activation_shards").glob("shard_*.npy")))
    if shard_count == 0:
        raise FileNotFoundError("Experiment 1 activation_shards contains no .npy shards")
    return {
        "valid": True,
        "data_path": str(config.data.path),
        "experiment1_dir": str(config.experiment1_dir),
        "output_dir": str(config.output_dir),
        "experiment1_activation_shards": shard_count,
    }


def write_resolved_config(config: Experiment2Config) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    payload = _serialize(asdict(config))
    path = config.output_dir / "experiment_config.yaml"
    atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))
    return path


def experiment2_run_identity(config: Experiment2Config) -> str:
    payload = _serialize(asdict(config))
    # Batch sizes change scheduling and peak memory, not samples or estimands. Keeping them
    # out of the durable identity lets run-all reuse completed stages after safe OOM tuning.
    for section in ("semantic_extraction", "verdict", "causal"):
        payload[section].pop("batch_size", None)
    payload = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def run_all_experiment2(
    config: Experiment2Config,
    *,
    force: bool = False,
) -> dict[str, object]:
    write_resolved_config(config)
    identity = experiment2_run_identity(config)
    stages = (
        ("validate-config", lambda: validate_experiment2_inputs(config)),
        ("analyze-robustness", lambda: run_marker_robustness(config)),
        ("extract-semantic", lambda: extract_semantic_activation_shards(config)),
        ("fit-semantic", lambda: fit_semantic_boundary_probes(config)),
        ("audit-verdict", lambda: audit_verdict_readout(config)),
        ("causal-validation", lambda: run_causal_validation(config)),
        ("plot", lambda: [str(path) for path in plot_experiment2(config)]),
    )
    results = {}
    for name, operation in stages:
        results[name] = run_stage(
            config.output_dir,
            name,
            operation,
            skip_completed=not force,
            identity=identity,
        )
    atomic_write_json(config.output_dir / "run_summary.json", results)
    return results


def plot_experiment2(config: Experiment2Config) -> list[Path]:
    """Create the compact Experiment 2 robustness and causal-validation figure."""
    import matplotlib.pyplot as plt

    marker_dir = config.output_dir / "marker_robustness"
    semantic_dir = config.output_dir / "semantic_boundary"
    verdict_dir = config.output_dir / "verdict_audit"
    causal_dir = config.output_dir / "causal_validation"
    required = [
        marker_dir / "layer_metrics.csv",
        semantic_dir / "layer_metrics.csv",
        verdict_dir / "summary.csv",
        causal_dir / "intervention_summary.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Run all Experiment 2 stages before plotting: " + ", ".join(missing)
        )

    marker_metrics = pd.read_csv(marker_dir / "layer_metrics.csv")
    semantic_metrics = pd.read_csv(semantic_dir / "layer_metrics.csv")
    marker_artifact = np.load(marker_dir / "directions.npz")
    semantic_artifact = np.load(semantic_dir / "directions.npz")
    marker_layer = int(marker_artifact["selected_layer"])
    semantic_layer = int(semantic_artifact["selected_layer"])
    verdict = pd.read_csv(verdict_dir / "summary.csv")
    causal = pd.read_csv(causal_dir / "intervention_summary.csv")

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axis = axes[0]
    for frame, label, color in (
        (marker_metrics, "marker boundary", "C0"),
        (semantic_metrics, "natural step end", "C2"),
    ):
        test = frame[frame["split"] == "test"]
        axis.plot(test["layer"], test["auroc"], label=label, color=color)
    axis.axvline(marker_layer, color="C0", linestyle=":", alpha=0.6)
    axis.axvline(semantic_layer, color="C2", linestyle=":", alpha=0.6)
    axis.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8)
    axis.set(xlabel="Hidden-state index", ylabel="Test AUROC", title="Boundary robustness")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1]
    verdict_test = verdict[verdict["partition"] == "test"]
    axis.bar(
        verdict_test["mapping"],
        verdict_test["balanced_accuracy"],
        color=["C1", "C4", "C3"][: len(verdict_test)],
    )
    axis.axhline(0.5, color="0.5", linestyle="--", linewidth=0.8)
    axis.set(
        ylabel="Balanced accuracy",
        title="Single-token verdict audit",
        ylim=(0, 1),
    )
    axis.tick_params(axis="x", rotation=25)

    axis = axes[2]
    for direction_type, group in causal.groupby("direction_type", sort=True):
        group = group.sort_values("alpha")
        axis.plot(group["alpha"], group["mean_delta"], marker="o", label=direction_type)
        axis.fill_between(
            group["alpha"],
            group["ci_low"],
            group["ci_high"],
            alpha=0.15,
        )
    axis.axhline(0, color="0.5", linewidth=0.8)
    axis.set(xlabel=r"Dose $\alpha$", ylabel="Change in invalid margin", title="Causal assay")
    axis.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    output = config.output_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / "experiment2_summary.pdf"
    png = output / "experiment2_summary.png"
    figure.savefig(pdf, bbox_inches="tight")
    figure.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return [pdf, png]


def _serialize(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value
