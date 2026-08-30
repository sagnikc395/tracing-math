"""Resumable experiment stages and artifact I/O."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from causal_circuits.analysis import ProbeResults, binary_metrics, fit_layer_probes
from causal_circuits.circuits import (
    causal_effect_statistics,
    run_interventions,
    summarize_interventions,
)
from causal_circuits.config import ExperimentConfig
from causal_circuits.data import (
    assign_partitions,
    iter_step_metadata,
    load_huggingface_traces,
    load_traces,
    save_traces,
)
from causal_circuits.models import HuggingFaceMathModel, TraceTooLongError


def download_data(config: ExperimentConfig, *, force: bool = False) -> Path:
    output = config.data.output_path
    if output.exists() and not force:
        return output
    traces = load_huggingface_traces(
        config.data.dataset,
        config.data.splits,
        max_examples_per_split=config.data.max_examples_per_split,
    )
    save_traces(traces, output)
    return output


def extract_activation_shards(config: ExperimentConfig) -> dict[str, int]:
    traces = load_traces(config.data.output_path)
    partitions = assign_partitions(
        traces,
        seed=config.seed,
        train_fraction=config.probe.train_fraction,
        validation_fraction=config.probe.validation_fraction,
    )
    output_dir = config.extraction.output_dir
    shard_dir = output_dir / "activation_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    _check_extraction_identity(config, output_dir)
    model = HuggingFaceMathModel(
        config.model.name,
        device=config.model.device,
        dtype=config.model.dtype,
        max_length=config.model.max_length,
    )

    totals = {"traces": len(traces), "extracted": 0, "skipped_long": 0, "cached": 0}
    chunk_size = config.extraction.save_every
    for start in tqdm(range(0, len(traces), chunk_size), desc="activation shards"):
        end = min(start + chunk_size, len(traces))
        stem = f"shard_{start:05d}_{end - 1:05d}"
        array_path = shard_dir / f"{stem}.npy"
        metadata_path = shard_dir / f"{stem}.csv"
        manifest_path = shard_dir / f"{stem}.json"
        if array_path.exists() and metadata_path.exists() and manifest_path.exists():
            cached = json.loads(manifest_path.read_text())
            totals["cached"] += int(cached["extracted"])
            totals["skipped_long"] += int(cached["skipped_long"])
            continue

        arrays: list[np.ndarray] = []
        rows: list[dict[str, object]] = []
        skipped: list[dict[str, object]] = []
        for trace in tqdm(traces[start:end], leave=False, desc=stem):
            try:
                result = model.extract_trace(trace)
            except TraceTooLongError as error:
                skipped.append({"trace_id": trace.trace_id, "reason": str(error)})
                continue
            arrays.append(result.values)
            rows.extend(
                iter_step_metadata(
                    trace, partitions[trace.trace_id], token_count=result.token_count
                )
            )
        if not arrays:
            raise RuntimeError(f"No extractable traces in {stem}; raise model.max_length")
        activations = np.concatenate(arrays, axis=0)
        metadata = pd.DataFrame(rows)
        if len(activations) != len(metadata):
            raise RuntimeError(f"Activation/metadata mismatch while writing {stem}")
        _atomic_save_array(array_path, activations)
        _atomic_save_csv(metadata_path, metadata)
        shard_manifest = {
            "attempted": end - start,
            "extracted": len(arrays),
            "skipped_long": len(skipped),
            "skipped": skipped,
            "activation_rows": len(activations),
        }
        _atomic_write_text(manifest_path, json.dumps(shard_manifest, indent=2))
        totals["extracted"] += len(arrays)
        totals["skipped_long"] += len(skipped)
    return totals


def load_activation_shards(output_dir: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    shard_dir = Path(output_dir) / "activation_shards"
    array_paths = sorted(shard_dir.glob("shard_*.npy"))
    if not array_paths:
        raise FileNotFoundError(f"No activation shards found under {shard_dir}")
    arrays: list[np.ndarray] = []
    frames: list[pd.DataFrame] = []
    expected_tail = None
    for array_path in array_paths:
        metadata_path = array_path.with_suffix(".csv")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata paired with {array_path}")
        array = np.load(array_path, mmap_mode="r")
        frame = pd.read_csv(metadata_path)
        if len(array) != len(frame):
            raise ValueError(f"Row mismatch between {array_path} and {metadata_path}")
        if expected_tail is None:
            expected_tail = array.shape[1:]
        elif array.shape[1:] != expected_tail:
            raise ValueError("Activation shards have inconsistent layer/hidden dimensions")
        arrays.append(np.asarray(array))
        frames.append(frame)
    return np.concatenate(arrays), pd.concat(frames, ignore_index=True)


def fit_and_save_probes(config: ExperimentConfig) -> ProbeResults:
    activations, metadata = load_activation_shards(config.extraction.output_dir)
    results = fit_layer_probes(
        activations,
        metadata,
        config.probe,
        seed=config.seed,
        analysis_config=config.analysis,
    )
    output = config.extraction.output_dir / "probes"
    output.mkdir(parents=True, exist_ok=True)
    results.metrics.to_csv(output / "layer_metrics.csv", index=False)
    results.predictions.to_csv(output / "test_predictions.csv", index=False)
    results.controls.to_csv(output / "controls.csv", index=False)
    results.transfer.to_csv(output / "domain_transfer.csv", index=False)
    results.pca_curve.to_csv(output / "pca_subspace.csv", index=False)
    results.bootstrap.to_csv(output / "test_group_bootstrap.csv", index=False)
    results.bootstrap_summary.to_csv(output / "test_group_bootstrap_summary.csv", index=False)
    results.family_metrics.to_csv(output / "probe_family_metrics.csv", index=False)
    results.family_predictions.to_csv(output / "probe_family_predictions.csv", index=False)
    results.diagnostic_metrics.to_csv(output / "diagnostic_target_metrics.csv", index=False)
    results.calibration.to_csv(output / "calibration.csv", index=False)
    results.threshold_sensitivity.to_csv(output / "threshold_sensitivity.csv", index=False)
    results.trajectories.to_csv(output / "score_trajectories.csv", index=False)
    results.subgroups.to_csv(output / "subgroup_metrics.csv", index=False)
    results.comparisons.to_csv(output / "probe_family_comparisons.csv", index=False)
    np.savez(
        output / "directions.npz",
        directions=results.directions,
        projection_stds=results.projection_stds,
        thresholds=results.thresholds,
        c_values=results.c_values,
        selected_layer=results.selected_layer,
        selected_intervention_layer=results.selected_intervention_layer,
    )
    return results


def run_and_save_interventions(config: ExperimentConfig) -> pd.DataFrame:
    traces = load_traces(config.data.output_path)
    _, metadata = load_activation_shards(config.extraction.output_dir)
    direction_path = config.extraction.output_dir / "probes" / "directions.npz"
    if not direction_path.exists():
        raise FileNotFoundError("Run fit-probes before interventions")
    artifact = np.load(direction_path)
    layer = int(artifact["selected_intervention_layer"])
    direction = artifact["directions"][layer]
    projection_std = float(artifact["projection_stds"][layer])
    model = HuggingFaceMathModel(
        config.model.name,
        device=config.model.device,
        dtype=config.model.dtype,
        max_length=config.model.max_length,
    )
    results = run_interventions(
        model,
        traces,
        metadata,
        direction=direction,
        projection_std=projection_std,
        layer=layer,
        config=config.intervention,
        target=config.probe.target,
        seed=config.seed,
    )
    output = config.extraction.output_dir / "interventions"
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "individual.csv", index=False)
    summarize_interventions(results).to_csv(output / "summary.csv", index=False)
    causal_effect_statistics(
        results,
        confidence_level=config.analysis.confidence_level,
        bootstrap_samples=config.probe.bootstrap_samples,
        subgroup_min_traces=config.analysis.subgroup_min_traces,
        seed=config.seed,
    ).to_csv(output / "effect_statistics.csv", index=False)
    baseline = results[(results["direction_type"] == "learned") & (results["alpha"] == 0.0)]
    baseline_probabilities = 1 / (1 + np.exp(-baseline["verdict_score"].to_numpy()))
    baseline_metrics = binary_metrics(
        baseline[config.probe.target].to_numpy(),
        baseline_probabilities,
        threshold=0.5,
        calibration_bins=config.analysis.calibration_bins,
    )
    baseline_metrics["zero_threshold_accuracy"] = float(
        (
            (baseline["verdict_score"].to_numpy() >= 0) == baseline[config.probe.target].to_numpy()
        ).mean()
    )
    _atomic_write_text(output / "behavioral_verdict.json", json.dumps(baseline_metrics, indent=2))
    return results


def plot_artifacts(config: ExperimentConfig) -> list[Path]:
    import matplotlib.pyplot as plt

    probe_dir = config.extraction.output_dir / "probes"
    figure_dir = config.extraction.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    metrics = pd.read_csv(probe_dir / "layer_metrics.csv")
    test = metrics[metrics["split"] == "test"]
    figure, axis = plt.subplots(figsize=(6, 3.5))
    axis.plot(test["layer"], test["auroc"], marker="o", label="step AUROC")
    axis.plot(test["layer"], test["process_f1"], marker="s", label="first-error F1")
    axis.axhline(0.5, color="0.6", linestyle="--", linewidth=1)
    axis.set(xlabel="Hidden-state index", ylabel="Held-out score", ylim=(0, 1))
    axis.legend(frameon=False)
    outputs.append(_save_figure(figure, figure_dir / "layerwise_probe.pdf"))

    pca_path = probe_dir / "pca_subspace.csv"
    if pca_path.exists() and not pd.read_csv(pca_path).empty:
        pca = pd.read_csv(pca_path)
        figure, axis = plt.subplots(figsize=(5, 3.5))
        axis.plot(pca["dimensions"], pca["auroc"], marker="o")
        axis.set_xscale("log", base=2)
        axis.set(xlabel="Top-variance PCA dimensions", ylabel="Held-out AUROC", ylim=(0, 1))
        outputs.append(_save_figure(figure, figure_dir / "pca_subspace.pdf"))

    calibration_path = probe_dir / "calibration.csv"
    if calibration_path.exists() and not pd.read_csv(calibration_path).empty:
        calibration = pd.read_csv(calibration_path).dropna(subset=["mean_score", "positive_rate"])
        figure, axis = plt.subplots(figsize=(4, 4))
        axis.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
        axis.plot(calibration["mean_score"], calibration["positive_rate"], marker="o")
        axis.set(
            xlabel="Mean predicted score",
            ylabel="Observed invalid-step rate",
            xlim=(0, 1),
            ylim=(0, 1),
        )
        outputs.append(_save_figure(figure, figure_dir / "probe_calibration.pdf"))

    trajectories_path = probe_dir / "score_trajectories.csv"
    if trajectories_path.exists() and not pd.read_csv(trajectories_path).empty:
        trajectories = pd.read_csv(trajectories_path)
        relative = trajectories[trajectories["metric"] == "mean_score_at_relative_step"]
        figure, axis = plt.subplots(figsize=(5, 3.5))
        axis.plot(relative["relative_step"], relative["value"], marker="o")
        axis.axvline(0, color="0.6", linestyle="--", linewidth=1)
        axis.set(xlabel="Step relative to first error", ylabel="Mean probe score", ylim=(0, 1))
        outputs.append(_save_figure(figure, figure_dir / "error_aligned_trajectory.pdf"))

    intervention_path = config.extraction.output_dir / "interventions" / "summary.csv"
    if intervention_path.exists():
        intervention = pd.read_csv(intervention_path)
        learned = intervention[intervention["direction_type"] == "learned"]
        figure, axis = plt.subplots(figsize=(5, 3.5))
        axis.errorbar(
            learned["alpha"],
            learned["mean_delta"],
            yerr=learned["standard_error"],
            marker="o",
        )
        axis.axhline(0, color="0.6", linewidth=1)
        axis.set(
            xlabel=r"Intervention strength $\alpha$ (projection SD)",
            ylabel="Change in incorrect-vs-correct verdict",
        )
        outputs.append(_save_figure(figure, figure_dir / "causal_dose_response.pdf"))
    return outputs


def _check_extraction_identity(config: ExperimentConfig, output_dir: Path) -> None:
    dataset_hash = hashlib.sha256(config.data.output_path.read_bytes()).hexdigest()
    identity = {
        "dataset_sha256": dataset_hash,
        "model": config.model.name,
        "dtype": config.model.dtype,
        "max_length": config.model.max_length,
    }
    path = output_dir / "extraction_identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise RuntimeError(
            f"Existing shards under {output_dir} came from different data/model settings; "
            "choose a new extraction.output_dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(path, json.dumps(identity, indent=2))


def _atomic_save_array(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, array)
    temporary.replace(path)


def _atomic_save_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _save_figure(figure, path: Path) -> Path:
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)
    return path
