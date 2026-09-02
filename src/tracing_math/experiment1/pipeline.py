"""Experiment 1 orchestration, resumable stages, and artifact I/O."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from tracing_math.experiment1.config import ExperimentConfig
from tracing_math.experiment1.data import (
    assign_partitions,
    iter_step_metadata,
    load_huggingface_traces,
    load_traces,
    save_traces,
)
from tracing_math.experiment1.interventions import (
    causal_effect_statistics,
    run_interventions,
    score_intervention_baseline,
    summarize_interventions,
)
from tracing_math.experiment1.model import (
    VERDICT_QUESTION,
    VERDICT_READOUT_ID,
    HuggingFaceMathModel,
    TraceTooLongError,
)
from tracing_math.experiment1.probes import ProbeResults, binary_metrics, fit_layer_probes
from tracing_math.parallel import ordered_parallel_map


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
            _write_extraction_progress(output_dir, totals, start, end, len(traces))
            continue

        arrays: list[np.ndarray] = []
        rows: list[dict[str, object]] = []
        skipped: list[dict[str, object]] = []
        shard_traces = traces[start:end]
        batch_size = config.extraction.batch_size
        for batch_start in tqdm(
            range(0, len(shard_traces), batch_size), leave=False, desc=stem
        ):
            batch = shard_traces[batch_start : batch_start + batch_size]
            try:
                batch_results = model.extract_traces(batch)
            except TraceTooLongError:
                # Isolate over-length traces without throwing away the rest of the batch.
                batch_results = []
                retained = []
                for trace in batch:
                    try:
                        batch_results.append(model.extract_trace(trace))
                        retained.append(trace)
                    except TraceTooLongError as error:
                        skipped.append({"trace_id": trace.trace_id, "reason": str(error)})
                batch = retained
            for trace, result in zip(batch, batch_results, strict=True):
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
        _write_extraction_progress(output_dir, totals, start, end, len(traces))
    return totals


def load_activation_shards(
    output_dir: str | Path, *, workers: int = 1
) -> tuple[np.ndarray, pd.DataFrame]:
    shard_dir = Path(output_dir) / "activation_shards"
    array_paths = sorted(shard_dir.glob("shard_*.npy"))
    if not array_paths:
        raise FileNotFoundError(f"No activation shards found under {shard_dir}")

    def load_shard(array_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
        metadata_path = array_path.with_suffix(".csv")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata paired with {array_path}")
        array = np.load(array_path, mmap_mode="r")
        frame = pd.read_csv(metadata_path)
        if len(array) != len(frame):
            raise ValueError(f"Row mismatch between {array_path} and {metadata_path}")
        return np.asarray(array), frame

    loaded = ordered_parallel_map(load_shard, array_paths, workers=workers)
    arrays = [array for array, _ in loaded]
    frames = [frame for _, frame in loaded]
    expected_tail = None
    for array in arrays:
        if expected_tail is None:
            expected_tail = array.shape[1:]
        elif array.shape[1:] != expected_tail:
            raise ValueError("Activation shards have inconsistent layer/hidden dimensions")
    return np.concatenate(arrays), pd.concat(frames, ignore_index=True)


def load_activation_metadata(output_dir: str | Path, *, workers: int = 1) -> pd.DataFrame:
    """Load and validate shard metadata without reading activation tensor contents."""
    shard_dir = Path(output_dir) / "activation_shards"
    array_paths = sorted(shard_dir.glob("shard_*.npy"))
    if not array_paths:
        raise FileNotFoundError(f"No activation shards found under {shard_dir}")

    def load_metadata(array_path: Path) -> pd.DataFrame:
        metadata_path = array_path.with_suffix(".csv")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata paired with {array_path}")
        array = np.load(array_path, mmap_mode="r")
        frame = pd.read_csv(metadata_path)
        if len(array) != len(frame):
            raise ValueError(f"Row mismatch between {array_path} and {metadata_path}")
        return frame

    frames = ordered_parallel_map(load_metadata, array_paths, workers=workers)
    return pd.concat(frames, ignore_index=True)


def fit_and_save_probes(config: ExperimentConfig) -> ProbeResults:
    activations, metadata = load_activation_shards(
        config.extraction.output_dir,
        workers=config.analysis.workers,
    )
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
    selected_predictions = results.predictions[
        results.predictions["layer"] == results.selected_layer
    ]
    selected_predictions.to_csv(output / "test_predictions.csv", index=False)
    results.fit_predictions.to_csv(output / "fit_predictions.csv", index=False)
    results.controls.to_csv(output / "controls.csv", index=False)
    results.control_predictions.to_csv(output / "control_predictions.csv", index=False)
    results.transfer.to_csv(output / "domain_transfer.csv", index=False)
    results.bootstrap.to_csv(output / "test_group_bootstrap.csv", index=False)
    results.bootstrap_summary.to_csv(output / "test_group_bootstrap_summary.csv", index=False)
    if config.probe.pca_dimensions:
        results.pca_curve.to_csv(output / "pca_subspace.csv", index=False)
    if len(config.probe.families) > 1:
        results.family_metrics.to_csv(output / "probe_family_metrics.csv", index=False)
        results.family_predictions.to_csv(output / "probe_family_predictions.csv", index=False)
        results.comparisons.to_csv(output / "probe_family_comparisons.csv", index=False)
    if config.probe.diagnostic_targets:
        results.diagnostic_metrics.to_csv(output / "diagnostic_target_metrics.csv", index=False)
    if config.analysis.exploratory_bootstrap_samples > 0:
        results.calibration.to_csv(output / "calibration.csv", index=False)
        results.threshold_sensitivity.to_csv(output / "threshold_sensitivity.csv", index=False)
        results.trajectories.to_csv(output / "score_trajectories.csv", index=False)
        results.subgroups.to_csv(output / "subgroup_metrics.csv", index=False)
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
    metadata = load_activation_metadata(
        config.extraction.output_dir,
        workers=config.analysis.workers,
    )
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
    output = config.extraction.output_dir / "interventions"
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / f"individual.{VERDICT_READOUT_ID}.checkpoint.csv"

    baseline = score_intervention_baseline(
        model,
        traces,
        metadata,
        config=config.intervention,
        target=config.probe.target,
        seed=config.seed,
    )
    baseline_probabilities = (baseline["verdict_score"].to_numpy() + 1) / 2
    baseline_metrics = binary_metrics(
        baseline[config.probe.target].to_numpy(),
        baseline_probabilities,
        threshold=0.5,
        calibration_bins=config.analysis.calibration_bins,
    )
    behaviorally_valid = bool(
        baseline_metrics["auroc"] > 0.5 and baseline_metrics["specificity"] > 0
    )
    baseline_metrics.update(
        {
            "readout_id": VERDICT_READOUT_ID,
            "question": VERDICT_QUESTION,
            "positive_answer": config.intervention.incorrect_answer.strip(),
            "negative_answer": config.intervention.correct_answer.strip(),
            "score": "conditional_P(Yes)-conditional_P(No)",
            "zero_threshold_accuracy": float(
                (
                    (baseline["verdict_score"].to_numpy() >= 0)
                    == baseline[config.probe.target].to_numpy()
                ).mean()
            ),
            "n_boundaries": len(baseline),
            "specificity_gate_passed": bool(baseline_metrics["specificity"] > 0),
            "behavioral_validity_gate_passed": behaviorally_valid,
        }
    )
    _atomic_write_text(output / "behavioral_verdict.json", json.dumps(baseline_metrics, indent=2))
    if not behaviorally_valid:
        baseline.to_csv(output / "individual.csv", index=False)
        summarize_interventions(baseline).to_csv(output / "summary.csv", index=False)
        pd.DataFrame(
            columns=["statistic", "scope", "value", "status", "estimate", "n_traces"]
        ).to_csv(output / "effect_statistics.csv", index=False)
        _atomic_write_text(
            output / "progress.json",
            json.dumps(
                {
                    "status": "stopped_after_baseline",
                    "reason": "verdict readout failed AUROC or specificity gate",
                    "readout_id": VERDICT_READOUT_ID,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            ),
        )
        return baseline

    existing = pd.read_csv(checkpoint_path) if checkpoint_path.exists() else baseline

    def checkpoint(frame: pd.DataFrame) -> None:
        _atomic_save_csv(checkpoint_path, frame)
        progress = {
            "status": "running",
            "rows_completed": len(frame),
            "learned_alphas_completed": sorted(
                frame.loc[frame["direction_type"] == "learned", "alpha"].unique().tolist()
            ),
            "random_direction_alpha_groups_completed": int(
                frame.loc[frame["direction_type"] == "random_orthogonal"]
                .groupby(["direction_index", "alpha"])
                .ngroups
            ),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_write_text(output / "progress.json", json.dumps(progress, indent=2))

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
        existing_results=existing,
        checkpoint_callback=checkpoint,
    )
    results.to_csv(output / "individual.csv", index=False)
    summarize_interventions(results).to_csv(output / "summary.csv", index=False)
    causal_effect_statistics(
        results,
        confidence_level=config.analysis.confidence_level,
        bootstrap_samples=config.probe.bootstrap_samples,
        subgroup_min_traces=config.analysis.subgroup_min_traces,
        seed=config.seed,
    ).to_csv(output / "effect_statistics.csv", index=False)
    _atomic_write_text(
        output / "progress.json",
        json.dumps(
            {
                "status": "complete",
                "rows_completed": len(results),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        ),
    )
    return results


def _write_extraction_progress(
    output_dir: Path, totals: dict[str, int], start: int, end: int, trace_count: int
) -> None:
    completed = totals["cached"] + totals["extracted"] + totals["skipped_long"]
    progress = {
        "status": "complete" if end >= trace_count else "running",
        "last_shard": {"start": start, "end_exclusive": end},
        "traces_total": trace_count,
        "traces_completed": completed,
        "fraction_complete": completed / trace_count if trace_count else 1.0,
        **totals,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_write_text(output_dir / "extraction_progress.json", json.dumps(progress, indent=2))


def plot_artifacts(config: ExperimentConfig) -> list[Path]:
    import matplotlib.pyplot as plt

    probe_dir = config.extraction.output_dir / "probes"
    figure_dir = config.extraction.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    metrics = pd.read_csv(probe_dir / "layer_metrics.csv")
    predictions = pd.read_csv(probe_dir / "test_predictions.csv")
    controls = pd.read_csv(probe_dir / "controls.csv")
    transfer = pd.read_csv(probe_dir / "domain_transfer.csv")
    bootstrap = pd.read_csv(probe_dir / "test_group_bootstrap_summary.csv")
    directions = np.load(probe_dir / "directions.npz")
    selected_layer = int(directions["selected_layer"])
    selected_threshold = float(directions["thresholds"][selected_layer])

    # Figure 1: method schematic and one deterministic held-out trajectory.
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
    method_axis, trajectory_axis = axes
    method_axis.axis("off")
    stages = [
        (0.02, 0.58, "Reasoning steps\nwith end markers"),
        (0.35, 0.58, "Boundary residual\nstates at every layer"),
        (0.68, 0.58, "L2 invalidity\nprobe"),
        (0.35, 0.12, "Held-out\nlocalization"),
        (0.68, 0.12, "Post-hoc causal pilot\n(assay-gated)"),
    ]
    for x, y, label in stages:
        method_axis.text(
            x,
            y,
            label,
            transform=method_axis.transAxes,
            ha="left",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "0.25"},
        )
    for start, end in (
        ((0.26, 0.58), (0.34, 0.58)),
        ((0.59, 0.58), (0.67, 0.58)),
        ((0.51, 0.51), (0.45, 0.25)),
        ((0.76, 0.51), (0.76, 0.25)),
    ):
        method_axis.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->", "color": "0.3", "linewidth": 1.2},
        )
    method_axis.set_title("Hidden-state probe audit", fontsize=10)

    selected_predictions = predictions[predictions["layer"] == selected_layer].copy()
    erroneous = selected_predictions[selected_predictions["first_error"] >= 0]
    candidate_ids = (
        erroneous[erroneous["first_error"] > 0]["trace_id"].drop_duplicates().sort_values()
    )
    if candidate_ids.empty:
        candidate_ids = erroneous["trace_id"].drop_duplicates().sort_values()
    if candidate_ids.empty:
        raise ValueError("No held-out erroneous trace is available for the trajectory figure")
    example = erroneous[erroneous["trace_id"] == candidate_ids.iloc[0]].sort_values("step_index")
    first_error = int(example["first_error"].iloc[0])
    trajectory_axis.plot(example["step_index"], example["score"], marker="o", color="C0")
    trajectory_axis.axhline(selected_threshold, color="C1", linestyle="--", label="threshold")
    trajectory_axis.axvline(first_error, color="C3", linestyle=":", label="human first error")
    trajectory_axis.set(
        xlabel="Reasoning step",
        ylabel="Invalid-so-far score",
        ylim=(0, 1),
        title="Held-out example trajectory",
    )
    trajectory_axis.set_xticks(example["step_index"].to_numpy())
    trajectory_axis.legend(frameon=False, fontsize=8)
    outputs.append(_save_figure(figure, figure_dir / "method_and_trajectory.pdf"))

    # Figure 2: layer curve, selected-layer bootstrap intervals, and all required controls.
    test = metrics[metrics["split"] == "test"].sort_values("layer")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.15, 1]})
    curve_axis, table_axis = axes
    curve_axis.plot(test["layer"], test["auroc"], marker="o", label="step AUROC")
    curve_axis.plot(test["layer"], test["process_f1"], marker="s", label="first-error F1")
    curve_axis.axhline(0.5, color="0.6", linestyle="--", linewidth=1)
    curve_axis.axvline(selected_layer, color="0.25", linestyle=":", linewidth=1)
    for offset, metric, color in ((-0.18, "auroc", "C0"), (0.18, "process_f1", "C1")):
        interval = bootstrap[bootstrap["metric"] == metric]
        if not interval.empty:
            row = interval.iloc[0]
            curve_axis.errorbar(
                selected_layer + offset,
                row["estimate"],
                yerr=[[row["estimate"] - row["ci_low"]], [row["ci_high"] - row["estimate"]]],
                fmt="o",
                capsize=3,
                color=color,
            )
    curve_axis.set(
        xlabel="Hidden-state index",
        ylabel="Held-out score",
        ylim=(0, 1),
        title="Layer-wise decoding and localization",
    )
    curve_axis.legend(frameon=False, fontsize=8)

    selected_row = test[test["layer"] == selected_layer].iloc[0]
    embedding_row = test[test["layer"] == 0].iloc[0]
    within_error_row = metrics[
        (metrics["split"] == "test_error_traces") & (metrics["layer"] == selected_layer)
    ].iloc[0]
    table_rows = [
        [
            "Selected hidden state",
            f"{selected_row['auroc']:.3f}",
            f"{selected_row['process_f1']:.3f}",
        ],
        [
            "Embedding-index probe",
            f"{embedding_row['auroc']:.3f}",
            f"{embedding_row['process_f1']:.3f}",
        ],
        ["Within-error traces", f"{within_error_row['auroc']:.3f}", "N/A"],
    ]
    table_rows.extend(
        [row["control"], f"{row['auroc']:.3f}", f"{row['process_f1']:.3f}"]
        for _, row in controls.iterrows()
    )
    table_axis.axis("off")
    table = table_axis.table(
        cellText=table_rows,
        colLabels=["Held-out comparison", "AUROC", "First-error F1"],
        colWidths=[0.58, 0.2, 0.25],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    table_axis.set_title("Required shortcut controls", fontsize=10, pad=12)
    outputs.append(_save_figure(figure, figure_dir / "predictive_results.pdf"))

    # Figure 3: domain transfer and causal dose response with random-direction controls.
    intervention_path = config.extraction.output_dir / "interventions" / "summary.csv"
    if not intervention_path.exists():
        raise FileNotFoundError("Run interventions before generating the paper figures")
    intervention = pd.read_csv(intervention_path)
    learned = intervention[intervention["direction_type"] == "learned"].sort_values("alpha")
    random = intervention[
        intervention["direction_type"].isin(["random_orthogonal", "random"])
    ]
    figure, axes = plt.subplots(
        1, 3, figsize=(13.5, 4.1), gridspec_kw={"width_ratios": [1, 1, 1.25]}
    )
    for axis, metric, title in zip(
        axes[:2],
        ("auroc", "process_f1"),
        ("Cross-domain AUROC", "Cross-domain first-error F1"),
        strict=True,
    ):
        matrix = transfer.pivot(index="train_source", columns="test_source", values=metric)
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
        axis.set_yticks(range(len(matrix.index)), matrix.index)
        axis.set(xlabel="Test source", ylabel="Train source", title=title)
        for row_index in range(len(matrix.index)):
            for column_index in range(len(matrix.columns)):
                value = float(matrix.iloc[row_index, column_index])
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.65 else "black",
                    fontsize=7,
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    causal_axis = axes[2]

    if not random.empty:
        causal_axis.scatter(
            random["alpha"],
            random["mean_delta"],
            s=14,
            color="0.65",
            alpha=0.6,
            label="random orthogonal directions",
        )
    causal_axis.errorbar(
        learned["alpha"],
        learned["mean_delta"],
        yerr=learned["standard_error"],
        marker="o",
        capsize=3,
        color="C3",
        label="learned direction",
    )
    causal_axis.axhline(0, color="0.4", linewidth=1)
    causal_axis.set(
        xlabel=r"Intervention strength $\alpha$ (projection SD)",
        ylabel="Change in verdict score",
        title="Held-out causal intervention",
    )
    causal_axis.legend(frameon=False, fontsize=8)
    outputs.append(_save_figure(figure, figure_dir / "transfer_and_causal.pdf"))
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
