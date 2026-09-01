"""Experiment 1 linear probes, controls, and change-point metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from causal_circuits.experiment1.config import AnalysisConfig, ProbeConfig, ProbeFamilyConfig
from causal_circuits.localization import (
    localization_metrics,
    localization_outcomes,
    quantile_interval,
)

PREDICTION_COLUMNS = [
    "trace_id",
    "source",
    "generator",
    "step_index",
    "step_fraction",
    "first_error",
    "has_error_trace",
    "n_steps",
    "token_count",
    "final_answer_correct",
    "invalid_so_far",
    "error_onset",
]


@dataclass
class ProbeResults:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    controls: pd.DataFrame
    transfer: pd.DataFrame
    pca_curve: pd.DataFrame
    bootstrap: pd.DataFrame
    directions: np.ndarray
    projection_stds: np.ndarray
    thresholds: np.ndarray
    c_values: np.ndarray
    selected_layer: int
    selected_intervention_layer: int
    family_metrics: pd.DataFrame
    family_predictions: pd.DataFrame
    diagnostic_metrics: pd.DataFrame
    calibration: pd.DataFrame
    threshold_sensitivity: pd.DataFrame
    trajectories: pd.DataFrame
    subgroups: pd.DataFrame
    comparisons: pd.DataFrame
    bootstrap_summary: pd.DataFrame


def binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float = 0.5,
    calibration_bins: int = 10,
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.shape != scores.shape or labels.size == 0:
        raise ValueError("labels and scores must be non-empty arrays with equal shape")
    probabilities = np.clip(scores, 1e-7, 1 - 1e-7)
    predicted = scores >= threshold
    positive = labels == 1
    negative = ~positive
    true_positive = int(np.sum(predicted & positive))
    false_positive = int(np.sum(predicted & negative))
    true_negative = int(np.sum(~predicted & negative))
    false_negative = int(np.sum(~predicted & positive))
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    specificity_denominator = true_negative + false_positive
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else float("nan")
    specificity = (
        true_negative / specificity_denominator if specificity_denominator else float("nan")
    )
    f1_denominator = precision + recall
    f1 = 2 * precision * recall / f1_denominator if f1_denominator > 0 else 0.0
    balanced_accuracy = (
        (recall + specificity) / 2
        if np.isfinite(recall) and np.isfinite(specificity)
        else float("nan")
    )
    ece = 0.0
    edges = np.linspace(0.0, 1.0, calibration_bins + 1)
    for bin_index in range(calibration_bins):
        if bin_index == calibration_bins - 1:
            member = (scores >= edges[bin_index]) & (scores <= edges[bin_index + 1])
        else:
            member = (scores >= edges[bin_index]) & (scores < edges[bin_index + 1])
        if member.any():
            ece += float(member.mean()) * abs(float(scores[member].mean() - labels[member].mean()))
    discrimination = {"auroc": float("nan"), "average_precision": float("nan")}
    if len(np.unique(labels)) >= 2:
        discrimination = {
            "auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
        }
    return {
        **discrimination,
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "step_f1": float(f1),
        "brier_score": float(np.mean((scores - labels) ** 2)),
        "log_loss": float(
            -np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))
        ),
        "expected_calibration_error": float(ece),
    }


def change_point_metrics(
    metadata: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    *,
    tolerances: tuple[int, ...] = (0, 1, 2),
) -> dict[str, float]:
    """ProcessBench exact first-error score induced by the first threshold crossing."""
    expected, predicted = localization_outcomes(metadata, scores, threshold)
    return localization_metrics(expected, predicted, tolerances)


def choose_threshold(
    metadata: pd.DataFrame,
    scores: np.ndarray,
    *,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_points: int = 181,
) -> float:
    candidates = np.linspace(threshold_min, threshold_max, threshold_points)
    frame = metadata[["trace_id", "step_index", "first_error"]].copy()
    frame["score"] = np.asarray(scores, dtype=float)
    expected: list[int] = []
    predictions: list[np.ndarray] = []
    for _, trace_rows in frame.groupby("trace_id", sort=False):
        ordered = trace_rows.sort_values("step_index")
        expected.append(int(ordered["first_error"].iloc[0]))
        crossings = ordered["score"].to_numpy()[:, None] >= candidates[None, :]
        any_crossing = crossings.any(axis=0)
        first_crossing = np.argmax(crossings, axis=0)
        predictions.append(
            np.where(any_crossing, ordered["step_index"].to_numpy()[first_crossing], -1)
        )
    expected_array = np.asarray(expected)
    prediction_matrix = np.asarray(predictions)
    error_mask = expected_array >= 0
    correct_mask = ~error_mask
    error_accuracy = np.mean(
        prediction_matrix[error_mask] == expected_array[error_mask, None], axis=0
    )
    correct_accuracy = np.mean(prediction_matrix[correct_mask] == -1, axis=0)
    denominator = error_accuracy + correct_accuracy
    process_f1 = np.divide(
        2 * error_accuracy * correct_accuracy,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    ranked = [
        (float(score), -abs(float(threshold) - 0.5), float(threshold))
        for threshold, score in zip(candidates, process_f1, strict=True)
    ]
    return max(ranked)[2]


def fit_layer_probes(
    activations: np.ndarray,
    metadata: pd.DataFrame,
    config: ProbeConfig,
    *,
    seed: int,
    analysis_config: AnalysisConfig | None = None,
) -> ProbeResults:
    """Fit one regularized logistic probe per hidden-state index."""
    analysis_config = analysis_config or AnalysisConfig()
    _validate_inputs(activations, metadata, config.target)
    metadata = metadata.copy()
    if "n_steps" not in metadata:
        metadata["n_steps"] = metadata.groupby("trace_id")["step_index"].transform("max") + 1
    if "token_count" not in metadata:
        metadata["token_count"] = metadata["n_steps"]
    if "final_answer_correct" not in metadata:
        metadata["final_answer_correct"] = 0
    if "error_onset" not in metadata:
        metadata["error_onset"] = (
            (metadata["first_error"] >= 0) & (metadata["step_index"] == metadata["first_error"])
        ).astype(int)
    labels = metadata[config.target].to_numpy(dtype=int)
    masks = _partition_masks(metadata)
    if any(labels[mask].min() == labels[mask].max() for mask in masks.values()):
        raise ValueError("Every data partition must contain both target classes")

    n_layers, hidden_size = activations.shape[1:]
    directions = np.zeros((n_layers, hidden_size), dtype=np.float32)
    projection_stds = np.zeros(n_layers, dtype=np.float32)
    thresholds = np.zeros(n_layers, dtype=np.float32)
    selected_cs = np.zeros(n_layers, dtype=np.float32)
    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    validation_prediction_rows: list[pd.DataFrame] = []
    primary_family = next(
        family for family in config.families if family.name == config.primary_family
    )

    for layer in tqdm(range(n_layers), desc="primary probe layers"):
        x = np.asarray(activations[:, layer, :], dtype=np.float32)
        best_c, _, validation_scores = _select_hyperparameters(
            x[masks["train"]],
            labels[masks["train"]],
            x[masks["validation"]],
            labels[masks["validation"]],
            primary_family,
            config.c_values,
            config.max_iter,
            seed,
        )
        threshold = choose_threshold(
            metadata.loc[masks["validation"]],
            validation_scores,
            threshold_min=analysis_config.threshold_min,
            threshold_max=analysis_config.threshold_max,
            threshold_points=analysis_config.threshold_points,
        )
        selected_cs[layer] = best_c
        thresholds[layer] = threshold
        metrics_rows.append(
            _metric_row(
                layer,
                "validation",
                metadata.loc[masks["validation"]],
                labels[masks["validation"]],
                validation_scores,
                threshold,
                best_c,
                analysis_config,
            )
        )

        fit_mask = masks["train"] | masks["validation"]
        scaler, classifier = _fit_logistic(
            x[fit_mask], labels[fit_mask], best_c, config.max_iter, seed
        )
        test_scores = classifier.predict_proba(scaler.transform(x[masks["test"]]))[:, 1]
        direction = classifier.coef_[0] / scaler.scale_
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise RuntimeError(f"Layer {layer} produced a zero probe direction")
        directions[layer] = direction / norm
        projection_stds[layer] = max(float(np.std(x[fit_mask] @ directions[layer], ddof=1)), 1e-8)
        metrics_rows.append(
            _metric_row(
                layer,
                "test",
                metadata.loc[masks["test"]],
                labels[masks["test"]],
                test_scores,
                threshold,
                best_c,
                analysis_config,
            )
        )
        erroneous_test = masks["test"] & metadata["has_error_trace"].eq(1).to_numpy()
        erroneous_scores = classifier.predict_proba(scaler.transform(x[erroneous_test]))[:, 1]
        metrics_rows.append(
            _metric_row(
                layer,
                "test_error_traces",
                metadata.loc[erroneous_test],
                labels[erroneous_test],
                erroneous_scores,
                threshold,
                best_c,
                analysis_config,
            )
        )
        layer_predictions = metadata.loc[
            masks["test"],
            PREDICTION_COLUMNS,
        ].copy()
        layer_predictions["layer"] = layer
        layer_predictions["label"] = labels[masks["test"]]
        layer_predictions["score"] = test_scores
        prediction_rows.append(layer_predictions)
        if analysis_config.exploratory_bootstrap_samples > 0:
            validation_predictions = metadata.loc[
                masks["validation"],
                ["trace_id", "step_index", "first_error", "source", "generator"],
            ].copy()
            validation_predictions["layer"] = layer
            validation_predictions["label"] = labels[masks["validation"]]
            validation_predictions["score"] = validation_scores
            validation_prediction_rows.append(validation_predictions)

    metrics = pd.DataFrame(metrics_rows)
    validation = metrics[metrics["split"] == "validation"].sort_values(
        ["process_f1", "auroc", "layer"], ascending=[False, False, True]
    )
    selected_layer = int(validation.iloc[0]["layer"])
    eligible = validation[validation["layer"] < n_layers - 1]
    selected_intervention_layer = int(eligible.iloc[0]["layer"])

    controls = evaluate_controls(
        np.asarray(activations[:, selected_layer, :], dtype=np.float32),
        metadata,
        labels,
        max_iter=config.max_iter,
        seed=seed,
        analysis_config=analysis_config,
    )
    transfer = domain_transfer(
        np.asarray(activations[:, selected_layer, :], dtype=np.float32),
        metadata,
        labels,
        c_value=float(selected_cs[selected_layer]),
        max_iter=config.max_iter,
        seed=seed,
        analysis_config=analysis_config,
    )
    pca_curve = pca_subspace_curve(
        np.asarray(activations[:, selected_layer, :], dtype=np.float32),
        metadata,
        labels,
        config.pca_dimensions,
        c_value=float(selected_cs[selected_layer]),
        max_iter=config.max_iter,
        seed=seed,
        analysis_config=analysis_config,
    )
    selected_predictions = pd.concat(prediction_rows, ignore_index=True)
    selected_predictions = selected_predictions[
        selected_predictions["layer"] == selected_layer
    ].copy()
    bootstrap = group_bootstrap_metrics(
        selected_predictions,
        selected_predictions["label"].to_numpy(),
        selected_predictions["score"].to_numpy(),
        float(thresholds[selected_layer]),
        samples=config.bootstrap_samples,
        seed=seed,
        analysis_config=analysis_config,
    )
    if len(config.families) > 1 or config.diagnostic_targets:
        family_metrics, family_predictions, diagnostic_metrics = _fit_exploratory_probes(
            activations,
            metadata,
            config,
            analysis_config,
            primary_metrics=metrics,
            primary_predictions=pd.concat(prediction_rows, ignore_index=True),
            seed=seed,
        )
    else:
        family_metrics = pd.DataFrame()
        family_predictions = pd.DataFrame()
        diagnostic_metrics = pd.DataFrame()

    if analysis_config.exploratory_bootstrap_samples > 0:
        validation_predictions = pd.concat(validation_prediction_rows, ignore_index=True)
        selected_validation = validation_predictions[
            validation_predictions["layer"] == selected_layer
        ].copy()
        threshold_sensitivity = threshold_sensitivity_curve(
            selected_validation,
            selected_predictions,
            analysis_config,
            selected_threshold=float(thresholds[selected_layer]),
        )
        calibration = calibration_curve_table(
            selected_predictions["label"].to_numpy(),
            selected_predictions["score"].to_numpy(),
            bins=analysis_config.calibration_bins,
        )
        trajectories = trajectory_metrics(selected_predictions)
        subgroups = subgroup_metrics(
            selected_predictions,
            metadata,
            threshold=float(thresholds[selected_layer]),
            analysis_config=analysis_config,
            bootstrap_samples=analysis_config.exploratory_bootstrap_samples,
            seed=seed,
        )
    else:
        threshold_sensitivity = pd.DataFrame()
        calibration = pd.DataFrame()
        trajectories = pd.DataFrame()
        subgroups = pd.DataFrame()

    if len(config.families) > 1:
        comparisons = compare_probe_families(
            family_metrics,
            family_predictions,
            primary_family=config.primary_family,
            threshold_by_layer=thresholds,
            analysis_config=analysis_config,
            bootstrap_samples=analysis_config.exploratory_bootstrap_samples,
            seed=seed,
        )
    else:
        comparisons = pd.DataFrame()
    bootstrap_summary = summarize_bootstrap(bootstrap, analysis_config.confidence_level)
    return ProbeResults(
        metrics=metrics,
        predictions=pd.concat(prediction_rows, ignore_index=True),
        controls=controls,
        transfer=transfer,
        pca_curve=pca_curve,
        bootstrap=bootstrap,
        directions=directions,
        projection_stds=projection_stds,
        thresholds=thresholds,
        c_values=selected_cs,
        selected_layer=selected_layer,
        selected_intervention_layer=selected_intervention_layer,
        family_metrics=family_metrics,
        family_predictions=family_predictions,
        diagnostic_metrics=diagnostic_metrics,
        calibration=calibration,
        threshold_sensitivity=threshold_sensitivity,
        trajectories=trajectories,
        subgroups=subgroups,
        comparisons=comparisons,
        bootstrap_summary=bootstrap_summary,
    )


def _fit_exploratory_probes(
    activations: np.ndarray,
    metadata: pd.DataFrame,
    config: ProbeConfig,
    analysis_config: AnalysisConfig,
    *,
    primary_metrics: pd.DataFrame,
    primary_predictions: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    family_frames = []
    prediction_frames = []
    primary_frame = primary_metrics.copy()
    primary_frame["family"] = config.primary_family
    primary_frame["penalty"] = "l2"
    primary_frame["l1_ratio"] = 0.0
    primary_frame["target"] = config.target
    family_frames.append(primary_frame)
    primary_prediction_frame = primary_predictions.copy()
    primary_prediction_frame["family"] = config.primary_family
    primary_prediction_frame["target"] = config.target
    prediction_frames.append(primary_prediction_frame)

    for family in config.families:
        if family.name == config.primary_family:
            continue
        metrics, predictions = _fit_family_layers(
            activations,
            metadata,
            target=config.target,
            family=family,
            config=config,
            analysis_config=analysis_config,
            seed=seed,
        )
        family_frames.append(metrics)
        prediction_frames.append(predictions)

    diagnostic_frames = []
    primary_family = next(
        family for family in config.families if family.name == config.primary_family
    )
    for target in config.diagnostic_targets:
        metrics, _ = _fit_family_layers(
            activations,
            metadata,
            target=target,
            family=primary_family,
            config=config,
            analysis_config=analysis_config,
            seed=seed,
        )
        diagnostic_frames.append(metrics)

    family_metrics = pd.concat(family_frames, ignore_index=True)
    family_predictions = pd.concat(prediction_frames, ignore_index=True)
    selected = (
        family_metrics[family_metrics["split"] == "validation"]
        .sort_values(
            ["family", "process_f1", "auroc", "layer"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates("family")[["family", "layer"]]
    )
    selected_lookup = dict(zip(selected["family"], selected["layer"], strict=True))
    family_metrics["is_family_selected"] = (
        family_metrics["layer"] == family_metrics["family"].map(selected_lookup)
    ).astype(int)
    family_predictions = family_predictions[
        family_predictions["layer"] == family_predictions["family"].map(selected_lookup)
    ].copy()
    family_predictions["is_family_selected"] = 1
    diagnostics = (
        pd.concat(diagnostic_frames, ignore_index=True) if diagnostic_frames else pd.DataFrame()
    )
    return family_metrics, family_predictions, diagnostics


def _fit_family_layers(
    activations: np.ndarray,
    metadata: pd.DataFrame,
    *,
    target: str,
    family: ProbeFamilyConfig,
    config: ProbeConfig,
    analysis_config: AnalysisConfig,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = metadata[target].to_numpy(dtype=int)
    masks = _partition_masks(metadata)
    if any(len(np.unique(labels[mask])) < 2 for mask in masks.values()):
        return pd.DataFrame(), pd.DataFrame()
    metric_rows = []
    for layer in tqdm(
        range(activations.shape[1]),
        desc=f"{family.name} / {target} layers",
    ):
        hidden = np.asarray(activations[:, layer, :], dtype=np.float32)
        best_c, best_ratio, validation_scores = _select_hyperparameters(
            hidden[masks["train"]],
            labels[masks["train"]],
            hidden[masks["validation"]],
            labels[masks["validation"]],
            family,
            config.c_values,
            config.max_iter,
            seed,
        )
        threshold = choose_threshold(
            metadata.loc[masks["validation"]],
            validation_scores,
            threshold_min=analysis_config.threshold_min,
            threshold_max=analysis_config.threshold_max,
            threshold_points=analysis_config.threshold_points,
        )
        reported_ratio = (
            best_ratio if best_ratio is not None else (1.0 if family.penalty == "l1" else 0.0)
        )
        metric_rows.append(
            {
                **_metric_row(
                    layer,
                    "validation",
                    metadata.loc[masks["validation"]],
                    labels[masks["validation"]],
                    validation_scores,
                    threshold,
                    best_c,
                    analysis_config,
                    l1_ratio=reported_ratio,
                ),
                "family": family.name,
                "penalty": family.penalty,
                "target": target,
            }
        )
        fit = masks["train"] | masks["validation"]
        scaler, classifier = _fit_logistic(
            hidden[fit],
            labels[fit],
            best_c,
            config.max_iter,
            seed,
            penalty=family.penalty,
            l1_ratio=best_ratio,
        )
        test_scores = classifier.predict_proba(scaler.transform(hidden[masks["test"]]))[:, 1]
        metric_rows.append(
            {
                **_metric_row(
                    layer,
                    "test",
                    metadata.loc[masks["test"]],
                    labels[masks["test"]],
                    test_scores,
                    threshold,
                    best_c,
                    analysis_config,
                    l1_ratio=reported_ratio,
                ),
                "family": family.name,
                "penalty": family.penalty,
                "target": target,
            }
        )
        erroneous = masks["test"] & metadata["has_error_trace"].eq(1).to_numpy()
        if len(np.unique(labels[erroneous])) >= 2:
            erroneous_scores = classifier.predict_proba(scaler.transform(hidden[erroneous]))[:, 1]
            metric_rows.append(
                {
                    **_metric_row(
                        layer,
                        "test_error_traces",
                        metadata.loc[erroneous],
                        labels[erroneous],
                        erroneous_scores,
                        threshold,
                        best_c,
                        analysis_config,
                        l1_ratio=reported_ratio,
                    ),
                    "family": family.name,
                    "penalty": family.penalty,
                    "target": target,
                }
            )
    metrics = pd.DataFrame(metric_rows)
    selected = (
        metrics[metrics["split"] == "validation"]
        .sort_values(["process_f1", "auroc", "layer"], ascending=[False, False, True])
        .iloc[0]
    )
    selected_layer = int(selected["layer"])
    metrics["is_family_selected"] = (metrics["layer"] == selected_layer).astype(int)
    hidden = np.asarray(activations[:, selected_layer, :], dtype=np.float32)
    fit = masks["train"] | masks["validation"]
    best_ratio = float(selected["l1_ratio"]) if family.penalty == "elasticnet" else None
    scaler, classifier = _fit_logistic(
        hidden[fit],
        labels[fit],
        float(selected["c_value"]),
        config.max_iter,
        seed,
        penalty=family.penalty,
        l1_ratio=best_ratio,
    )
    predictions = metadata.loc[
        masks["test"],
        PREDICTION_COLUMNS,
    ].copy()
    predictions["layer"] = selected_layer
    predictions["label"] = labels[masks["test"]]
    predictions["score"] = classifier.predict_proba(scaler.transform(hidden[masks["test"]]))[:, 1]
    predictions["threshold"] = float(selected["threshold"])
    predictions["family"] = family.name
    predictions["target"] = target
    return metrics, predictions


def _select_hyperparameters(
    x_train,
    y_train,
    x_validation,
    y_validation,
    family,
    c_values,
    max_iter,
    seed,
):
    ratios = family.l1_ratios if family.penalty == "elasticnet" else (None,)
    candidates = []
    for c_value in c_values:
        for ratio in ratios:
            scaler, classifier = _fit_logistic(
                x_train,
                y_train,
                c_value,
                max_iter,
                seed,
                penalty=family.penalty,
                l1_ratio=ratio,
            )
            scores = classifier.predict_proba(scaler.transform(x_validation))[:, 1]
            auroc = binary_metrics(y_validation, scores)["auroc"]
            ratio_tie_break = -abs((ratio if ratio is not None else 0.5) - 0.5)
            candidates.append(
                (auroc, -abs(np.log10(c_value)), ratio_tie_break, c_value, ratio, scores)
            )
    _, _, _, best_c, best_ratio, scores = max(candidates, key=lambda item: item[:3])
    return float(best_c), best_ratio, scores


def group_bootstrap_metrics(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    samples: int,
    seed: int,
    analysis_config: AnalysisConfig | None = None,
) -> pd.DataFrame:
    """Bootstrap whole traces, preserving within-trace step dependence."""
    analysis_config = analysis_config or AnalysisConfig()
    if samples < 1:
        return pd.DataFrame()
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(metadata) != len(labels) or labels.shape != scores.shape:
        raise ValueError("metadata, labels, and scores must have equal lengths")
    trace_indices = [
        group.to_numpy(dtype=int)
        for _, group in pd.Series(np.arange(len(metadata))).groupby(
            metadata["trace_id"].to_numpy(), sort=False
        )
    ]
    expected, predicted = localization_outcomes(metadata, scores, threshold)
    n_traces = len(trace_indices)
    rng = np.random.default_rng(seed)
    rows = []
    for sample_index in range(samples):
        draws = rng.choice(n_traces, size=n_traces, replace=True)
        sampled_indices = np.concatenate([trace_indices[index] for index in draws])
        row = {
            "sample": sample_index,
            **binary_metrics(
                labels[sampled_indices],
                scores[sampled_indices],
                threshold=threshold,
                calibration_bins=analysis_config.calibration_bins,
            ),
            **localization_metrics(
                expected[draws],
                predicted[draws],
                analysis_config.localization_tolerances,
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_bootstrap(frame: pd.DataFrame, confidence_level: float) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["metric", "estimate", "ci_low", "ci_high"])
    tail = (1 - confidence_level) / 2
    rows = []
    for column in frame.columns.difference(["sample"]):
        values = frame[column].dropna().to_numpy(dtype=float)
        if values.size:
            ci_low, ci_high = quantile_interval(values, tail)
            rows.append(
                {
                    "metric": column,
                    "estimate": float(np.mean(values)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "confidence_level": confidence_level,
                }
            )
    return pd.DataFrame(rows)


def threshold_sensitivity_curve(
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    config: AnalysisConfig,
    *,
    selected_threshold: float,
) -> pd.DataFrame:
    rows = []
    thresholds = np.linspace(config.threshold_min, config.threshold_max, config.threshold_points)
    for split, frame in (("validation", validation_predictions), ("test", test_predictions)):
        for threshold in thresholds:
            rows.append(
                {
                    "split": split,
                    "threshold": float(threshold),
                    "is_validation_selected": int(np.isclose(threshold, selected_threshold)),
                    **binary_metrics(
                        frame["label"].to_numpy(),
                        frame["score"].to_numpy(),
                        threshold=float(threshold),
                        calibration_bins=config.calibration_bins,
                    ),
                    **change_point_metrics(
                        frame,
                        frame["score"].to_numpy(),
                        float(threshold),
                        tolerances=config.localization_tolerances,
                    ),
                }
            )
    return pd.DataFrame(rows)


def calibration_curve_table(labels: np.ndarray, scores: np.ndarray, *, bins: int) -> pd.DataFrame:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for bin_index in range(bins):
        if bin_index == bins - 1:
            member = (scores >= edges[bin_index]) & (scores <= edges[bin_index + 1])
        else:
            member = (scores >= edges[bin_index]) & (scores < edges[bin_index + 1])
        rows.append(
            {
                "bin": bin_index,
                "lower": float(edges[bin_index]),
                "upper": float(edges[bin_index + 1]),
                "n": int(member.sum()),
                "mean_score": float(scores[member].mean()) if member.any() else float("nan"),
                "positive_rate": float(labels[member].mean()) if member.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def trajectory_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    erroneous = predictions[predictions["first_error"] >= 0].copy()
    if erroneous.empty:
        return pd.DataFrame()
    erroneous["relative_step"] = erroneous["step_index"] - erroneous["first_error"]
    window = erroneous[erroneous["relative_step"].between(-3, 3)]
    rows = [
        {
            "metric": "mean_score_at_relative_step",
            "relative_step": int(relative_step),
            "value": float(group["score"].mean()),
            "standard_deviation": float(group["score"].std(ddof=1)),
            "n": len(group),
        }
        for relative_step, group in window.groupby("relative_step", sort=True)
    ]
    onset_jumps = []
    pre_post_differences = []
    for _, trace in erroneous.groupby("trace_id", sort=False):
        trace = trace.sort_values("step_index")
        onset = int(trace["first_error"].iloc[0])
        onset_score = trace.loc[trace["step_index"] == onset, "score"]
        previous_score = trace.loc[trace["step_index"] == onset - 1, "score"]
        if not onset_score.empty and not previous_score.empty:
            onset_jumps.append(float(onset_score.iloc[0] - previous_score.iloc[0]))
        before = trace.loc[trace["step_index"] < onset, "score"]
        after = trace.loc[trace["step_index"] >= onset, "score"]
        if not before.empty and not after.empty:
            pre_post_differences.append(float(after.mean() - before.mean()))
    for name, values in (
        ("mean_onset_jump", onset_jumps),
        ("mean_post_minus_pre_score", pre_post_differences),
    ):
        rows.append(
            {
                "metric": name,
                "relative_step": np.nan,
                "value": _safe_mean(np.asarray(values)),
                "standard_deviation": (
                    float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
                ),
                "n": len(values),
            }
        )
    return pd.DataFrame(rows)


def subgroup_metrics(
    predictions: pd.DataFrame,
    all_metadata: pd.DataFrame,
    *,
    threshold: float,
    analysis_config: AnalysisConfig,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    frame = predictions.copy()
    fit_traces = all_metadata[
        all_metadata["partition"].isin(["train", "validation"])
    ].drop_duplicates("trace_id")
    test_traces = frame.drop_duplicates("trace_id")
    frame["trace_length_bin"] = frame["trace_id"].map(
        _quantile_bin_mapping(fit_traces, test_traces, "n_steps")
    )
    frame["token_count_bin"] = frame["trace_id"].map(
        _quantile_bin_mapping(fit_traces, test_traces, "token_count")
    )
    error_fraction = frame["first_error"] / frame["n_steps"].clip(lower=1)
    frame["error_position_bin"] = np.select(
        [
            frame["first_error"] < 0,
            error_fraction <= 1 / 3,
            error_fraction <= 2 / 3,
        ],
        ["correct", "early", "middle"],
        default="late",
    )
    rows = []
    subgroup_columns = [
        "source",
        "generator",
        "final_answer_correct",
        "error_position_bin",
        "trace_length_bin",
        "token_count_bin",
    ]
    tail = (1 - analysis_config.confidence_level) / 2
    for column in subgroup_columns:
        for value, group in frame.groupby(column, dropna=False, sort=True):
            n_traces = group["trace_id"].nunique()
            row = {
                "subgroup": column,
                "value": str(value),
                "n_traces": n_traces,
                "n_steps": len(group),
                "status": (
                    "reported" if n_traces >= analysis_config.subgroup_min_traces else "suppressed"
                ),
            }
            if n_traces >= analysis_config.subgroup_min_traces:
                metrics = {
                    **binary_metrics(
                        group["label"].to_numpy(),
                        group["score"].to_numpy(),
                        threshold=threshold,
                        calibration_bins=analysis_config.calibration_bins,
                    ),
                    **change_point_metrics(
                        group,
                        group["score"].to_numpy(),
                        threshold,
                        tolerances=analysis_config.localization_tolerances,
                    ),
                }
                row.update(metrics)
                bootstrap = group_bootstrap_metrics(
                    group,
                    group["label"].to_numpy(),
                    group["score"].to_numpy(),
                    threshold,
                    samples=bootstrap_samples,
                    seed=seed,
                    analysis_config=analysis_config,
                )
                for metric in ("auroc", "process_f1", "first_error_exact"):
                    values = bootstrap.get(metric, pd.Series(dtype=float)).dropna()
                    if not values.empty:
                        bounds = quantile_interval(values.to_numpy(dtype=float), tail)
                        row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = bounds
            rows.append(row)
    return pd.DataFrame(rows)


def _quantile_bin_mapping(
    fit_traces: pd.DataFrame, test_traces: pd.DataFrame, column: str
) -> dict[str, str]:
    values = fit_traces[column].to_numpy(dtype=float)
    edges = np.unique(np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0]))
    if len(edges) < 2:
        return dict.fromkeys(test_traces["trace_id"].astype(str), "all")
    edges[0] = -np.inf
    edges[-1] = np.inf
    labels = [f"Q{index + 1}" for index in range(len(edges) - 1)]
    bins = pd.cut(test_traces[column], bins=edges, labels=labels, include_lowest=True)
    return dict(zip(test_traces["trace_id"].astype(str), bins.astype(str), strict=True))


def compare_probe_families(
    family_metrics: pd.DataFrame,
    family_predictions: pd.DataFrame,
    *,
    primary_family: str,
    threshold_by_layer: np.ndarray,
    analysis_config: AnalysisConfig,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    selected_rows = family_metrics[
        (family_metrics["split"] == "validation") & (family_metrics["is_family_selected"] == 1)
    ]
    selected_layers = dict(zip(selected_rows["family"], selected_rows["layer"], strict=True))
    primary_layer = int(selected_layers[primary_family])
    primary = family_predictions[
        (family_predictions["family"] == primary_family)
        & (family_predictions["layer"] == primary_layer)
    ]
    primary_threshold = float(threshold_by_layer[primary_layer])
    rows = []
    for family, family_layer in selected_layers.items():
        if family == primary_family:
            continue
        candidate = family_predictions[
            (family_predictions["family"] == family) & (family_predictions["layer"] == family_layer)
        ]
        candidate_threshold = float(candidate["threshold"].iloc[0])
        merged = primary.merge(
            candidate[["trace_id", "step_index", "score"]],
            on=["trace_id", "step_index"],
            suffixes=("_primary", "_candidate"),
            validate="one_to_one",
        )
        primary_values = _comparison_values(
            merged, "score_primary", primary_threshold, analysis_config
        )
        candidate_values = _comparison_values(
            merged, "score_candidate", candidate_threshold, analysis_config
        )
        bootstraps = _paired_comparison_bootstrap(
            merged,
            primary_threshold,
            candidate_threshold,
            analysis_config,
            samples=bootstrap_samples,
            seed=seed,
        )
        tail = (1 - analysis_config.confidence_level) / 2
        for metric in primary_values:
            deltas = bootstraps.get(metric, pd.Series(dtype=float)).dropna()
            bounds = (
                quantile_interval(deltas.to_numpy(dtype=float), tail)
                if not deltas.empty
                else (np.nan, np.nan)
            )
            rows.append(
                {
                    "family": family,
                    "primary_layer": primary_layer,
                    "family_layer": int(family_layer),
                    "metric": metric,
                    "primary_value": primary_values[metric],
                    "family_value": candidate_values[metric],
                    "delta": candidate_values[metric] - primary_values[metric],
                    "delta_ci_low": bounds[0],
                    "delta_ci_high": bounds[1],
                }
            )
    return pd.DataFrame(rows)


def _comparison_values(
    frame: pd.DataFrame, score_column: str, threshold: float, config: AnalysisConfig
) -> dict[str, float]:
    scores = frame[score_column].to_numpy()
    binary = binary_metrics(
        frame["label"].to_numpy(),
        scores,
        threshold=threshold,
        calibration_bins=config.calibration_bins,
    )
    change = change_point_metrics(
        frame,
        scores,
        threshold,
        tolerances=config.localization_tolerances,
    )
    return {
        "auroc": binary["auroc"],
        "average_precision": binary["average_precision"],
        "process_f1": change["process_f1"],
        "first_error_exact": change["first_error_exact"],
    }


def _paired_comparison_bootstrap(
    frame: pd.DataFrame,
    primary_threshold: float,
    candidate_threshold: float,
    config: AnalysisConfig,
    *,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    if samples < 1:
        return pd.DataFrame()
    groups = {trace_id: group for trace_id, group in frame.groupby("trace_id", sort=False)}
    trace_ids = np.asarray(list(groups))
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(samples):
        pieces = []
        for draw_index, trace_id in enumerate(rng.choice(trace_ids, len(trace_ids), replace=True)):
            piece = groups[trace_id].copy()
            piece["trace_id"] = f"bootstrap-{draw_index}"
            pieces.append(piece)
        sampled = pd.concat(pieces, ignore_index=True)
        primary = _comparison_values(sampled, "score_primary", primary_threshold, config)
        candidate = _comparison_values(sampled, "score_candidate", candidate_threshold, config)
        rows.append({metric: candidate[metric] - value for metric, value in primary.items()})
    return pd.DataFrame(rows)


def evaluate_controls(
    hidden: np.ndarray,
    metadata: pd.DataFrame,
    labels: np.ndarray,
    *,
    max_iter: int,
    seed: int,
    analysis_config: AnalysisConfig | None = None,
) -> pd.DataFrame:
    analysis_config = analysis_config or AnalysisConfig()
    train = metadata["partition"].eq("train").to_numpy()
    validation = metadata["partition"].eq("validation").to_numpy()
    test = metadata["partition"].eq("test").to_numpy()
    rows: list[dict[str, object]] = []

    position = metadata[["step_index", "step_fraction"]].to_numpy(dtype=np.float32)
    rows.append(
        _fit_control(
            "position",
            position,
            metadata,
            labels,
            train,
            validation,
            test,
            max_iter,
            seed,
            analysis_config,
        )
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000)
    train_text = metadata.loc[train, "step_text"].astype(str)
    x_train = vectorizer.fit_transform(train_text)
    x_validation = vectorizer.transform(metadata.loc[validation, "step_text"].astype(str))
    x_test = vectorizer.transform(metadata.loc[test, "step_text"].astype(str))
    classifier = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=max_iter, solver="liblinear", random_state=seed
    ).fit(x_train, labels[train])
    validation_scores = classifier.predict_proba(x_validation)[:, 1]
    threshold = _choose_configured_threshold(
        metadata.loc[validation], validation_scores, analysis_config
    )
    scores = classifier.predict_proba(x_test)[:, 1]
    rows.append(
        _control_row(
            "current-step TF-IDF",
            metadata.loc[test],
            labels[test],
            scores,
            threshold,
            analysis_config,
        )
    )

    rng = np.random.default_rng(seed)
    shuffled = labels[train].copy()
    rng.shuffle(shuffled)
    scaler, classifier = _fit_logistic(hidden[train], shuffled, 1.0, max_iter, seed)
    validation_scores = classifier.predict_proba(scaler.transform(hidden[validation]))[:, 1]
    threshold = _choose_configured_threshold(
        metadata.loc[validation], validation_scores, analysis_config
    )
    scores = classifier.predict_proba(scaler.transform(hidden[test]))[:, 1]
    rows.append(
        _control_row(
            "shuffled-label hidden",
            metadata.loc[test],
            labels[test],
            scores,
            threshold,
            analysis_config,
        )
    )
    return pd.DataFrame(rows)


def domain_transfer(
    hidden: np.ndarray,
    metadata: pd.DataFrame,
    labels: np.ndarray,
    *,
    c_value: float,
    max_iter: int,
    seed: int,
    analysis_config: AnalysisConfig | None = None,
) -> pd.DataFrame:
    analysis_config = analysis_config or AnalysisConfig()
    rows: list[dict[str, object]] = []
    sources = sorted(metadata["source"].unique())
    for train_source in sources:
        train = metadata["source"].eq(train_source) & metadata["partition"].eq("train")
        validation = metadata["source"].eq(train_source) & metadata["partition"].eq("validation")
        if labels[train].min() == labels[train].max():
            continue
        scaler, classifier = _fit_logistic(hidden[train], labels[train], c_value, max_iter, seed)
        validation_scores = classifier.predict_proba(scaler.transform(hidden[validation]))[:, 1]
        threshold = _choose_configured_threshold(
            metadata.loc[validation], validation_scores, analysis_config
        )
        for test_source in sources:
            test = metadata["source"].eq(test_source) & metadata["partition"].eq("test")
            scores = classifier.predict_proba(scaler.transform(hidden[test]))[:, 1]
            row = _control_row(
                "domain transfer",
                metadata.loc[test],
                labels[test],
                scores,
                threshold,
                analysis_config,
            )
            row.update({"train_source": train_source, "test_source": test_source})
            rows.append(row)
    return pd.DataFrame(rows)


def pca_subspace_curve(
    hidden: np.ndarray,
    metadata: pd.DataFrame,
    labels: np.ndarray,
    dimensions: tuple[int, ...],
    *,
    c_value: float,
    max_iter: int,
    seed: int,
    analysis_config: AnalysisConfig | None = None,
) -> pd.DataFrame:
    """Test accessibility in top-variance PCA subspaces (not intrinsic dimension)."""
    analysis_config = analysis_config or AnalysisConfig()
    train = metadata["partition"].eq("train").to_numpy()
    validation = metadata["partition"].eq("validation").to_numpy()
    test = metadata["partition"].eq("test").to_numpy()
    maximum = min(max(dimensions, default=0), hidden.shape[1], int(train.sum()))
    if maximum < 1:
        return pd.DataFrame()
    pca = PCA(n_components=maximum, svd_solver="randomized", random_state=seed).fit(hidden[train])
    transformed = pca.transform(hidden)
    rows = []
    for dimension in sorted({value for value in dimensions if value <= maximum}):
        scaler, classifier = _fit_logistic(
            transformed[train, :dimension], labels[train], c_value, max_iter, seed
        )
        validation_scores = classifier.predict_proba(
            scaler.transform(transformed[validation, :dimension])
        )[:, 1]
        threshold = _choose_configured_threshold(
            metadata.loc[validation], validation_scores, analysis_config
        )
        scores = classifier.predict_proba(scaler.transform(transformed[test, :dimension]))[:, 1]
        row = _control_row(
            "PCA subspace",
            metadata.loc[test],
            labels[test],
            scores,
            threshold,
            analysis_config,
        )
        row["dimensions"] = dimension
        row["variance_explained"] = float(pca.explained_variance_ratio_[:dimension].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_control(
    name,
    features,
    metadata,
    labels,
    train,
    validation,
    test,
    max_iter,
    seed,
    analysis_config,
):
    scaler, classifier = _fit_logistic(features[train], labels[train], 1.0, max_iter, seed)
    validation_scores = classifier.predict_proba(scaler.transform(features[validation]))[:, 1]
    threshold = _choose_configured_threshold(
        metadata.loc[validation], validation_scores, analysis_config
    )
    scores = classifier.predict_proba(scaler.transform(features[test]))[:, 1]
    return _control_row(name, metadata.loc[test], labels[test], scores, threshold, analysis_config)


def _control_row(name, metadata, labels, scores, threshold, analysis_config=None):
    analysis_config = analysis_config or AnalysisConfig()
    return {
        "control": name,
        **binary_metrics(
            labels,
            scores,
            threshold=threshold,
            calibration_bins=analysis_config.calibration_bins,
        ),
        **change_point_metrics(
            metadata,
            scores,
            threshold,
            tolerances=analysis_config.localization_tolerances,
        ),
        "threshold": threshold,
    }


def _choose_configured_threshold(
    metadata: pd.DataFrame, scores: np.ndarray, config: AnalysisConfig
) -> float:
    return choose_threshold(
        metadata,
        scores,
        threshold_min=config.threshold_min,
        threshold_max=config.threshold_max,
        threshold_points=config.threshold_points,
    )


def _fit_logistic(
    features,
    labels,
    c_value,
    max_iter,
    seed,
    *,
    penalty="l2",
    l1_ratio=None,
):
    scaler = StandardScaler().fit(features)
    solver = "saga" if penalty == "elasticnet" else "liblinear"
    kwargs = {}
    sklearn_version = tuple(int(part) for part in sklearn.__version__.split(".")[:2])
    if sklearn_version >= (1, 8):
        kwargs["l1_ratio"] = {
            "l2": 0.0,
            "l1": 1.0,
            "elasticnet": l1_ratio,
        }[penalty]
    else:
        kwargs["penalty"] = penalty
        if l1_ratio is not None:
            kwargs["l1_ratio"] = l1_ratio
    classifier = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=max_iter,
        solver=solver,
        random_state=seed,
        **kwargs,
    ).fit(scaler.transform(features), labels)
    return scaler, classifier


def _metric_row(
    layer,
    split,
    metadata,
    labels,
    scores,
    threshold,
    c_value,
    analysis_config=None,
    *,
    l1_ratio=None,
):
    analysis_config = analysis_config or AnalysisConfig()
    return {
        "layer": layer,
        "split": split,
        "c_value": c_value,
        "l1_ratio": l1_ratio,
        "threshold": threshold,
        **binary_metrics(
            labels,
            scores,
            threshold=threshold,
            calibration_bins=analysis_config.calibration_bins,
        ),
        **change_point_metrics(
            metadata,
            scores,
            threshold,
            tolerances=analysis_config.localization_tolerances,
        ),
    }


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def _validate_inputs(activations: np.ndarray, metadata: pd.DataFrame, target: str) -> None:
    if activations.ndim != 3:
        raise ValueError("activations must have shape [examples, layers, hidden]")
    required = {
        target,
        "partition",
        "trace_id",
        "source",
        "step_index",
        "step_fraction",
        "step_text",
        "first_error",
        "generator",
        "has_error_trace",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    if len(metadata) != len(activations):
        raise ValueError("Activation and metadata row counts differ")


def _partition_masks(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        split: metadata["partition"].eq(split).to_numpy()
        for split in ("train", "validation", "test")
    }
