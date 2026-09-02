"""CPU-only follow-up analyses over frozen Experiment 1 artifacts."""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tracing_math.experiment1.data import ProcessTrace, assign_partitions, load_traces
from tracing_math.experiment1.probes import binary_metrics, change_point_metrics, choose_threshold
from tracing_math.followup.config import FollowupConfig
from tracing_math.localization import (
    assert_process_f1_identity,
    first_crossing,
    localization_metrics,
    localization_outcomes,
    outcome_label,
    quantile_interval,
)

LOGGER = logging.getLogger(__name__)

CONTROL_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
"""Regularization grid shared with the hidden-state probe's validation selection."""


@dataclass(frozen=True)
class TraceSeries:
    trace_id: str
    source: str
    generator: str
    first_error: int
    n_steps: int
    token_count: int
    final_answer_correct: int
    scores: np.ndarray


def run_followup(config: FollowupConfig) -> dict[str, Any]:
    """Run the complete follow-up without loading a language model or activation shard."""
    LOGGER.info("Loading frozen Experiment 1 predictions")
    predictions = _load_predictions(config.experiment1_dir)
    threshold = _load_threshold(config.experiment1_dir, predictions)
    traces = _trace_series(predictions)
    source_traces = load_traces(config.data_path)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Running temporal randomization (%d draws)", config.permutation_samples)
    permutation_summary, permutation_draws = temporal_randomization_test(
        traces,
        threshold=threshold,
        samples=config.permutation_samples,
        seed=config.seed,
        confidence_level=config.confidence_level,
    )
    permutation_summary.to_csv(output / "temporal_randomization_summary.csv", index=False)
    permutation_draws.to_csv(output / "temporal_randomization_draws.csv", index=False)

    LOGGER.info("Running error-aligned event study (%d draws)", config.bootstrap_samples)
    trajectory = event_study(
        traces,
        samples=config.bootstrap_samples,
        seed=config.seed,
        confidence_level=config.confidence_level,
    )
    trajectory.to_csv(output / "error_aligned_trajectory.csv", index=False)

    LOGGER.info("Running matched-placebo analysis (%d draws)", config.bootstrap_samples)
    placebo = matched_placebo_analysis(
        traces,
        samples=config.bootstrap_samples,
        seed=config.seed,
        confidence_level=config.confidence_level,
    )
    placebo.to_csv(output / "matched_placebo_onset.csv", index=False)

    LOGGER.info("Running within-trace discrimination (%d draws)", config.bootstrap_samples)
    centered = centered_discrimination(
        predictions,
        samples=config.bootstrap_samples,
        seed=config.seed,
        confidence_level=config.confidence_level,
    )
    centered.to_csv(output / "within_trace_discrimination.csv", index=False)

    availability: dict[str, str] = {}
    fit_predictions_path = config.experiment1_dir / "probes" / "fit_predictions.csv"
    if fit_predictions_path.exists():
        fit_predictions = pd.read_csv(fit_predictions_path)
        length_thresholds, length_results = length_aware_threshold_analysis(
            fit_predictions,
            predictions,
            global_threshold=threshold,
        )
        length_thresholds.to_csv(output / "length_aware_thresholds.csv", index=False)
        length_results.to_csv(output / "length_aware_threshold_results.csv", index=False)
        availability["length_aware_thresholding"] = "complete"

        LOGGER.info("Running tuned prefix-text, metadata, and joint shortcut controls")
        shortcut_controls, shortcut_predictions = shortcut_control_analysis(
            fit_predictions,
            predictions,
            source_traces,
            seed=config.seed,
            c_grid=config.control_c_values,
        )
        shortcut_controls.to_csv(output / "shortcut_controls.csv", index=False)
        shortcut_predictions.to_csv(output / "shortcut_control_predictions.csv", index=False)
        main_comparison = main_comparison_table(
            predictions,
            shortcut_controls,
            hidden_threshold=threshold,
        )
        main_comparison.to_csv(output / "main_comparison_table.csv", index=False)
        correct_overlap = correct_trace_overlap_table(
            predictions,
            shortcut_predictions,
            hidden_threshold=threshold,
        )
        correct_overlap.to_csv(output / "correct_trace_overlap.csv", index=False)
        availability["shortcut_controls"] = "complete"
        availability["metric_audit"] = "complete"

        LOGGER.info("Running trace-equal-weight sensitivity (%d draws)", config.bootstrap_samples)
        trace_weighting = trace_equal_weight_sensitivity(
            predictions,
            threshold=threshold,
            samples=config.bootstrap_samples,
            seed=config.seed,
            confidence_level=config.confidence_level,
        )
        trace_weighting.to_csv(output / "trace_weighting_sensitivity.csv", index=False)
        availability["trace_weighting_sensitivity"] = "complete"

        train_fraction, validation_fraction = _frozen_split_fractions(config.experiment1_dir)
        data_flow = data_flow_table(
            fit_predictions,
            predictions,
            source_traces,
            seed=config.seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        data_flow.to_csv(output / "data_flow.csv", index=False)
        availability["data_flow"] = "complete"
    else:
        length_results = pd.DataFrame()
        shortcut_controls = pd.DataFrame()
        shortcut_predictions = pd.DataFrame()
        main_comparison = pd.DataFrame()
        correct_overlap = pd.DataFrame()
        trace_weighting = pd.DataFrame()
        data_flow = pd.DataFrame()
        availability["length_aware_thresholding"] = (
            "not run: probes/fit_predictions.csv is absent; test labels were not used for tuning"
        )

    control_predictions_path = config.experiment1_dir / "probes" / "control_predictions.csv"
    if control_predictions_path.exists():
        LOGGER.info("Running paired probe-control intervals (%d draws)", config.bootstrap_samples)
        control_predictions = pd.read_csv(control_predictions_path)
        if not shortcut_predictions.empty:
            control_predictions = pd.concat(
                [control_predictions, shortcut_predictions], ignore_index=True
            )
        paired_controls = paired_probe_control_intervals(
            predictions,
            control_predictions,
            probe_threshold=threshold,
            samples=config.bootstrap_samples,
            seed=config.seed,
            confidence_level=config.confidence_level,
        )
        paired_controls.to_csv(output / "probe_control_paired_intervals.csv", index=False)
        availability["probe_control_paired_intervals"] = "complete"

        LOGGER.info(
            "Running outcome-stratified probe-control intervals (%d draws)",
            config.bootstrap_samples,
        )
        outcome_intervals = stratified_probe_control_intervals(
            predictions,
            control_predictions,
            stratum_column="final_answer_correct",
            probe_threshold=threshold,
            samples=config.bootstrap_samples,
            seed=config.seed,
            confidence_level=config.confidence_level,
        )
        outcome_intervals.to_csv(
            output / "probe_control_intervals_by_final_answer.csv", index=False
        )
        source_intervals = stratified_probe_control_intervals(
            predictions,
            control_predictions,
            stratum_column="source",
            probe_threshold=threshold,
            samples=config.bootstrap_samples,
            seed=config.seed,
            confidence_level=config.confidence_level,
        )
        source_intervals.to_csv(output / "probe_control_intervals_by_source.csv", index=False)
        availability["probe_control_intervals_by_stratum"] = "complete"
    else:
        paired_controls = pd.DataFrame()
        availability["probe_control_paired_intervals"] = (
            "not run: probes/control_predictions.csv is absent; aggregate control metrics "
            "cannot be paired"
        )
    (output / "analysis_availability.json").write_text(json.dumps(availability, indent=2))

    LOGGER.info("Auditing onset-task eligibility (step-0 exclusions)")
    onset_audit = onset_eligibility_audit(traces)
    onset_audit.to_csv(output / "onset_eligibility_audit.csv", index=False)

    LOGGER.info("Running subgroup intervals (%d draws)", config.bootstrap_samples)
    subgroups = subgroup_outcomes(
        traces,
        threshold=threshold,
        samples=config.bootstrap_samples,
        seed=config.seed,
        confidence_level=config.confidence_level,
        min_traces=config.subgroup_min_traces,
    )
    subgroups.to_csv(output / "subgroup_outcomes.csv", index=False)

    LOGGER.info("Summarizing causal sensitivity and writing audit outputs")
    sensitivity = causal_sensitivity_analysis(
        config.experiment1_dir,
        confidence_level=config.confidence_level,
    )
    sensitivity.to_csv(output / "causal_assay_sensitivity.csv", index=False)

    audit_path = write_failure_audit_sample(
        traces,
        threshold=threshold,
        data_path=config.data_path,
        output_path=output / "failure_audit_sample.jsonl",
        examples_per_category=config.audit_examples_per_category,
        seed=config.seed,
    )
    _plot_followup(trajectory, permutation_draws, permutation_summary, output)

    summary = _build_summary(
        environment=_environment_metadata(),
        predictions=predictions,
        threshold=threshold,
        permutation=permutation_summary,
        placebo=placebo,
        centered=centered,
        sensitivity=sensitivity,
        length_results=length_results,
        paired_controls=paired_controls,
        shortcut_controls=shortcut_controls,
        main_comparison=main_comparison,
        correct_overlap=correct_overlap,
        trace_weighting=trace_weighting,
        data_flow=data_flow,
        availability=availability,
        audit_path=audit_path,
        config=config,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "results.md").write_text(
        _results_markdown(summary, placebo, centered, sensitivity)
    )
    LOGGER.info("CPU follow-up complete: %s", output)
    return summary


def _trace_equal_train_weights(frame: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    """Inverse boundary-count weights giving every trace equal total training loss."""
    codes = pd.factorize(frame.loc[mask, "trace_id"], sort=False)[0]
    counts = np.bincount(codes).astype(float)
    return 1.0 / counts[codes]


def _select_control_c(
    make_model,
    train_features,
    train_labels: np.ndarray,
    validation_features,
    validation_labels: np.ndarray,
    *,
    c_grid: tuple[float, ...],
    sample_weight: np.ndarray | None = None,
) -> float:
    """Pick C by validation AUROC using the hidden probe's grid and tie-breaking."""
    candidates = []
    for c_value in c_grid:
        model = make_model(c_value)
        _fit_with_weight(model, train_features, train_labels, sample_weight)
        scores = model.predict_proba(validation_features)[:, 1]
        auroc = float(roc_auc_score(validation_labels, scores))
        candidates.append((auroc, -abs(float(np.log10(c_value))), c_value))
    return float(max(candidates, key=lambda item: item[:2])[2])


def _fit_with_weight(model, features, labels: np.ndarray, weight: np.ndarray | None) -> None:
    """Fit a bare estimator or a pipeline, routing sample weights to the final step."""
    if weight is None:
        model.fit(features, labels)
    elif hasattr(model, "steps"):
        final_step = model.steps[-1][0]
        model.fit(features, labels, **{f"{final_step}__sample_weight": weight})
    else:
        model.fit(features, labels, sample_weight=weight)


def _fit_final_control(
    make_model,
    c_value: float,
    fit_features,
    fit_labels: np.ndarray,
    *,
    sample_weight: np.ndarray | None,
):
    """Refit one control on train plus validation, mirroring the hidden probe."""
    model = make_model(c_value)
    _fit_with_weight(model, fit_features, fit_labels, sample_weight)
    return model


def shortcut_control_analysis(
    fit_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    traces: list[ProcessTrace],
    *,
    seed: int,
    max_iter: int = 2_000,
    c_grid: tuple[float, ...] = CONTROL_C_GRID,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit tuned prefix-text, metadata, and joint controls without test labels.

    Every control receives the same model-selection budget as the hidden-state
    probe: L2 regularization is chosen from ``c_grid`` by validation AUROC, the
    threshold is chosen on the same validation scores, and the selected model is
    refit on train plus validation. Each control is fit with inverse
    boundary-count sample weights so every trace contributes equal training
    loss, addressing the boundary-weighted-training asymmetry of the primary
    probe.
    """
    required = {
        "trace_id",
        "source",
        "generator",
        "partition",
        "step_index",
        "step_fraction",
        "first_error",
        "n_steps",
        "token_count",
        "final_answer_correct",
        "label",
    }
    missing_fit = required.difference(fit_predictions.columns)
    missing_test = (required - {"partition"}).difference(test_predictions.columns)
    if missing_fit or missing_test:
        raise ValueError(
            "Missing shortcut-control columns: "
            f"fit={sorted(missing_fit)}, test={sorted(missing_test)}"
        )
    if set(fit_predictions["partition"].astype(str)) != {"train", "validation"}:
        raise ValueError("Shortcut controls require frozen train and validation predictions")

    fit = _attach_prefix_text(fit_predictions, traces)
    test = _attach_prefix_text(test_predictions.assign(partition="test"), traces)
    train = fit["partition"].eq("train").to_numpy()
    validation = fit["partition"].eq("validation").to_numpy()
    fit_mask = train | validation
    labels = fit["label"].to_numpy(dtype=int)
    test_labels = test["label"].to_numpy(dtype=int)
    if any(np.unique(labels[mask]).size < 2 for mask in (train, validation)):
        raise ValueError("Train and validation shortcut-control splits need both classes")

    def tuned(c_value: float):
        return LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=max_iter,
            solver="liblinear",
            random_state=seed,
        )

    rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    def evaluate(
        name: str,
        make_model,
        train_features,
        validation_features,
        test_features,
        fit_features,
        fit_weight: np.ndarray | None,
    ) -> None:
        c_value = _select_control_c(
            make_model,
            train_features,
            labels[train],
            validation_features,
            labels[validation],
            c_grid=c_grid,
            sample_weight=_trace_equal_train_weights(fit, train),
        )
        selection_model = _fit_final_control(
            make_model,
            c_value,
            train_features,
            labels[train],
            sample_weight=_trace_equal_train_weights(fit, train),
        )
        validation_scores = selection_model.predict_proba(validation_features)[:, 1]
        threshold = choose_threshold(fit.loc[validation], validation_scores)
        final_model = _fit_final_control(
            make_model, c_value, fit_features, labels[fit_mask], sample_weight=fit_weight
        )
        scores = final_model.predict_proba(test_features)[:, 1]
        row, control_predictions = _shortcut_control_outputs(
            name, test, test_labels, scores, threshold, c_value=c_value
        )
        rows.append(row)
        prediction_rows.append(control_predictions)

    fit_weights = _trace_equal_train_weights(fit, fit_mask)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000)
    x_train = vectorizer.fit_transform(fit.loc[train, "prefix_text"])
    x_validation = vectorizer.transform(fit.loc[validation, "prefix_text"])
    x_fit = vectorizer.transform(fit.loc[fit_mask, "prefix_text"])
    x_test = vectorizer.transform(test["prefix_text"])
    evaluate(
        "prefix TF-IDF",
        tuned,
        x_train,
        x_validation,
        x_test,
        x_fit,
        fit_weights,
    )

    numeric = ["step_index", "step_fraction", "n_steps", "token_count"]
    categorical = ["source", "generator"]
    for name, include_outcome in (
        ("structural metadata", False),
        ("metadata plus final outcome", True),
    ):
        numeric_features = [*numeric, "final_answer_correct"] if include_outcome else numeric
        features = [*numeric_features, *categorical]

        def make_metadata(c_value: float, numeric_features=numeric_features):
            return make_pipeline(
                ColumnTransformer(
                    [
                        ("numeric", StandardScaler(), numeric_features),
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                    ]
                ),
                tuned(c_value),
            )

        evaluate(
            name,
            make_metadata,
            fit.loc[train, features],
            fit.loc[validation, features],
            test[features],
            fit.loc[fit_mask, features],
            fit_weights,
        )

    for name, include_outcome in (
        ("joint text + metadata", False),
        ("joint text + metadata + outcome", True),
    ):
        numeric_features = [*numeric, "final_answer_correct"] if include_outcome else numeric
        features = [*numeric_features, *categorical, "prefix_text"]

        def make_joint(c_value: float, numeric_features=numeric_features):
            return make_pipeline(
                ColumnTransformer(
                    [
                        (
                            "text",
                            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000),
                            "prefix_text",
                        ),
                        ("numeric", StandardScaler(), numeric_features),
                        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
                    ]
                ),
                tuned(c_value),
            )

        evaluate(
            name,
            make_joint,
            fit.loc[train, features],
            fit.loc[validation, features],
            test[features],
            fit.loc[fit_mask, features],
            fit_weights,
        )
    return pd.DataFrame(rows), pd.concat(prediction_rows, ignore_index=True)


def trace_equal_weight_sensitivity(
    predictions: pd.DataFrame,
    *,
    threshold: float,
    samples: int,
    seed: int,
    confidence_level: float,
) -> pd.DataFrame:
    """Compare boundary-weighted metrics with metrics giving every trace equal total weight."""
    required = {"trace_id", "label", "score"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing trace-weighting columns: {sorted(missing)}")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")

    labels = predictions["label"].to_numpy(dtype=int)
    scores = predictions["score"].to_numpy(dtype=float)
    trace_codes, trace_ids = pd.factorize(predictions["trace_id"], sort=False)
    row_counts = np.bincount(trace_codes).astype(float)
    boundary_weights = np.ones(len(predictions), dtype=float)
    equal_weights = 1.0 / row_counts[trace_codes]
    boundary = _weighted_binary_metrics(labels, scores, threshold, boundary_weights)
    equal = _weighted_binary_metrics(labels, scores, threshold, equal_weights)

    metrics = ("auroc", "average_precision", "step_f1")
    rng = np.random.default_rng(seed)
    draws = {metric: [] for metric in metrics}
    for _ in range(samples):
        trace_weights = np.bincount(
            rng.integers(0, len(trace_ids), size=len(trace_ids)), minlength=len(trace_ids)
        ).astype(float)
        draw_boundary = _weighted_binary_metrics(
            labels, scores, threshold, trace_weights[trace_codes]
        )
        draw_equal = _weighted_binary_metrics(
            labels,
            scores,
            threshold,
            trace_weights[trace_codes] / row_counts[trace_codes],
        )
        for metric in metrics:
            difference = draw_equal[metric] - draw_boundary[metric]
            if np.isfinite(difference):
                draws[metric].append(difference)

    tail = (1 - confidence_level) / 2
    rows = []
    for metric in metrics:
        ci_low, ci_high = quantile_interval(np.asarray(draws[metric]), tail)
        rows.append(
            {
                "metric": metric,
                "boundary_weighted": boundary[metric],
                "trace_equal_weighted": equal[metric],
                "difference": equal[metric] - boundary[metric],
                "difference_ci_low": ci_low,
                "difference_ci_high": ci_high,
                "n_traces": len(trace_ids),
                "bootstrap_samples": samples,
            }
        )
    return pd.DataFrame(rows)


def data_flow_table(
    fit_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    traces: list[ProcessTrace],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> pd.DataFrame:
    """Count attempted, retained, and excluded traces and boundaries by frozen partition."""
    assignments = assign_partitions(
        traces,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    attempted = pd.DataFrame(
        [
            {
                "trace_id": trace.trace_id,
                "source": trace.source,
                "partition": assignments[trace.trace_id],
                "has_error": int(trace.has_error),
                "n_steps": len(trace.steps),
            }
            for trace in traces
        ]
    )
    retained_rows = pd.concat(
        [fit_predictions, test_predictions.assign(partition="test")], ignore_index=True
    )
    retained = (
        retained_rows.groupby(["trace_id", "partition", "source"], as_index=False)
        .agg(retained_boundaries=("step_index", "size"))
    )
    attempted = attempted.merge(
        retained[["trace_id", "retained_boundaries"]],
        on="trace_id",
        how="left",
        validate="one_to_one",
    )
    attempted["retained_boundaries"] = attempted["retained_boundaries"].fillna(0).astype(int)
    attempted["retained"] = attempted["retained_boundaries"].gt(0)

    rows = []
    for partition in ("train", "validation", "test"):
        frame = attempted[attempted["partition"].eq(partition)]
        rows.append(_data_flow_row(partition, "all", frame))
        for source, group in frame.groupby("source", sort=True):
            rows.append(_data_flow_row(partition, str(source), group))
    rows.append(_data_flow_row("all", "all", attempted))
    return pd.DataFrame(rows)


def _attach_prefix_text(
    predictions: pd.DataFrame, traces: list[ProcessTrace]
) -> pd.DataFrame:
    prefix_text: dict[tuple[str, int], str] = {}
    for trace in traces:
        blocks = [f"Problem:\n{trace.problem.strip()}\n\nReasoning:"]
        for step_index, step in enumerate(trace.steps):
            blocks.append(f"[Step {step_index}]\n{step.strip()}")
            prefix_text[(trace.trace_id, step_index)] = "\n\n".join(blocks)
    keys = list(
        zip(
            predictions["trace_id"].astype(str),
            predictions["step_index"].astype(int),
            strict=True,
        )
    )
    missing = [key for key in keys if key not in prefix_text]
    if missing:
        raise ValueError(f"Missing source text for prediction row {missing[0]}")
    frame = predictions.copy()
    frame["prefix_text"] = [prefix_text[key] for key in keys]
    return frame


def _shortcut_control_outputs(
    name: str,
    metadata: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    c_value: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    row = {
        "control": name,
        **binary_metrics(labels, scores, threshold=threshold),
        **change_point_metrics(metadata, scores, threshold),
        "threshold": threshold,
        "c_value": c_value,
        "training_weighting": "trace_equal",
    }
    assert_process_f1_identity(row)
    columns = [
        "trace_id",
        "source",
        "generator",
        "step_index",
        "step_fraction",
        "first_error",
        "n_steps",
        "token_count",
        "final_answer_correct",
    ]
    predictions = metadata[columns].copy()
    predictions["control"] = name
    predictions["label"] = labels
    predictions["score"] = scores
    predictions["threshold"] = threshold
    predictions["c_value"] = c_value
    return row, predictions


def main_comparison_table(
    hidden_predictions: pd.DataFrame,
    shortcut_controls: pd.DataFrame,
    *,
    hidden_threshold: float,
) -> pd.DataFrame:
    """Build the audited headline comparison table from one metric schema."""
    required = {"trace_id", "step_index", "first_error", "label", "score"}
    missing = required.difference(hidden_predictions.columns)
    if missing:
        raise ValueError(f"Missing hidden prediction columns: {sorted(missing)}")
    hidden = {
        "predictor": "hidden state",
        **binary_metrics(
            hidden_predictions["label"].to_numpy(dtype=int),
            hidden_predictions["score"].to_numpy(dtype=float),
            threshold=hidden_threshold,
        ),
        **change_point_metrics(
            hidden_predictions,
            hidden_predictions["score"].to_numpy(dtype=float),
            hidden_threshold,
        ),
        "threshold": hidden_threshold,
        "training_weighting": "boundary_equal",
    }
    rows = [hidden]
    for control in shortcut_controls.to_dict(orient="records"):
        control = dict(control)
        control["predictor"] = control.pop("control")
        rows.append(control)
    for row in rows:
        assert_process_f1_identity(row)
    columns = [
        "predictor",
        "auroc",
        "average_precision",
        "log_loss",
        "brier_score",
        "error_exact",
        "correct_rejection",
        "process_f1",
        "error_within_1_accuracy",
        "error_within_2_accuracy",
        "complete_accuracy",
        "threshold",
        "c_value",
        "training_weighting",
    ]
    frame = pd.DataFrame(rows)
    return frame.reindex(columns=columns)


def correct_trace_overlap_table(
    hidden_predictions: pd.DataFrame,
    control_predictions: pd.DataFrame,
    *,
    hidden_threshold: float,
) -> pd.DataFrame:
    """Count paired false-alarm overlap on held-out correct traces."""
    required = {"trace_id", "step_index", "first_error", "score"}
    if missing := required.difference(hidden_predictions.columns):
        raise ValueError(f"Missing hidden prediction columns: {sorted(missing)}")
    if missing := (required | {"control", "threshold"}).difference(
        control_predictions.columns
    ):
        raise ValueError(f"Missing control prediction columns: {sorted(missing)}")
    hidden_expected, hidden_outcomes = localization_outcomes(
        hidden_predictions,
        hidden_predictions["score"].to_numpy(dtype=float),
        hidden_threshold,
    )
    trace_ids = (
        hidden_predictions.groupby("trace_id", sort=False).size().index.astype(str).to_numpy()
    )
    rows: list[dict[str, object]] = []
    for name, group in control_predictions.groupby("control", sort=True):
        if group["threshold"].nunique() != 1:
            raise ValueError(f"Control {name!r} has multiple held-out thresholds")
        expected, outcomes = localization_outcomes(
            group,
            group["score"].to_numpy(dtype=float),
            float(group["threshold"].iloc[0]),
        )
        control_ids = group.groupby("trace_id", sort=False).size().index.astype(str).to_numpy()
        if not np.array_equal(trace_ids, control_ids) or not np.array_equal(
            hidden_expected, expected
        ):
            raise ValueError(f"Control {name!r} does not align with hidden predictions")
        correct = expected < 0
        hidden_alarm = hidden_outcomes >= 0
        nuisance_alarm = outcomes >= 0
        categories = {
            "both_reject": ~hidden_alarm & ~nuisance_alarm,
            "hidden_only_alarm": hidden_alarm & ~nuisance_alarm,
            "nuisance_only_alarm": ~hidden_alarm & nuisance_alarm,
            "both_alarm": hidden_alarm & nuisance_alarm,
        }
        n_correct = int(correct.sum())
        for outcome, mask in categories.items():
            count = int((correct & mask).sum())
            rows.append(
                {
                    "control": name,
                    "outcome": outcome,
                    "count": count,
                    "rate": count / n_correct if n_correct else float("nan"),
                    "n_correct_traces": n_correct,
                }
            )
    return pd.DataFrame(rows)


def stratified_probe_control_intervals(
    probe_predictions: pd.DataFrame,
    control_predictions: pd.DataFrame,
    *,
    stratum_column: str,
    probe_threshold: float,
    samples: int,
    seed: int,
    confidence_level: float,
) -> pd.DataFrame:
    """Probe-minus-control paired intervals within one metadata stratum."""
    if stratum_column not in probe_predictions.columns:
        raise ValueError(f"Probe predictions lack the stratum column {stratum_column!r}")
    frames = []
    for stratum_value, group in probe_predictions.groupby(stratum_column, sort=True):
        if group["label"].nunique() < 2:
            continue
        matched = control_predictions[
            control_predictions["trace_id"].isin(set(group["trace_id"].astype(str)))
        ]
        if matched.empty:
            continue
        result = paired_probe_control_intervals(
            group,
            matched,
            probe_threshold=probe_threshold,
            samples=samples,
            seed=seed,
            confidence_level=confidence_level,
        )
        result.insert(1, "stratum", f"{stratum_column}={stratum_value}")
        frames.append(result)
    if not frames:
        return pd.DataFrame(
            columns=[
                "control",
                "stratum",
                "metric",
                "probe_minus_control",
                "ci_low",
                "ci_high",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _data_flow_row(partition: str, source: str, frame: pd.DataFrame) -> dict[str, object]:
    retained = frame["retained"]
    return {
        "partition": partition,
        "source": source,
        "attempted_traces": len(frame),
        "retained_traces": int(retained.sum()),
        "excluded_over_length": int((~retained).sum()),
        "attempted_boundaries": int(frame["n_steps"].sum()),
        "retained_boundaries": int(frame["retained_boundaries"].sum()),
        "retained_erroneous_traces": int((retained & frame["has_error"].eq(1)).sum()),
        "retained_correct_traces": int((retained & frame["has_error"].eq(0)).sum()),
    }


def _frozen_split_fractions(experiment1_dir: Path) -> tuple[float, float]:
    path = experiment1_dir / "experiment_config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen Experiment 1 configuration: {path}")
    raw = yaml.safe_load(path.read_text())
    probe = raw.get("probe", {}) if isinstance(raw, dict) else {}
    return float(probe["train_fraction"]), float(probe["validation_fraction"])


def length_aware_threshold_analysis(
    fit_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    *,
    global_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit trace-length-bin thresholds on train+validation and evaluate once on test."""
    required = {"trace_id", "step_index", "first_error", "n_steps", "label", "score"}
    for name, frame in (("fit", fit_predictions), ("test", test_predictions)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing {name} prediction columns: {sorted(missing)}")
        if frame.duplicated(["trace_id", "step_index"]).any():
            raise ValueError(f"{name} predictions must be unique by trace and step")
    if "partition" not in fit_predictions:
        raise ValueError("Fit predictions must identify train and validation partitions")
    partitions = set(fit_predictions["partition"].astype(str))
    if not partitions or not partitions.issubset({"train", "validation"}):
        raise ValueError("Length thresholds may only use train and validation predictions")

    fit = fit_predictions.copy()
    test = test_predictions.copy()
    edges, labels = _length_bin_specification(fit)
    fit["length_bin"] = pd.cut(
        fit["n_steps"], bins=edges, labels=labels, include_lowest=True
    ).astype(str)
    test["length_bin"] = pd.cut(
        test["n_steps"], bins=edges, labels=labels, include_lowest=True
    ).astype(str)

    threshold_rows = []
    threshold_by_bin: dict[str, float] = {}
    for index, label in enumerate(labels):
        group = fit[fit["length_bin"] == label]
        trace_labels = group.drop_duplicates("trace_id")["first_error"].ge(0)
        usable = not group.empty and trace_labels.nunique() == 2
        threshold = (
            choose_threshold(group, group["score"].to_numpy())
            if usable
            else float(global_threshold)
        )
        threshold_by_bin[label] = threshold
        threshold_rows.append(
            {
                "length_bin": label,
                "lower_n_steps": edges[index],
                "upper_n_steps": edges[index + 1],
                "threshold": threshold,
                "fit_traces": group["trace_id"].nunique(),
                "status": "fit" if usable else "global_fallback",
            }
        )

    global_metrics = _thresholded_process_metrics(
        test, np.full(len(test), global_threshold, dtype=float)
    )
    length_metrics = _thresholded_process_metrics(
        test, test["length_bin"].map(threshold_by_bin).to_numpy(dtype=float)
    )
    metric_names = ("process_f1", "correct_rejection", "error_exact", "complete_accuracy")
    results = pd.DataFrame(
        [
            {
                "metric": metric,
                "global_threshold": global_metrics[metric],
                "length_aware": length_metrics[metric],
                "difference": length_metrics[metric] - global_metrics[metric],
            }
            for metric in metric_names
        ]
    )
    return pd.DataFrame(threshold_rows), results


def paired_probe_control_intervals(
    probe_predictions: pd.DataFrame,
    control_predictions: pd.DataFrame,
    *,
    probe_threshold: float,
    samples: int,
    seed: int,
    confidence_level: float,
) -> pd.DataFrame:
    """Whole-trace paired bootstrap intervals for probe-minus-control metrics."""
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    required = {"trace_id", "step_index", "first_error", "label", "score"}
    missing_probe = required.difference(probe_predictions.columns)
    missing_control = (required | {"control", "threshold"}).difference(
        control_predictions.columns
    )
    if missing_probe or missing_control:
        raise ValueError(
            "Missing paired prediction columns: "
            f"probe={sorted(missing_probe)}, control={sorted(missing_control)}"
        )

    rng = np.random.default_rng(seed)
    tail = (1 - confidence_level) / 2
    rows = []
    keys = ["trace_id", "step_index"]
    probe = probe_predictions[list(required)].rename(
        columns={"score": "probe_score", "label": "probe_label", "first_error": "probe_error"}
    )
    for control_name, control in control_predictions.groupby("control", sort=True):
        if control["threshold"].nunique() != 1:
            raise ValueError(f"Control {control_name!r} has multiple held-out thresholds")
        merged = probe.merge(
            control[list(required | {"threshold"})].rename(
                columns={
                    "score": "control_score",
                    "label": "control_label",
                    "first_error": "control_error",
                }
            ),
            on=keys,
            validate="one_to_one",
        )
        if len(merged) != len(probe_predictions):
            raise ValueError(f"Control {control_name!r} is missing held-out prediction rows")
        if not (
            merged["probe_label"].eq(merged["control_label"]).all()
            and merged["probe_error"].eq(merged["control_error"]).all()
        ):
            raise ValueError(f"Control {control_name!r} labels do not match probe labels")
        control_threshold = float(merged["threshold"].iloc[0])
        point = _probe_minus_control_metrics(merged, probe_threshold, control_threshold)
        trace_codes, trace_ids = pd.factorize(merged["trace_id"], sort=False)
        expected, probe_predicted, control_predicted = _paired_trace_predictions(
            merged,
            trace_codes,
            trace_count=len(trace_ids),
            probe_threshold=probe_threshold,
            control_threshold=control_threshold,
        )
        labels = merged["probe_label"].to_numpy(dtype=int)
        probe_scores = merged["probe_score"].to_numpy(dtype=float)
        control_scores = merged["control_score"].to_numpy(dtype=float)
        draws: dict[str, list[float]] = {metric: [] for metric in point}
        for _ in range(samples):
            trace_weights = np.bincount(
                rng.integers(0, len(trace_ids), size=len(trace_ids)),
                minlength=len(trace_ids),
            )
            draw = _weighted_probe_minus_control_metrics(
                labels=labels,
                probe_scores=probe_scores,
                control_scores=control_scores,
                probe_threshold=probe_threshold,
                control_threshold=control_threshold,
                row_weights=trace_weights[trace_codes],
                trace_weights=trace_weights,
                expected=expected,
                probe_predicted=probe_predicted,
                control_predicted=control_predicted,
            )
            for metric, value in draw.items():
                if np.isfinite(value):
                    draws[metric].append(value)
        for metric, estimate in point.items():
            values = np.asarray(draws[metric], dtype=float)
            if values.size:
                ci_low, ci_high = quantile_interval(values, tail)
            else:
                # Metrics that are undefined in every draw (for example correct
                # rejection inside an all-erroneous stratum) get missing bounds.
                ci_low = ci_high = float("nan")
            rows.append(
                {
                    "control": control_name,
                    "metric": metric,
                    "probe_minus_control": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "confidence_level": confidence_level,
                    "bootstrap_unit": "trace",
                    "bootstrap_samples": samples,
                    "n_traces": len(trace_ids),
                }
            )
    return pd.DataFrame(rows)


def temporal_randomization_test(
    traces: list[TraceSeries],
    *,
    threshold: float,
    samples: int,
    seed: int,
    confidence_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare observed timing with within-trace circular shifts of the frozen scores."""
    expected = np.asarray([trace.first_error for trace in traces], dtype=int)
    observed_predicted = np.asarray(
        [first_crossing(trace.scores, threshold) for trace in traces], dtype=int
    )
    observed = _trace_metrics(expected, observed_predicted)
    crossing_by_offset = [
        np.asarray(
            [
                first_crossing(np.roll(trace.scores, offset), threshold)
                for offset in range(len(trace.scores))
            ],
            dtype=int,
        )
        for trace in traces
    ]
    rng = np.random.default_rng(seed)
    draw_batches = []
    batch_size = 512
    variable_indices = np.asarray(
        [index for index, crossings in enumerate(crossing_by_offset) if len(crossings) > 1]
    )
    variable_lengths = np.asarray(
        [len(crossing_by_offset[index]) for index in variable_indices]
    )
    for start in range(0, samples, batch_size):
        size = min(batch_size, samples - start)
        predicted = np.empty((size, len(traces)), dtype=int)
        offsets = (
            rng.integers(0, variable_lengths, size=(size, len(variable_indices)))
            if len(variable_indices)
            else np.empty((size, 0), dtype=int)
        )
        variable_column = 0
        for trace_index, crossings in enumerate(crossing_by_offset):
            if len(crossings) == 1:
                predicted[:, trace_index] = crossings[0]
            else:
                predicted[:, trace_index] = crossings[offsets[:, variable_column]]
                variable_column += 1
        batch = _trace_metric_draws(expected, predicted)
        batch.insert(0, "sample", np.arange(start, start + size))
        draw_batches.append(batch)
    draws = pd.concat(draw_batches, ignore_index=True)
    tail = (1 - confidence_level) / 2
    specifications = {
        "error_exact": "higher",
        "error_within_1_accuracy": "higher",
        "error_within_2_accuracy": "higher",
        "process_f1": "higher",
        "mean_absolute_localization_error": "lower",
        "mean_signed_localization_error": "two_sided",
    }
    summary_rows = []
    for metric, direction in specifications.items():
        values = draws[metric].dropna().to_numpy(dtype=float)
        estimate = float(observed[metric])
        if direction == "higher":
            exceedances = int(np.sum(values >= estimate))
        elif direction == "lower":
            exceedances = int(np.sum(values <= estimate))
        else:
            center = float(np.mean(values))
            exceedances = int(np.sum(np.abs(values - center) >= abs(estimate - center)))
        ci_low, ci_high = quantile_interval(values, tail)
        summary_rows.append(
            {
                "metric": metric,
                "observed": estimate,
                "null_mean": float(np.mean(values)),
                "null_ci_low": ci_low,
                "null_ci_high": ci_high,
                "p_value": (exceedances + 1) / (len(values) + 1),
                "direction": direction,
                "permutations": len(values),
            }
        )
    return pd.DataFrame(summary_rows), draws


def _trace_metric_draws(expected: np.ndarray, predicted: np.ndarray) -> pd.DataFrame:
    """Vectorized counterpart of ``_trace_metrics`` for permutation batches."""
    expected = np.asarray(expected, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    if predicted.ndim != 2 or predicted.shape[1] != len(expected):
        raise ValueError("predicted draws must have shape (draws, traces)")
    erroneous = expected >= 0
    correct = ~erroneous
    error_expected = expected[erroneous]
    error_predictions = predicted[:, erroneous]
    correct_predictions = predicted[:, correct]
    detected = error_predictions >= 0
    localization_error = error_predictions - error_expected
    detected_count = detected.sum(axis=1)

    def detected_mean(values: np.ndarray) -> np.ndarray:
        return np.divide(
            np.where(detected, values, 0).sum(axis=1),
            detected_count,
            out=np.full(len(predicted), np.nan),
            where=detected_count > 0,
        )

    error_accuracy = (error_predictions == error_expected).mean(axis=1)
    correct_accuracy = (correct_predictions == -1).mean(axis=1)
    denominator = error_accuracy + correct_accuracy
    process_f1 = np.divide(
        2 * error_accuracy * correct_accuracy,
        denominator,
        out=np.zeros(len(predicted), dtype=float),
        where=denominator > 0,
    )
    output = {
        "error_exact": error_accuracy,
        "correct_rejection": correct_accuracy,
        "process_f1": process_f1,
        "complete_accuracy": (predicted == expected).mean(axis=1),
        "error_detection_rate": detected.mean(axis=1),
        "mean_signed_localization_error": detected_mean(localization_error),
        "mean_absolute_localization_error": detected_mean(np.abs(localization_error)),
    }
    for tolerance in (1, 2):
        output[f"error_within_{tolerance}_accuracy"] = (
            detected & (np.abs(localization_error) <= tolerance)
        ).mean(axis=1)
    return pd.DataFrame(output)


def event_study(
    traces: list[TraceSeries],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
    window: int = 3,
) -> pd.DataFrame:
    """Estimate the frozen score trajectory around the human-annotated first error."""
    erroneous = [trace for trace in traces if trace.first_error >= 0]
    rng = np.random.default_rng(seed)
    tail = (1 - confidence_level) / 2
    rows = []
    for relative_step in range(-window, window + 1):
        values = np.asarray(
            [
                trace.scores[trace.first_error + relative_step]
                for trace in erroneous
                if 0 <= trace.first_error + relative_step < len(trace.scores)
            ],
            dtype=float,
        )
        boot = np.asarray(
            [rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)]
        )
        ci_low, ci_high = quantile_interval(boot, tail)
        rows.append(
            {
                "relative_step": relative_step,
                "mean_score": float(values.mean()),
                "median_score": float(np.median(values)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_traces": len(values),
            }
        )
    return pd.DataFrame(rows)


def matched_placebo_analysis(
    traces: list[TraceSeries],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> pd.DataFrame:
    """Compare error-onset score jumps with metadata-matched transitions on correct traces."""
    erroneous = [trace for trace in traces if trace.first_error > 0]
    correct = [trace for trace in traces if trace.first_error < 0 and len(trace.scores) > 1]
    pairs = []
    for trace in erroneous:
        match, tier, placebo_step = _match_placebo(trace, correct)
        real_jump = float(trace.scores[trace.first_error] - trace.scores[trace.first_error - 1])
        placebo_jump = float(match.scores[placebo_step] - match.scores[placebo_step - 1])
        prior_jumps = np.diff(trace.scores[: trace.first_error])
        prior_mean = float(prior_jumps.mean()) if len(prior_jumps) else float("nan")
        pairs.append(
            {
                "trace_id": trace.trace_id,
                "placebo_trace_id": match.trace_id,
                "match_tier": tier,
                "real_jump": real_jump,
                "placebo_jump": placebo_jump,
                "paired_difference": real_jump - placebo_jump,
                "prior_mean_jump": prior_mean,
                "onset_minus_prior": real_jump - prior_mean,
            }
        )
    frame = pd.DataFrame(pairs)
    rng = np.random.default_rng(seed)
    tail = (1 - confidence_level) / 2
    rows = []
    for metric in ("real_jump", "placebo_jump", "paired_difference", "onset_minus_prior"):
        values = frame[metric].dropna().to_numpy(dtype=float)
        metric_frame = frame.loc[frame[metric].notna()].reset_index(drop=True)
        boot = _matched_bootstrap_draws(
            metric_frame,
            metric=metric,
            samples=samples,
            rng=rng,
        )
        ci_low, ci_high = quantile_interval(boot, tail)
        rows.append(
            {
                "metric": metric,
                "estimate": float(values.mean()),
                "median": float(np.median(values)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_traces": len(values),
                "exact_source_generator_matches": int(
                    (frame["match_tier"] == "source_generator").sum()
                ),
                "unique_placebo_traces": int(metric_frame["placebo_trace_id"].nunique()),
                "bootstrap_unit": (
                    "error_and_placebo_trace"
                    if metric in {"placebo_jump", "paired_difference"}
                    else "error_trace"
                ),
            }
        )
    return pd.DataFrame(rows)


def centered_discrimination(
    predictions: pd.DataFrame,
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> pd.DataFrame:
    """Measure discrimination after removing each erroneous trace's first-step score."""
    frame = predictions[
        predictions["has_error_trace"].eq(1) & predictions["first_error"].gt(0)
    ].copy()
    first_scores = frame.sort_values("step_index").groupby("trace_id")["score"].first()
    frame["centered_score"] = frame["score"] - frame["trace_id"].map(first_scores)
    trace_ids = frame["trace_id"].drop_duplicates().to_numpy()
    raw = roc_auc_score(frame["label"], frame["score"])
    centered = roc_auc_score(frame["label"], frame["centered_score"])
    macro_values = []
    for _, trace in frame.groupby("trace_id", sort=False):
        if trace["label"].nunique() == 2:
            macro_values.append(roc_auc_score(trace["label"], trace["score"]))
    trace_codes, _ = pd.factorize(frame["trace_id"], sort=False)
    labels = frame["label"].to_numpy(dtype=int)
    raw_scores = frame["score"].to_numpy(dtype=float)
    centered_scores = frame["centered_score"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    raw_draws = []
    centered_draws = []
    for _ in range(samples):
        counts = np.bincount(
            rng.integers(0, len(trace_ids), size=len(trace_ids)), minlength=len(trace_ids)
        )
        weights = counts[trace_codes]
        raw_draws.append(roc_auc_score(labels, raw_scores, sample_weight=weights))
        centered_draws.append(roc_auc_score(labels, centered_scores, sample_weight=weights))
    macro_draws = [
        float(rng.choice(macro_values, size=len(macro_values), replace=True).mean())
        for _ in range(samples)
    ]
    tail = (1 - confidence_level) / 2
    return pd.DataFrame(
        [
            _interval_row("pooled_raw_auroc", raw, raw_draws, tail, len(trace_ids)),
            _interval_row(
                "pooled_first_step_centered_auroc",
                centered,
                centered_draws,
                tail,
                len(trace_ids),
            ),
            _interval_row(
                "mean_within_trace_auroc",
                float(np.mean(macro_values)),
                macro_draws,
                tail,
                len(macro_values),
            ),
        ]
    )


def onset_eligibility_audit(traces: list[TraceSeries]) -> pd.DataFrame:
    """Count traces excluded from the transition task because the error starts at step 0."""
    rows = []
    for eligible, group in pd.DataFrame(
        [
            {
                "onset_eligible": int(trace.first_error > 0),
                "first_error": trace.first_error,
                "source": trace.source,
            }
            for trace in traces
        ]
    ).groupby("onset_eligible"):
        row: dict[str, object] = {
            "onset_eligible": int(eligible),
            "n_traces": len(group),
            "n_step0_errors": int((group["first_error"] == 0).sum()),
            "n_later_onsets": int((group["first_error"] > 0).sum()),
            "sources": ", ".join(sorted(group["source"].unique())),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def subgroup_outcomes(
    traces: list[TraceSeries],
    *,
    threshold: float,
    samples: int,
    seed: int,
    confidence_level: float,
    min_traces: int,
) -> pd.DataFrame:
    """Report trace-level localization outcomes across frozen metadata groups."""
    frame = _trace_outcome_frame(traces, threshold)
    frame["error_position"] = np.select(
        [
            frame["first_error"].lt(0),
            frame["first_error"] / frame["n_steps"].clip(lower=1) <= 1 / 3,
            frame["first_error"] / frame["n_steps"].clip(lower=1) <= 2 / 3,
        ],
        ["correct", "early", "middle"],
        default="late",
    )
    frame["trace_length_bin"] = pd.qcut(
        frame["n_steps"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    ).astype(str)
    frame["token_count_bin"] = pd.qcut(
        frame["token_count"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    ).astype(str)
    rng = np.random.default_rng(seed)
    tail = (1 - confidence_level) / 2
    rows = []
    for column in (
        "source",
        "generator",
        "final_answer_correct",
        "error_position",
        "trace_length_bin",
        "token_count_bin",
    ):
        for value, group in frame.groupby(column, dropna=False, sort=True):
            if len(group) < min_traces:
                continue
            draw_frame = _bootstrap_outcome_summaries(group, samples=samples, rng=rng)
            point = _outcome_summary(group)
            row: dict[str, Any] = {
                "subgroup": column,
                "value": str(value),
                "n_traces": len(group),
                **point,
            }
            for metric in ("complete_accuracy", "error_exact", "correct_rejection"):
                values = draw_frame[metric].dropna()
                bounds = (
                    quantile_interval(values.to_numpy(dtype=float), tail)
                    if not values.empty
                    else (float("nan"), float("nan"))
                )
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = bounds
            rows.append(row)
    return pd.DataFrame(rows)


def causal_sensitivity_analysis(
    experiment1_dir: Path,
    *,
    confidence_level: float,
    power: float = 0.8,
) -> pd.DataFrame:
    """Estimate paired mean effects and approximate minimum detectable effects."""
    path = experiment1_dir / "interventions" / "individual.csv"
    frame = pd.read_csv(path)
    learned = frame[frame["direction_type"].eq("learned") & frame["alpha"].ne(0)].copy()
    baseline_scale = float(
        frame[frame["direction_type"].eq("learned")]
        .drop_duplicates(["trace_id", "step_index"])["baseline_verdict_score"]
        .std(ddof=1)
    )
    z_alpha = norm.ppf(0.5 + confidence_level / 2)
    z_power = norm.ppf(power)
    rows = []
    for alpha, group in learned.groupby("alpha", sort=True):
        values = group["delta_verdict_score"].to_numpy(dtype=float)
        standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
        estimate = float(values.mean())
        half_width = z_alpha * standard_error
        detectable = (z_alpha + z_power) * standard_error
        rows.append(
            {
                "alpha": float(alpha),
                "estimate": estimate,
                "ci_low": estimate - half_width,
                "ci_high": estimate + half_width,
                "standard_error": standard_error,
                "minimum_detectable_effect_80pct": detectable,
                "minimum_detectable_standardized_effect_80pct": detectable / baseline_scale,
                "baseline_score_standard_deviation": baseline_scale,
                "n": len(values),
            }
        )
    return pd.DataFrame(rows)


def write_failure_audit_sample(
    traces: list[TraceSeries],
    *,
    threshold: float,
    data_path: Path,
    output_path: Path,
    examples_per_category: int,
    seed: int,
) -> Path:
    """Freeze a stratified qualitative sample before manual inspection."""
    outcomes = _trace_outcome_frame(traces, threshold)
    rng = np.random.default_rng(seed)
    selected_ids = []
    for category, group in outcomes.groupby("outcome", sort=True):
        count = min(examples_per_category, len(group))
        positions = rng.choice(len(group), size=count, replace=False)
        selected_ids.extend((category, trace_id) for trace_id in group.iloc[positions].trace_id)
    lookup = {trace_id: category for category, trace_id in selected_ids}
    details = {trace.trace_id: trace for trace in traces if trace.trace_id in lookup}
    records = []
    with data_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            trace_id = str(row["trace_id"])
            if trace_id not in lookup:
                continue
            trace = details[trace_id]
            records.append(
                {
                    "trace_id": trace_id,
                    "sampling_category": lookup[trace_id],
                    "source": trace.source,
                    "generator": trace.generator,
                    "first_error": trace.first_error,
                    "predicted_first_error": first_crossing(trace.scores, threshold),
                    "problem": row["problem"],
                    "steps": row["steps"],
                    "manual_error_type": "",
                    "manual_notes": "",
                }
            )
    records.sort(key=lambda row: (row["sampling_category"], row["trace_id"]))
    with output_path.open("w") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    return output_path


def _load_predictions(experiment1_dir: Path) -> pd.DataFrame:
    path = experiment1_dir / "probes" / "test_predictions.csv"
    frame = pd.read_csv(path)
    required = {
        "trace_id",
        "source",
        "generator",
        "step_index",
        "first_error",
        "n_steps",
        "token_count",
        "final_answer_correct",
        "has_error_trace",
        "label",
        "score",
        "layer",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if frame["layer"].nunique() != 1:
        raise ValueError("CPU follow-up expects predictions from exactly one frozen layer")
    if frame.duplicated(["trace_id", "step_index"]).any():
        raise ValueError("Prediction rows must be unique by trace and step")
    return frame.sort_values(["trace_id", "step_index"]).reset_index(drop=True)


def _load_threshold(experiment1_dir: Path, predictions: pd.DataFrame) -> float:
    artifact = np.load(experiment1_dir / "probes" / "directions.npz")
    layer = int(predictions["layer"].iloc[0])
    if layer != int(artifact["selected_layer"]):
        raise ValueError("Prediction layer does not match the frozen selected layer")
    return float(artifact["thresholds"][layer])


def _trace_series(predictions: pd.DataFrame) -> list[TraceSeries]:
    traces = []
    for trace_id, frame in predictions.groupby("trace_id", sort=False):
        ordered = frame.sort_values("step_index")
        expected_steps = np.arange(len(ordered))
        if not np.array_equal(ordered["step_index"].to_numpy(), expected_steps):
            raise ValueError(f"Non-contiguous step indices for {trace_id}")
        first = ordered.iloc[0]
        traces.append(
            TraceSeries(
                trace_id=str(trace_id),
                source=str(first["source"]),
                generator=str(first["generator"]),
                first_error=int(first["first_error"]),
                n_steps=int(first["n_steps"]),
                token_count=int(first["token_count"]),
                final_answer_correct=int(first["final_answer_correct"]),
                scores=ordered["score"].to_numpy(dtype=float),
            )
        )
    return traces


def _trace_metrics(expected: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    metrics = localization_metrics(expected, predicted, (1, 2))
    return {
        "error_exact": metrics["error_exact"],
        "correct_rejection": metrics["correct_rejection"],
        "process_f1": metrics["process_f1"],
        "complete_accuracy": metrics["complete_accuracy"],
        "error_detection_rate": metrics["error_detection_rate"],
        "error_within_1_accuracy": metrics["error_within_1_accuracy"],
        "error_within_2_accuracy": metrics["error_within_2_accuracy"],
        "mean_signed_localization_error": metrics["mean_signed_localization_error"],
        "mean_absolute_localization_error": metrics["mean_absolute_localization_error"],
    }


def _match_placebo(
    target: TraceSeries,
    correct: list[TraceSeries],
) -> tuple[TraceSeries, str, int]:
    exact = [
        trace
        for trace in correct
        if trace.source == target.source and trace.generator == target.generator
    ]
    candidates = exact
    tier = "source_generator"
    if not candidates:
        candidates = [trace for trace in correct if trace.source == target.source]
        tier = "source"
    if not candidates:
        candidates = correct
        tier = "global"
    target_fraction = (target.first_error + 1) / target.n_steps
    ranked = []
    for trace in candidates:
        step = int(np.clip(round(target_fraction * trace.n_steps) - 1, 1, len(trace.scores) - 1))
        fraction = (step + 1) / trace.n_steps
        distance = (
            abs(np.log(trace.n_steps / target.n_steps))
            + abs(np.log(max(trace.token_count, 1) / max(target.token_count, 1)))
            + 2 * abs(fraction - target_fraction)
        )
        ranked.append((distance, trace.trace_id, step, trace))
    _, _, step, match = min(ranked, key=lambda item: item[:2])
    return match, tier, step


def _interval_row(
    metric: str,
    estimate: float,
    values: list[float],
    tail: float,
    n_traces: int,
) -> dict[str, float | int | str]:
    ci_low, ci_high = quantile_interval(np.asarray(values, dtype=float), tail)
    return {
        "metric": metric,
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_traces": n_traces,
    }


def _matched_bootstrap_draws(
    frame: pd.DataFrame,
    *,
    metric: str,
    samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    values = frame[metric].to_numpy(dtype=float)
    if metric in {"real_jump", "onset_minus_prior"}:
        return np.asarray(
            [rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)]
        )

    placebo_codes, placebo_ids = pd.factorize(frame["placebo_trace_id"], sort=False)
    draws = []
    for _ in range(samples):
        placebo_counts = np.bincount(
            rng.integers(0, len(placebo_ids), size=len(placebo_ids)),
            minlength=len(placebo_ids),
        )
        weights = placebo_counts[placebo_codes]
        if metric == "paired_difference":
            error_counts = np.bincount(
                rng.integers(0, len(frame), size=len(frame)), minlength=len(frame)
            )
            weights = weights * error_counts
        if weights.sum() == 0:
            draws.append(float(values.mean()))
        else:
            draws.append(float(np.average(values, weights=weights)))
    return np.asarray(draws)


def _trace_outcome_frame(traces: list[TraceSeries], threshold: float) -> pd.DataFrame:
    rows = []
    for trace in traces:
        predicted = first_crossing(trace.scores, threshold)
        rows.append(
            {
                "trace_id": trace.trace_id,
                "source": trace.source,
                "generator": trace.generator,
                "first_error": trace.first_error,
                "predicted_first_error": predicted,
                "n_steps": trace.n_steps,
                "token_count": trace.token_count,
                "final_answer_correct": trace.final_answer_correct,
                "outcome": outcome_label(trace.first_error, predicted),
            }
        )
    return pd.DataFrame(rows)


def _outcome_summary(frame: pd.DataFrame) -> dict[str, float]:
    erroneous = frame["first_error"].ge(0)
    correct = ~erroneous
    complete = frame["predicted_first_error"].eq(frame["first_error"])
    return {
        "complete_accuracy": float(complete.mean()),
        "error_exact": (float(complete[erroneous].mean()) if erroneous.any() else float("nan")),
        "correct_rejection": (
            float(frame.loc[correct, "predicted_first_error"].lt(0).mean())
            if correct.any()
            else float("nan")
        ),
    }


def _bootstrap_outcome_summaries(
    frame: pd.DataFrame,
    *,
    samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Vectorize trace-level subgroup resampling without constructing DataFrames per draw."""
    erroneous = frame["first_error"].ge(0).to_numpy()
    correct = ~erroneous
    complete = frame["predicted_first_error"].eq(frame["first_error"]).to_numpy()
    rejected = frame["predicted_first_error"].lt(0).to_numpy()
    positions = rng.integers(0, len(frame), size=(samples, len(frame)))
    sampled_erroneous = erroneous[positions]
    sampled_correct = correct[positions]

    def conditional_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        denominator = mask.sum(axis=1)
        return np.divide(
            np.where(mask, values, False).sum(axis=1),
            denominator,
            out=np.full(samples, np.nan),
            where=denominator > 0,
        )

    return pd.DataFrame(
        {
            "complete_accuracy": complete[positions].mean(axis=1),
            "error_exact": conditional_mean(complete[positions], sampled_erroneous),
            "correct_rejection": conditional_mean(rejected[positions], sampled_correct),
        }
    )


def _length_bin_specification(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    trace_lengths = frame.drop_duplicates("trace_id")["n_steps"].to_numpy(dtype=float)
    edges = np.unique(np.quantile(trace_lengths, [0.0, 0.25, 0.5, 0.75, 1.0]))
    if len(edges) < 2:
        edges = np.asarray([-np.inf, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    labels = [f"Q{index + 1}" for index in range(len(edges) - 1)]
    return edges, labels


def _thresholded_process_metrics(
    frame: pd.DataFrame, row_thresholds: np.ndarray
) -> dict[str, float]:
    if len(frame) != len(row_thresholds):
        raise ValueError("A threshold is required for every prediction row")
    work = frame[["trace_id", "step_index", "first_error", "score"]].copy()
    work["threshold"] = np.asarray(row_thresholds, dtype=float)
    expected, predicted = [], []
    for _, trace in work.groupby("trace_id", sort=False):
        ordered = trace.sort_values("step_index")
        crossings = ordered["score"].to_numpy() >= ordered["threshold"].to_numpy()
        expected.append(int(ordered["first_error"].iloc[0]))
        predicted.append(
            int(ordered.loc[crossings, "step_index"].iloc[0]) if crossings.any() else -1
        )
    metrics = localization_metrics(np.asarray(expected), np.asarray(predicted))
    return {
        "process_f1": metrics["process_f1"],
        "correct_rejection": metrics["correct_rejection"],
        "error_exact": metrics["error_exact"],
        "complete_accuracy": metrics["complete_accuracy"],
    }


def _probe_minus_control_metrics(
    frame: pd.DataFrame, probe_threshold: float, control_threshold: float
) -> dict[str, float]:
    metadata = frame[["trace_id", "step_index", "probe_error"]].rename(
        columns={"probe_error": "first_error"}
    )
    labels = frame["probe_label"].to_numpy(dtype=int)
    probe_scores = frame["probe_score"].to_numpy(dtype=float)
    control_scores = frame["control_score"].to_numpy(dtype=float)
    probe_binary = binary_metrics(labels, probe_scores, threshold=probe_threshold)
    control_binary = binary_metrics(labels, control_scores, threshold=control_threshold)
    probe_process = _thresholded_process_metrics(
        metadata.assign(score=probe_scores), np.full(len(frame), probe_threshold)
    )
    control_process = _thresholded_process_metrics(
        metadata.assign(score=control_scores), np.full(len(frame), control_threshold)
    )
    return {
        "auroc": probe_binary["auroc"] - control_binary["auroc"],
        "average_precision": (
            probe_binary["average_precision"] - control_binary["average_precision"]
        ),
        "step_f1": probe_binary["step_f1"] - control_binary["step_f1"],
        **{
            metric: probe_process[metric] - control_process[metric]
            for metric in probe_process
        },
    }


def _paired_trace_predictions(
    frame: pd.DataFrame,
    trace_codes: np.ndarray,
    *,
    trace_count: int,
    probe_threshold: float,
    control_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute trace-level outcomes once for the paired bootstrap."""
    expected = np.empty(trace_count, dtype=int)
    probe_predicted = np.empty(trace_count, dtype=int)
    control_predicted = np.empty(trace_count, dtype=int)
    step_indices = frame["step_index"].to_numpy(dtype=int)
    first_errors = frame["probe_error"].to_numpy(dtype=int)
    probe_scores = frame["probe_score"].to_numpy(dtype=float)
    control_scores = frame["control_score"].to_numpy(dtype=float)
    for trace_code in range(trace_count):
        positions = np.flatnonzero(trace_codes == trace_code)
        order = positions[np.argsort(step_indices[positions], kind="stable")]
        expected[trace_code] = first_errors[order[0]]
        probe_predicted[trace_code] = first_crossing(
            probe_scores[order], probe_threshold, step_indices[order]
        )
        control_predicted[trace_code] = first_crossing(
            control_scores[order], control_threshold, step_indices[order]
        )
    return expected, probe_predicted, control_predicted


def _weighted_probe_minus_control_metrics(
    *,
    labels: np.ndarray,
    probe_scores: np.ndarray,
    control_scores: np.ndarray,
    probe_threshold: float,
    control_threshold: float,
    row_weights: np.ndarray,
    trace_weights: np.ndarray,
    expected: np.ndarray,
    probe_predicted: np.ndarray,
    control_predicted: np.ndarray,
) -> dict[str, float]:
    """Calculate one paired bootstrap draw without rebuilding pandas frames."""
    probe_binary = _weighted_binary_metrics(
        labels, probe_scores, probe_threshold, row_weights
    )
    control_binary = _weighted_binary_metrics(
        labels, control_scores, control_threshold, row_weights
    )
    probe_process = _weighted_process_metrics(expected, probe_predicted, trace_weights)
    control_process = _weighted_process_metrics(expected, control_predicted, trace_weights)
    return {
        metric: probe_binary[metric] - control_binary[metric]
        for metric in ("auroc", "average_precision", "step_f1")
    } | {
        metric: probe_process[metric] - control_process[metric]
        for metric in probe_process
    }


def _weighted_binary_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    weights: np.ndarray,
) -> dict[str, float]:
    positive = labels == 1
    negative = ~positive
    positive_weight = float(weights[positive].sum())
    negative_weight = float(weights[negative].sum())
    predicted = scores >= threshold
    true_positive = float(weights[predicted & positive].sum())
    false_positive = float(weights[predicted & negative].sum())
    false_negative = float(weights[~predicted & positive].sum())
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else float("nan")
    )
    step_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    if positive_weight == 0 or negative_weight == 0:
        auroc = average_precision = float("nan")
    else:
        auroc = float(roc_auc_score(labels, scores, sample_weight=weights))
        average_precision = float(
            average_precision_score(labels, scores, sample_weight=weights)
        )
    return {
        "auroc": auroc,
        "average_precision": average_precision,
        "step_f1": float(step_f1),
    }


def _weighted_process_metrics(
    expected: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    erroneous = expected >= 0
    correct = ~erroneous

    def weighted_mean(values: np.ndarray, mask: np.ndarray) -> float:
        denominator = float(weights[mask].sum())
        return (
            float(np.dot(values[mask].astype(float), weights[mask]) / denominator)
            if denominator
            else float("nan")
        )

    error_accuracy = weighted_mean(predicted == expected, erroneous)
    correct_rejection = weighted_mean(predicted == -1, correct)
    denominator = error_accuracy + correct_rejection
    return {
        "process_f1": (
            2 * error_accuracy * correct_rejection / denominator if denominator > 0 else 0.0
        ),
        "correct_rejection": correct_rejection,
        "error_exact": error_accuracy,
        "complete_accuracy": weighted_mean(
            predicted == expected, np.ones(len(expected), dtype=bool)
        ),
    }


def _plot_followup(
    trajectory: pd.DataFrame,
    permutation_draws: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    axes[0].plot(trajectory["relative_step"], trajectory["mean_score"], marker="o")
    axes[0].fill_between(
        trajectory["relative_step"], trajectory["ci_low"], trajectory["ci_high"], alpha=0.2
    )
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Step relative to first error", ylabel="Mean frozen probe score")

    observed = float(
        permutation_summary.loc[
            permutation_summary["metric"].eq("error_exact"), "observed"
        ].iloc[0]
    )
    axes[1].hist(permutation_draws["error_exact"], bins=30, color="#777777")
    axes[1].axvline(observed, color="#b2182b", linewidth=2, label="Observed")
    axes[1].set(xlabel="Exact localization under circular shifts", ylabel="Count")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "temporal_validity.png", dpi=200)
    figure.savefig(output / "temporal_validity.pdf")
    plt.close(figure)


def _environment_metadata() -> dict[str, str]:
    """Record the exact software environment used by this analysis run."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }


def _build_summary(
    *,
    environment: dict[str, str],
    predictions: pd.DataFrame,
    threshold: float,
    permutation: pd.DataFrame,
    placebo: pd.DataFrame,
    centered: pd.DataFrame,
    sensitivity: pd.DataFrame,
    length_results: pd.DataFrame,
    paired_controls: pd.DataFrame,
    shortcut_controls: pd.DataFrame,
    main_comparison: pd.DataFrame,
    correct_overlap: pd.DataFrame,
    trace_weighting: pd.DataFrame,
    data_flow: pd.DataFrame,
    availability: dict[str, str],
    audit_path: Path,
    config: FollowupConfig,
) -> dict[str, Any]:
    permutation_lookup = permutation.set_index("metric")
    placebo_lookup = placebo.set_index("metric")
    centered_lookup = centered.set_index("metric")
    verdict = json.loads(
        (config.experiment1_dir / "interventions" / "behavioral_verdict.json").read_text()
    )
    summary = {
        "status": "complete",
        "analysis_type": "post_hoc_cpu_followup_on_frozen_experiment1_outputs",
        "environment": environment,
        "selected_layer": int(predictions["layer"].iloc[0]),
        "threshold": threshold,
        "test_rows": len(predictions),
        "test_traces": int(predictions["trace_id"].nunique()),
        "confidence_level": config.confidence_level,
        "temporal_randomization": {
            "observed_exact": _metric_value(permutation_lookup, "error_exact", "observed"),
            "null_exact_mean": _metric_value(permutation_lookup, "error_exact", "null_mean"),
            "exact_p_value": _metric_value(permutation_lookup, "error_exact", "p_value"),
            "observed_within_1": _metric_value(
                permutation_lookup, "error_within_1_accuracy", "observed"
            ),
            "null_within_1_mean": _metric_value(
                permutation_lookup, "error_within_1_accuracy", "null_mean"
            ),
            "within_1_p_value": _metric_value(
                permutation_lookup, "error_within_1_accuracy", "p_value"
            ),
        },
        "matched_placebo": {
            "onset_jump": _metric_value(placebo_lookup, "real_jump"),
            "placebo_jump": _metric_value(placebo_lookup, "placebo_jump"),
            "difference": _metric_value(placebo_lookup, "paired_difference"),
            "difference_ci_low": _metric_value(placebo_lookup, "paired_difference", "ci_low"),
            "difference_ci_high": _metric_value(placebo_lookup, "paired_difference", "ci_high"),
        },
        "within_trace_discrimination": {
            "raw_auroc": _metric_value(centered_lookup, "pooled_raw_auroc"),
            "centered_auroc": _metric_value(centered_lookup, "pooled_first_step_centered_auroc"),
            "centered_ci_low": _metric_value(
                centered_lookup, "pooled_first_step_centered_auroc", "ci_low"
            ),
            "centered_ci_high": _metric_value(
                centered_lookup, "pooled_first_step_centered_auroc", "ci_high"
            ),
            "mean_within_trace_auroc": _metric_value(centered_lookup, "mean_within_trace_auroc"),
        },
        "causal_assay": {
            "baseline_auroc": float(verdict["auroc"]),
            "baseline_specificity": float(verdict["specificity"]),
            "smallest_mde_80pct": float(sensitivity["minimum_detectable_effect_80pct"].min()),
            "behaviorally_valid": bool(verdict["auroc"] > 0.5 and verdict["specificity"] > 0),
        },
        "decision": {
            "temporal_alignment_above_shift_null": bool(
                _metric_value(permutation_lookup, "error_exact", "p_value") < 0.05
                and _metric_value(permutation_lookup, "error_exact", "observed")
                > _metric_value(permutation_lookup, "error_exact", "null_mean")
            ),
            "onset_jump_exceeds_placebo": bool(
                _metric_value(placebo_lookup, "paired_difference", "ci_low") > 0
            ),
            "centered_discrimination_above_chance": bool(
                _metric_value(centered_lookup, "pooled_first_step_centered_auroc", "ci_low") > 0.5
            ),
            "causal_assay_valid": bool(verdict["auroc"] > 0.5 and verdict["specificity"] > 0),
        },
        "failure_audit_sample": str(audit_path),
        "output_dir": str(config.output_dir),
    }
    summary["analysis_availability"] = availability
    if not length_results.empty:
        length_lookup = length_results.set_index("metric")
        summary["length_aware_thresholding"] = {
            "global_process_f1": _metric_value(length_lookup, "process_f1", "global_threshold"),
            "length_aware_process_f1": _metric_value(length_lookup, "process_f1", "length_aware"),
            "global_correct_rejection": _metric_value(
                length_lookup, "correct_rejection", "global_threshold"
            ),
            "length_aware_correct_rejection": _metric_value(
                length_lookup, "correct_rejection", "length_aware"
            ),
        }
    if not paired_controls.empty:
        summary["probe_control_paired_intervals"] = paired_controls.to_dict(orient="records")
    if not shortcut_controls.empty:
        summary["shortcut_controls"] = shortcut_controls.to_dict(orient="records")
    if not main_comparison.empty:
        summary["main_comparison_table"] = _json_records(main_comparison)
    if not correct_overlap.empty:
        summary["correct_trace_overlap"] = correct_overlap.to_dict(orient="records")
    if not trace_weighting.empty:
        summary["trace_weighting_sensitivity"] = trace_weighting.to_dict(orient="records")
    if not data_flow.empty:
        summary["data_flow"] = data_flow.to_dict(orient="records")
    return summary


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return strict-JSON-compatible records with missing values represented by null."""
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


def _metric_value(frame: pd.DataFrame, metric: str, column: str = "estimate") -> float:
    return float(frame.loc[metric, column])


def _results_markdown(
    summary: dict[str, Any],
    placebo: pd.DataFrame,
    centered: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> str:
    temporal = summary["temporal_randomization"]
    matched = summary["matched_placebo"]
    within = summary["within_trace_discrimination"]
    causal = summary["causal_assay"]
    availability = summary["analysis_availability"]
    shortcut_rows = summary.get("shortcut_controls", [])
    shortcut_lines = "\n".join(
        f'- {row["control"]}: AUROC {row["auroc"]:.3f}, Process F1 '
        f'{row["process_f1"]:.3f} (C={row.get("c_value", float("nan")):g}, '
        f'trace-equal training weights)'
        for row in shortcut_rows
    )
    shortcut_section = (
        "## Prefix and metadata shortcut controls\n\n"
        f"{shortcut_lines}\n"
        if shortcut_lines
        else "## Prefix and metadata shortcut controls\n\nNot available.\n"
    )
    trace_rows = {
        row["metric"]: row for row in summary.get("trace_weighting_sensitivity", [])
    }
    if "auroc" in trace_rows:
        trace_row = trace_rows["auroc"]
        trace_section = f"""## Trace-equal weighting

Boundary-weighted AUROC was {trace_row["boundary_weighted"]:.3f}; assigning every trace equal
total weight gave {trace_row["trace_equal_weighted"]:.3f} (difference
{trace_row["difference"]:+.3f}, 95% interval [{trace_row["difference_ci_low"]:+.3f},
{trace_row["difference_ci_high"]:+.3f}]).
"""
    else:
        trace_section = "## Trace-equal weighting\n\nNot available.\n"
    if "length_aware_thresholding" in summary:
        length = summary["length_aware_thresholding"]
        length_section = f"""## Length-aware thresholding

Thresholds fitted within train+validation trace-length bins changed test Process F1 from
{length["global_process_f1"]:.3f} to {length["length_aware_process_f1"]:.3f} and correct rejection
from {length["global_correct_rejection"]:.3f} to
{length["length_aware_correct_rejection"]:.3f}.
"""
    else:
        length_section = (
            "## Length-aware thresholding\n\n"
            f"{availability['length_aware_thresholding']}.\n"
        )
    if "probe_control_paired_intervals" in summary:
        paired_section = (
            "## Probe-versus-control paired intervals\n\n"
            "Whole-trace paired bootstrap intervals are in "
            "`probe_control_paired_intervals.csv`.\n"
        )
    else:
        paired_section = (
            "## Probe-versus-control paired intervals\n\n"
            f"{availability['probe_control_paired_intervals']}.\n"
        )
    if "probe_control_intervals_by_stratum" in availability:
        stratum_section = (
            "## Stratified probe-versus-control intervals\n\n"
            "Hidden-minus-control intervals within final-answer-correctness and source strata "
            "are in `probe_control_intervals_by_final_answer.csv` and "
            "`probe_control_intervals_by_source.csv`.\n"
        )
    else:
        stratum_section = (
            "## Stratified probe-versus-control intervals\n\n"
            f"{availability.get('probe_control_intervals_by_stratum', 'not run')}.\n"
        )
    environment = summary.get("environment", {})
    environment_section = (
        "## Environment\n\n"
        f"Python {environment.get('python', 'unknown')}, numpy "
        f"{environment.get('numpy', 'unknown')}, pandas {environment.get('pandas', 'unknown')}, "
        f"scipy {environment.get('scipy', 'unknown')}, scikit-learn "
        f"{environment.get('scikit_learn', 'unknown')}.\n"
    )
    return f"""# CPU-only follow-up results

This analysis reuses the frozen Experiment 1 test predictions and intervention tables. It does not
load the language model, fit a new hidden-state probe, or inspect activation tensors. The tests are
post-hoc and are reported as robustness analyses rather than preregistered confirmation.

## Temporal alignment

The original detector localized {temporal["observed_exact"]:.1%} of first errors exactly. Circularly
shifting each score trajectory within its trace reduced the null mean to
{temporal["null_exact_mean"]:.1%} (permutation p = {temporal["exact_p_value"]:.4g}). Within-one-step
localization was {temporal["observed_within_1"]:.1%}, compared with a shifted null mean of
{temporal["null_within_1_mean"]:.1%} (p = {temporal["within_1_p_value"]:.4g}). The frozen score is
temporally related to the annotation, although exact localization remains low in absolute terms.

## Onset change and trace-level offsets

The mean score jump at the annotated onset was {matched["onset_jump"]:.3f}. Metadata-matched
transitions from correct traces changed by {matched["placebo_jump"]:.3f}; the paired difference was
{matched["difference"]:.3f} with a
{100 * summary.get("confidence_level", 0.95):.0f}% interval of
[{matched["difference_ci_low"]:.3f}, {matched["difference_ci_high"]:.3f}].

Among erroneous traces whose first error occurred after step 0, pooled AUROC was
{within["raw_auroc"]:.3f}. Subtracting each trace's first-step score gave AUROC
{within["centered_auroc"]:.3f} [{within["centered_ci_low"]:.3f},
{within["centered_ci_high"]:.3f}]. Mean AUROC calculated separately within each eligible trace was
{within["mean_within_trace_auroc"]:.3f}. Stable trace-level offsets do not explain all of the frozen
probe's discrimination.

## Causal assay

The unmodified verdict score had AUROC {causal["baseline_auroc"]:.3f} and specificity
{causal["baseline_specificity"]:.3f}. The smallest approximate effect detectable with 80% power
across the tested doses was {causal["smallest_mde_80pct"]:.5f} verdict-margin units. Statistical
sensitivity does not repair a readout that fails to distinguish valid from invalid boundaries.

{length_section}

{paired_section}

{stratum_section}

{shortcut_section}

{trace_section}

{environment_section}

## Files

The output directory contains the permutation draws, error-aligned trajectory, matched-placebo
results, centered discrimination, subgroup outcomes, causal sensitivity table, a frozen qualitative
audit sample, and a two-panel temporal-validity figure.
"""
