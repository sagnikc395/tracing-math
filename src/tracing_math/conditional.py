"""Leakage-safe conditional hidden-state comparisons."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tracing_math.data import ProcessTrace
from tracing_math.localization import assert_process_f1_identity, localization_outcomes
from tracing_math.probes import binary_metrics, change_point_metrics, choose_threshold

CONDITIONS = ("N", "N+H", "O", "O+H", "H")
STRUCTURAL_COLUMNS = ("step_index", "step_fraction", "n_steps", "token_count")
CATEGORICAL_COLUMNS = ("source", "generator")


@dataclass(frozen=True)
class ConditionalAnalysisResult:
    metrics: pd.DataFrame
    selection: pd.DataFrame
    predictions: pd.DataFrame
    paired_intervals: pd.DataFrame
    feature_blocks: dict[str, list[str]]


def conditional_hidden_state_analysis(
    activations: np.ndarray,
    metadata: pd.DataFrame,
    traces: Sequence[ProcessTrace],
    *,
    layer: int,
    c_values: tuple[float, ...],
    max_iter: int,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
    tfidf_min_df: int = 2,
    tfidf_max_features: int = 20_000,
) -> ConditionalAnalysisResult:
    """Fit N, N+H, O, O+H, and H with held-out selection and early fusion."""
    _validate_inputs(activations, metadata, layer, c_values)
    frame = _attach_prefix_text(metadata, traces)
    masks = {
        partition: frame["partition"].eq(partition).to_numpy()
        for partition in ("train", "validation", "test")
    }
    labels = frame["invalid_so_far"].to_numpy(dtype=int)
    for partition, mask in masks.items():
        if np.unique(labels[mask]).size != 2:
            raise ValueError(f"Conditional {partition} split must contain both classes")
    _assert_group_separation(frame)

    hidden = np.asarray(activations[:, layer, :], dtype=np.float32)
    selection_blocks = _feature_blocks(
        frame,
        hidden,
        fit_mask=masks["train"],
        transform_masks=(masks["train"], masks["validation"]),
        tfidf_min_df=tfidf_min_df,
        tfidf_max_features=tfidf_max_features,
    )
    fit_mask = masks["train"] | masks["validation"]
    final_blocks = _feature_blocks(
        frame,
        hidden,
        fit_mask=fit_mask,
        transform_masks=(fit_mask, masks["test"]),
        tfidf_min_df=tfidf_min_df,
        tfidf_max_features=tfidf_max_features,
    )

    metrics_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for condition in CONDITIONS:
        x_train, x_validation = selection_blocks[condition]
        c_value, validation_scores, condition_selection = _select_c(
            x_train,
            labels[masks["train"]],
            x_validation,
            labels[masks["validation"]],
            frame.loc[masks["train"]],
            c_values=c_values,
            max_iter=max_iter,
            seed=seed,
        )
        threshold = choose_threshold(frame.loc[masks["validation"]], validation_scores)
        for row in condition_selection:
            row.update(
                {
                    "condition": condition,
                    "selected": bool(row["c_value"] == c_value),
                    "selected_threshold": threshold if row["c_value"] == c_value else np.nan,
                }
            )
            selection_rows.append(row)

        x_fit, x_test = final_blocks[condition]
        model = _classifier(c_value, max_iter, seed)
        model.fit(
            x_fit,
            labels[fit_mask],
            sample_weight=_trace_equal_weights(frame.loc[fit_mask]),
        )
        scores = model.predict_proba(x_test)[:, 1]
        test = frame.loc[masks["test"]].copy()
        test_labels = labels[masks["test"]]
        row = {
            "condition": condition,
            "layer": layer,
            "c_value": c_value,
            "threshold": threshold,
            "feature_count": int(x_test.shape[1]),
            "training_weighting": "trace_equal",
            **binary_metrics(test_labels, scores, threshold=threshold),
            **change_point_metrics(test, scores, threshold, tolerances=(0, 1, 2)),
        }
        assert_process_f1_identity(row)
        metrics_rows.append(row)
        predictions = test[
            [
                "trace_id",
                "problem_group",
                "source",
                "generator",
                "step_index",
                "step_fraction",
                "first_error",
                "n_steps",
                "token_count",
                "final_answer_correct",
            ]
        ].copy()
        predictions["condition"] = condition
        predictions["label"] = test_labels
        predictions["score"] = scores
        predictions["threshold"] = threshold
        predictions["layer"] = layer
        prediction_rows.append(predictions)

    all_predictions = pd.concat(prediction_rows, ignore_index=True)
    paired = paired_condition_intervals(
        all_predictions,
        comparisons=(("N+H", "N"), ("O+H", "O")),
        samples=bootstrap_samples,
        seed=seed,
        confidence_level=confidence_level,
    )
    feature_blocks = {
        "N": ["prefix_tfidf", "structural_metadata"],
        "N+H": ["prefix_tfidf", "structural_metadata", f"hidden_layer_{layer}"],
        "O": ["prefix_tfidf", "structural_metadata", "final_answer_correct_oracle"],
        "O+H": [
            "prefix_tfidf",
            "structural_metadata",
            "final_answer_correct_oracle",
            f"hidden_layer_{layer}",
        ],
        "H": [f"hidden_layer_{layer}"],
    }
    return ConditionalAnalysisResult(
        metrics=pd.DataFrame(metrics_rows),
        selection=pd.DataFrame(selection_rows),
        predictions=all_predictions,
        paired_intervals=paired,
        feature_blocks=feature_blocks,
    )


def paired_condition_intervals(
    predictions: pd.DataFrame,
    *,
    comparisons: tuple[tuple[str, str], ...],
    samples: int,
    seed: int,
    confidence_level: float,
) -> pd.DataFrame:
    """Whole-trace paired intervals for literal left-minus-right differences."""
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    required = {
        "condition",
        "trace_id",
        "step_index",
        "first_error",
        "label",
        "score",
        "threshold",
    }
    if missing := required.difference(predictions.columns):
        raise ValueError(f"Missing conditional prediction columns: {sorted(missing)}")
    rng = np.random.default_rng(seed)
    tail = (1 - confidence_level) / 2
    rows: list[dict[str, object]] = []
    for left_name, right_name in comparisons:
        left = predictions[predictions["condition"].eq(left_name)]
        right = predictions[predictions["condition"].eq(right_name)]
        merged = left.merge(
            right,
            on=["trace_id", "step_index"],
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        if len(merged) != len(left) or len(left) != len(right):
            raise ValueError(f"Comparison {left_name} versus {right_name} is not paired")
        if not (
            merged["label_left"].eq(merged["label_right"]).all()
            and merged["first_error_left"].eq(merged["first_error_right"]).all()
        ):
            raise ValueError(f"Comparison {left_name} versus {right_name} has label mismatch")
        left_threshold = _single_threshold(left, left_name)
        right_threshold = _single_threshold(right, right_name)
        trace_codes, trace_ids = pd.factorize(merged["trace_id"], sort=False)
        expected, left_predicted, right_predicted = _paired_trace_outcomes(
            merged,
            len(trace_ids),
            left_threshold,
            right_threshold,
        )
        point = _paired_metrics(
            merged["label_left"].to_numpy(dtype=int),
            merged["score_left"].to_numpy(dtype=float),
            merged["score_right"].to_numpy(dtype=float),
            expected,
            left_predicted,
            right_predicted,
            np.ones(len(merged)),
            np.ones(len(trace_ids)),
        )
        draws = {metric: [] for metric in point}
        for _ in range(samples):
            trace_weights = np.bincount(
                rng.integers(0, len(trace_ids), len(trace_ids)),
                minlength=len(trace_ids),
            ).astype(float)
            draw = _paired_metrics(
                merged["label_left"].to_numpy(dtype=int),
                merged["score_left"].to_numpy(dtype=float),
                merged["score_right"].to_numpy(dtype=float),
                expected,
                left_predicted,
                right_predicted,
                trace_weights[trace_codes],
                trace_weights,
            )
            for metric, value in draw.items():
                if np.isfinite(value):
                    draws[metric].append(value)
        for metric, estimate in point.items():
            values = np.asarray(draws[metric], dtype=float)
            ci_low, ci_high = (
                (float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail)))
                if values.size
                else (float("nan"), float("nan"))
            )
            rows.append(
                {
                    "comparison": f"{left_name} - {right_name}",
                    "left_condition": left_name,
                    "right_condition": right_name,
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "favorable_direction": (
                        "lower" if metric in {"log_loss", "brier_score"} else "higher"
                    ),
                    "confidence_level": confidence_level,
                    "bootstrap_unit": "trace",
                    "bootstrap_samples": samples,
                    "n_traces": len(trace_ids),
                }
            )
    return pd.DataFrame(rows)


def _feature_blocks(
    frame: pd.DataFrame,
    hidden: np.ndarray,
    *,
    fit_mask: np.ndarray,
    transform_masks: tuple[np.ndarray, np.ndarray],
    tfidf_min_df: int,
    tfidf_max_features: int,
) -> dict[str, tuple[sparse.csr_matrix, sparse.csr_matrix]]:
    text = TfidfVectorizer(
        ngram_range=(1, 2), min_df=tfidf_min_df, max_features=tfidf_max_features
    )
    text.fit(frame.loc[fit_mask, "prefix_text"].astype(str))
    numeric = StandardScaler().fit(frame.loc[fit_mask, list(STRUCTURAL_COLUMNS)])
    categorical = OneHotEncoder(handle_unknown="ignore").fit(
        frame.loc[fit_mask, list(CATEGORICAL_COLUMNS)]
    )
    oracle = StandardScaler().fit(frame.loc[fit_mask, ["final_answer_correct"]])
    hidden_scaler = StandardScaler().fit(hidden[fit_mask])

    output: dict[str, list[sparse.csr_matrix]] = {name: [] for name in CONDITIONS}
    for mask in transform_masks:
        nuisance = sparse.hstack(
            [
                text.transform(frame.loc[mask, "prefix_text"].astype(str)),
                sparse.csr_matrix(
                    numeric.transform(frame.loc[mask, list(STRUCTURAL_COLUMNS)])
                ),
                categorical.transform(frame.loc[mask, list(CATEGORICAL_COLUMNS)]),
            ],
            format="csr",
        )
        oracle_block = sparse.csr_matrix(
            oracle.transform(frame.loc[mask, ["final_answer_correct"]])
        )
        hidden_block = sparse.csr_matrix(hidden_scaler.transform(hidden[mask]))
        output["N"].append(nuisance)
        output["N+H"].append(sparse.hstack([nuisance, hidden_block], format="csr"))
        output["O"].append(sparse.hstack([nuisance, oracle_block], format="csr"))
        output["O+H"].append(
            sparse.hstack([nuisance, oracle_block, hidden_block], format="csr")
        )
        output["H"].append(hidden_block)
    return {name: (blocks[0], blocks[1]) for name, blocks in output.items()}


def _select_c(
    x_train,
    y_train: np.ndarray,
    x_validation,
    y_validation: np.ndarray,
    train_metadata: pd.DataFrame,
    *,
    c_values: tuple[float, ...],
    max_iter: int,
    seed: int,
) -> tuple[float, np.ndarray, list[dict[str, float]]]:
    rows = []
    scores_by_c = {}
    weights = _trace_equal_weights(train_metadata)
    for c_value in c_values:
        model = _classifier(c_value, max_iter, seed)
        model.fit(x_train, y_train, sample_weight=weights)
        scores = model.predict_proba(x_validation)[:, 1]
        auroc = float(roc_auc_score(y_validation, scores))
        rows.append(
            {
                "c_value": float(c_value),
                "validation_auroc": auroc,
                "validation_average_precision": float(
                    average_precision_score(y_validation, scores)
                ),
            }
        )
        scores_by_c[float(c_value)] = scores
    selected = max(
        rows,
        key=lambda row: (
            row["validation_auroc"],
            -abs(float(np.log10(row["c_value"]))),
        ),
    )
    c_value = float(selected["c_value"])
    return c_value, scores_by_c[c_value], rows


def _classifier(c_value: float, max_iter: int, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=max_iter,
        solver="liblinear",
        random_state=seed,
    )


def _trace_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame["trace_id"].astype(str).value_counts()
    return 1.0 / frame["trace_id"].astype(str).map(counts).to_numpy(dtype=float)


def _attach_prefix_text(
    metadata: pd.DataFrame, traces: Sequence[ProcessTrace]
) -> pd.DataFrame:
    prefix_by_step: dict[tuple[str, int], str] = {}
    for trace in traces:
        blocks = [f"Problem:\n{trace.problem.strip()}\n\nReasoning:"]
        for step_index, step in enumerate(trace.steps):
            blocks.append(f"[Step {step_index}]\n{step.strip()}")
            prefix_by_step[(trace.trace_id, step_index)] = "\n\n".join(blocks)
    keys = list(
        zip(
            metadata["trace_id"].astype(str),
            metadata["step_index"].astype(int),
            strict=True,
        )
    )
    if missing := [key for key in keys if key not in prefix_by_step]:
        raise ValueError(f"Missing source text for activation row {missing[0]}")
    frame = metadata.copy()
    frame["prefix_text"] = [prefix_by_step[key] for key in keys]
    return frame


def _validate_inputs(
    activations: np.ndarray,
    metadata: pd.DataFrame,
    layer: int,
    c_values: tuple[float, ...],
) -> None:
    if activations.ndim != 3 or len(activations) != len(metadata):
        raise ValueError("Activations must align with metadata and have three dimensions")
    if not 0 <= layer < activations.shape[1]:
        raise ValueError(f"Selected layer {layer} is outside the activation tensor")
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("C values must be positive")
    required = {
        "trace_id",
        "problem_group",
        "partition",
        "source",
        "generator",
        "step_index",
        "step_fraction",
        "first_error",
        "n_steps",
        "token_count",
        "final_answer_correct",
        "invalid_so_far",
    }
    if missing := required.difference(metadata.columns):
        raise ValueError(f"Missing conditional-analysis columns: {sorted(missing)}")
    if metadata.duplicated(["trace_id", "step_index"]).any():
        raise ValueError("Activation rows must be unique by trace and step")


def _assert_group_separation(metadata: pd.DataFrame) -> None:
    counts = metadata.groupby("problem_group")["partition"].nunique()
    if (counts > 1).any():
        group = str(counts[counts > 1].index[0])
        raise ValueError(f"Problem group {group} crosses data partitions")


def _single_threshold(frame: pd.DataFrame, condition: str) -> float:
    if frame.empty or frame["threshold"].nunique() != 1:
        raise ValueError(f"Condition {condition!r} must have one held-out threshold")
    return float(frame["threshold"].iloc[0])


def _paired_trace_outcomes(
    frame: pd.DataFrame,
    trace_count: int,
    left_threshold: float,
    right_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    metadata = frame[["trace_id", "step_index", "first_error_left"]].rename(
        columns={"first_error_left": "first_error"}
    )
    expected, left = localization_outcomes(
        metadata, frame["score_left"].to_numpy(dtype=float), left_threshold
    )
    right_expected, right = localization_outcomes(
        metadata, frame["score_right"].to_numpy(dtype=float), right_threshold
    )
    if len(expected) != trace_count or not np.array_equal(expected, right_expected):
        raise ValueError("Paired conditions produced inconsistent trace outcomes")
    return expected, left, right


def _paired_metrics(
    labels: np.ndarray,
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    expected: np.ndarray,
    left_predicted: np.ndarray,
    right_predicted: np.ndarray,
    row_weights: np.ndarray,
    trace_weights: np.ndarray,
) -> dict[str, float]:
    left_binary = _weighted_binary_metrics(labels, left_scores, row_weights)
    right_binary = _weighted_binary_metrics(labels, right_scores, row_weights)
    left_process = _weighted_process_metrics(
        expected, left_predicted, trace_weights
    )
    right_process = _weighted_process_metrics(
        expected, right_predicted, trace_weights
    )
    return {
        metric: left_binary[metric] - right_binary[metric]
        for metric in left_binary
    } | {
        metric: left_process[metric] - right_process[metric]
        for metric in left_process
    }


def _weighted_binary_metrics(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    positive_weight = float(weights[labels == 1].sum())
    negative_weight = float(weights[labels == 0].sum())
    if positive_weight == 0 or negative_weight == 0:
        auroc = average_precision = float("nan")
    else:
        auroc = float(roc_auc_score(labels, scores, sample_weight=weights))
        average_precision = float(
            average_precision_score(labels, scores, sample_weight=weights)
        )
    probabilities = np.clip(scores, 1e-12, 1 - 1e-12)
    denominator = float(weights.sum())
    return {
        "auroc": auroc,
        "average_precision": average_precision,
        "log_loss": float(
            np.dot(
                -(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities)),
                weights,
            )
            / denominator
        ),
        "brier_score": float(np.dot((scores - labels) ** 2, weights) / denominator),
    }


def _weighted_process_metrics(
    expected: np.ndarray, predicted: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    erroneous = expected >= 0
    correct = ~erroneous

    def weighted_mean(values: np.ndarray, mask: np.ndarray) -> float:
        denominator = float(weights[mask].sum())
        if denominator == 0:
            return float("nan")
        return float(np.dot(values[mask].astype(float), weights[mask]) / denominator)

    error_exact = weighted_mean(predicted == expected, erroneous)
    correct_rejection = weighted_mean(predicted == -1, correct)
    denominator = error_exact + correct_rejection
    metrics = {
        "error_exact": error_exact,
        "correct_rejection": correct_rejection,
        "process_f1": (
            2 * error_exact * correct_rejection / denominator if denominator > 0 else 0.0
        ),
        "error_within_1_accuracy": weighted_mean(
            (predicted >= 0) & (np.abs(predicted - expected) <= 1), erroneous
        ),
        "error_within_2_accuracy": weighted_mean(
            (predicted >= 0) & (np.abs(predicted - expected) <= 2), erroneous
        ),
        "complete_accuracy": weighted_mean(
            predicted == expected, np.ones(len(expected), dtype=bool)
        ),
    }
    return metrics
