"""Shared first-error localization calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def first_crossing(
    scores: np.ndarray,
    threshold: float,
    step_indices: np.ndarray | None = None,
) -> int:
    """Return the first step at or above ``threshold``, or ``-1`` when absent."""
    crossings = np.flatnonzero(np.asarray(scores) >= threshold)
    if not crossings.size:
        return -1
    return int(crossings[0] if step_indices is None else step_indices[crossings[0]])


def localization_outcomes(
    metadata: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return annotated and predicted first-error steps for each trace."""
    frame = metadata[["trace_id", "step_index", "first_error"]].copy()
    frame["score"] = np.asarray(scores, dtype=float)
    expected: list[int] = []
    predicted: list[int] = []
    for _, trace_rows in frame.groupby("trace_id", sort=False):
        ordered = trace_rows.sort_values("step_index")
        expected.append(int(ordered["first_error"].iloc[0]))
        predicted.append(
            first_crossing(
                ordered["score"].to_numpy(),
                threshold,
                ordered["step_index"].to_numpy(),
            )
        )
    return np.asarray(expected), np.asarray(predicted)


def localization_metrics(
    expected: np.ndarray,
    predicted: np.ndarray,
    tolerances: tuple[int, ...] = (0, 1, 2),
) -> dict[str, float]:
    """Calculate detection and localization metrics from trace-level outcomes."""
    expected = np.asarray(expected, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    erroneous = expected >= 0
    correct = ~erroneous
    error_predictions = predicted[erroneous]
    error_expected = expected[erroneous]
    detected = erroneous & (predicted >= 0)
    localization_error = predicted[detected] - expected[detected]
    error_accuracy = _safe_mean(error_predictions == error_expected)
    correct_accuracy = _safe_mean(predicted[correct] == -1)
    denominator = error_accuracy + correct_accuracy
    output = {
        "error_accuracy": error_accuracy,
        "correct_accuracy": correct_accuracy,
        "process_f1": (
            2 * error_accuracy * correct_accuracy / denominator if denominator > 0 else 0.0
        ),
        "first_error_exact": _safe_mean(predicted == expected),
        "error_detection_rate": _safe_mean(error_predictions >= 0),
        "error_miss_rate": _safe_mean(error_predictions < 0),
        "correct_false_alarm_rate": _safe_mean(predicted[correct] >= 0),
        "early_detection_rate": _safe_mean(
            (error_predictions >= 0) & (error_predictions < error_expected)
        ),
        "on_time_detection_rate": _safe_mean(error_predictions == error_expected),
        "late_detection_rate": _safe_mean(error_predictions > error_expected),
        "mean_signed_localization_error": _safe_mean(localization_error),
        "mean_absolute_localization_error": _safe_mean(np.abs(localization_error)),
    }
    for tolerance in sorted(set(tolerances)):
        within = (error_predictions >= 0) & (
            np.abs(error_predictions - error_expected) <= tolerance
        )
        output[f"error_within_{tolerance}_accuracy"] = _safe_mean(within)
    return output


def outcome_label(expected: int, predicted: int) -> str:
    """Classify a trace-level localization outcome."""
    if expected < 0:
        return "correct_rejection" if predicted < 0 else "false_alarm"
    if predicted < 0:
        return "miss"
    if predicted < expected:
        return "early"
    if predicted == expected:
        return "on_time"
    return "late"


def quantile_interval(values: np.ndarray, tail: float) -> tuple[float, float]:
    """Return lower and upper quantiles for a two-sided interval."""
    return float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail))


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")
