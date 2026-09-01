"""Tests for the CPU-only follow-up analyses."""

import numpy as np
import pandas as pd

from tracing_math.followup.analysis import (
    TraceSeries,
    centered_discrimination,
    length_aware_threshold_analysis,
    matched_placebo_analysis,
    paired_probe_control_intervals,
    temporal_randomization_test,
)


def _trace(
    trace_id: str,
    scores: list[float],
    first_error: int,
    *,
    source: str = "math",
    generator: str = "generator",
) -> TraceSeries:
    return TraceSeries(
        trace_id=trace_id,
        source=source,
        generator=generator,
        first_error=first_error,
        n_steps=len(scores),
        token_count=100,
        final_answer_correct=0,
        scores=np.asarray(scores),
    )


def test_temporal_randomization_uses_frozen_within_trace_scores() -> None:
    traces = [
        _trace("bad-a", [0.1, 0.2, 0.9, 0.95], 2),
        _trace("bad-b", [0.1, 0.8, 0.9], 1),
        _trace("good", [0.1, 0.2, 0.3], -1),
    ]
    summary, draws = temporal_randomization_test(
        traces,
        threshold=0.5,
        samples=50,
        seed=42,
        confidence_level=0.95,
    )
    exact = summary.set_index("metric").loc["first_error_exact"]
    assert exact["observed"] == 1.0
    assert exact["null_mean"] < 1.0
    assert len(draws) == 50


def test_matched_placebo_detects_onset_jump_beyond_correct_trace_drift() -> None:
    traces = [
        _trace("bad-a", [0.1, 0.2, 0.9], 2),
        _trace("bad-b", [0.2, 0.8, 0.9], 1),
        _trace("good", [0.1, 0.15, 0.2], -1),
    ]
    result = matched_placebo_analysis(
        traces,
        samples=50,
        seed=42,
        confidence_level=0.95,
    ).set_index("metric")
    assert result.loc["real_jump", "estimate"] > 0.5
    assert result.loc["placebo_jump", "estimate"] < 0.1
    assert result.loc["paired_difference", "ci_low"] > 0


def test_first_step_centering_retains_within_trace_discrimination() -> None:
    rows = []
    for trace_id, offset in (("a", 0.0), ("b", 2.0), ("c", -1.0)):
        for step, score in enumerate((0.1, 0.2, 0.9)):
            rows.append(
                {
                    "trace_id": trace_id,
                    "step_index": step,
                    "first_error": 2,
                    "has_error_trace": 1,
                    "label": int(step >= 2),
                    "score": score + offset,
                }
            )
    result = centered_discrimination(
        pd.DataFrame(rows),
        samples=20,
        seed=42,
        confidence_level=0.95,
    ).set_index("metric")
    assert result.loc["pooled_first_step_centered_auroc", "estimate"] == 1.0
    assert result.loc["mean_within_trace_auroc", "estimate"] == 1.0


def _prediction_rows(partition: str) -> list[dict[str, object]]:
    rows = []
    specifications = [
        ("short-good", 2, -1, [0.1, 0.2]),
        ("short-bad", 2, 1, [0.1, 0.7]),
        ("long-good", 4, -1, [0.6, 0.7, 0.7, 0.7]),
        ("long-bad", 4, 2, [0.6, 0.7, 0.9, 0.9]),
    ]
    for trace_id, n_steps, first_error, scores in specifications:
        for step_index, score in enumerate(scores):
            rows.append(
                {
                    "trace_id": f"{partition}-{trace_id}",
                    "partition": partition,
                    "step_index": step_index,
                    "first_error": first_error,
                    "n_steps": n_steps,
                    "label": int(first_error >= 0 and step_index >= first_error),
                    "score": score,
                }
            )
    return rows


def test_length_thresholds_are_fit_without_test_labels_and_improve_rejection() -> None:
    fit = pd.DataFrame(_prediction_rows("train") + _prediction_rows("validation"))
    test = pd.DataFrame(_prediction_rows("test"))
    thresholds, results = length_aware_threshold_analysis(
        fit, test, global_threshold=0.5
    )
    comparison = results.set_index("metric")
    assert set(thresholds["status"]) == {"fit"}
    assert comparison.loc["correct_rejection", "length_aware"] == 1.0
    assert comparison.loc["process_f1", "length_aware"] > comparison.loc[
        "process_f1", "global_threshold"
    ]


def test_probe_control_bootstrap_is_paired_by_trace() -> None:
    probe_rows = []
    control_rows = []
    for index in range(20):
        first_error = 1 if index % 2 else -1
        for step_index in range(2):
            label = int(first_error >= 0 and step_index >= first_error)
            base = {
                "trace_id": f"trace-{index}",
                "step_index": step_index,
                "first_error": first_error,
                "label": label,
            }
            probe_rows.append({**base, "score": 0.9 if label else 0.1})
            control_rows.append(
                {**base, "control": "surface", "score": 0.5, "threshold": 0.5}
            )
    result = paired_probe_control_intervals(
        pd.DataFrame(probe_rows),
        pd.DataFrame(control_rows),
        probe_threshold=0.5,
        samples=100,
        seed=42,
        confidence_level=0.95,
    ).set_index("metric")
    assert result.loc["auroc", "probe_minus_control"] == 0.5
    assert result.loc["auroc", "ci_low"] == 0.5
