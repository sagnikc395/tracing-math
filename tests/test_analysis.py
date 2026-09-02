"""Tests for frozen-artifact analysis helpers."""

import numpy as np
import pandas as pd

from tracing_math.analysis import (
    TraceSeries,
    _bootstrap_outcome_summaries,
    _outcome_summary,
    _paired_trace_predictions,
    _probe_minus_control_metrics,
    _trace_metric_draws,
    _trace_metrics,
    _weighted_probe_minus_control_metrics,
    centered_discrimination,
    length_aware_threshold_analysis,
    matched_placebo_analysis,
    paired_probe_control_intervals,
    shortcut_control_analysis,
    temporal_randomization_test,
    trace_equal_weight_sensitivity,
)
from tracing_math.data import ProcessTrace
from tracing_math.localization import first_crossing


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
        samples=515,
        seed=42,
        confidence_level=0.95,
    )
    exact = summary.set_index("metric").loc["error_exact"]
    assert exact["observed"] == 1.0
    assert exact["null_mean"] < 1.0
    assert len(draws) == 515
    reference_rng = np.random.default_rng(42)
    expected = np.asarray([trace.first_error for trace in traces])
    reference_rows = []
    for sample in range(515):
        predicted = np.asarray(
            [
                first_crossing(
                    np.roll(
                        trace.scores,
                        int(reference_rng.integers(0, len(trace.scores)))
                        if len(trace.scores) > 1
                        else 0,
                    ),
                    0.5,
                )
                for trace in traces
            ]
        )
        reference_rows.append({"sample": sample, **_trace_metrics(expected, predicted)})
    reference = pd.DataFrame(reference_rows)[draws.columns]
    pd.testing.assert_frame_equal(draws, reference, check_exact=False, rtol=1e-15)


def test_vectorized_trace_metric_draws_match_single_draw_metrics() -> None:
    expected = np.asarray([-1, 1, 2, 1])
    predicted = np.asarray(
        [
            [-1, 1, 2, -1],
            [0, 0, 3, 2],
            [-1, -1, -1, -1],
        ]
    )
    vectorized = _trace_metric_draws(expected, predicted)
    for draw_index, row in vectorized.iterrows():
        reference = _trace_metrics(expected, predicted[draw_index])
        for metric, value in reference.items():
            np.testing.assert_allclose(row[metric], value, equal_nan=True)


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


def test_weighted_probe_control_draw_matches_duplicated_trace_rows() -> None:
    rows = []
    for trace_id, first_error, probe_scores, control_scores in (
        ("a", -1, [0.1, 0.2], [0.3, 0.7]),
        ("b", 1, [0.2, 0.9], [0.6, 0.7]),
        ("c", 2, [0.1, 0.4, 0.8], [0.2, 0.8, 0.9]),
    ):
        for step_index, (probe_score, control_score) in enumerate(
            zip(probe_scores, control_scores, strict=True)
        ):
            rows.append(
                {
                    "trace_id": trace_id,
                    "step_index": step_index,
                    "probe_error": first_error,
                    "probe_label": int(first_error >= 0 and step_index >= first_error),
                    "probe_score": probe_score,
                    "control_score": control_score,
                }
            )
    frame = pd.DataFrame(rows)
    trace_codes, trace_ids = pd.factorize(frame["trace_id"], sort=False)
    trace_weights = np.asarray([2, 0, 1])
    expected, probe_predicted, control_predicted = _paired_trace_predictions(
        frame,
        trace_codes,
        trace_count=len(trace_ids),
        probe_threshold=0.5,
        control_threshold=0.5,
    )
    weighted = _weighted_probe_minus_control_metrics(
        labels=frame["probe_label"].to_numpy(),
        probe_scores=frame["probe_score"].to_numpy(),
        control_scores=frame["control_score"].to_numpy(),
        probe_threshold=0.5,
        control_threshold=0.5,
        row_weights=trace_weights[trace_codes],
        trace_weights=trace_weights,
        expected=expected,
        probe_predicted=probe_predicted,
        control_predicted=control_predicted,
    )
    duplicated = pd.concat(
        [
            frame.loc[frame["trace_id"].eq(trace_id)].assign(trace_id=f"draw-{draw_index}")
            for draw_index, trace_id in enumerate(("a", "a", "c"))
        ],
        ignore_index=True,
    )
    duplicated["control_label"] = duplicated["probe_label"]
    duplicated["control_error"] = duplicated["probe_error"]
    reference = _probe_minus_control_metrics(duplicated, 0.5, 0.5)
    assert weighted == reference


def test_vectorized_subgroup_bootstrap_matches_row_resampling() -> None:
    frame = pd.DataFrame(
        {
            "first_error": [-1, -1, 1, 2],
            "predicted_first_error": [-1, 0, 1, -1],
        }
    )
    vectorized = _bootstrap_outcome_summaries(
        frame, samples=20, rng=np.random.default_rng(42)
    )
    reference_rng = np.random.default_rng(42)
    reference = pd.DataFrame(
        [
            _outcome_summary(
                frame.iloc[reference_rng.integers(0, len(frame), len(frame))]
            )
            for _ in range(20)
        ]
    )
    pd.testing.assert_frame_equal(vectorized, reference)


def test_shortcut_controls_use_prefix_text_and_frozen_partitions() -> None:
    traces = []
    fit_rows = []
    test_rows = []
    for partition in ("train", "validation", "test"):
        for status, label in (("good", -1), ("bad", 1)):
            trace_id = f"{partition}-{status}"
            steps = ("start neutral", "wrong contradiction" if label >= 0 else "valid follows")
            traces.append(
                ProcessTrace(
                    trace_id=trace_id,
                    source="math",
                    generator="generator",
                    problem=f"problem {trace_id}",
                    steps=steps,
                    label=label,
                    final_answer_correct=label < 0,
                )
            )
            for step_index in range(2):
                row = {
                    "trace_id": trace_id,
                    "source": "math",
                    "generator": "generator",
                    "partition": partition,
                    "step_index": step_index,
                    "step_fraction": (step_index + 1) / 2,
                    "first_error": label,
                    "n_steps": 2,
                    "token_count": 20,
                    "final_answer_correct": int(label < 0),
                    "label": int(label >= 0 and step_index >= label),
                }
                (test_rows if partition == "test" else fit_rows).append(row)

    controls, predictions = shortcut_control_analysis(
        pd.DataFrame(fit_rows),
        pd.DataFrame(test_rows).drop(columns="partition"),
        traces,
        seed=42,
    )

    assert set(controls["control"]) == {
        "prefix TF-IDF",
        "structural metadata",
        "metadata plus final outcome",
        "joint text + metadata",
        "joint text + metadata + outcome",
    }
    assert controls["c_value"].isin([0.01, 0.1, 1.0, 10.0]).all()
    assert (controls["training_weighting"] == "trace_equal").all()
    assert len(predictions) == 5 * len(test_rows)
    assert predictions.groupby("control")["threshold"].nunique().eq(1).all()


def test_trace_equal_weighting_reports_paired_sensitivity() -> None:
    predictions = pd.DataFrame(
        {
            "trace_id": ["short", "long", "long", "long", "long"],
            "label": [0, 0, 1, 1, 1],
            "score": [0.2, 0.8, 0.9, 0.7, 0.6],
        }
    )
    result = trace_equal_weight_sensitivity(
        predictions,
        threshold=0.5,
        samples=50,
        seed=42,
        confidence_level=0.95,
    ).set_index("metric")

    assert set(result.index) == {"auroc", "average_precision", "step_f1"}
    assert result.loc["auroc", "trace_equal_weighted"] != result.loc[
        "auroc", "boundary_weighted"
    ]
