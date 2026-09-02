"""Tests for tuned joint shortcut controls and stratified paired intervals."""

import numpy as np
import pandas as pd

from tracing_math.analysis import (
    _trace_equal_train_weights,
    correct_trace_overlap_table,
    main_comparison_table,
    shortcut_control_analysis,
    stratified_probe_control_intervals,
)
from tracing_math.data import ProcessTrace


def _shortcut_fixture() -> tuple[list[ProcessTrace], pd.DataFrame, pd.DataFrame]:
    traces = []
    fit_rows = []
    test_rows = []
    for partition in ("train", "validation", "test"):
        for status, label in (("good", -1), ("bad", 1)):
            trace_id = f"{partition}-{status}"
            steps = (
                "start neutral",
                "wrong contradiction" if label >= 0 else "valid follows",
            )
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
    return traces, pd.DataFrame(fit_rows), pd.DataFrame(test_rows)


def test_shortcut_controls_add_tuned_joint_baselines() -> None:
    traces, fit_rows, test_rows = _shortcut_fixture()

    controls, predictions = shortcut_control_analysis(
        fit_rows,
        test_rows.drop(columns="partition"),
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
    assert predictions.groupby("control")["threshold"].nunique().eq(1).all()
    assert len(predictions) == 5 * len(test_rows)


def test_generated_comparison_table_uses_consistent_metric_schema() -> None:
    traces, fit_rows, test_rows = _shortcut_fixture()
    test = test_rows.drop(columns="partition")
    controls, control_predictions = shortcut_control_analysis(
        fit_rows, test, traces, seed=42
    )
    hidden = test.copy()
    hidden["score"] = np.where(hidden["label"].eq(1), 0.9, 0.1)

    table = main_comparison_table(hidden, controls, hidden_threshold=0.5)
    expected = 2 * table["error_exact"] * table["correct_rejection"] / (
        table["error_exact"] + table["correct_rejection"]
    )
    np.testing.assert_allclose(table["process_f1"], expected)
    assert {
        "error_exact",
        "correct_rejection",
        "process_f1",
        "complete_accuracy",
    }.issubset(table.columns)

    overlap = correct_trace_overlap_table(
        hidden, control_predictions, hidden_threshold=0.5
    )
    assert set(overlap["outcome"]) == {
        "both_reject",
        "hidden_only_alarm",
        "nuisance_only_alarm",
        "both_alarm",
    }
    assert overlap.groupby("control")["count"].sum().eq(1).all()


def test_joint_control_with_outcome_uses_final_answer_field() -> None:
    traces, fit_rows, test_rows = _shortcut_fixture()

    controls, predictions = shortcut_control_analysis(
        fit_rows,
        test_rows.drop(columns="partition"),
        traces,
        seed=42,
    )

    outcome_row = controls.set_index("control").loc["joint text + metadata + outcome"]
    plain_row = controls.set_index("control").loc["joint text + metadata"]
    assert predictions[predictions["control"] == "joint text + metadata + outcome"][
        "final_answer_correct"
    ].notna().all()
    # The fixture ties outcome to the label, so the oracle baseline should not
    # underperform the joint model without the outcome field.
    assert outcome_row["auroc"] >= plain_row["auroc"] - 1e-9


def test_trace_equal_weights_invert_boundary_counts() -> None:
    frame = pd.DataFrame(
        {"trace_id": ["a", "a", "a", "b"], "value": [1, 1, 1, 1]}
    )
    weights = _trace_equal_train_weights(frame, np.ones(len(frame), dtype=bool))
    assert np.allclose(weights, [1 / 3, 1 / 3, 1 / 3, 1.0])


def test_stratified_intervals_split_by_stratum() -> None:
    probe_rows = []
    control_rows = []
    for stratum in (0, 1):
        for index in range(10):
            first_error = 1 if index % 2 else -1
            for step_index in range(2):
                label = int(first_error >= 0 and step_index >= first_error)
                base = {
                    "trace_id": f"{stratum}-trace-{index}",
                    "step_index": step_index,
                    "first_error": first_error,
                    "final_answer_correct": stratum,
                    "label": label,
                }
                probe_rows.append({**base, "score": 0.9 if label else 0.1})
                control_rows.append(
                    {**base, "control": "surface", "score": 0.5, "threshold": 0.5}
                )
    result = stratified_probe_control_intervals(
        pd.DataFrame(probe_rows),
        pd.DataFrame(control_rows),
        stratum_column="final_answer_correct",
        probe_threshold=0.5,
        samples=20,
        seed=42,
        confidence_level=0.95,
    )
    assert set(result["stratum"]) == {
        "final_answer_correct=0",
        "final_answer_correct=1",
    }
    assert (result["metric"] == "auroc").groupby(result["stratum"]).sum().eq(1).all()
