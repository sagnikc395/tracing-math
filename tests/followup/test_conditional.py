"""Tests for the leakage-safe E1 nested-model comparison."""

import numpy as np
import pandas as pd

from tracing_math.experiment1.data import ProcessTrace
from tracing_math.followup.conditional import (
    CONDITIONS,
    conditional_hidden_state_analysis,
)


def _conditional_fixture() -> tuple[np.ndarray, pd.DataFrame, list[ProcessTrace]]:
    rows = []
    traces = []
    hidden_rows = []
    rng = np.random.default_rng(42)
    for partition_index, partition in enumerate(("train", "validation", "test")):
        for trace_index in range(8):
            erroneous = trace_index % 2 == 1
            first_error = 1 if erroneous else -1
            trace_id = f"{partition}-{trace_index}"
            steps = ("begin calculation", "continue arithmetic", "finish result")
            traces.append(
                ProcessTrace(
                    trace_id=trace_id,
                    source="math" if trace_index < 4 else "gsm8k",
                    generator="fixture",
                    problem=f"unique problem {partition_index} {trace_index}",
                    steps=steps,
                    label=first_error,
                    final_answer_correct=not erroneous,
                )
            )
            for step_index in range(3):
                label = int(erroneous and step_index >= first_error)
                rows.append(
                    {
                        "trace_id": trace_id,
                        "problem_group": f"group-{partition_index}-{trace_index}",
                        "partition": partition,
                        "source": "math" if trace_index < 4 else "gsm8k",
                        "generator": "fixture",
                        "step_index": step_index,
                        "step_fraction": (step_index + 1) / 3,
                        "first_error": first_error,
                        "n_steps": 3,
                        "token_count": 30 + trace_index,
                        "final_answer_correct": int(not erroneous),
                        "invalid_so_far": label,
                    }
                )
                vector = rng.normal(scale=0.1, size=(2, 4))
                vector[1, 0] += 3 * label
                hidden_rows.append(vector)
    return np.asarray(hidden_rows, dtype=np.float32), pd.DataFrame(rows), traces


def test_conditional_analysis_fits_all_conditions_and_pairs_test_rows() -> None:
    activations, metadata, traces = _conditional_fixture()
    result = conditional_hidden_state_analysis(
        activations,
        metadata,
        traces,
        layer=1,
        c_values=(0.1, 1.0),
        max_iter=500,
        bootstrap_samples=20,
        confidence_level=0.95,
        seed=42,
        tfidf_min_df=1,
        tfidf_max_features=100,
    )

    assert set(result.metrics["condition"]) == set(CONDITIONS)
    assert result.selection.groupby("condition").size().eq(2).all()
    assert result.selection.groupby("condition")["selected"].sum().eq(1).all()
    assert result.predictions.groupby("condition").size().nunique() == 1
    assert set(result.paired_intervals["comparison"]) == {"N+H - N", "O+H - O"}
    assert {
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
    }.issubset(set(result.paired_intervals["metric"]))
    assert result.metrics.set_index("condition").loc["H", "auroc"] > 0.9
