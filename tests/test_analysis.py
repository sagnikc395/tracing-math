import numpy as np
import pandas as pd
import pytest

from causal_circuits.analysis import binary_metrics, change_point_metrics, choose_threshold


def test_binary_metrics() -> None:
    result = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert result["auroc"] == 1.0
    assert result["average_precision"] == 1.0


def test_change_point_metrics_recover_first_error_and_correct_trace() -> None:
    metadata = pd.DataFrame(
        {
            "trace_id": ["bad", "bad", "bad", "good", "good"],
            "step_index": [0, 1, 2, 0, 1],
            "first_error": [1, 1, 1, -1, -1],
        }
    )
    scores = np.array([0.1, 0.8, 0.9, 0.1, 0.2])
    result = change_point_metrics(metadata, scores, threshold=0.5)
    assert result["error_accuracy"] == 1.0
    assert result["correct_accuracy"] == 1.0
    assert result["process_f1"] == 1.0
    assert choose_threshold(metadata, scores) == pytest.approx(0.5, abs=0.45)
