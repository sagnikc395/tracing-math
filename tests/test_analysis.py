import numpy as np
import pandas as pd
import pytest

from causal_circuits.analysis import (
    binary_metrics,
    change_point_metrics,
    choose_threshold,
    fit_layer_probes,
    group_bootstrap_metrics,
)
from causal_circuits.config import ProbeConfig


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


def test_layer_probe_pipeline_smoke() -> None:
    rows = []
    labels = []
    partitions = ["train"] * 12 + ["validation"] * 12 + ["test"] * 12
    for index, partition in enumerate(partitions):
        label = index % 2
        labels.append(label)
        rows.append(
            {
                "trace_id": f"trace-{index}",
                "source": "math" if index % 4 < 2 else "gsm8k",
                "generator": "test-model",
                "partition": partition,
                "step_index": 0,
                "step_fraction": 1.0,
                "step_text": "wrong arithmetic" if label else "valid arithmetic",
                "first_error": 0 if label else -1,
                "has_error_trace": label,
                "invalid_so_far": label,
            }
        )
    rng = np.random.default_rng(42)
    activations = rng.normal(size=(36, 3, 6)).astype(np.float32)
    activations[:, 1:, 0] += np.asarray(labels)[:, None] * 3
    config = ProbeConfig(
        target="invalid_so_far",
        train_fraction=0.6,
        validation_fraction=0.2,
        test_fraction=0.2,
        c_values=(0.1, 1.0),
        max_iter=200,
        bootstrap_samples=0,
        pca_dimensions=(1, 2),
    )
    result = fit_layer_probes(activations, pd.DataFrame(rows), config, seed=42)
    assert result.directions.shape == (3, 6)
    assert result.selected_intervention_layer in {0, 1}
    assert len(result.transfer) == 4


def test_bootstrap_samples_whole_traces() -> None:
    metadata = pd.DataFrame(
        {
            "trace_id": ["bad", "bad", "good", "good"],
            "step_index": [0, 1, 0, 1],
            "first_error": [1, 1, -1, -1],
        }
    )
    result = group_bootstrap_metrics(
        metadata,
        np.array([0, 1, 0, 0]),
        np.array([0.1, 0.9, 0.1, 0.2]),
        0.5,
        samples=10,
        seed=42,
    )
    assert len(result) == 10
    assert set(result.columns) >= {"auroc", "process_f1", "first_error_exact"}
