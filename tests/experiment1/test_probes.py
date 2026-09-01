"""Tests for Experiment 1 probe fitting and evaluation."""

import numpy as np
import pandas as pd
import pytest

from causal_circuits.experiment1.config import AnalysisConfig, ProbeConfig, ProbeFamilyConfig
from causal_circuits.experiment1.probes import (
    binary_metrics,
    calibration_curve_table,
    change_point_metrics,
    choose_threshold,
    fit_layer_probes,
    group_bootstrap_metrics,
)


def test_binary_metrics() -> None:
    result = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert result["auroc"] == 1.0
    assert result["average_precision"] == 1.0
    assert result["balanced_accuracy"] == 1.0
    assert result["brier_score"] == pytest.approx(0.025)
    assert result["expected_calibration_error"] == pytest.approx(0.15)


def test_change_point_metrics_measure_early_late_missed_and_false_alarms() -> None:
    rows = []
    scores = []
    cases = {
        "early": (2, [0.1, 0.8, 0.9]),
        "late": (1, [0.1, 0.2, 0.8]),
        "missed": (1, [0.1, 0.2, 0.3]),
        "correct": (-1, [0.8, 0.2, 0.1]),
    }
    for trace_id, (first_error, trace_scores) in cases.items():
        for step_index, score in enumerate(trace_scores):
            rows.append(
                {"trace_id": trace_id, "step_index": step_index, "first_error": first_error}
            )
            scores.append(score)
    result = change_point_metrics(pd.DataFrame(rows), np.asarray(scores), 0.5)
    assert result["error_detection_rate"] == pytest.approx(2 / 3)
    assert result["error_miss_rate"] == pytest.approx(1 / 3)
    assert result["early_detection_rate"] == pytest.approx(1 / 3)
    assert result["late_detection_rate"] == pytest.approx(1 / 3)
    assert result["correct_false_alarm_rate"] == 1.0
    assert result["mean_signed_localization_error"] == 0.0
    assert result["mean_absolute_localization_error"] == 1.0
    assert result["error_within_1_accuracy"] == pytest.approx(2 / 3)


def test_calibration_table_keeps_empty_bins() -> None:
    result = calibration_curve_table(np.array([0, 1]), np.array([0.1, 0.9]), bins=4)
    assert len(result) == 4
    assert result["n"].sum() == 2


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


def test_change_point_metrics_uses_recorded_step_indices() -> None:
    metadata = pd.DataFrame(
        {"trace_id": ["bad", "bad"], "step_index": [4, 7], "first_error": [7, 7]}
    )
    result = change_point_metrics(metadata, np.array([0.1, 0.9]), threshold=0.5)
    assert result["error_accuracy"] == 1.0


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
        families=(
            ProbeFamilyConfig(name="l2", penalty="l2"),
            ProbeFamilyConfig(name="l1", penalty="l1"),
            ProbeFamilyConfig(name="elastic_net", penalty="elasticnet", l1_ratios=(0.25, 0.5)),
        ),
        diagnostic_targets=("error_onset",),
    )
    result = fit_layer_probes(
        activations,
        pd.DataFrame(rows),
        config,
        seed=42,
        analysis_config=AnalysisConfig(subgroup_min_traces=2),
    )
    assert result.directions.shape == (3, 6)
    assert result.selected_intervention_layer in {0, 1}
    assert set(result.fit_predictions["partition"]) == {"train", "validation"}
    assert set(result.control_predictions["control"]) == {
        "position",
        "current-step TF-IDF",
        "shuffled-label hidden",
    }
    assert len(result.transfer) == 4
    assert set(result.family_metrics["family"]) == {"l2", "l1", "elastic_net"}
    assert set(result.diagnostic_metrics["target"]) == {"error_onset"}
    assert not result.threshold_sensitivity.empty
    assert not result.calibration.empty


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
