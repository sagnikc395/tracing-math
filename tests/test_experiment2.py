from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from causal_circuits.experiment2_analysis import (
    bootstrap_onset_jump,
    choose_binary_threshold,
    domain_calibration_analysis,
    fit_surface_metadata_control,
)
from causal_circuits.experiment2_causal import (
    balanced_boundary_sample,
    counterbalance_interventions,
    counterbalance_verdict_rows,
    verdict_mappings,
)
from causal_circuits.experiment2_config import Experiment2Config
from causal_circuits.experiment2_semantic import semantic_token_before_marker


def test_experiment2_config_and_path_overrides(tmp_path) -> None:
    config = Experiment2Config.from_yaml("configs/experiment2.yaml")
    overridden = config.with_paths(
        experiment1_dir=tmp_path / "experiment1",
        output_dir=tmp_path / "experiment2",
        data_path=tmp_path / "data.jsonl",
    )
    assert overridden.experiment1_dir == tmp_path / "experiment1"
    assert overridden.output_dir == tmp_path / "experiment2"
    assert overridden.data.path == tmp_path / "data.jsonl"
    assert overridden.output_dir != config.output_dir


def test_semantic_token_before_marker_skips_whitespace() -> None:
    rendered = "Step text\n\n<<END_STEP_0>>"
    offsets = [(0, 4), (4, 9), (9, 11), (11, 25)]
    assert semantic_token_before_marker(rendered, "<<END_STEP_0>>", offsets) == 1


def test_semantic_token_requires_unique_marker() -> None:
    rendered = "x <<END_STEP_0>> y <<END_STEP_0>>"
    with pytest.raises(ValueError, match="exactly one"):
        semantic_token_before_marker(rendered, "<<END_STEP_0>>", [(0, 1)])


def test_counterbalanced_verdict_margin_cancels_mapping_bias() -> None:
    frame = pd.DataFrame(
        [
            {
                "partition": "test",
                "trace_id": "a",
                "source": "math",
                "generator": "g",
                "step_index": 1,
                "first_error": 1,
                "invalid_so_far": 1,
                "mapping": mapping,
                "margin": margin,
                "probability_invalid": 0.5,
                "margin_prediction_correct": int(margin >= 0),
                "greedy_matches_expected": int(margin >= 0),
            }
            for mapping, margin in (("fixed", 2.0), ("reversed", -1.0))
        ]
    )
    result = counterbalance_verdict_rows(frame)
    assert len(result) == 1
    assert result.loc[0, "margin"] == pytest.approx(0.5)
    assert result.loc[0, "margin_prediction_correct"] == 1
    assert result.loc[0, "both_mappings_correct"] == 0
    assert verdict_mappings(("A", "B")) == [
        ("fixed", "A", "B"),
        ("reversed", "B", "A"),
    ]


def test_counterbalance_interventions_averages_mappings() -> None:
    common = {
        "trace_id": "a",
        "source": "math",
        "generator": "g",
        "step_index": 1,
        "first_error": 1,
        "invalid_so_far": 1,
        "direction_type": "learned_probe",
        "layer": 2,
        "alpha": 1.0,
        "baseline_margin": 0.2,
    }
    frame = pd.DataFrame(
        [
            {**common, "mapping": "fixed", "margin": 0.5, "delta_margin": 0.3},
            {**common, "mapping": "reversed", "margin": 0.3, "delta_margin": 0.1},
        ]
    )
    result = counterbalance_interventions(frame)
    assert result.loc[0, "margin"] == pytest.approx(0.4)
    assert result.loc[0, "delta_margin"] == pytest.approx(0.2)


def test_balanced_boundary_sample_uses_distinct_traces() -> None:
    rows = []
    for label in (0, 1):
        for trace_index in range(6):
            for step in range(2):
                rows.append(
                    {
                        "partition": "test",
                        "trace_id": f"{label}-{trace_index}",
                        "invalid_so_far": label,
                        "step_index": step,
                    }
                )
    sample = balanced_boundary_sample(
        pd.DataFrame(rows),
        partition="test",
        per_class=4,
        seed=42,
    )
    assert len(sample) == 8
    assert sample["trace_id"].nunique() == 8
    assert sample["invalid_so_far"].value_counts().to_dict() == {0: 4, 1: 4}


def test_binary_threshold_and_onset_jump_bootstrap() -> None:
    base = Experiment2Config.from_yaml("configs/experiment2.yaml")
    config = replace(
        base,
        analysis=replace(base.analysis, bootstrap_samples=20, threshold_points=11),
    )
    threshold = choose_binary_threshold(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        config,
    )
    assert 0.2 < threshold <= 0.8
    predictions = pd.DataFrame(
        [
            {"trace_id": trace, "step_index": step, "first_error": 1, "score": score}
            for trace, scores in (("a", [0.1, 0.7]), ("b", [0.2, 0.8]))
            for step, score in enumerate(scores)
        ]
    )
    result = bootstrap_onset_jump(predictions, config)
    assert result.loc[0, "n_traces"] == 2
    assert result.loc[0, "mean_onset_jump"] == pytest.approx(0.6)
    assert result.loc[0, "ci_low"] > 0


def test_domain_calibration_and_combined_surface_control_smoke() -> None:
    base = Experiment2Config.from_yaml("configs/experiment2.yaml")
    config = replace(
        base,
        analysis=replace(
            base.analysis,
            bootstrap_samples=5,
            threshold_points=11,
            c_values=(0.1,),
        ),
    )
    rows = []
    labels = []
    for source in ("math", "gsm8k"):
        for partition in ("train", "validation", "test"):
            for index in range(6):
                label = index % 2
                labels.append(label)
                rows.append(
                    {
                        "trace_id": f"{source}-{partition}-{index}",
                        "partition": partition,
                        "source": source,
                        "generator": "generator-a" if index < 3 else "generator-b",
                        "step_index": 0,
                        "step_fraction": 1.0,
                        "n_steps": 1,
                        "token_count": 100 + index,
                        "step_text": "invalid arithmetic" if label else "valid arithmetic",
                        "first_error": 0 if label else -1,
                        "has_error_trace": label,
                        "invalid_so_far": label,
                    }
                )
    rng = np.random.default_rng(42)
    hidden = rng.normal(size=(len(rows), 5)).astype(np.float32)
    hidden[:, 0] += np.asarray(labels) * 3
    metadata = pd.DataFrame(rows)
    transfer, cosine = domain_calibration_analysis(hidden, metadata, config)
    assert len(transfer) == 4
    assert len(cosine) == 4
    assert set(transfer.columns) >= {
        "auroc_ci_low",
        "source_threshold_process_f1",
        "target_threshold_process_f1",
    }
    predictions, metrics = fit_surface_metadata_control(metadata, config)
    assert len(predictions) == 12
    assert metrics.loc[0, "auroc"] >= 0.5
