"""Tests for Experiment 1 orchestration and artifact generation."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from causal_circuits.experiment1 import pipeline
from causal_circuits.experiment1.config import ExperimentConfig
from causal_circuits.experiment1.pipeline import plot_artifacts


def test_plot_artifacts_creates_only_essential_paper_figures(tmp_path) -> None:
    base_config = ExperimentConfig.from_yaml("configs/experiment.yaml")
    config = replace(base_config, extraction=replace(base_config.extraction, output_dir=tmp_path))
    probe_dir = tmp_path / "probes"
    intervention_dir = tmp_path / "interventions"
    probe_dir.mkdir()
    intervention_dir.mkdir()

    pd.DataFrame(
        [
            {"layer": 0, "split": "test", "auroc": 0.55, "process_f1": 0.50},
            {"layer": 1, "split": "test", "auroc": 0.80, "process_f1": 0.70},
            {
                "layer": 1,
                "split": "test_error_traces",
                "auroc": 0.75,
                "process_f1": 0.65,
            },
        ]
    ).to_csv(probe_dir / "layer_metrics.csv", index=False)
    pd.DataFrame(
        [
            {
                "trace_id": "trace-a",
                "layer": layer,
                "step_index": step,
                "first_error": 1,
                "score": score,
            }
            for layer in (0, 1)
            for step, score in ((0, 0.2), (1, 0.8), (2, 0.9))
        ]
    ).to_csv(probe_dir / "test_predictions.csv", index=False)
    pd.DataFrame(
        [
            {"control": "position", "auroc": 0.52, "process_f1": 0.48},
            {"control": "current-step TF-IDF", "auroc": 0.60, "process_f1": 0.55},
            {"control": "shuffled-label hidden", "auroc": 0.49, "process_f1": 0.45},
        ]
    ).to_csv(probe_dir / "controls.csv", index=False)
    pd.DataFrame(
        [
            {
                "train_source": train,
                "test_source": test,
                "auroc": value,
                "process_f1": value - 0.1,
            }
            for train, row in (("gsm8k", (0.8, 0.7)), ("math", (0.65, 0.85)))
            for test, value in zip(("gsm8k", "math"), row, strict=True)
        ]
    ).to_csv(probe_dir / "domain_transfer.csv", index=False)
    pd.DataFrame(
        [
            {"metric": "auroc", "estimate": 0.8, "ci_low": 0.75, "ci_high": 0.85},
            {"metric": "process_f1", "estimate": 0.7, "ci_low": 0.63, "ci_high": 0.77},
        ]
    ).to_csv(probe_dir / "test_group_bootstrap_summary.csv", index=False)
    np.savez(
        probe_dir / "directions.npz",
        selected_layer=1,
        thresholds=np.asarray([0.5, 0.6]),
    )
    pd.DataFrame(
        [
            {
                "direction_type": "learned",
                "direction_index": -1,
                "alpha": alpha,
                "mean_delta": delta,
                "standard_error": 0.02,
            }
            for alpha, delta in ((-2.0, -0.2), (0.0, 0.0), (2.0, 0.25))
        ]
        + [
            {
                "direction_type": "random",
                "direction_index": 0,
                "alpha": alpha,
                "mean_delta": delta,
                "standard_error": 0.02,
            }
            for alpha, delta in ((-2.0, -0.03), (2.0, 0.04))
        ]
    ).to_csv(intervention_dir / "summary.csv", index=False)

    outputs = plot_artifacts(config)

    assert {path.name for path in outputs} == {
        "method_and_trajectory.pdf",
        "predictive_results.pdf",
        "transfer_and_causal.pdf",
    }
    assert all(path.exists() for path in outputs)


def test_interventions_stop_when_readout_specificity_is_zero(tmp_path, monkeypatch) -> None:
    base_config = ExperimentConfig.from_yaml("configs/experiment.yaml")
    config = replace(base_config, extraction=replace(base_config.extraction, output_dir=tmp_path))
    probe_dir = tmp_path / "probes"
    probe_dir.mkdir()
    np.savez(
        probe_dir / "directions.npz",
        selected_intervention_layer=0,
        directions=np.ones((1, 2)),
        projection_stds=np.ones(1),
    )
    metadata = pd.DataFrame({"partition": ["test"]})
    baseline = pd.DataFrame(
        {
            "invalid_so_far": [1, 0],
            "verdict_score": [0.8, 0.8],
        }
    )
    monkeypatch.setattr(pipeline, "load_traces", lambda _path: [])
    monkeypatch.setattr(pipeline, "load_activation_shards", lambda _path: (np.empty(0), metadata))
    monkeypatch.setattr(pipeline, "HuggingFaceMathModel", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(pipeline, "score_intervention_baseline", lambda *_args, **_kwargs: baseline)

    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(pipeline, "run_interventions", fail_if_called)
    with pytest.raises(RuntimeError, match="specificity is zero"):
        pipeline.run_and_save_interventions(config)
    assert not called
