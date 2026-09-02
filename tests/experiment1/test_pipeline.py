"""Tests for Experiment 1 orchestration and artifact generation."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from tracing_math.experiment1 import pipeline
from tracing_math.experiment1.config import ExperimentConfig
from tracing_math.experiment1.pipeline import load_activation_metadata, plot_artifacts


def test_load_activation_metadata_skips_tensor_contents_and_preserves_order(tmp_path) -> None:
    shard_dir = tmp_path / "activation_shards"
    shard_dir.mkdir()
    for index in (2, 1):
        stem = shard_dir / f"shard_{index:05d}_{index:05d}"
        np.save(stem.with_suffix(".npy"), np.full((1, 2, 3), index, dtype=np.float32))
        pd.DataFrame({"trace_id": [f"trace-{index}"]}).to_csv(
            stem.with_suffix(".csv"), index=False
        )

    metadata = load_activation_metadata(tmp_path, workers=2)

    assert metadata["trace_id"].tolist() == ["trace-1", "trace-2"]


def test_plot_artifacts_creates_only_essential_paper_figures(tmp_path) -> None:
    base_config = ExperimentConfig.from_yaml("configs/experiment1.yaml")
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
    base_config = ExperimentConfig.from_yaml("configs/experiment1.yaml")
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
            "trace_id": ["a", "b"],
            "step_index": [0, 0],
            "invalid_so_far": [1, 0],
            "direction_type": ["learned", "learned"],
            "direction_index": [-1, -1],
            "alpha": [0.0, 0.0],
            "verdict_score": [0.8, 0.8],
            "delta_verdict_score": [0.0, 0.0],
        }
    )
    monkeypatch.setattr(pipeline, "load_traces", lambda _path: [])
    monkeypatch.setattr(pipeline, "load_activation_metadata", lambda _path, **_kwargs: metadata)
    monkeypatch.setattr(pipeline, "HuggingFaceMathModel", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(pipeline, "score_intervention_baseline", lambda *_args, **_kwargs: baseline)

    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(pipeline, "run_interventions", fail_if_called)
    result = pipeline.run_and_save_interventions(config)
    assert not called
    assert len(result) == 2
    assert (tmp_path / "interventions" / "individual.csv").exists()
    progress = json.loads((tmp_path / "interventions" / "progress.json").read_text())
    assert progress["status"] == "stopped_after_baseline"


def test_interventions_stop_when_readout_ranks_labels_below_chance(
    tmp_path, monkeypatch
) -> None:
    base_config = ExperimentConfig.from_yaml("configs/experiment1.yaml")
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
            "trace_id": list("abcde"),
            "step_index": [0] * 5,
            "invalid_so_far": [1, 1, 0, 0, 0],
            "direction_type": ["learned"] * 5,
            "direction_index": [-1] * 5,
            "alpha": [0.0] * 5,
            "verdict_score": [0.1, 0.2, 0.9, 0.8, -0.9],
            "delta_verdict_score": [0.0] * 5,
        }
    )
    monkeypatch.setattr(pipeline, "load_traces", lambda _path: [])
    monkeypatch.setattr(pipeline, "load_activation_metadata", lambda _path, **_kwargs: metadata)
    monkeypatch.setattr(pipeline, "HuggingFaceMathModel", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(pipeline, "score_intervention_baseline", lambda *_args, **_kwargs: baseline)

    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(pipeline, "run_interventions", fail_if_called)
    pipeline.run_and_save_interventions(config)

    assert not called
    verdict = json.loads((tmp_path / "interventions" / "behavioral_verdict.json").read_text())
    assert verdict["specificity_gate_passed"]
    assert verdict["auroc"] < 0.5
    assert not verdict["behavioral_validity_gate_passed"]
