import json
from dataclasses import replace

import numpy as np
import pandas as pd

from tracing_math import operations as pipeline
from tracing_math.conditional import ConditionalAnalysisResult
from tracing_math.config import ProjectConfig


class _PatchModel:
    def verdict_scores(self, requests, *, replacement_states=None, **_kwargs):
        if replacement_states is not None:
            return np.asarray(replacement_states)[:, 0].tolist()
        return [0.8 if trace.has_error else -0.8 for trace, _step in requests]

    def boundary_input_states(self, requests, **_kwargs):
        return np.asarray([[1.0 if trace.has_error else -1.0] for trace, _step in requests])


def test_conditional_pipeline_writes_auditable_artifacts(tmp_path, monkeypatch) -> None:
    base = ProjectConfig.from_yaml("configs/project.yaml")
    artifact_dir = tmp_path / "artifacts"
    probe_dir = artifact_dir / "probes"
    probe_dir.mkdir(parents=True)
    np.savez(probe_dir / "directions.npz", selected_layer=1)
    paired = pd.DataFrame(
        [
            {
                "comparison": "N+H - N",
                "metric": metric,
                "estimate": estimate,
                "ci_low": estimate - 0.01,
                "ci_high": estimate + 0.01,
            }
            for metric, estimate in (("auroc", 0.01), ("log_loss", -0.01))
        ]
    )
    predictions = pd.DataFrame({"trace_id": ["test-a", "test-a"], "condition": ["N", "N+H"]})
    fake = ConditionalAnalysisResult(
        metrics=pd.DataFrame({"condition": ["N", "N+H"]}),
        selection=pd.DataFrame({"condition": ["N", "N+H"]}),
        predictions=predictions,
        paired_intervals=paired,
        feature_blocks={"N": ["prefix_tfidf"], "N+H": ["prefix_tfidf", "hidden"]},
    )
    monkeypatch.setattr(pipeline, "load_activation_shards", lambda _path: (np.zeros(1), None))
    monkeypatch.setattr(pipeline, "load_traces", lambda _path: [])
    monkeypatch.setattr(
        pipeline, "conditional_hidden_state_analysis", lambda *_args, **_kwargs: fake
    )
    config = replace(
        base,
        extraction=replace(base.extraction, output_dir=artifact_dir),
        artifacts=replace(base.artifacts, analysis_dir=tmp_path / "output"),
        data=replace(base.data, output_path=tmp_path / "traces.jsonl"),
    )

    summary = pipeline.fit_and_save_conditional_hidden_state(config)

    output = config.artifacts.analysis_dir / "conditional_hidden_state"
    assert summary["status"] == "complete"
    assert len(summary["config_sha256"]) == 64
    for name in (
        "metrics.csv",
        "validation_selection.csv",
        "test_predictions.csv",
        "paired_differences.csv",
        "feature_blocks.json",
        "resolved_config.json",
        "result_record.md",
        "summary.json",
    ):
        assert (output / name).exists()


def test_counterfactual_patching_checks_baseline_then_checkpoints(tmp_path, monkeypatch) -> None:
    base = ProjectConfig.from_yaml("configs/project.yaml")
    artifact_dir = tmp_path / "artifacts"
    probe_dir = artifact_dir / "probes"
    probe_dir.mkdir(parents=True)
    np.savez(probe_dir / "directions.npz", selected_intervention_layer=0)
    pairs_path = tmp_path / "pairs.jsonl"
    rows = [
        {
            "pair_id": f"pair-{index}",
            "source": "math",
            "problem": f"problem {index}",
            "prefix_steps": ["valid step"],
            "error_step": "1 + 1 = 3",
            "corrected_step": "1 + 1 = 2",
            "verified": True,
        }
        for index in range(2)
    ]
    pairs_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    config = replace(
        base,
        extraction=replace(base.extraction, output_dir=artifact_dir),
        artifacts=replace(
            base.artifacts,
            analysis_dir=tmp_path / "output",
            counterfactual_pairs_path=pairs_path,
        ),
        analysis=replace(base.analysis, transition_bootstrap_samples=20),
    )
    monkeypatch.setattr(pipeline, "HuggingFaceMathModel", lambda *_args, **_kwargs: _PatchModel())

    status = pipeline.run_counterfactual_patching(config)

    assert status["status"] == "complete"
    metrics = json.loads(
        (
            config.artifacts.analysis_dir / "counterfactual_patching/baseline_metrics.json"
        ).read_text()
    )
    assert metrics["gate_passed"] is True
    individual = pd.read_csv(
        config.artifacts.analysis_dir / "counterfactual_patching/individual.csv"
    )
    assert set(individual["condition"]) == set(pipeline.PATCH_CONDITIONS)
    assert (
        config.artifacts.analysis_dir / "counterfactual_patching/individual.checkpoint.csv"
    ).exists()


def test_transition_sensitivity_writes_intervals_and_matching_diagnostics(
    tmp_path, monkeypatch
) -> None:
    base = ProjectConfig.from_yaml("configs/project.yaml")
    metadata = pd.DataFrame(
        [
            {
                "pair_id": "pair-1",
                "role": "error_onset",
                "onset_trace_id": "error-1",
                "placebo_trace_id": "correct-1",
                "trace_id": "error-1",
                "partition": "test",
                "label": 1,
                "step_fraction": 0.5,
                "n_steps": 2,
                "token_count": 20,
            },
            {
                "pair_id": "pair-1",
                "role": "correct_placebo",
                "onset_trace_id": "error-1",
                "placebo_trace_id": "correct-1",
                "trace_id": "correct-1",
                "partition": "test",
                "label": 0,
                "step_fraction": 0.5,
                "n_steps": 2,
                "token_count": 20,
            },
        ]
    )
    bootstrap = pd.DataFrame(
        {
            "metric": ["auroc", "average_precision", "paired_accuracy"],
            "estimate": [0.75, 0.7, 0.75],
            "ci_low": [0.6, 0.55, 0.6],
            "ci_high": [0.9, 0.85, 0.9],
        }
    )
    monkeypatch.setattr(pipeline, "load_activation_shards", lambda _path: (np.zeros(1), None))
    monkeypatch.setattr(
        pipeline,
        "build_matched_transition_dataset",
        lambda *_args, **_kwargs: (np.zeros((2, 1, 1)), metadata.copy()),
    )
    monkeypatch.setattr(
        pipeline,
        "fit_transition_probes",
        lambda *_args, **_kwargs: {
            "predictions": metadata.copy(),
            "bootstrap": bootstrap.copy(),
            "selected_layer": 0,
            "selected_c": 0.1,
        },
    )
    config = replace(
        base,
        artifacts=replace(base.artifacts, analysis_dir=tmp_path / "output"),
    )

    summary = pipeline.run_transition_matching_sensitivity(config)

    output = config.artifacts.analysis_dir / "transition_probe"
    assert summary["variants"]["with_reuse"]["auroc_ci_low"] == 0.6
    assert set(summary["artifacts"]) == {"summary", "intervals", "matching_diagnostics"}
    for name in (
        "matching_sensitivity.csv",
        "matching_sensitivity_bootstrap.csv",
        "matching_sensitivity_diagnostics.csv",
    ):
        assert (output / name).exists()
