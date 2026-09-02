import json
from dataclasses import replace

import numpy as np
import pandas as pd

from tracing_math.experiment3 import pipeline
from tracing_math.experiment3.config import ExtendedFollowupConfig
from tracing_math.followup.conditional import ConditionalAnalysisResult


class _PatchModel:
    def verdict_scores(self, requests, *, replacement_states=None, **_kwargs):
        if replacement_states is not None:
            return np.asarray(replacement_states)[:, 0].tolist()
        return [0.8 if trace.has_error else -0.8 for trace, _step in requests]

    def boundary_input_states(self, requests, **_kwargs):
        return np.asarray([[1.0 if trace.has_error else -1.0] for trace, _step in requests])


def test_conditional_pipeline_writes_auditable_artifacts(tmp_path, monkeypatch) -> None:
    base = ExtendedFollowupConfig.from_yaml("configs/experiment3.yaml")
    experiment1_dir = tmp_path / "experiment1"
    probe_dir = experiment1_dir / "probes"
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
    predictions = pd.DataFrame(
        {"trace_id": ["test-a", "test-a"], "condition": ["N", "N+H"]}
    )
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
        experiment1_dir=experiment1_dir,
        output_dir=tmp_path / "output",
        data_path=tmp_path / "traces.jsonl",
    )

    summary = pipeline.fit_and_save_conditional_hidden_state(config)

    output = config.output_dir / "conditional_hidden_state"
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


def test_counterfactual_patching_checks_baseline_then_checkpoints(
    tmp_path, monkeypatch
) -> None:
    base = ExtendedFollowupConfig.from_yaml("configs/experiment3.yaml")
    experiment1_dir = tmp_path / "experiment1"
    probe_dir = experiment1_dir / "probes"
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
        experiment1_dir=experiment1_dir,
        output_dir=tmp_path / "output",
        counterfactual_pairs_path=pairs_path,
        bootstrap_samples=20,
    )
    monkeypatch.setattr(pipeline, "HuggingFaceMathModel", lambda *_args, **_kwargs: _PatchModel())

    status = pipeline.run_counterfactual_patching(config)

    assert status["status"] == "complete"
    metrics = json.loads(
        (config.output_dir / "counterfactual_patching/baseline_metrics.json").read_text()
    )
    assert metrics["gate_passed"] is True
    individual = pd.read_csv(config.output_dir / "counterfactual_patching/individual.csv")
    assert set(individual["condition"]) == set(pipeline.PATCH_CONDITIONS)
    assert (config.output_dir / "counterfactual_patching/individual.checkpoint.csv").exists()
