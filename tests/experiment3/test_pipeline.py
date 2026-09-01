import json
from dataclasses import replace

import numpy as np
import pandas as pd

from causal_circuits.experiment3 import pipeline
from causal_circuits.experiment3.config import ExtendedFollowupConfig


class _PatchModel:
    def verdict_scores(self, requests, *, replacement_states=None, **_kwargs):
        if replacement_states is not None:
            return np.asarray(replacement_states)[:, 0].tolist()
        return [0.8 if trace.has_error else -0.8 for trace, _step in requests]

    def boundary_input_states(self, requests, **_kwargs):
        return np.asarray([[1.0 if trace.has_error else -1.0] for trace, _step in requests])


def test_counterfactual_patching_checks_baseline_then_checkpoints(
    tmp_path, monkeypatch
) -> None:
    base = ExtendedFollowupConfig.from_yaml("configs/experiment3_extended.yaml")
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
