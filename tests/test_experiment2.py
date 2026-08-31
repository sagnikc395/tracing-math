from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from causal_circuits.experiment2_analysis import (
    bootstrap_onset_jump,
    choose_binary_threshold,
    domain_calibration_analysis,
    fit_surface_metadata_control,
)
from causal_circuits.experiment2_causal import (
    _completed_causal_jobs,
    _decoder_final_hidden,
    _deduplicate_causal_rows,
    _ensure_causal_checkpoint_identity,
    _label_margins,
    balanced_boundary_sample,
    counterbalance_interventions,
    counterbalance_verdict_rows,
    intervened_next_token_margins,
    verdict_mappings,
)
from causal_circuits.experiment2_config import Experiment2Config
from causal_circuits.experiment2_pipeline import experiment2_run_identity
from causal_circuits.experiment2_runtime import (
    ensure_checkpoint_identity,
    run_stage,
    stage_status,
)
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


def test_experiment2_run_identity_ignores_operational_batch_sizes() -> None:
    config = Experiment2Config.from_yaml("configs/experiment2.yaml")
    changed = replace(
        config,
        semantic_extraction=replace(config.semantic_extraction, batch_size=1),
        verdict=replace(config.verdict, batch_size=1),
        causal=replace(config.causal, batch_size=1),
    )
    assert experiment2_run_identity(changed) == experiment2_run_identity(config)


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


def test_causal_checkpoint_jobs_resume_only_complete_pairs() -> None:
    alignment = pd.DataFrame(
        [
            {"trace_id": "a", "step_index": 1, "mapping": "fixed", "layer": layer}
            for layer in range(3)
        ]
    )
    interventions = pd.DataFrame(
        [
            {
                "trace_id": "a",
                "step_index": 1,
                "mapping": "fixed",
                "direction_type": direction,
                "layer": 2,
                "alpha": alpha,
            }
            for direction in ("learned_probe", "gradient_positive_control")
            for alpha in (-1.0, 0.0, 1.0)
        ]
    )
    assert _completed_causal_jobs(alignment) & _completed_causal_jobs(interventions) == {
        ("a", 1, "fixed")
    }
    duplicated = pd.concat([interventions, interventions.iloc[[0]]], ignore_index=True)
    assert len(_deduplicate_causal_rows(duplicated, alignment=False)) == len(interventions)


def test_stage_status_skips_only_matching_completed_identity(tmp_path) -> None:
    calls = []

    def operation():
        calls.append("called")
        return {"value": len(calls)}

    assert run_stage(tmp_path, "stage", operation, identity="one") == {"value": 1}
    assert run_stage(
        tmp_path,
        "stage",
        operation,
        skip_completed=True,
        identity="one",
    ) == {"value": 1}
    assert calls == ["called"]
    assert run_stage(
        tmp_path,
        "stage",
        operation,
        skip_completed=True,
        identity="two",
    ) == {"value": 2}
    assert stage_status(tmp_path)["stages"]["stage"]["status"] == "complete"


def test_stage_status_records_failure_and_checkpoint_identity_is_guarded(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        run_stage(
            tmp_path,
            "failing",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            identity="run",
        )
    failed = stage_status(tmp_path)["stages"]["failing"]
    assert failed["status"] == "failed"
    assert failed["error"] == {"type": "RuntimeError", "message": "boom"}

    identity_path = tmp_path / "checkpoint_identity.json"
    ensure_checkpoint_identity(identity_path, {"model": "a", "labels": ("A", "B")})
    ensure_checkpoint_identity(identity_path, {"model": "a", "labels": ("A", "B")})
    with pytest.raises(RuntimeError, match="identity mismatch"):
        ensure_checkpoint_identity(identity_path, {"model": "b"})


def test_causal_checkpoint_identity_ignores_only_operational_batch_size(tmp_path) -> None:
    identity_path = tmp_path / "checkpoint_identity.json"
    original = {"model": "a", "stage_config": {"examples_per_class": 32, "batch_size": 1}}
    ensure_checkpoint_identity(identity_path, original)
    _ensure_causal_checkpoint_identity(
        identity_path,
        {"model": "a", "stage_config": {"examples_per_class": 32, "batch_size": 8}},
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _ensure_causal_checkpoint_identity(
            identity_path,
            {"model": "a", "stage_config": {"examples_per_class": 16, "batch_size": 8}},
        )


def test_final_state_only_logits_match_full_sequence_logits() -> None:
    class TinyDecoder(torch.nn.Module):
        def forward(self, input_ids, attention_mask, **_kwargs):
            hidden = torch.nn.functional.one_hot(input_ids, num_classes=4).float()
            return SimpleNamespace(last_hidden_state=hidden)

    lm_head = torch.nn.Linear(4, 6, bias=False)
    adapter = SimpleNamespace(
        _torch=torch,
        model=SimpleNamespace(model=TinyDecoder(), lm_head=lm_head),
    )
    inputs = {
        "input_ids": torch.tensor([[1, 2, 0], [3, 1, 2]]),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
    }
    full_hidden = adapter.model.model(**inputs).last_hidden_state
    full_logits = adapter.model.lm_head(full_hidden)
    final_hidden = _decoder_final_hidden(adapter, inputs)
    margins = _label_margins(adapter, final_hidden, [1, 2], [4, 5])
    expected = torch.stack(
        [full_logits[0, 1, 4] - full_logits[0, 1, 1], full_logits[1, 2, 5] - full_logits[1, 2, 2]]
    )
    torch.testing.assert_close(margins, expected)


def test_batched_interventions_match_one_at_a_time() -> None:
    class TinyTokenizer:
        padding_side = "right"

        def __call__(self, prompts, **_kwargs):
            encoded = {
                "short": [1, 2, 3],
                "long": [2, 1, 3, 2],
            }
            rows = [encoded[prompt] for prompt in prompts]
            width = max(map(len, rows))
            return {
                "input_ids": torch.tensor([row + [0] * (width - len(row)) for row in rows]),
                "attention_mask": torch.tensor(
                    [[1] * len(row) + [0] * (width - len(row)) for row in rows]
                ),
            }

    class CausalMix(torch.nn.Module):
        def forward(self, hidden):
            return hidden + hidden.cumsum(dim=1)

    class TinyDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 4)
            self.layers = torch.nn.ModuleList([CausalMix()])

        def forward(self, input_ids, attention_mask, **_kwargs):
            hidden = self.embedding(input_ids)
            for layer in self.layers:
                hidden = layer(hidden)
            return SimpleNamespace(last_hidden_state=hidden)

    torch.manual_seed(42)
    decoder = TinyDecoder()
    adapter = SimpleNamespace(
        _torch=torch,
        tokenizer=TinyTokenizer(),
        model=SimpleNamespace(model=decoder, lm_head=torch.nn.Linear(4, 6, bias=False)),
        decoder_layers=decoder.layers,
        device="cpu",
        max_length=16,
    )
    common = {
        "layer": 0,
        "valid_token_ids": [1, 1],
        "invalid_token_ids": [4, 4],
    }
    directions = [
        np.array([1, 0, 0, 0], dtype=np.float32),
        np.array([0, 1, 0, 0], dtype=np.float32),
    ]
    batched = intervened_next_token_margins(
        adapter,
        ["short", "long"],
        [1, 2],
        directions=directions,
        magnitudes=[0.5, -0.25],
        **common,
    )
    individual = [
        intervened_next_token_margins(
            adapter,
            [prompt],
            [boundary],
            layer=0,
            directions=[direction],
            magnitudes=[magnitude],
            valid_token_ids=[1],
            invalid_token_ids=[4],
        )[0]
        for prompt, boundary, direction, magnitude in zip(
            ["short", "long"], [1, 2], directions, [0.5, -0.25], strict=True
        )
    ]
    np.testing.assert_allclose(batched, individual, rtol=1e-6, atol=1e-6)


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
