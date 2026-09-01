import numpy as np
import pandas as pd

from tracing_math.experiment3.analysis import (
    build_matched_transition_dataset,
    fit_transition_probes,
    summarize_counterfactual_patching,
)


def _transition_fixture() -> tuple[np.ndarray, pd.DataFrame]:
    rows = []
    values = []
    for partition_index, partition in enumerate(("train", "validation", "test")):
        for kind, first_error in (("error", 1), ("correct", -1)):
            trace_id = f"{partition}-{kind}"
            for step_index in range(2):
                rows.append(
                    {
                        "trace_id": trace_id,
                        "partition": partition,
                        "source": "math",
                        "generator": "generator",
                        "step_index": step_index,
                        "step_fraction": (step_index + 1) / 2,
                        "first_error": first_error,
                        "has_error_trace": int(first_error >= 0),
                        "n_steps": 2,
                        "token_count": 20,
                        "step_text": f"step {step_index}",
                    }
                )
                jump = 2.0 if kind == "error" and step_index == 1 else 0.1 * step_index
                values.append(np.full((2, 3), jump + partition_index * 0.01))
    return np.asarray(values), pd.DataFrame(rows)


def test_transition_dataset_pairs_error_onsets_with_correct_placebos() -> None:
    activations, metadata = _transition_fixture()

    transitions, matched = build_matched_transition_dataset(activations, metadata)

    assert transitions.shape == (6, 2, 3)
    assert set(matched["role"]) == {"error_onset", "correct_placebo"}
    assert matched.groupby("pair_id")["label"].sum().eq(1).all()
    assert matched.groupby("partition")["label"].nunique().eq(2).all()


def test_transition_probe_selects_and_evaluates_a_layer() -> None:
    activations, metadata = _transition_fixture()
    transitions, matched = build_matched_transition_dataset(activations, metadata)

    result = fit_transition_probes(
        transitions,
        matched,
        c_values=(0.1,),
        max_iter=200,
        bootstrap_samples=10,
        confidence_level=0.95,
        seed=42,
    )

    assert result["selected_layer"] in {0, 1}
    assert len(result["predictions"]) == 2
    assert set(result["controls"]["control"]) == {
        "position",
        "current_step_tfidf",
        "shuffled_label_hidden",
    }


def test_counterfactual_summary_uses_within_pair_effects() -> None:
    rows = []
    for pair_id in ("a", "b"):
        rows.extend(
            [
                {"pair_id": pair_id, "condition": "correct_baseline", "verdict_score": -0.5},
                {"pair_id": pair_id, "condition": "error_baseline", "verdict_score": 0.5},
                {
                    "pair_id": pair_id,
                    "condition": "error_state_into_correct",
                    "verdict_score": 0.0,
                },
                {
                    "pair_id": pair_id,
                    "condition": "correct_state_into_error",
                    "verdict_score": 0.0,
                },
            ]
        )

    summary = summarize_counterfactual_patching(
        pd.DataFrame(rows), samples=20, confidence_level=0.95, seed=42
    ).set_index("effect")

    assert summary.loc["baseline_error_minus_correct", "estimate"] == 1.0
    assert summary.loc["error_state_into_correct", "estimate"] == 0.5
    assert summary.loc["correct_state_into_error", "estimate"] == -0.5
