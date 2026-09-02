"""Tests for transition matching diagnostics and sensitivity options."""

import numpy as np
import pandas as pd
import pytest

from tracing_math.transitions import (
    build_matched_transition_dataset,
    fit_transition_probes,
    inverse_reuse_weights,
    transition_matching_diagnostics,
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


def _diagnostic(frame: pd.DataFrame, name: str) -> float:
    return float(frame.set_index("diagnostic").loc[name, "value"])


def test_matching_diagnostics_report_reuse_and_balance() -> None:
    activations, metadata = _transition_fixture()
    _, matched = build_matched_transition_dataset(activations, metadata)

    diagnostics = transition_matching_diagnostics(matched)

    assert _diagnostic(diagnostics, "pairs") == 3.0
    assert _diagnostic(diagnostics, "unique_placebo_traces") == 3.0
    assert _diagnostic(diagnostics, "placebo_reuse_max") == 1.0
    for column in ("step_fraction", "n_steps", "token_count"):
        assert np.isfinite(_diagnostic(diagnostics, f"{column}_standardized_difference"))


def test_matching_diagnostics_detect_heavy_reuse() -> None:
    rows = []
    for pair in range(6):
        pair_id = f"pair-{pair}"
        rows.append(
            {
                "pair_id": pair_id,
                "role": "error_onset",
                "onset_trace_id": f"err-{pair}",
                "placebo_trace_id": "shared-placebo",
                "step_fraction": 0.5,
                "n_steps": 4,
                "token_count": 100,
            }
        )
        rows.append(
            {
                "pair_id": pair_id,
                "role": "correct_placebo",
                "onset_trace_id": f"err-{pair}",
                "placebo_trace_id": "shared-placebo",
                "step_fraction": 0.5,
                "n_steps": 4,
                "token_count": 100,
            }
        )
    diagnostics = transition_matching_diagnostics(pd.DataFrame(rows))
    assert _diagnostic(diagnostics, "unique_placebo_traces") == 1.0
    assert _diagnostic(diagnostics, "placebo_reuse_max") == 6.0
    assert _diagnostic(diagnostics, "step_fraction_standardized_difference") == 0.0


def test_inverse_reuse_weights_downweight_reused_placebos() -> None:
    rows = []
    for pair in range(4):
        pair_id = f"pair-{pair}"
        rows.append({"pair_id": pair_id, "role": "error_onset", "placebo_trace_id": "p"})
        rows.append({"pair_id": pair_id, "role": "correct_placebo", "placebo_trace_id": "p"})
    rows.append({"pair_id": "pair-4", "role": "error_onset", "placebo_trace_id": "q"})
    rows.append({"pair_id": "pair-4", "role": "correct_placebo", "placebo_trace_id": "q"})
    weights = inverse_reuse_weights(pd.DataFrame(rows))
    # All four p placebos get 1/4; the error rows and the single-use q placebo get 1.
    assert np.allclose(weights, [1.0, 0.25, 1.0, 0.25, 1.0, 0.25, 1.0, 0.25, 1.0, 1.0])


def test_one_to_one_matching_limits_placebo_reuse() -> None:
    rows = []
    values = []
    for partition_index, partition in enumerate(("train", "validation", "test")):
        for error_index in range(3):
            error_id = f"{partition}-err-{error_index}"
            correct_id = f"{partition}-correct-{error_index}"
            for trace_id, first_error in ((error_id, 1), (correct_id, -1)):
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
                    jump = 2.0 if first_error == 1 and step_index == 1 else 0.0
                    values.append(np.full((2, 3), jump + partition_index * 0.01))
    activations = np.asarray(values)
    metadata = pd.DataFrame(rows)

    _, matched = build_matched_transition_dataset(
        activations, metadata, one_to_one=True, rng_seed=42
    )

    placebos = matched[matched["role"].eq("correct_placebo")]
    assert placebos["trace_id"].nunique() == placebos["pair_id"].nunique()
    per_partition = placebos.groupby("partition").agg(
        pairs=("pair_id", "nunique"), traces=("trace_id", "nunique")
    )
    assert (per_partition["pairs"] == per_partition["traces"]).all()
    # With reuse allowed, the same fixture collapses onto fewer placebos.
    _, reused = build_matched_transition_dataset(activations, metadata)
    reused_placebos = reused[reused["role"].eq("correct_placebo")]
    assert reused_placebos["trace_id"].nunique() < reused_placebos["pair_id"].nunique()


def test_transition_probe_accepts_inverse_reuse_weights() -> None:
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
        inverse_reuse=True,
    )

    assert result["selected_layer"] in {0, 1}
    assert len(result["predictions"]) == 2
    assert set(result["bootstrap"]["bootstrap_scheme"]) == {"two_way_pair_and_placebo_trace"}


def test_matching_diagnostics_reject_missing_columns() -> None:
    with pytest.raises(ValueError, match="Missing matching columns"):
        transition_matching_diagnostics(pd.DataFrame({"pair_id": []}))
