"""Tests for Experiment 1 activation interventions."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from causal_circuits.experiment1.config import ExperimentConfig
from causal_circuits.experiment1.data import ProcessTrace
from causal_circuits.experiment1.interventions import (
    causal_effect_statistics,
    random_orthogonal_directions,
    run_interventions,
)


def test_random_controls_are_unit_and_orthogonal() -> None:
    direction = np.array([1.0, 0.0, 0.0])
    controls = random_orthogonal_directions(direction, 5, seed=42)
    assert controls.shape == (5, 3)
    np.testing.assert_allclose(np.linalg.norm(controls, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(controls @ direction, 0.0, atol=1e-6)


def test_causal_effect_statistics_measure_dose_and_random_comparison() -> None:
    rows = []
    for trace_index in range(4):
        for alpha in (-1.0, 0.0, 1.0):
            rows.append(
                {
                    "trace_id": f"trace-{trace_index}",
                    "source": "math",
                    "step_index": trace_index,
                    "invalid_so_far": trace_index % 2,
                    "direction_type": "learned",
                    "direction_index": -1,
                    "alpha": alpha,
                    "delta_verdict_score": alpha,
                }
            )
        for direction_index in range(2):
            for alpha in (-1.0, 1.0):
                rows.append(
                    {
                        "trace_id": f"trace-{trace_index}",
                        "source": "math",
                        "step_index": trace_index,
                        "invalid_so_far": trace_index % 2,
                        "direction_type": "random_orthogonal",
                        "direction_index": direction_index,
                        "alpha": alpha,
                        "delta_verdict_score": alpha * 0.1,
                    }
                )
    result = causal_effect_statistics(
        pd.DataFrame(rows), bootstrap_samples=10, subgroup_min_traces=2, seed=42
    )
    slope = result[result["statistic"] == "dose_slope"].iloc[0]
    assert slope["estimate"] == pytest.approx(1.0)
    comparisons = result[result["statistic"] == "learned_vs_random_empirical_p"]
    assert len(comparisons) == 2


class _BatchModel:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, float]] = []

    def verdict_scores(self, requests, *, batch_size, magnitude=0.0, **_kwargs):
        if requests:
            self.calls.append((len(requests), batch_size, magnitude))
        return [float(index) + magnitude for index, _request in enumerate(requests)]


def test_interventions_batch_and_resume_from_group_checkpoints() -> None:
    traces = [
        ProcessTrace(
            trace_id=f"trace-{index}",
            source="math",
            generator="test",
            problem=f"problem {index}",
            steps=("one",),
            label=0 if index < 2 else -1,
            final_answer_correct=index >= 2,
        )
        for index in range(4)
    ]
    metadata = pd.DataFrame(
        [
            {
                "trace_id": trace.trace_id,
                "source": trace.source,
                "partition": "test",
                "step_index": 0,
                "first_error": trace.label,
                "invalid_so_far": int(trace.label == 0),
                "error_onset": int(trace.label == 0),
            }
            for trace in traces
        ]
    )
    base = ExperimentConfig.from_yaml("configs/experiment.yaml").intervention
    config = replace(
        base,
        alphas=(-1.0, 0.0, 1.0),
        examples_per_class=2,
        random_directions=2,
        batch_size=2,
    )
    checkpoints = []
    model = _BatchModel()
    result = run_interventions(
        model,
        traces,
        metadata,
        direction=np.array([1.0, 0.0, 0.0]),
        projection_std=1.0,
        layer=0,
        config=config,
        target="invalid_so_far",
        seed=42,
        checkpoint_callback=lambda frame: checkpoints.append(frame.copy()),
    )

    assert all(batch_size == 2 for _size, batch_size, _magnitude in model.calls)
    assert len(checkpoints) == 7
    assert len(result) == 28
    assert not result.duplicated(
        ["trace_id", "step_index", "direction_type", "direction_index", "alpha"]
    ).any()

    resumed_model = _BatchModel()
    resumed = run_interventions(
        resumed_model,
        traces,
        metadata,
        direction=np.array([1.0, 0.0, 0.0]),
        projection_std=1.0,
        layer=0,
        config=config,
        target="invalid_so_far",
        seed=42,
        existing_results=result,
    )
    assert resumed_model.calls == []
    assert len(resumed) == len(result)
