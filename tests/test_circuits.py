import numpy as np
import pandas as pd
import pytest

from causal_circuits.circuits import causal_effect_statistics, random_orthogonal_directions


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
