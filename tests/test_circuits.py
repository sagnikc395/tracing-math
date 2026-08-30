import numpy as np

from causal_circuits.circuits import random_orthogonal_directions


def test_random_controls_are_unit_and_orthogonal() -> None:
    direction = np.array([1.0, 0.0, 0.0])
    controls = random_orthogonal_directions(direction, 5, seed=42)
    assert controls.shape == (5, 3)
    np.testing.assert_allclose(np.linalg.norm(controls, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(controls @ direction, 0.0, atol=1e-6)
