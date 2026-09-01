"""Tests for shared bounded parallel execution."""

import pytest

from tracing_math.parallel import ordered_parallel_map, resolve_workers


def test_parallel_map_preserves_input_order() -> None:
    assert ordered_parallel_map(lambda value: value**2, range(6), workers=3) == [
        0,
        1,
        4,
        9,
        16,
        25,
    ]


@pytest.mark.parametrize("workers", [0, -2])
def test_invalid_workers_are_rejected(workers: int) -> None:
    with pytest.raises(ValueError, match="workers must be positive"):
        resolve_workers(workers)


def test_all_workers_resolves_to_a_positive_count() -> None:
    assert resolve_workers(-1) >= 1
