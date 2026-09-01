"""Small, deterministic parallel-execution helpers for in-memory analyses."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from tqdm.auto import tqdm

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


def resolve_workers(workers: int) -> int:
    """Resolve ``-1`` to all available CPUs and reject other non-positive values."""
    if workers == -1:
        return os.cpu_count() or 1
    if workers < 1:
        raise ValueError("workers must be positive or -1 for all available CPUs")
    return workers


def ordered_parallel_map(
    function: Callable[[InputT], OutputT],
    items: Iterable[InputT],
    *,
    workers: int,
    description: str | None = None,
) -> list[OutputT]:
    """Map with worker threads while preserving input order and exception behavior.

    Threads share large NumPy activation arrays instead of copying them into subprocesses.
    The scientific routines called here spend most of their time in NumPy and scikit-learn,
    which release the Python GIL during their expensive numerical work.
    """
    item_list = list(items)
    worker_count = min(resolve_workers(workers), max(len(item_list), 1))

    def with_progress(iterator: Iterable[OutputT]) -> Iterable[OutputT]:
        if description is None:
            return iterator
        return tqdm(iterator, total=len(item_list), desc=description)

    if worker_count == 1:
        iterator = map(function, item_list)
        return list(with_progress(iterator))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        iterator = executor.map(function, item_list)
        return list(with_progress(iterator))
