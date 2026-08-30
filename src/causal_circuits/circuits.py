"""Causal probe-direction interventions and matched random controls."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from causal_circuits.config import InterventionConfig
from causal_circuits.data import ProcessTrace
from causal_circuits.models import HuggingFaceMathModel


def random_orthogonal_directions(direction: np.ndarray, count: int, *, seed: int) -> np.ndarray:
    """Create unit random directions orthogonal to the learned probe direction."""
    direction = np.asarray(direction, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    rng = np.random.default_rng(seed)
    controls = []
    for _ in range(count):
        vector = rng.normal(size=direction.shape)
        vector -= np.dot(vector, direction) * direction
        vector /= np.linalg.norm(vector)
        controls.append(vector.astype(np.float32))
    return np.stack(controls) if controls else np.empty((0, len(direction)), dtype=np.float32)


def run_interventions(
    model: HuggingFaceMathModel,
    traces: Sequence[ProcessTrace],
    metadata: pd.DataFrame,
    *,
    direction: np.ndarray,
    projection_std: float,
    layer: int,
    config: InterventionConfig,
    target: str,
    seed: int,
) -> pd.DataFrame:
    """Run a held-out dose response plus cheaper matched random-direction controls."""
    test = metadata[metadata["partition"] == "test"].copy()
    selected = _balanced_sample(test, target, config.examples_per_class, seed)
    trace_lookup = {trace.trace_id: trace for trace in traces}
    missing = set(selected["trace_id"]).difference(trace_lookup)
    if missing:
        raise ValueError(f"Missing {len(missing)} intervention traces from the local dataset")

    rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        trace = trace_lookup[row.trace_id]
        baseline = model.verdict_score(
            trace,
            int(row.step_index),
            correct_answer=config.correct_answer,
            incorrect_answer=config.incorrect_answer,
        )
        rows.append(_intervention_row(row, "learned", -1, 0.0, baseline, baseline))
        for alpha in config.alphas:
            if alpha == 0:
                continue
            score = model.verdict_score(
                trace,
                int(row.step_index),
                correct_answer=config.correct_answer,
                incorrect_answer=config.incorrect_answer,
                layer=layer,
                direction=direction,
                magnitude=float(alpha) * projection_std,
            )
            rows.append(_intervention_row(row, "learned", -1, alpha, score, baseline))

    random_directions = random_orthogonal_directions(
        direction, config.random_directions, seed=seed + 1
    )
    control_size = min(config.examples_per_class, 16)
    control_examples = _balanced_sample(test, target, control_size, seed + 1)
    extreme_alphas = sorted({min(config.alphas), max(config.alphas)} - {0.0})
    control_rows = list(control_examples.itertuples(index=False))
    control_baselines = {
        (row.trace_id, int(row.step_index)): model.verdict_score(
            trace_lookup[row.trace_id],
            int(row.step_index),
            correct_answer=config.correct_answer,
            incorrect_answer=config.incorrect_answer,
        )
        for row in control_rows
    }
    for direction_index, control_direction in enumerate(random_directions):
        for row in control_rows:
            trace = trace_lookup[row.trace_id]
            baseline = control_baselines[(row.trace_id, int(row.step_index))]
            for alpha in extreme_alphas:
                score = model.verdict_score(
                    trace,
                    int(row.step_index),
                    correct_answer=config.correct_answer,
                    incorrect_answer=config.incorrect_answer,
                    layer=layer,
                    direction=control_direction,
                    magnitude=float(alpha) * projection_std,
                )
                rows.append(
                    _intervention_row(
                        row, "random_orthogonal", direction_index, alpha, score, baseline
                    )
                )
    return pd.DataFrame(rows)


def summarize_interventions(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["direction_type", "direction_index", "alpha"], as_index=False)
        .agg(
            mean_verdict_score=("verdict_score", "mean"),
            mean_delta=("delta_verdict_score", "mean"),
            standard_error=(
                "verdict_score",
                lambda values: values.std(ddof=1) / np.sqrt(len(values)),
            ),
            n=("verdict_score", "size"),
        )
        .sort_values(["direction_type", "direction_index", "alpha"])
    )


def _balanced_sample(frame: pd.DataFrame, target: str, per_class: int, seed: int) -> pd.DataFrame:
    if target not in frame:
        raise ValueError(f"Missing intervention target column {target!r}")
    groups = []
    for label in (0, 1):
        group = frame[frame[target] == label]
        if group.empty:
            raise ValueError(f"No held-out intervention rows for class {label}")
        groups.append(group.sample(n=min(per_class, len(group)), random_state=seed + label))
    return pd.concat(groups, ignore_index=True).sample(frac=1, random_state=seed)


def _intervention_row(row, direction_type, direction_index, alpha, score, baseline):
    return {
        "trace_id": row.trace_id,
        "source": row.source,
        "step_index": int(row.step_index),
        "first_error": int(row.first_error),
        "invalid_so_far": int(row.invalid_so_far),
        "error_onset": int(row.error_onset),
        "direction_type": direction_type,
        "direction_index": direction_index,
        "alpha": float(alpha),
        "verdict_score": float(score),
        "baseline_verdict_score": float(baseline),
        "delta_verdict_score": float(score - baseline),
    }
