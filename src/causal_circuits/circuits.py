"""Causal probe-direction interventions and matched random controls."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
    # Use a subset of the learned-direction sample so learned and random effects are trace matched.
    control_examples = _balanced_sample(selected, target, control_size, seed + 1)
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
                "delta_verdict_score",
                lambda values: values.std(ddof=1) / np.sqrt(len(values)),
            ),
            n=("verdict_score", "size"),
        )
        .sort_values(["direction_type", "direction_index", "alpha"])
    )


def causal_effect_statistics(
    frame: pd.DataFrame,
    *,
    confidence_level: float = 0.95,
    bootstrap_samples: int = 1000,
    subgroup_min_traces: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Summarize paired effects, dose shape, and matched random-direction evidence."""
    learned = frame[frame["direction_type"] == "learned"].copy()
    rows: list[dict[str, object]] = []
    scopes = [("overall", "all", learned)]
    scopes.extend(
        ("starting_class", str(value), group)
        for value, group in learned.groupby("invalid_so_far", sort=True)
    )
    scopes.extend(
        ("source", str(value), group) for value, group in learned.groupby("source", sort=True)
    )
    tail = (1 - confidence_level) / 2
    rng = np.random.default_rng(seed)
    for scope, value, scoped in scopes:
        n_traces = scoped["trace_id"].nunique()
        if scope != "overall" and n_traces < subgroup_min_traces:
            rows.append(
                {
                    "statistic": "paired_effect",
                    "scope": scope,
                    "value": value,
                    "status": "suppressed",
                    "n_traces": n_traces,
                }
            )
            continue
        for alpha, group in scoped.groupby("alpha", sort=True):
            values = group["delta_verdict_score"].to_numpy(dtype=float)
            bootstrap_means = np.asarray([], dtype=float)
            if bootstrap_samples > 0:
                bootstrap_means = np.asarray(
                    [
                        rng.choice(values, size=len(values), replace=True).mean()
                        for _ in range(bootstrap_samples)
                    ]
                )
            rows.append(
                {
                    "statistic": "paired_effect",
                    "scope": scope,
                    "value": value,
                    "status": "reported",
                    "alpha": float(alpha),
                    "estimate": float(values.mean()),
                    "ci_low": (
                        float(np.quantile(bootstrap_means, tail))
                        if bootstrap_means.size
                        else np.nan
                    ),
                    "ci_high": (
                        float(np.quantile(bootstrap_means, 1 - tail))
                        if bootstrap_means.size
                        else np.nan
                    ),
                    "n_traces": len(values),
                }
            )

    dose = learned.groupby("alpha", as_index=False)["delta_verdict_score"].mean()
    nonzero = learned[learned["alpha"] != 0]
    slope = float(np.polyfit(dose["alpha"], dose["delta_verdict_score"], 1)[0])
    monotonicity = float(spearmanr(dose["alpha"], dose["delta_verdict_score"]).statistic)
    sign_consistency = float(((nonzero["alpha"] * nonzero["delta_verdict_score"]) > 0).mean())
    symmetric_pairs = []
    dose_lookup = dict(zip(dose["alpha"], dose["delta_verdict_score"], strict=True))
    for alpha in sorted(value for value in dose_lookup if value > 0 and -value in dose_lookup):
        symmetric_pairs.append(abs(float(dose_lookup[alpha] + dose_lookup[-alpha])))
    rows.extend(
        [
            {
                "statistic": "dose_slope",
                "scope": "overall",
                "value": "all",
                "status": "reported",
                "estimate": slope,
                "n_traces": learned["trace_id"].nunique(),
            },
            {
                "statistic": "dose_rank_monotonicity",
                "scope": "overall",
                "value": "all",
                "status": "reported",
                "estimate": monotonicity,
                "n_traces": learned["trace_id"].nunique(),
            },
            {
                "statistic": "signed_effect_consistency",
                "scope": "overall",
                "value": "all",
                "status": "reported",
                "estimate": sign_consistency,
                "n_traces": nonzero["trace_id"].nunique(),
            },
            {
                "statistic": "mean_symmetry_error",
                "scope": "overall",
                "value": "all",
                "status": "reported",
                "estimate": float(np.mean(symmetric_pairs)) if symmetric_pairs else np.nan,
                "n_traces": learned["trace_id"].nunique(),
            },
        ]
    )

    random = frame[frame["direction_type"] == "random_orthogonal"].copy()
    if not random.empty:
        matched_keys = random[["trace_id", "step_index"]].drop_duplicates()
        matched_learned = learned.merge(
            matched_keys, on=["trace_id", "step_index"], how="inner", validate="many_to_one"
        )
        for alpha, random_alpha in random.groupby("alpha", sort=True):
            learned_alpha = matched_learned[matched_learned["alpha"] == alpha]
            if learned_alpha.empty:
                continue
            learned_effect = float(learned_alpha["delta_verdict_score"].mean())
            random_effects = random_alpha.groupby("direction_index")["delta_verdict_score"].mean()
            if alpha > 0:
                more_extreme = int((random_effects >= learned_effect).sum())
            else:
                more_extreme = int((random_effects <= learned_effect).sum())
            rows.append(
                {
                    "statistic": "learned_vs_random_empirical_p",
                    "scope": "matched_extreme_alpha",
                    "value": "all",
                    "status": "reported",
                    "alpha": float(alpha),
                    "estimate": learned_effect,
                    "random_mean": float(random_effects.mean()),
                    "empirical_p": float((1 + more_extreme) / (1 + len(random_effects))),
                    "n_traces": learned_alpha["trace_id"].nunique(),
                    "n_random_directions": len(random_effects),
                }
            )
    return pd.DataFrame(rows)


def _balanced_sample(frame: pd.DataFrame, target: str, per_class: int, seed: int) -> pd.DataFrame:
    if target not in frame:
        raise ValueError(f"Missing intervention target column {target!r}")
    groups = []
    used_traces: set[str] = set()
    for label in (1, 0):
        group = frame[(frame[target] == label) & ~frame["trace_id"].isin(used_traces)]
        if group.empty:
            raise ValueError(f"No held-out intervention rows for class {label}")
        group = group.sample(frac=1, random_state=seed + label).drop_duplicates("trace_id")
        chosen = group.sample(n=min(per_class, len(group)), random_state=seed + 10 + label)
        groups.append(chosen)
        used_traces.update(chosen["trace_id"].astype(str))
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
