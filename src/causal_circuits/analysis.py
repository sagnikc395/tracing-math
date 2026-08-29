"""Metrics for model fidelity, experimental validity, and residue constraint."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def spearman(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    """Spearman correlation with finite paired observations only."""
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if mask.sum() < 2:
        return float("nan")
    return float(spearmanr(x_values[mask], y_values[mask]).statistic)


def residue_intolerance(frame: pd.DataFrame, aggregation: str = "median") -> pd.DataFrame:
    """Aggregate experimental fitness into intolerance (-fitness) per residue."""
    if aggregation not in {"median", "mean"}:
        raise ValueError("aggregation must be 'median' or 'mean'")
    required = {"position", "fitness"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    grouped = frame.groupby("position", as_index=False)["fitness"].agg(aggregation)
    return grouped.rename(columns={"fitness": "aggregate_fitness"}).assign(
        intolerance=lambda values: -values["aggregate_fitness"]
    )


def summarize_scores(frame: pd.DataFrame) -> dict[str, float | int]:
    required = {"fitness", "zero_shot_score"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    return {
        "n_variants": int(len(frame)),
        "n_positions": int(frame["position"].nunique()) if "position" in frame else 0,
        "dms_spearman": spearman(frame["zero_shot_score"], frame["fitness"]),
    }
