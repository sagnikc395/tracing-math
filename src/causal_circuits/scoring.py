"""Masked-marginal zero-shot scoring."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

import pandas as pd

from causal_circuits.data import Substitution


class PositionLogProbModel(Protocol):
    def position_log_probs(self, sequence: str, position: int) -> Mapping[str, float]:
        """Return amino-acid log probabilities at a zero-indexed masked position."""


def masked_marginal_score(
    model: PositionLogProbModel, sequence: str, substitution: Substitution
) -> float:
    """Compute log P(mutant | masked context) - log P(WT | masked context)."""
    if sequence[substitution.position - 1] != substitution.wild_type:
        raise ValueError(f"{substitution} does not match the supplied wild-type sequence")
    log_probs = model.position_log_probs(sequence, substitution.position - 1)
    try:
        return float(log_probs[substitution.mutant] - log_probs[substitution.wild_type])
    except KeyError as error:
        raise ValueError(f"Model did not return a probability for {error.args[0]}") from error


def score_dms(
    model: PositionLogProbModel, wild_type_sequence: str, frame: pd.DataFrame
) -> pd.DataFrame:
    """Score normalized DMS rows, caching the model call for each residue position."""
    required = {"mutation", "wild_type", "position", "mutant"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing normalized DMS columns: {sorted(missing)}")

    position_cache: dict[int, Mapping[str, float]] = {}
    scores: list[float] = []
    for row in frame.itertuples(index=False):
        position = int(row.position)
        if wild_type_sequence[position - 1] != row.wild_type:
            raise ValueError(f"{row.mutation} does not match the supplied wild-type sequence")
        if position not in position_cache:
            position_cache[position] = model.position_log_probs(wild_type_sequence, position - 1)
        probabilities = position_cache[position]
        scores.append(float(probabilities[row.mutant] - probabilities[row.wild_type]))

    result = frame.copy()
    result["zero_shot_score"] = scores
    return result
