"""ProteinGym-style single-mutant data loading and normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_SUBSTITUTION = re.compile(r"^(?P<wild_type>[A-Z])(?P<position>[1-9][0-9]*)(?P<mutant>[A-Z])$")


@dataclass(frozen=True)
class Substitution:
    wild_type: str
    position: int
    mutant: str

    @classmethod
    def parse(cls, value: str) -> Substitution:
        match = _SUBSTITUTION.fullmatch(str(value).strip().upper())
        if match is None:
            raise ValueError(f"Expected a single substitution like A42G, got {value!r}")
        substitution = cls(
            wild_type=match["wild_type"],
            position=int(match["position"]),
            mutant=match["mutant"],
        )
        if substitution.wild_type not in AMINO_ACIDS or substitution.mutant not in AMINO_ACIDS:
            raise ValueError(f"Non-canonical amino acid in {value!r}")
        if substitution.wild_type == substitution.mutant:
            raise ValueError(f"Mutation does not change the residue: {value!r}")
        return substitution

    def __str__(self) -> str:
        return f"{self.wild_type}{self.position}{self.mutant}"


def load_dms(
    path: str | Path,
    *,
    mutation_column: str = "mutation",
    fitness_column: str = "DMS_score",
    directionality: int = 1,
) -> pd.DataFrame:
    """Load a DMS CSV and return a normalized single-substitution table."""
    frame = pd.read_csv(path)
    missing = {mutation_column, fitness_column}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required DMS columns: {sorted(missing)}")
    if directionality not in {-1, 1}:
        raise ValueError("directionality must be either -1 or 1")

    substitutions = frame[mutation_column].map(Substitution.parse)
    normalized = frame.copy()
    normalized["mutation"] = substitutions.map(str)
    normalized["wild_type"] = substitutions.map(lambda item: item.wild_type)
    normalized["position"] = substitutions.map(lambda item: item.position)
    normalized["mutant"] = substitutions.map(lambda item: item.mutant)
    fitness = pd.to_numeric(normalized[fitness_column], errors="raise")
    normalized["fitness"] = fitness * directionality
    return normalized


def validate_against_sequence(frame: pd.DataFrame, wild_type_sequence: str) -> None:
    """Raise when mutation coordinates disagree with a supplied wild-type sequence."""
    sequence = wild_type_sequence.strip().upper()
    if not sequence or set(sequence).difference(AMINO_ACIDS):
        raise ValueError("Wild-type sequence must contain only canonical amino acids")
    for row in frame[["mutation", "wild_type", "position"]].itertuples(index=False):
        if row.position > len(sequence):
            raise ValueError(f"{row.mutation} is outside a sequence of length {len(sequence)}")
        observed = sequence[row.position - 1]
        if observed != row.wild_type:
            raise ValueError(
                f"{row.mutation} expects {row.wild_type} at position {row.position}, "
                f"but the sequence contains {observed}"
            )
