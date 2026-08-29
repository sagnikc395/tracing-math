"""Backend-neutral data structures for circuit discovery and interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class CircuitNode:
    layer: int
    latent: int
    token: int
    attribution: float


class SparseCircuitBackend(Protocol):
    """Contract the ProtoMech adapter must implement."""

    def attribute(
        self, sequence: str, position: int, wild_type: str, mutant: str
    ) -> list[CircuitNode]:
        """Attribute a mutant-versus-WT logit margin to latent-token nodes."""

    def score_with_nodes(
        self, sequence: str, position: int, wild_type: str, mutant: str, nodes: list[CircuitNode]
    ) -> float:
        """Score while retaining only selected nodes (sufficiency)."""

    def score_without_nodes(
        self, sequence: str, position: int, wild_type: str, mutant: str, nodes: list[CircuitNode]
    ) -> float:
        """Score while ablating selected nodes (necessity)."""


def rank_nodes(node_lists: list[list[CircuitNode]]) -> list[CircuitNode]:
    """Aggregate mean absolute attribution and rank unique latent-token nodes."""
    totals: dict[tuple[int, int, int], list[float]] = {}
    for nodes in node_lists:
        for node in nodes:
            totals.setdefault((node.layer, node.latent, node.token), []).append(
                abs(node.attribution)
            )
    ranked = [
        CircuitNode(layer, latent, token, float(np.mean(values)))
        for (layer, latent, token), values in totals.items()
    ]
    return sorted(ranked, key=lambda node: node.attribution, reverse=True)


def select_top_fraction(nodes: list[CircuitNode], fraction: float) -> list[CircuitNode]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    count = max(1, int(np.ceil(len(nodes) * fraction))) if nodes else 0
    return nodes[:count]


class ProtoMechBackend:
    """Integration boundary for the released ProtoMech CLT checkpoint.

    ProtoMech is external research code rather than a stable package. Keep its
    checkpoint/model-loading logic isolated here and record a pinned revision.
    """

    def __init__(self, checkpoint: str) -> None:
        self.checkpoint = checkpoint
        raise NotImplementedError(
            "Wire the pinned ProtoMech checkout into ProtoMechBackend after recording "
            "its commit and checkpoint hash in checkpoints/protomech/README.md."
        )
