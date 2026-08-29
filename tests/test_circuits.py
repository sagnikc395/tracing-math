from causal_circuits.circuits import CircuitNode, rank_nodes, select_top_fraction


def test_rank_and_select_nodes() -> None:
    ranked = rank_nodes(
        [
            [CircuitNode(0, 1, 2, -4.0), CircuitNode(0, 2, 2, 2.0)],
            [CircuitNode(0, 1, 2, 2.0), CircuitNode(0, 2, 2, 1.0)],
        ]
    )
    assert [(node.latent, node.attribution) for node in ranked] == [(1, 3.0), (2, 1.5)]
    assert select_top_fraction(ranked, 0.5) == ranked[:1]
