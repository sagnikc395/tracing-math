import pandas as pd
import pytest

from causal_circuits.analysis import residue_intolerance, summarize_scores
from causal_circuits.scoring import score_dms


class FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def position_log_probs(self, sequence: str, position: int) -> dict[str, float]:
        self.calls += 1
        return {"A": -2.0, "C": -1.0, "D": -0.5, "G": -3.0}


def test_scores_and_caches_by_position() -> None:
    frame = pd.DataFrame(
        {
            "mutation": ["A1C", "A1D", "C2A"],
            "wild_type": ["A", "A", "C"],
            "position": [1, 1, 2],
            "mutant": ["C", "D", "A"],
            "fitness": [0.1, 0.8, -0.4],
        }
    )
    model = FakeModel()
    result = score_dms(model, "AC", frame)
    assert result["zero_shot_score"].tolist() == pytest.approx([1.0, 1.5, -1.0])
    assert model.calls == 2
    assert summarize_scores(result)["n_variants"] == 3


def test_residue_intolerance() -> None:
    frame = pd.DataFrame({"position": [1, 1, 2], "fitness": [-1.0, -3.0, 0.5]})
    result = residue_intolerance(frame)
    assert result["intolerance"].tolist() == pytest.approx([2.0, -0.5])
