import pandas as pd
import pytest

from causal_circuits.data import Substitution, load_dms, validate_against_sequence


def test_parse_substitution() -> None:
    mutation = Substitution.parse("A12G")
    assert mutation.wild_type == "A"
    assert mutation.position == 12
    assert mutation.mutant == "G"


@pytest.mark.parametrize("value", ["A0G", "A12A", "B12G", "A12*", "A12G:C13D"])
def test_reject_invalid_substitution(value: str) -> None:
    with pytest.raises(ValueError):
        Substitution.parse(value)


def test_load_and_validate_dms(tmp_path) -> None:
    path = tmp_path / "assay.csv"
    pd.DataFrame({"mutant": ["A1G", "C2D"], "score": [0.5, -1.0]}).to_csv(path, index=False)
    frame = load_dms(path, mutation_column="mutant", fitness_column="score", directionality=-1)
    assert frame["fitness"].tolist() == [-0.5, 1.0]
    validate_against_sequence(frame, "AC")


def test_sequence_mismatch_is_detected(tmp_path) -> None:
    path = tmp_path / "assay.csv"
    pd.DataFrame({"mutation": ["A1G"], "DMS_score": [0.5]}).to_csv(path, index=False)
    frame = load_dms(path)
    with pytest.raises(ValueError, match="sequence contains C"):
        validate_against_sequence(frame, "C")
