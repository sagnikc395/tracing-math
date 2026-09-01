"""Tests for the behavioral verdict readout."""

from types import SimpleNamespace

import pytest
import torch

from causal_circuits.experiment1.model import HuggingFaceMathModel


class _Tokenizer:
    def __call__(self, text, **_kwargs):
        if isinstance(text, str):
            return {"input_ids": {" Yes": [2], " No": [3], " many": [2, 3]}[text]}
        return {
            "input_ids": torch.tensor([[1, 1], [1, 0]]),
            "attention_mask": torch.tensor([[1, 1], [1, 0]]),
        }


class _Model:
    def __call__(self, *, attention_mask, **_kwargs):
        logits = torch.zeros((2, 2, 4))
        logits[0, 1, 2] = 2.0
        logits[1, 0, 3] = 2.0
        return SimpleNamespace(logits=logits)


def _adapter() -> HuggingFaceMathModel:
    adapter = object.__new__(HuggingFaceMathModel)
    adapter._torch = torch
    adapter.device = "cpu"
    adapter.max_length = 8
    adapter.tokenizer = _Tokenizer()
    adapter.model = _Model()
    adapter.decoder_layers = []
    return adapter


def test_yes_no_readout_uses_final_non_padding_token() -> None:
    scores = _adapter()._answer_probability_differences(
        ["first", "second"],
        [0, 0],
        correct_answer=" No",
        incorrect_answer=" Yes",
        layer=None,
        direction=None,
        magnitude=0.0,
    )
    assert scores[0] == pytest.approx(0.761594, abs=1e-6)
    assert scores[1] == pytest.approx(-0.761594, abs=1e-6)


def test_yes_no_readout_rejects_multitoken_answers() -> None:
    with pytest.raises(ValueError, match="exactly one token"):
        _adapter()._single_token_id(" many")
