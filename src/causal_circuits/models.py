"""Model adapters. Heavy dependencies are imported only when used."""

from __future__ import annotations

from collections.abc import Mapping

from causal_circuits.data import AMINO_ACIDS


class HuggingFaceESM2:
    """Hugging Face ESM-2 adapter for masked-position log probabilities."""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "ESM-2 dependencies are missing; install with `uv sync --extra model`."
            ) from error

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu" and getattr(torch.backends, "mps", None):
                device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(device).eval()

    def position_log_probs(self, sequence: str, position: int) -> Mapping[str, float]:
        if not 0 <= position < len(sequence):
            raise IndexError(f"Position {position} is outside a sequence of length {len(sequence)}")
        masked = f"{sequence[:position]}{self.tokenizer.mask_token}{sequence[position + 1 :]}"
        encoded = self.tokenizer(masked, return_tensors="pt", add_special_tokens=True)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        mask_locations = (encoded["input_ids"][0] == self.tokenizer.mask_token_id).nonzero()
        if mask_locations.shape[0] != 1:
            raise RuntimeError("Expected exactly one mask token")
        mask_index = int(mask_locations[0, 0])
        with self._torch.inference_mode():
            logits = self.model(**encoded).logits[0, mask_index]
            log_probs = self._torch.log_softmax(logits, dim=-1)
        return {
            amino_acid: float(log_probs[self.tokenizer.convert_tokens_to_ids(amino_acid)].item())
            for amino_acid in AMINO_ACIDS
        }
