"""Hugging Face adapter for activation extraction and causal interventions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from causal_circuits.data import SYSTEM_PROMPT, ProcessTrace, format_user_content


class TraceTooLongError(ValueError):
    """Raised when a complete trace does not fit the preregistered context limit."""


@dataclass(frozen=True)
class TraceActivations:
    values: np.ndarray  # [step, hidden-state index, hidden dimension]
    token_count: int


class HuggingFaceMathModel:
    """Small causal-LM wrapper designed for a single Colab T4."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        dtype: str = "float16",
        max_length: int = 2048,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Install the project dependencies before loading the model"
            ) from error

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu" and getattr(torch.backends, "mps", None):
                device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "cpu" and dtype == "float16":
            dtype = "float32"

        self._torch = torch
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError("A fast tokenizer is required to locate exact step boundaries")
        torch_dtype = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()

        try:
            self.decoder_layers = self.model.model.layers
        except AttributeError as error:
            raise RuntimeError(
                "This intervention adapter expects a Qwen/Llama-style model.model.layers stack"
            ) from error

    @property
    def n_decoder_layers(self) -> int:
        return len(self.decoder_layers)

    def _render(self, trace: ProcessTrace, steps: Sequence[str]) -> tuple[str, list[str]]:
        user_content, markers = format_user_content(trace.problem, steps)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return rendered, markers

    def _encode_with_boundaries(
        self, rendered: str, markers: Sequence[str]
    ) -> tuple[dict[str, object], list[int]]:
        encoded = self.tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        token_count = int(encoded["input_ids"].shape[1])
        if token_count > self.max_length:
            raise TraceTooLongError(
                f"Complete prompt has {token_count} tokens (limit {self.max_length})"
            )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        boundaries = [self._marker_token_index(rendered, marker, offsets) for marker in markers]
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        return model_inputs, boundaries

    @staticmethod
    def _marker_token_index(rendered: str, marker: str, offsets: Sequence[Sequence[int]]) -> int:
        start = rendered.find(marker)
        if start < 0 or rendered.find(marker, start + 1) >= 0:
            raise ValueError(f"Expected exactly one marker {marker!r} in the rendered prompt")
        end = start + len(marker)
        candidates = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_end > start and token_end <= end and token_end > token_start
        ]
        if not candidates:
            raise RuntimeError(f"Tokenizer offsets did not cover step marker {marker!r}")
        return candidates[-1]

    def extract_trace(self, trace: ProcessTrace) -> TraceActivations:
        """Extract every step boundary with a single causal forward pass."""
        rendered, markers = self._render(trace, trace.steps)
        model_inputs, boundaries = self._encode_with_boundaries(rendered, markers)
        with self._torch.inference_mode():
            output = self.model(
                **model_inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        values = self._torch.stack(
            [hidden[0, boundaries, :].detach().float().cpu() for hidden in output.hidden_states],
            dim=1,
        ).numpy()
        return TraceActivations(
            values=values.astype(np.float16),
            token_count=len(model_inputs["input_ids"][0]),
        )

    def verdict_score(
        self,
        trace: ProcessTrace,
        step_index: int,
        *,
        correct_answer: str,
        incorrect_answer: str,
        layer: int | None = None,
        direction: np.ndarray | None = None,
        magnitude: float = 0.0,
    ) -> float:
        """Return mean log P(INCORRECT) - mean log P(CORRECT), optionally intervened."""
        if not 0 <= step_index < len(trace.steps):
            raise IndexError(f"Step {step_index} is outside trace {trace.trace_id}")
        rendered, markers = self._render(trace, trace.steps[: step_index + 1])
        _, boundaries = self._encode_with_boundaries(rendered, markers)
        boundary = boundaries[-1]
        kwargs = {"layer": layer, "direction": direction, "magnitude": magnitude}
        incorrect = self._answer_log_probability(
            rendered, incorrect_answer, boundary=boundary, **kwargs
        )
        correct = self._answer_log_probability(
            rendered, correct_answer, boundary=boundary, **kwargs
        )
        return incorrect - correct

    def _answer_log_probability(
        self,
        rendered: str,
        answer: str,
        *,
        boundary: int,
        layer: int | None,
        direction: np.ndarray | None,
        magnitude: float,
    ) -> float:
        full_text = rendered + answer
        encoded = self.tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[1] > self.max_length:
            raise TraceTooLongError("Verdict prompt exceeds the configured context limit")
        offsets = encoded.pop("offset_mapping")[0].tolist()
        answer_positions = [index for index, (_, end) in enumerate(offsets) if end > len(rendered)]
        if not answer_positions or answer_positions[0] == 0:
            raise RuntimeError(f"Could not identify tokens for answer {answer!r}")
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}

        handle = None
        if direction is not None and magnitude != 0.0:
            if layer is None or not 0 <= layer < self.n_decoder_layers:
                raise ValueError(f"Intervention layer must be in [0, {self.n_decoder_layers - 1}]")
            vector = self._torch.as_tensor(direction, device=self.device)

            def inject(_module, args):
                hidden = args[0].clone()
                hidden[:, boundary, :] += magnitude * vector.to(hidden.dtype)
                return (hidden, *args[1:])

            handle = self.decoder_layers[layer].register_forward_pre_hook(inject)
        try:
            with self._torch.inference_mode():
                logits = self.model(**model_inputs, use_cache=False, return_dict=True).logits[0]
                log_probs = self._torch.log_softmax(logits.float(), dim=-1)
            token_ids = model_inputs["input_ids"][0]
            scores = [log_probs[position - 1, token_ids[position]] for position in answer_positions]
            return float(self._torch.stack(scores).mean().item())
        finally:
            if handle is not None:
                handle.remove()
