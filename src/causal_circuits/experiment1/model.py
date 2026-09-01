"""Experiment 1 Hugging Face adapter for extraction and interventions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from causal_circuits.experiment1.data import SYSTEM_PROMPT, ProcessTrace, format_user_content


class TraceTooLongError(ValueError):
    """Raised when a complete trace does not fit the preregistered context limit."""


@dataclass(frozen=True)
class TraceActivations:
    values: np.ndarray  # [step, hidden-state index, hidden dimension]
    token_count: int


class HuggingFaceMathModel:
    """Causal-LM wrapper with exact-boundary batching for a single GPU."""

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
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
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
        return self.extract_traces([trace])[0]

    def extract_traces(self, traces: Sequence[ProcessTrace]) -> list[TraceActivations]:
        """Extract a batch of traces while retaining each trace's exact boundaries."""
        if not traces:
            return []
        rendered_batch: list[str] = []
        marker_batch: list[list[str]] = []
        for trace in traces:
            rendered, markers = self._render(trace, trace.steps)
            rendered_batch.append(rendered)
            marker_batch.append(markers)

        encoded = self.tokenizer(
            rendered_batch,
            add_special_tokens=False,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets_batch = encoded.pop("offset_mapping").tolist()
        token_counts = encoded["attention_mask"].sum(dim=1).tolist()
        for token_count in token_counts:
            if token_count > self.max_length:
                raise TraceTooLongError(
                    f"Complete prompt has {token_count} tokens (limit {self.max_length})"
                )
        boundaries_batch = [
            [self._marker_token_index(rendered, marker, offsets) for marker in markers]
            for rendered, markers, offsets in zip(
                rendered_batch, marker_batch, offsets_batch, strict=True
            )
        ]
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            # Call the decoder directly: activation extraction does not need the very large
            # [batch, sequence, vocabulary] LM-head logits tensor.
            output = self.model.model(
                **model_inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        results = []
        for batch_index, (boundaries, token_count) in enumerate(
            zip(boundaries_batch, token_counts, strict=True)
        ):
            values = self._torch.stack(
                [
                    hidden[batch_index, boundaries, :].detach().float().cpu()
                    for hidden in output.hidden_states
                ],
                dim=1,
            ).numpy()
            results.append(
                TraceActivations(values=values.astype(np.float16), token_count=int(token_count))
            )
        return results

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
        return self.verdict_scores(
            [(trace, step_index)],
            correct_answer=correct_answer,
            incorrect_answer=incorrect_answer,
            layer=layer,
            direction=direction,
            magnitude=magnitude,
            batch_size=1,
        )[0]

    def verdict_scores(
        self,
        requests: Sequence[tuple[ProcessTrace, int]],
        *,
        correct_answer: str,
        incorrect_answer: str,
        layer: int | None = None,
        direction: np.ndarray | None = None,
        magnitude: float = 0.0,
        batch_size: int = 1,
    ) -> list[float]:
        """Score multiple trace boundaries, batching both candidate answers together."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        scores: list[float] = []
        for start in range(0, len(requests), batch_size):
            chunk = requests[start : start + batch_size]
            rendered_batch: list[str] = []
            boundary_batch: list[int] = []
            for trace, step_index in chunk:
                if not 0 <= step_index < len(trace.steps):
                    raise IndexError(f"Step {step_index} is outside trace {trace.trace_id}")
                rendered, markers = self._render(trace, trace.steps[: step_index + 1])
                boundary_encoding = self.tokenizer(
                    rendered,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                )
                if len(boundary_encoding["input_ids"]) > self.max_length:
                    raise TraceTooLongError(
                        "Verdict prompt exceeds the configured context limit"
                    )
                offsets = boundary_encoding["offset_mapping"]
                boundaries = [
                    self._marker_token_index(rendered, marker, offsets) for marker in markers
                ]
                rendered_batch.append(rendered)
                boundary_batch.append(boundaries[-1])
            scores.extend(
                self._answer_log_probability_differences(
                    rendered_batch,
                    boundary_batch,
                    correct_answer=correct_answer,
                    incorrect_answer=incorrect_answer,
                    layer=layer,
                    direction=direction,
                    magnitude=magnitude,
                )
            )
        return scores

    def _answer_log_probability_differences(
        self,
        rendered_batch: Sequence[str],
        boundaries: Sequence[int],
        *,
        correct_answer: str,
        incorrect_answer: str,
        layer: int | None,
        direction: np.ndarray | None,
        magnitude: float,
    ) -> list[float]:
        texts = [
            rendered + answer
            for rendered in rendered_batch
            for answer in (incorrect_answer, correct_answer)
        ]
        prompt_lengths = [len(rendered) for rendered in rendered_batch for _ in range(2)]
        encoded = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        if int(encoded["attention_mask"].sum(dim=1).max()) > self.max_length:
            raise TraceTooLongError("Verdict prompt exceeds the configured context limit")
        offsets_batch = encoded.pop("offset_mapping").tolist()
        answer_positions = [
            [index for index, (_, end) in enumerate(offsets) if end > prompt_length]
            for offsets, prompt_length in zip(offsets_batch, prompt_lengths, strict=True)
        ]
        if any(not positions or positions[0] == 0 for positions in answer_positions):
            raise RuntimeError("Could not identify tokens for a verdict answer")
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}

        handle = None
        if direction is not None and magnitude != 0.0:
            if layer is None or not 0 <= layer < self.n_decoder_layers:
                raise ValueError(f"Intervention layer must be in [0, {self.n_decoder_layers - 1}]")
            vector = self._torch.as_tensor(direction, device=self.device)
            injection_boundaries = self._torch.as_tensor(
                [boundary for boundary in boundaries for _ in range(2)], device=self.device
            )

            def inject(_module, args):
                hidden = args[0].clone()
                batch_indices = self._torch.arange(hidden.shape[0], device=self.device)
                hidden[batch_indices, injection_boundaries, :] += magnitude * vector.to(
                    hidden.dtype
                )
                return (hidden, *args[1:])

            handle = self.decoder_layers[layer].register_forward_pre_hook(inject)
        try:
            with self._torch.inference_mode():
                logits = self.model(**model_inputs, use_cache=False, return_dict=True).logits
            token_ids = model_inputs["input_ids"]
            selected_logits = self._torch.stack(
                [
                    logits[batch_index, position - 1, :]
                    for batch_index, positions in enumerate(answer_positions)
                    for position in positions
                ]
            )
            selected_token_ids = self._torch.stack(
                [
                    token_ids[batch_index, position]
                    for batch_index, positions in enumerate(answer_positions)
                    for position in positions
                ]
            )
            del logits
            selected_log_probs = self._torch.log_softmax(selected_logits.float(), dim=-1)
            token_log_probs = selected_log_probs.gather(
                1, selected_token_ids[:, None]
            ).squeeze(1)
            answer_scores = []
            offset = 0
            for positions in answer_positions:
                count = len(positions)
                answer_scores.append(float(token_log_probs[offset : offset + count].mean().item()))
                offset += count
            return [
                answer_scores[index] - answer_scores[index + 1]
                for index in range(0, len(answer_scores), 2)
            ]
        finally:
            if handle is not None:
                handle.remove()
