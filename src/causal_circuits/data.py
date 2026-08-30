"""ProcessBench loading, validation, prompting, and leakage-safe partitions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a mathematical reasoning verifier. Read the problem and numbered reasoning "
    "steps. When asked, answer with exactly CORRECT or INCORRECT."
)
VERDICT_QUESTION = "Is the reasoning valid up to and including the last displayed step?"


@dataclass(frozen=True)
class ProcessTrace:
    """One human-annotated ProcessBench solution trace."""

    trace_id: str
    source: str
    generator: str
    problem: str
    steps: tuple[str, ...]
    label: int
    final_answer_correct: bool

    def __post_init__(self) -> None:
        if not self.trace_id or not self.problem.strip() or not self.steps:
            raise ValueError("A trace needs a non-empty id, problem, and step list")
        if self.label < -1 or self.label >= len(self.steps):
            raise ValueError(
                f"Trace {self.trace_id}: label {self.label} is invalid for {len(self.steps)} steps"
            )
        if any(not str(step).strip() for step in self.steps):
            raise ValueError(f"Trace {self.trace_id} contains an empty step")

    @property
    def has_error(self) -> bool:
        return self.label >= 0

    @property
    def problem_group(self) -> str:
        normalized = " ".join(self.problem.split()).casefold()
        return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_record(cls, record: dict, source: str | None = None) -> ProcessTrace:
        return cls(
            trace_id=str(record.get("trace_id", record.get("id", ""))),
            source=str(source or record.get("source", "unknown")),
            generator=str(record.get("generator", "unknown")),
            problem=str(record["problem"]),
            steps=tuple(map(str, record["steps"])),
            label=int(record["label"]),
            final_answer_correct=bool(record["final_answer_correct"]),
        )

    def to_record(self) -> dict:
        record = asdict(self)
        record["steps"] = list(self.steps)
        return record


def load_huggingface_traces(
    dataset_name: str,
    splits: Sequence[str],
    *,
    max_examples_per_split: int | None = None,
) -> list[ProcessTrace]:
    """Download ProcessBench through Hugging Face and normalize all requested splits."""
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Install project dependencies before downloading ProcessBench"
        ) from error

    traces: list[ProcessTrace] = []
    for split in splits:
        dataset = load_dataset(dataset_name, split=split)
        if max_examples_per_split is not None:
            dataset = dataset.select(range(min(max_examples_per_split, len(dataset))))
        traces.extend(ProcessTrace.from_record(dict(record), source=split) for record in dataset)
    ids = [trace.trace_id for trace in traces]
    if len(ids) != len(set(ids)):
        raise ValueError("Trace ids are not unique across the requested dataset splits")
    return traces


def save_traces(traces: Iterable[ProcessTrace], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace.to_record(), ensure_ascii=False) + "\n")


def load_traces(path: str | Path) -> list[ProcessTrace]:
    with Path(path).open(encoding="utf-8") as handle:
        traces = [ProcessTrace.from_record(json.loads(line)) for line in handle if line.strip()]
    if not traces:
        raise ValueError(f"No traces found in {path}")
    return traces


def assign_partitions(
    traces: Sequence[ProcessTrace],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, str]:
    """Assign whole problem groups to deterministic, approximately stratified partitions."""
    grouped: dict[str, list[ProcessTrace]] = {}
    for trace in traces:
        grouped.setdefault(trace.problem_group, []).append(trace)

    assignments: dict[str, str] = {}
    for group_id, members in grouped.items():
        sources = "+".join(sorted({member.source for member in members}))
        status = "error" if any(member.has_error for member in members) else "correct"
        digest = hashlib.sha256(f"{seed}:{sources}:{status}:{group_id}".encode()).digest()
        draw = int.from_bytes(digest[:8], "big") / 2**64
        if draw < train_fraction:
            partition = "train"
        elif draw < train_fraction + validation_fraction:
            partition = "validation"
        else:
            partition = "test"
        assignments.update({member.trace_id: partition for member in members})
    return assignments


def step_marker(step_index: int) -> str:
    return f"<<END_STEP_{step_index}>>"


def format_user_content(
    problem: str, steps: Sequence[str], *, include_question: bool = True
) -> tuple[str, list[str]]:
    """Format a trace and return both its text and unique step-end markers."""
    blocks = [f"Problem:\n{problem.strip()}\n\nReasoning:"]
    markers: list[str] = []
    for index, step in enumerate(steps):
        marker = step_marker(index)
        markers.append(marker)
        blocks.append(f"[Step {index}]\n{step.strip()}\n{marker}")
    if include_question:
        blocks.append(f"Question: {VERDICT_QUESTION}")
    return "\n\n".join(blocks), markers


def iter_step_metadata(
    trace: ProcessTrace, partition: str, *, token_count: int
) -> Iterator[dict[str, object]]:
    for step_index, step in enumerate(trace.steps):
        yield {
            "trace_id": trace.trace_id,
            "problem_group": trace.problem_group,
            "source": trace.source,
            "generator": trace.generator,
            "partition": partition,
            "step_index": step_index,
            "n_steps": len(trace.steps),
            "step_fraction": (step_index + 1) / len(trace.steps),
            "first_error": trace.label,
            "has_error_trace": int(trace.has_error),
            "invalid_so_far": int(trace.has_error and step_index >= trace.label),
            "error_onset": int(trace.has_error and step_index == trace.label),
            "final_answer_correct": int(trace.final_answer_correct),
            "token_count": token_count,
            "step_text": step,
        }
