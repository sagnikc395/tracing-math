"""Tests for Experiment 1 ProcessBench data handling."""

import pytest

from causal_circuits.experiment1.data import (
    ProcessTrace,
    assign_partitions,
    format_user_content,
    iter_step_metadata,
    load_traces,
    save_traces,
)


def make_trace(trace_id: str = "math-0", *, problem: str = "What is 1+1?") -> ProcessTrace:
    return ProcessTrace(
        trace_id=trace_id,
        source="math",
        generator="test-model",
        problem=problem,
        steps=("We add the numbers.", "The answer is 3."),
        label=1,
        final_answer_correct=False,
    )


def test_trace_validation_and_step_labels() -> None:
    trace = make_trace()
    rows = list(iter_step_metadata(trace, "train", token_count=50))
    assert [row["invalid_so_far"] for row in rows] == [0, 1]
    assert [row["error_onset"] for row in rows] == [0, 1]
    with pytest.raises(ValueError, match="label"):
        ProcessTrace(**{**trace.__dict__, "label": 2})


def test_prompt_has_unique_boundaries() -> None:
    text, markers = format_user_content("p", ("first", "second"))
    assert markers == ["<<END_STEP_0>>", "<<END_STEP_1>>"]
    assert all(text.count(marker) == 1 for marker in markers)
    assert text.index(markers[0]) < text.index(markers[1])


def test_jsonl_round_trip(tmp_path) -> None:
    path = tmp_path / "traces.jsonl"
    save_traces([make_trace()], path)
    assert load_traces(path) == [make_trace()]


def test_problem_duplicates_stay_in_the_same_partition() -> None:
    traces = [make_trace("a"), make_trace("b"), make_trace("c", problem="Different")]
    assignments = assign_partitions(traces, seed=42, train_fraction=0.6, validation_fraction=0.2)
    assert assignments["a"] == assignments["b"]
    assert set(assignments.values()).issubset({"train", "validation", "test"})


def test_partitions_are_balanced_within_source_and_error_status() -> None:
    traces = []
    for index in range(20):
        trace = make_trace(f"error-{index}", problem=f"Error problem {index}")
        traces.append(trace)
        correct = ProcessTrace(
            **{
                **trace.__dict__,
                "trace_id": f"correct-{index}",
                "problem": f"Correct problem {index}",
                "label": -1,
                "final_answer_correct": True,
            }
        )
        traces.append(correct)
    assignments = assign_partitions(traces, seed=42, train_fraction=0.6, validation_fraction=0.2)
    for prefix in ("error", "correct"):
        counts = [
            sum(assignments[f"{prefix}-{index}"] == partition for index in range(20))
            for partition in ("train", "validation", "test")
        ]
        assert counts == [12, 4, 4]
