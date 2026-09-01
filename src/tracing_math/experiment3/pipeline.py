"""Resumable orchestration for the exploratory workshop follow-ups."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from tracing_math.experiment1.data import (
    ProcessTrace,
    assign_partitions,
    iter_step_metadata,
    load_traces,
)
from tracing_math.experiment1.model import HuggingFaceMathModel, TraceTooLongError
from tracing_math.experiment1.pipeline import load_activation_shards
from tracing_math.experiment1.probes import binary_metrics
from tracing_math.experiment3.analysis import (
    analyze_fixed_boundary_locations,
    build_matched_transition_dataset,
    fit_transition_probes,
    summarize_counterfactual_patching,
)
from tracing_math.experiment3.config import ExtendedFollowupConfig

BOUNDARY_LOCATIONS = ("step_content", "marker")
PATCH_CONDITIONS = (
    "correct_baseline",
    "error_baseline",
    "error_state_into_correct",
    "correct_state_into_error",
)


def fit_and_save_transition_probe(config: ExtendedFollowupConfig) -> dict[str, Any]:
    activations, metadata = load_activation_shards(config.experiment1_dir)
    transitions, transition_metadata = build_matched_transition_dataset(
        activations, metadata
    )
    result = fit_transition_probes(
        transitions,
        transition_metadata,
        c_values=config.c_values,
        max_iter=config.max_iter,
        bootstrap_samples=config.bootstrap_samples,
        confidence_level=config.confidence_level,
        seed=config.seed,
    )
    output = config.output_dir / "transition_probe"
    output.mkdir(parents=True, exist_ok=True)
    result["metrics"].to_csv(output / "layer_metrics.csv", index=False)
    result["predictions"].to_csv(output / "test_predictions.csv", index=False)
    result["controls"].to_csv(output / "controls.csv", index=False)
    result["bootstrap"].to_csv(output / "bootstrap_summary.csv", index=False)
    transition_metadata.to_csv(output / "matched_transitions.csv", index=False)
    np.savez(
        output / "direction.npz",
        direction=result["direction"],
        selected_layer=result["selected_layer"],
        selected_c=result["selected_c"],
    )
    summary = {
        "status": "complete",
        "analysis_scope": "post_hoc",
        "selected_layer": result["selected_layer"],
        "selected_c": result["selected_c"],
        "test_pairs": int(result["predictions"]["pair_id"].nunique()),
        "updated_at": _utc_now(),
    }
    _atomic_write_text(output / "summary.json", json.dumps(summary, indent=2))
    return summary


def extract_boundary_control_shards(config: ExtendedFollowupConfig) -> dict[str, int]:
    traces = load_traces(config.data_path)
    partitions = assign_partitions(
        traces, seed=config.seed, train_fraction=0.6, validation_fraction=0.2
    )
    output = config.output_dir / "boundary_control"
    shard_dir = output / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    _check_boundary_identity(config, output)
    model = HuggingFaceMathModel(
        config.model_name,
        device=config.device,
        dtype=config.dtype,
        max_length=config.max_length,
    )
    totals = {"traces": len(traces), "extracted": 0, "skipped_long": 0, "cached": 0}
    shard_size = config.extraction_save_every
    for start in tqdm(range(0, len(traces), shard_size), desc="boundary-control shards"):
        end = min(start + shard_size, len(traces))
        stem = f"shard_{start:05d}_{end - 1:05d}"
        array_path = shard_dir / f"{stem}.npy"
        metadata_path = shard_dir / f"{stem}.csv"
        manifest_path = shard_dir / f"{stem}.json"
        if all(path.exists() for path in (array_path, metadata_path, manifest_path)):
            manifest = json.loads(manifest_path.read_text())
            totals["cached"] += int(manifest["extracted"])
            totals["skipped_long"] += int(manifest["skipped_long"])
            _write_progress(output, totals, start, end, len(traces))
            continue

        arrays: list[np.ndarray] = []
        rows: list[dict[str, object]] = []
        skipped = []
        shard_traces = traces[start:end]
        for batch_start in range(0, len(shard_traces), config.extraction_batch_size):
            batch = shard_traces[
                batch_start : batch_start + config.extraction_batch_size
            ]
            try:
                results = model.extract_traces_at_locations(batch)
                retained = batch
            except TraceTooLongError:
                results = []
                retained = []
                for trace in batch:
                    try:
                        results.extend(model.extract_traces_at_locations([trace]))
                        retained.append(trace)
                    except TraceTooLongError as error:
                        skipped.append({"trace_id": trace.trace_id, "reason": str(error)})
            for trace, result in zip(retained, results, strict=True):
                arrays.append(
                    np.stack([result.values[name] for name in BOUNDARY_LOCATIONS], axis=1)
                )
                rows.extend(
                    iter_step_metadata(
                        trace, partitions[trace.trace_id], token_count=result.token_count
                    )
                )
        if not arrays:
            raise RuntimeError(f"No extractable traces in {stem}")
        values = np.concatenate(arrays)
        frame = pd.DataFrame(rows)
        if len(values) != len(frame):
            raise RuntimeError(f"Boundary activation/metadata mismatch in {stem}")
        _atomic_save_array(array_path, values)
        _atomic_save_csv(metadata_path, frame)
        _atomic_write_text(
            manifest_path,
            json.dumps(
                {
                    "attempted": end - start,
                    "extracted": len(arrays),
                    "skipped_long": len(skipped),
                    "skipped": skipped,
                    "activation_rows": len(values),
                    "locations": list(BOUNDARY_LOCATIONS),
                },
                indent=2,
            ),
        )
        totals["extracted"] += len(arrays)
        totals["skipped_long"] += len(skipped)
        _write_progress(output, totals, start, end, len(traces))
    return totals


def analyze_boundary_controls(config: ExtendedFollowupConfig) -> dict[str, Any]:
    activations, metadata = load_boundary_control_shards(config.output_dir)
    direction_artifact = np.load(config.experiment1_dir / "probes" / "directions.npz")
    layer = int(direction_artifact["selected_layer"])
    c_value = float(direction_artifact["c_values"][layer])
    metrics, predictions, comparison = analyze_fixed_boundary_locations(
        activations,
        metadata,
        BOUNDARY_LOCATIONS,
        layer=layer,
        c_value=c_value,
        max_iter=config.max_iter,
        bootstrap_samples=config.bootstrap_samples,
        confidence_level=config.confidence_level,
        seed=config.seed,
    )
    output = config.output_dir / "boundary_control"
    metrics.to_csv(output / "metrics.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    comparison.to_csv(output / "paired_differences.csv", index=False)
    summary = {
        "status": "complete",
        "analysis_scope": "post_hoc",
        "fixed_layer": layer,
        "fixed_c": c_value,
        "locations": list(BOUNDARY_LOCATIONS),
        "updated_at": _utc_now(),
    }
    _atomic_write_text(output / "summary.json", json.dumps(summary, indent=2))
    return summary


def load_boundary_control_shards(output_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    shard_dir = output_dir / "boundary_control" / "shards"
    array_paths = sorted(shard_dir.glob("shard_*.npy"))
    if not array_paths:
        raise FileNotFoundError(f"No boundary-control shards found under {shard_dir}")
    arrays = []
    frames = []
    expected_tail = None
    for array_path in array_paths:
        metadata_path = array_path.with_suffix(".csv")
        manifest_path = array_path.with_suffix(".json")
        if not metadata_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"Incomplete boundary-control shard {array_path.stem}")
        manifest = json.loads(manifest_path.read_text())
        if tuple(manifest["locations"]) != BOUNDARY_LOCATIONS:
            raise ValueError(f"Unexpected boundary locations in {manifest_path}")
        array = np.load(array_path, mmap_mode="r")
        frame = pd.read_csv(metadata_path)
        if len(array) != len(frame):
            raise ValueError(f"Row mismatch between {array_path} and {metadata_path}")
        if expected_tail is None:
            expected_tail = array.shape[1:]
        elif array.shape[1:] != expected_tail:
            raise ValueError("Boundary-control shards have inconsistent dimensions")
        arrays.append(np.asarray(array))
        frames.append(frame)
    return np.concatenate(arrays), pd.concat(frames, ignore_index=True)


def prepare_counterfactual_template(
    config: ExtendedFollowupConfig, *, force: bool = False
) -> Path:
    output = config.counterfactual_pairs_path
    if output.exists() and not force:
        return output
    traces = [trace for trace in load_traces(config.data_path) if trace.has_error]
    ordered = sorted(
        traces,
        key=lambda trace: hashlib.sha256(
            f"{config.seed}:{trace.source}:{trace.trace_id}".encode()
        ).digest(),
    )[: config.counterfactual_template_size]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for trace in ordered:
            handle.write(
                json.dumps(
                    {
                        "pair_id": trace.trace_id,
                        "source": trace.source,
                        "problem": trace.problem,
                        "prefix_steps": list(trace.steps[: trace.label]),
                        "error_step": trace.steps[trace.label],
                        "corrected_step": "",
                        "verified": False,
                        "annotation_notes": "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    temporary.replace(output)
    return output


def run_counterfactual_patching(config: ExtendedFollowupConfig) -> dict[str, Any]:
    pairs = _load_counterfactual_pairs(config.counterfactual_pairs_path)
    artifact = np.load(config.experiment1_dir / "probes" / "directions.npz")
    layer = int(artifact["selected_intervention_layer"])
    model = HuggingFaceMathModel(
        config.model_name,
        device=config.device,
        dtype=config.dtype,
        max_length=config.max_length,
    )
    output = config.output_dir / "counterfactual_patching"
    output.mkdir(parents=True, exist_ok=True)
    _check_counterfactual_identity(config, pairs, layer, output)
    kwargs = {"correct_answer": " No", "incorrect_answer": " Yes"}
    baseline_path = output / "baseline.checkpoint.csv"
    baseline_rows = (
        pd.read_csv(baseline_path).to_dict(orient="records")
        if baseline_path.exists()
        else []
    )
    baseline_completed = (
        set(pd.DataFrame(baseline_rows)["pair_id"].astype(str)) if baseline_rows else set()
    )
    baseline_pending = [pair for pair in pairs if pair["pair_id"] not in baseline_completed]
    for start in tqdm(
        range(0, len(baseline_pending), config.patching_batch_size),
        desc="counterfactual baseline",
    ):
        chunk = baseline_pending[start : start + config.patching_batch_size]
        correct_traces = [_pair_trace(pair, correct=True) for pair in chunk]
        error_traces = [_pair_trace(pair, correct=False) for pair in chunk]
        correct_requests = [(trace, len(trace.steps) - 1) for trace in correct_traces]
        error_requests = [(trace, len(trace.steps) - 1) for trace in error_traces]
        correct_scores = model.verdict_scores(
            correct_requests, batch_size=config.patching_batch_size, **kwargs
        )
        error_scores = model.verdict_scores(
            error_requests, batch_size=config.patching_batch_size, **kwargs
        )
        for index, pair in enumerate(chunk):
            baseline_rows.extend(
                [
                    {
                        "pair_id": pair["pair_id"],
                        "source": pair["source"],
                        "layer": layer,
                        "condition": "correct_baseline",
                        "verdict_score": float(correct_scores[index]),
                    },
                    {
                        "pair_id": pair["pair_id"],
                        "source": pair["source"],
                        "layer": layer,
                        "condition": "error_baseline",
                        "verdict_score": float(error_scores[index]),
                    },
                ]
            )
        _atomic_save_csv(baseline_path, pd.DataFrame(baseline_rows))

    baseline = pd.DataFrame(baseline_rows)
    baseline_labels = baseline["condition"].eq("error_baseline").to_numpy(dtype=int)
    baseline_probabilities = (baseline["verdict_score"].to_numpy(dtype=float) + 1) / 2
    baseline_metrics = binary_metrics(baseline_labels, baseline_probabilities)
    baseline_metrics["n_pairs"] = len(pairs)
    baseline_metrics["gate_passed"] = bool(
        baseline_metrics["specificity"] > 0 and baseline_metrics["auroc"] > 0.5
    )
    _atomic_write_text(
        output / "baseline_metrics.json", json.dumps(baseline_metrics, indent=2)
    )
    if not baseline_metrics["gate_passed"]:
        baseline.to_csv(output / "individual.csv", index=False)
        pd.DataFrame(
            columns=["effect", "estimate", "ci_low", "ci_high", "n_pairs"]
        ).to_csv(output / "summary.csv", index=False)
        status = {
            "status": "stopped_after_baseline",
            "reason": "counterfactual verdict did not pass AUROC and specificity gates",
            "analysis_scope": "post_hoc",
            "layer": layer,
            "pairs_completed": len(pairs),
            "updated_at": _utc_now(),
        }
        _atomic_write_text(output / "progress.json", json.dumps(status, indent=2))
        return status

    checkpoint_path = output / "individual.checkpoint.csv"
    patch_rows = (
        pd.read_csv(checkpoint_path).to_dict(orient="records")
        if checkpoint_path.exists()
        else []
    )
    completed = set(pd.DataFrame(patch_rows)["pair_id"].astype(str)) if patch_rows else set()
    pending = [pair for pair in pairs if pair["pair_id"] not in completed]
    for start in tqdm(
        range(0, len(pending), config.patching_batch_size), desc="counterfactual patches"
    ):
        chunk = pending[start : start + config.patching_batch_size]
        correct_traces = [_pair_trace(pair, correct=True) for pair in chunk]
        error_traces = [_pair_trace(pair, correct=False) for pair in chunk]
        correct_requests = [(trace, len(trace.steps) - 1) for trace in correct_traces]
        error_requests = [(trace, len(trace.steps) - 1) for trace in error_traces]
        correct_states = model.boundary_input_states(
            correct_requests, layer=layer, batch_size=config.patching_batch_size
        )
        error_states = model.boundary_input_states(
            error_requests, layer=layer, batch_size=config.patching_batch_size
        )
        error_into_correct = model.verdict_scores(
            correct_requests,
            layer=layer,
            replacement_states=error_states,
            batch_size=config.patching_batch_size,
            **kwargs,
        )
        correct_into_error = model.verdict_scores(
            error_requests,
            layer=layer,
            replacement_states=correct_states,
            batch_size=config.patching_batch_size,
            **kwargs,
        )
        for index, pair in enumerate(chunk):
            patch_rows.extend(
                {
                    "pair_id": pair["pair_id"],
                    "source": pair["source"],
                    "layer": layer,
                    "condition": condition,
                    "verdict_score": float(score),
                }
                for condition, score in (
                    ("error_state_into_correct", error_into_correct[index]),
                    ("correct_state_into_error", correct_into_error[index]),
                )
            )
        _atomic_save_csv(checkpoint_path, pd.DataFrame(patch_rows))
        _atomic_write_text(
            output / "progress.json",
            json.dumps(
                {
                    "status": "running",
                    "pairs_completed": len(completed) + start + len(chunk),
                    "pairs_total": len(pairs),
                    "updated_at": _utc_now(),
                },
                indent=2,
            ),
        )
    individual = pd.concat([baseline, pd.DataFrame(patch_rows)], ignore_index=True)
    if set(individual["condition"]) != set(PATCH_CONDITIONS):
        raise RuntimeError("Counterfactual patching did not produce all required conditions")
    individual.to_csv(output / "individual.csv", index=False)
    summary = summarize_counterfactual_patching(
        individual,
        samples=config.bootstrap_samples,
        confidence_level=config.confidence_level,
        seed=config.seed,
    )
    summary.to_csv(output / "summary.csv", index=False)
    status = {
        "status": "complete",
        "analysis_scope": "post_hoc",
        "layer": layer,
        "pairs_completed": len(pairs),
        "updated_at": _utc_now(),
    }
    _atomic_write_text(output / "progress.json", json.dumps(status, indent=2))
    return status


def _pair_trace(pair: dict[str, Any], *, correct: bool) -> ProcessTrace:
    step = pair["corrected_step"] if correct else pair["error_step"]
    steps = (*pair["prefix_steps"], step)
    return ProcessTrace(
        trace_id=f"{pair['pair_id']}:{'correct' if correct else 'error'}",
        source=str(pair["source"]),
        generator="counterfactual",
        problem=str(pair["problem"]),
        steps=tuple(map(str, steps)),
        label=-1 if correct else len(steps) - 1,
        final_answer_correct=correct,
    )


def _load_counterfactual_pairs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Create and annotate {path} with prepare-counterfactual-template first"
        )
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    verified = []
    for row in rows:
        if not row.get("verified"):
            continue
        required = {"pair_id", "source", "problem", "prefix_steps", "error_step", "corrected_step"}
        if missing := required.difference(row):
            raise ValueError(f"Counterfactual row is missing fields: {sorted(missing)}")
        if not str(row["corrected_step"]).strip():
            raise ValueError(f"Verified pair {row['pair_id']} has no corrected step")
        if str(row["corrected_step"]).strip() == str(row["error_step"]).strip():
            raise ValueError(f"Pair {row['pair_id']} has identical corrected and error steps")
        verified.append(row)
    if not verified:
        raise ValueError("No verified counterfactual pairs are available for patching")
    ids = [str(row["pair_id"]) for row in verified]
    if len(ids) != len(set(ids)):
        raise ValueError("Counterfactual pair ids must be unique")
    return verified


def _check_boundary_identity(config: ExtendedFollowupConfig, output: Path) -> None:
    identity = {
        "dataset_sha256": hashlib.sha256(config.data_path.read_bytes()).hexdigest(),
        "model": config.model_name,
        "dtype": config.dtype,
        "max_length": config.max_length,
        "locations": list(BOUNDARY_LOCATIONS),
    }
    path = output / "extraction_identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Boundary-control output contains shards from a different run")
    _atomic_write_text(path, json.dumps(identity, indent=2))


def _check_counterfactual_identity(
    config: ExtendedFollowupConfig,
    pairs: list[dict[str, Any]],
    layer: int,
    output: Path,
) -> None:
    pair_bytes = json.dumps(pairs, sort_keys=True, ensure_ascii=False).encode()
    identity = {
        "pairs_sha256": hashlib.sha256(pair_bytes).hexdigest(),
        "model": config.model_name,
        "dtype": config.dtype,
        "max_length": config.max_length,
        "layer": layer,
    }
    path = output / "identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError(
            "Counterfactual pairs or model settings changed after checkpointing; "
            "use a new output directory"
        )
    _atomic_write_text(path, json.dumps(identity, indent=2))


def _write_progress(
    output: Path, totals: dict[str, int], start: int, end: int, trace_count: int
) -> None:
    completed = totals["cached"] + totals["extracted"] + totals["skipped_long"]
    _atomic_write_text(
        output / "progress.json",
        json.dumps(
            {
                "status": "complete" if end >= trace_count else "running",
                "last_shard": {"start": start, "end_exclusive": end},
                "traces_total": trace_count,
                "traces_completed": completed,
                **totals,
                "updated_at": _utc_now(),
            },
            indent=2,
        ),
    )


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _atomic_save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values)
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
