"""Natural step-end activation extraction for Experiment 2."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from causal_circuits.analysis import _paired_comparison_bootstrap, fit_layer_probes
from causal_circuits.config import AnalysisConfig
from causal_circuits.data import assign_partitions, iter_step_metadata, load_traces
from causal_circuits.experiment2_analysis import expanded_analysis_configs, save_probe_results
from causal_circuits.experiment2_config import Experiment2Config
from causal_circuits.experiment2_runtime import update_stage_progress, utc_now
from causal_circuits.models import HuggingFaceMathModel, TraceActivations, TraceTooLongError


def extract_semantic_activation_shards(config: Experiment2Config) -> dict[str, int]:
    """Extract hidden states at each step's last non-whitespace semantic token."""
    traces = load_traces(config.data.path)
    partitions = assign_partitions(
        traces,
        seed=config.seed,
        train_fraction=config.data.train_fraction,
        validation_fraction=config.data.validation_fraction,
    )
    shard_dir = config.output_dir / "semantic_activation_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    _check_identity(config)
    model = HuggingFaceMathModel(
        config.model.name,
        device=config.model.device,
        dtype=config.model.dtype,
        max_length=config.model.max_length,
    )
    totals = {"traces": len(traces), "extracted": 0, "skipped_long": 0, "cached": 0}
    chunk_size = config.semantic_extraction.save_every
    for start in tqdm(range(0, len(traces), chunk_size), desc="semantic activation shards"):
        end = min(start + chunk_size, len(traces))
        stem = f"shard_{start:05d}_{end - 1:05d}"
        array_path = shard_dir / f"{stem}.npy"
        metadata_path = shard_dir / f"{stem}.csv"
        manifest_path = shard_dir / f"{stem}.json"
        if array_path.exists() and metadata_path.exists() and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            totals["cached"] += int(manifest["extracted"])
            totals["skipped_long"] += int(manifest["skipped_long"])
            _write_progress(config.output_dir, totals, end, len(traces))
            continue

        arrays = []
        rows = []
        skipped = []
        # Similar-length prompts belong in the same batch.  Decoder cost is set by the
        # longest prompt in a padded batch, so the previous dataset order could make a
        # short trace pay for almost 2,048 tokens of padding.  Ordering is immaterial to
        # the trace-grouped analysis, and rows remain paired with their activations.
        shard_traces = sorted(
            traces[start:end],
            key=lambda trace: len(trace.problem) + sum(map(len, trace.steps)),
        )
        batch_size = config.semantic_extraction.batch_size
        for batch_start in tqdm(
            range(0, len(shard_traces), batch_size),
            leave=False,
            desc=stem,
        ):
            batch = shard_traces[batch_start : batch_start + batch_size]
            try:
                batch_results = extract_semantic_traces(model, batch)
            except TraceTooLongError:
                retained = []
                batch_results = []
                for trace in batch:
                    try:
                        batch_results.extend(extract_semantic_traces(model, [trace]))
                        retained.append(trace)
                    except TraceTooLongError as error:
                        skipped.append({"trace_id": trace.trace_id, "reason": str(error)})
                batch = retained
            for trace, result in zip(batch, batch_results, strict=True):
                arrays.append(result.values)
                rows.extend(
                    iter_step_metadata(
                        trace,
                        partitions[trace.trace_id],
                        token_count=result.token_count,
                    )
                )
        if not arrays:
            raise RuntimeError(f"No extractable traces in semantic shard {stem}")
        activations = np.concatenate(arrays, axis=0)
        metadata = pd.DataFrame(rows)
        if len(activations) != len(metadata):
            raise RuntimeError(f"Activation/metadata mismatch while writing {stem}")
        _atomic_save_array(array_path, activations)
        _atomic_save_csv(metadata_path, metadata)
        _atomic_write_text(
            manifest_path,
            json.dumps(
                {
                    "attempted": end - start,
                    "extracted": len(arrays),
                    "skipped_long": len(skipped),
                    "skipped": skipped,
                    "activation_rows": len(activations),
                    "boundary_mode": "last_non_whitespace_token_before_marker",
                },
                indent=2,
            ),
        )
        totals["extracted"] += len(arrays)
        totals["skipped_long"] += len(skipped)
        _write_progress(config.output_dir, totals, end, len(traces))
    return totals


def extract_semantic_traces(
    adapter: HuggingFaceMathModel,
    traces,
) -> list[TraceActivations]:
    """Extract natural step-end states from a batch using one causal pass per trace."""
    if not traces:
        return []
    rendered_batch = []
    marker_batch = []
    for trace in traces:
        rendered, markers = adapter._render(trace, trace.steps)
        rendered_batch.append(rendered)
        marker_batch.append(markers)
    encoded = adapter.tokenizer(
        rendered_batch,
        add_special_tokens=False,
        padding=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets_batch = encoded.pop("offset_mapping").tolist()
    token_counts = encoded["attention_mask"].sum(dim=1).tolist()
    if max(token_counts) > adapter.max_length:
        raise TraceTooLongError(
            f"Complete semantic-boundary prompt exceeds {adapter.max_length} tokens"
        )
    boundaries_batch = [
        [semantic_token_before_marker(rendered, marker, offsets) for marker in markers]
        for rendered, markers, offsets in zip(
            rendered_batch,
            marker_batch,
            offsets_batch,
            strict=True,
        )
    ]
    model_inputs = {key: value.to(adapter.device) for key, value in encoded.items()}
    with adapter._torch.inference_mode():
        output = adapter.model.model(
            **model_inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
    # Gather all requested step boundaries on the GPU and transfer them to the CPU once.
    # The former nested trace/layer loop caused batch_size * n_layers synchronizations.
    batch_indices = adapter._torch.cat(
        [
            adapter._torch.full(
                (len(boundaries),),
                batch_index,
                device=output.hidden_states[0].device,
                dtype=adapter._torch.long,
            )
            for batch_index, boundaries in enumerate(boundaries_batch)
        ]
    )
    boundary_indices = adapter._torch.as_tensor(
        [boundary for boundaries in boundaries_batch for boundary in boundaries],
        device=output.hidden_states[0].device,
        dtype=adapter._torch.long,
    )
    gathered = (
        adapter._torch.stack(
            [hidden[batch_indices, boundary_indices] for hidden in output.hidden_states],
            dim=1,
        )
        .detach()
        .to(device="cpu", dtype=adapter._torch.float32)
        .numpy()
    )
    split_points = np.cumsum([len(boundaries) for boundaries in boundaries_batch[:-1]])
    trace_values = np.split(gathered, split_points)

    results = []
    for values, token_count in zip(trace_values, token_counts, strict=True):
        results.append(
            TraceActivations(values=values.astype(np.float16), token_count=int(token_count))
        )
    return results


def semantic_token_before_marker(
    rendered: str,
    marker: str,
    offsets,
) -> int:
    """Locate the final token containing non-whitespace text before a unique marker."""
    marker_start = rendered.find(marker)
    if marker_start < 0 or rendered.find(marker, marker_start + 1) >= 0:
        raise ValueError(f"Expected exactly one marker {marker!r}")
    candidates = [
        index
        for index, (start, end) in enumerate(offsets)
        if end <= marker_start and end > start and rendered[start:end].strip()
    ]
    if not candidates:
        raise RuntimeError(f"Could not locate a semantic token before {marker!r}")
    return candidates[-1]


def load_semantic_activation_shards(
    output_dir: str | Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    shard_dir = Path(output_dir) / "semantic_activation_shards"
    array_paths = sorted(shard_dir.glob("shard_*.npy"))
    if not array_paths:
        raise FileNotFoundError(f"No semantic activation shards found under {shard_dir}")
    arrays = []
    frames = []
    expected_tail = None
    for array_path in array_paths:
        metadata_path = array_path.with_suffix(".csv")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata paired with {array_path}")
        array = np.load(array_path, mmap_mode="r")
        frame = pd.read_csv(metadata_path)
        if len(array) != len(frame):
            raise ValueError(f"Row mismatch between {array_path} and {metadata_path}")
        if expected_tail is None:
            expected_tail = array.shape[1:]
        elif array.shape[1:] != expected_tail:
            raise ValueError("Semantic activation shards have inconsistent dimensions")
        arrays.append(np.asarray(array))
        frames.append(frame)
    return np.concatenate(arrays), pd.concat(frames, ignore_index=True)


def fit_semantic_boundary_probes(config: Experiment2Config) -> dict[str, object]:
    activations, metadata = load_semantic_activation_shards(config.output_dir)
    probe_config, analysis_config = expanded_analysis_configs(config)
    results = fit_layer_probes(
        activations,
        metadata,
        probe_config,
        seed=config.seed,
        analysis_config=analysis_config,
    )
    output = config.output_dir / "semantic_boundary"
    save_probe_results(results, output)
    source_artifact = np.load(config.experiment1_dir / "probes" / "directions.npz")
    frozen_layer = int(source_artifact["selected_layer"])
    semantic_fixed = results.predictions[results.predictions["layer"] == frozen_layer].copy()
    semantic_fixed.to_csv(output / "fixed_experiment1_layer_predictions.csv", index=False)
    marker_fixed = pd.read_csv(config.experiment1_dir / "probes" / "test_predictions.csv")
    marker_threshold = float(source_artifact["thresholds"][frozen_layer])
    semantic_threshold = float(results.thresholds[frozen_layer])
    comparison = compare_boundaries(
        marker_fixed,
        semantic_fixed,
        marker_threshold=marker_threshold,
        semantic_threshold=semantic_threshold,
        config=config,
    )
    comparison.to_csv(output / "semantic_vs_marker_paired.csv", index=False)
    auroc_low = float(
        results.bootstrap_summary.loc[
            results.bootstrap_summary["metric"] == "auroc", "ci_low"
        ].iloc[0]
    )
    delta_lookup = comparison.set_index("metric")
    marker_invariant = bool(
        auroc_low > 0.5
        and delta_lookup.loc["auroc", "ci_low"] > -0.05
        and delta_lookup.loc["process_f1", "ci_low"] > -0.10
    )
    decision = {
        "semantic_boundary_decodable": auroc_low > 0.5,
        "marker_invariant": marker_invariant,
        "criterion": (
            "selected semantic-boundary AUROC CI is above 0.5; at frozen Experiment 1 layer, "
            "semantic-minus-marker AUROC CI lower bound exceeds -0.05 and Process F1 lower "
            "bound exceeds -0.10"
        ),
        "selected_semantic_auroc_ci_low": auroc_low,
    }
    (output / "decision.json").write_text(json.dumps(decision, indent=2))
    return {
        "selected_layer": results.selected_layer,
        "frozen_experiment1_layer": frozen_layer,
        **decision,
        "activation_rows": len(metadata),
        "traces": int(metadata["trace_id"].nunique()),
        "output_dir": str(output),
    }


def compare_boundaries(
    marker: pd.DataFrame,
    semantic: pd.DataFrame,
    *,
    marker_threshold: float,
    semantic_threshold: float,
    config: Experiment2Config,
) -> pd.DataFrame:
    """Paired whole-trace comparison at the frozen Experiment 1 layer."""
    merged = marker.merge(
        semantic[["trace_id", "step_index", "score"]],
        on=["trace_id", "step_index"],
        suffixes=("_primary", "_candidate"),
        validate="one_to_one",
    )
    bootstrap = _paired_comparison_bootstrap(
        merged,
        marker_threshold,
        semantic_threshold,
        AnalysisConfig(
            confidence_level=config.analysis.confidence_level,
            localization_tolerances=(0, 1, 2),
        ),
        samples=config.analysis.bootstrap_samples,
        seed=config.seed,
    )
    tail = (1 - config.analysis.confidence_level) / 2
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "semantic_minus_marker_estimate": values.mean(),
                "ci_low": values.quantile(tail),
                "ci_high": values.quantile(1 - tail),
                "bootstrap_samples": len(values),
            }
            for metric in bootstrap
            if not (values := bootstrap[metric].dropna()).empty
        ]
    )


def _check_identity(config: Experiment2Config) -> None:
    identity_path = config.output_dir / "semantic_extraction_identity.json"
    identity = {
        "dataset_sha256": _sha256_file(config.data.path),
        "model": config.model.name,
        "dtype": config.model.dtype,
        "max_length": config.model.max_length,
        "boundary_mode": "last_non_whitespace_token_before_marker",
    }
    if identity_path.exists():
        existing = json.loads(identity_path.read_text())
        if existing != identity:
            raise RuntimeError(
                "Existing semantic activation shards have a different extraction identity"
            )
    else:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(identity_path, json.dumps(identity, indent=2))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_progress(
    output_dir: Path,
    totals: dict[str, int],
    completed_index: int,
    trace_count: int,
) -> None:
    completed = totals["cached"] + totals["extracted"] + totals["skipped_long"]
    payload = {
        "status": "complete" if completed_index >= trace_count else "running",
        "traces_total": trace_count,
        "traces_completed": completed,
        "fraction_complete": completed / trace_count if trace_count else 1.0,
        "updated_at": utc_now(),
        **totals,
    }
    _atomic_write_text(
        output_dir / "semantic_extraction_progress.json",
        json.dumps(payload, indent=2),
    )
    update_stage_progress(
        output_dir,
        "extract-semantic",
        units_completed=completed,
        units_total=trace_count,
        fraction_complete=payload["fraction_complete"],
    )


def _atomic_save_array(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def _atomic_save_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(temporary, path)
