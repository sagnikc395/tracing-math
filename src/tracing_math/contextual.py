"""Visible-text contextual baseline for the conditional hidden-state comparison."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from tracing_math.data import ProcessTrace
from tracing_math.localization import assert_process_f1_identity
from tracing_math.probes import binary_metrics, change_point_metrics, choose_threshold


@dataclass(frozen=True)
class ContextualBaselineResult:
    metrics: pd.DataFrame
    selection: pd.DataFrame
    predictions: pd.DataFrame


def contextual_prefix_embeddings(
    texts: Sequence[str],
    *,
    model_name: str,
    revision: str | None,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """Mean-pool a frozen encoder over the visible problem and prefix text."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    kwargs = {"revision": revision} if revision is not None else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    model = AutoModel.from_pretrained(model_name, **kwargs)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                list(texts[start : start + batch_size]),
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            states = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(states.dtype)
            pooled = (states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            batches.append(pooled.cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(batches, axis=0)


def fit_contextual_text_baseline(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    traces: Sequence[ProcessTrace],
    *,
    c_values: tuple[float, ...],
    max_iter: int,
    seed: int,
) -> ContextualBaselineResult:
    """Fit the same held-out linear-head and threshold protocol on visible-text embeddings."""
    if embeddings.ndim != 2 or len(embeddings) != len(metadata):
        raise ValueError(
            "Contextual embeddings must be a two-dimensional array aligned to metadata"
        )
    required = {"trace_id", "partition", "problem_group", "invalid_so_far", "step_index"}
    if missing := required.difference(metadata.columns):
        raise ValueError(f"Missing contextual-baseline columns: {sorted(missing)}")
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("C values must be positive")
    frame = attach_prefix_text(metadata, traces)
    masks = {
        name: frame["partition"].eq(name).to_numpy() for name in ("train", "validation", "test")
    }
    labels = frame["invalid_so_far"].to_numpy(dtype=int)
    if any(np.unique(labels[mask]).size != 2 for mask in masks.values()):
        raise ValueError("Every contextual-baseline partition must contain both classes")
    if (frame.groupby("problem_group")["partition"].nunique() > 1).any():
        raise ValueError("A problem group crosses contextual-baseline partitions")

    train, validation, test = (masks[name] for name in ("train", "validation", "test"))
    scaler = StandardScaler().fit(embeddings[train])
    x_train, x_validation = (
        scaler.transform(embeddings[train]),
        scaler.transform(embeddings[validation]),
    )
    weights = _trace_equal_weights(frame.loc[train])
    selection_rows = []
    validation_scores: dict[float, np.ndarray] = {}
    for c_value in c_values:
        model = _classifier(c_value, max_iter, seed)
        model.fit(x_train, labels[train], sample_weight=weights)
        scores = model.predict_proba(x_validation)[:, 1]
        validation_scores[float(c_value)] = scores
        selection_rows.append(
            {
                "c_value": float(c_value),
                "validation_auroc": float(roc_auc_score(labels[validation], scores)),
                "validation_average_precision": float(
                    average_precision_score(labels[validation], scores)
                ),
            }
        )
    selected = max(
        selection_rows, key=lambda row: (row["validation_auroc"], -abs(np.log10(row["c_value"])))
    )
    c_value = float(selected["c_value"])
    threshold = choose_threshold(frame.loc[validation], validation_scores[c_value])
    for row in selection_rows:
        row["selected"] = row["c_value"] == c_value
        row["selected_threshold"] = threshold if row["selected"] else np.nan

    fit = train | validation
    final_scaler = StandardScaler().fit(embeddings[fit])
    model = _classifier(c_value, max_iter, seed)
    model.fit(
        final_scaler.transform(embeddings[fit]),
        labels[fit],
        sample_weight=_trace_equal_weights(frame.loc[fit]),
    )
    scores = model.predict_proba(final_scaler.transform(embeddings[test]))[:, 1]
    test_frame = frame.loc[test].copy()
    metrics = {
        "condition": "contextual_text",
        "c_value": c_value,
        "threshold": threshold,
        "feature_count": int(embeddings.shape[1]),
        "training_weighting": "trace_equal",
        **binary_metrics(labels[test], scores, threshold=threshold),
        **change_point_metrics(test_frame, scores, threshold, tolerances=(0, 1, 2)),
    }
    assert_process_f1_identity(metrics)
    predictions = test_frame[
        ["trace_id", "problem_group", "source", "generator", "step_index", "first_error", "n_steps"]
    ].copy()
    predictions["condition"] = "contextual_text"
    predictions["label"] = labels[test]
    predictions["score"] = scores
    predictions["threshold"] = threshold
    return ContextualBaselineResult(
        metrics=pd.DataFrame([metrics]),
        selection=pd.DataFrame(selection_rows),
        predictions=predictions,
    )


def attach_prefix_text(metadata: pd.DataFrame, traces: Sequence[ProcessTrace]) -> pd.DataFrame:
    prefixes: dict[tuple[str, int], str] = {}
    for trace in traces:
        blocks = [f"Problem:\n{trace.problem.strip()}\n\nReasoning:"]
        for step_index, step in enumerate(trace.steps):
            blocks.append(f"[Step {step_index}]\n{step.strip()}")
            prefixes[(trace.trace_id, step_index)] = "\n\n".join(blocks)
    keys = list(
        zip(metadata["trace_id"].astype(str), metadata["step_index"].astype(int), strict=True)
    )
    if missing := [key for key in keys if key not in prefixes]:
        raise ValueError(f"Missing source text for contextual activation row {missing[0]}")
    frame = metadata.copy()
    frame["prefix_text"] = [prefixes[key] for key in keys]
    return frame


def _trace_equal_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame["trace_id"].astype(str).value_counts()
    return 1.0 / frame["trace_id"].astype(str).map(counts).to_numpy(dtype=float)


def _classifier(c_value: float, max_iter: int, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=max_iter,
        solver="liblinear",
        random_state=seed,
    )
