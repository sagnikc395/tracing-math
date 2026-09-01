"""Statistical analyses for the exploratory workshop follow-ups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from tracing_math.experiment1.probes import (
    binary_metrics,
    change_point_metrics,
    choose_threshold,
)


@dataclass(frozen=True)
class FittedProbe:
    scaler: StandardScaler
    classifier: LogisticRegression
    c_value: float


def build_matched_transition_dataset(
    activations: np.ndarray, metadata: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    """Pair each first-error transition with a similar transition from a correct trace."""
    if activations.ndim != 3 or len(activations) != len(metadata):
        raise ValueError("Activations must align with metadata and have three dimensions")
    required = {
        "trace_id",
        "partition",
        "source",
        "generator",
        "step_index",
        "step_fraction",
        "first_error",
        "has_error_trace",
        "n_steps",
        "token_count",
        "step_text",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing transition metadata columns: {sorted(missing)}")

    indexed = metadata.reset_index(names="activation_index")
    lookup = {
        (str(row.trace_id), int(row.step_index)): int(row.activation_index)
        for row in indexed.itertuples(index=False)
    }
    errors = indexed[
        (indexed["first_error"] > 0)
        & (indexed["step_index"] == indexed["first_error"])
    ].copy()
    correct = indexed[
        (indexed["has_error_trace"] == 0) & (indexed["step_index"] > 0)
    ].copy()
    if errors.empty or correct.empty:
        raise ValueError("Transition analysis needs non-initial errors and correct transitions")

    rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    for error in errors.sort_values(["partition", "trace_id"]).itertuples(index=False):
        candidates = correct[correct["partition"] == error.partition]
        if candidates.empty:
            raise ValueError(f"No correct transition is available in {error.partition}")
        match = _closest_transition(error, candidates)
        pair_id = f"{error.trace_id}:{int(error.step_index)}"
        for label, row, role in ((1, error, "error_onset"), (0, match, "correct_placebo")):
            current = lookup[(str(row.trace_id), int(row.step_index))]
            previous = lookup[(str(row.trace_id), int(row.step_index) - 1)]
            vectors.append(
                np.asarray(activations[current], dtype=np.float32)
                - np.asarray(activations[previous], dtype=np.float32)
            )
            rows.append(
                {
                    "pair_id": pair_id,
                    "onset_trace_id": str(error.trace_id),
                    "placebo_trace_id": str(match.trace_id),
                    "role": role,
                    "label": label,
                    "trace_id": str(row.trace_id),
                    "partition": str(row.partition),
                    "source": str(row.source),
                    "generator": str(row.generator),
                    "step_index": int(row.step_index),
                    "step_fraction": float(row.step_fraction),
                    "n_steps": int(row.n_steps),
                    "token_count": int(row.token_count),
                    "step_text": str(row.step_text),
                }
            )
    return np.stack(vectors), pd.DataFrame(rows)


def fit_transition_probes(
    transitions: np.ndarray,
    metadata: pd.DataFrame,
    *,
    c_values: tuple[float, ...],
    max_iter: int,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    """Fit layer-wise onset-transition probes with matched held-out controls."""
    labels = metadata["label"].to_numpy(dtype=int)
    masks = {
        name: metadata["partition"].eq(name).to_numpy()
        for name in ("train", "validation", "test")
    }
    if any(len(np.unique(labels[mask])) != 2 for mask in masks.values()):
        raise ValueError("Every transition partition must contain both classes")

    metric_rows = []
    layer_models: dict[int, FittedProbe] = {}
    layer_test_scores: dict[int, np.ndarray] = {}
    for layer in range(transitions.shape[1]):
        features = np.asarray(transitions[:, layer, :], dtype=np.float32)
        c_value = _select_c(
            features[masks["train"]],
            labels[masks["train"]],
            features[masks["validation"]],
            labels[masks["validation"]],
            c_values,
            max_iter,
            seed,
        )
        validation_model = _fit_probe(
            features[masks["train"]], labels[masks["train"]], c_value, max_iter, seed
        )
        validation_scores = _scores(validation_model, features[masks["validation"]])
        fit_mask = masks["train"] | masks["validation"]
        fitted = _fit_probe(features[fit_mask], labels[fit_mask], c_value, max_iter, seed)
        test_scores = _scores(fitted, features[masks["test"]])
        layer_models[layer] = fitted
        layer_test_scores[layer] = test_scores
        for split, split_labels, scores in (
            ("validation", labels[masks["validation"]], validation_scores),
            ("test", labels[masks["test"]], test_scores),
        ):
            metric_rows.append(
                {
                    "layer": layer,
                    "split": split,
                    "c_value": c_value,
                    **_ranking_metrics(split_labels, scores),
                    "paired_accuracy": _paired_accuracy(
                        metadata.loc[masks[split]], scores
                    ),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    selected_row = (
        metrics[metrics["split"] == "validation"]
        .sort_values(
            ["auroc", "paired_accuracy", "average_precision", "layer"],
            ascending=[False, False, False, True],
        )
        .iloc[0]
    )
    selected_layer = int(selected_row["layer"])
    selected_scores = layer_test_scores[selected_layer]
    predictions = metadata.loc[masks["test"]].copy()
    predictions["score"] = selected_scores
    predictions["layer"] = selected_layer

    controls = transition_controls(
        transitions[:, selected_layer, :],
        metadata,
        c_values=c_values,
        max_iter=max_iter,
        seed=seed,
    )
    bootstrap = _transition_bootstrap(
        predictions,
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    fitted = layer_models[selected_layer]
    raw_weight = fitted.classifier.coef_[0] / fitted.scaler.scale_
    return {
        "metrics": metrics,
        "predictions": predictions,
        "controls": controls,
        "bootstrap": bootstrap,
        "selected_layer": selected_layer,
        "selected_c": float(selected_row["c_value"]),
        "direction": (raw_weight / np.linalg.norm(raw_weight)).astype(np.float32),
    }


def transition_controls(
    selected_transitions: np.ndarray,
    metadata: pd.DataFrame,
    *,
    c_values: tuple[float, ...],
    max_iter: int,
    seed: int,
) -> pd.DataFrame:
    """Evaluate position, current-step lexical, and shuffled-label controls."""
    labels = metadata["label"].to_numpy(dtype=int)
    train = metadata["partition"].eq("train").to_numpy()
    validation = metadata["partition"].eq("validation").to_numpy()
    test = metadata["partition"].eq("test").to_numpy()
    rows = []

    position = metadata[["step_index", "step_fraction", "n_steps", "token_count"]].to_numpy(
        dtype=np.float32
    )
    rows.append(
        _control_result(
            "position",
            position,
            labels,
            metadata,
            train,
            validation,
            test,
            c_values,
            max_iter,
            seed,
        )
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000)
    lexical_train = vectorizer.fit_transform(metadata.loc[train, "step_text"].astype(str))
    lexical_validation = vectorizer.transform(metadata.loc[validation, "step_text"].astype(str))
    lexical_test = vectorizer.transform(metadata.loc[test, "step_text"].astype(str))
    rows.append(
        _control_result_from_splits(
            "current_step_tfidf",
            lexical_train,
            lexical_validation,
            lexical_test,
            labels[train],
            labels[validation],
            labels[test],
            metadata.loc[test],
            c_values,
            max_iter,
            seed,
        )
    )

    shuffled = labels[train].copy()
    np.random.default_rng(seed).shuffle(shuffled)
    rows.append(
        _control_result_from_splits(
            "shuffled_label_hidden",
            selected_transitions[train],
            selected_transitions[validation],
            selected_transitions[test],
            shuffled,
            labels[validation],
            labels[test],
            metadata.loc[test],
            c_values,
            max_iter,
            seed,
        )
    )
    return pd.DataFrame(rows)


def analyze_fixed_boundary_locations(
    activations: np.ndarray,
    metadata: pd.DataFrame,
    locations: tuple[str, ...],
    *,
    layer: int,
    c_value: float,
    max_iter: int,
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare natural step endings and marker endings at one frozen layer."""
    if activations.ndim != 4 or activations.shape[1] != len(locations):
        raise ValueError("Boundary activations must have shape [row, location, layer, hidden]")
    labels = metadata["invalid_so_far"].to_numpy(dtype=int)
    train = metadata["partition"].eq("train").to_numpy()
    validation = metadata["partition"].eq("validation").to_numpy()
    test = metadata["partition"].eq("test").to_numpy()
    rows = []
    prediction_frames = []
    for location_index, location in enumerate(locations):
        features = np.asarray(activations[:, location_index, layer, :], dtype=np.float32)
        validation_model = _fit_probe(features[train], labels[train], c_value, max_iter, seed)
        validation_scores = _scores(validation_model, features[validation])
        threshold = choose_threshold(metadata.loc[validation], validation_scores)
        fitted = _fit_probe(
            features[train | validation], labels[train | validation], c_value, max_iter, seed
        )
        test_scores = _scores(fitted, features[test])
        rows.append(
            {
                "location": location,
                "layer": layer,
                "c_value": c_value,
                "threshold": threshold,
                **binary_metrics(labels[test], test_scores, threshold=threshold),
                **change_point_metrics(metadata.loc[test], test_scores, threshold),
            }
        )
        frame = metadata.loc[test, ["trace_id", "step_index", "first_error"]].copy()
        frame["label"] = labels[test]
        frame["location"] = location
        frame["score"] = test_scores
        frame["threshold"] = threshold
        prediction_frames.append(frame)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    comparison = _boundary_difference_bootstrap(
        predictions,
        locations=locations,
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed,
    )
    return pd.DataFrame(rows), predictions, comparison


def summarize_counterfactual_patching(
    frame: pd.DataFrame, *, samples: int, confidence_level: float, seed: int
) -> pd.DataFrame:
    """Summarize paired baseline separation and real-state patch effects."""
    required = {"pair_id", "condition", "verdict_score"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing patching columns: {sorted(missing)}")
    wide = frame.pivot(index="pair_id", columns="condition", values="verdict_score")
    conditions = {
        "baseline_error_minus_correct": wide["error_baseline"]
        - wide["correct_baseline"],
        "error_state_into_correct": wide["error_state_into_correct"]
        - wide["correct_baseline"],
        "correct_state_into_error": wide["correct_state_into_error"]
        - wide["error_baseline"],
    }
    rng = np.random.default_rng(seed)
    tail = (1 - confidence_level) / 2
    rows = []
    for name, values in conditions.items():
        array = values.to_numpy(dtype=float)
        draws = np.asarray(
            [rng.choice(array, len(array), replace=True).mean() for _ in range(samples)]
        )
        rows.append(
            {
                "effect": name,
                "estimate": float(array.mean()),
                "ci_low": float(np.quantile(draws, tail)),
                "ci_high": float(np.quantile(draws, 1 - tail)),
                "n_pairs": len(array),
            }
        )
    return pd.DataFrame(rows)


def _closest_transition(error: object, candidates: pd.DataFrame) -> object:
    work = candidates.copy()
    work["match_tier"] = (
        2 * work["source"].eq(error.source).astype(int)
        + work["generator"].eq(error.generator).astype(int)
    )
    work["distance"] = (
        (work["step_fraction"] - float(error.step_fraction)).abs()
        + (work["n_steps"] - int(error.n_steps)).abs() / max(int(error.n_steps), 1)
        + (work["token_count"] - int(error.token_count)).abs()
        / max(int(error.token_count), 1)
    )
    return work.sort_values(
        ["match_tier", "distance", "trace_id", "step_index"],
        ascending=[False, True, True, True],
    ).iloc[0]


def _fit_probe(
    features: object, labels: np.ndarray, c_value: float, max_iter: int, seed: int
) -> FittedProbe:
    scaler = StandardScaler(with_mean=not hasattr(features, "tocsr"))
    scaled = scaler.fit_transform(features)
    classifier = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=max_iter,
        solver="liblinear",
        random_state=seed,
    ).fit(scaled, labels)
    return FittedProbe(scaler=scaler, classifier=classifier, c_value=c_value)


def _scores(model: FittedProbe, features: object) -> np.ndarray:
    return model.classifier.predict_proba(model.scaler.transform(features))[:, 1]


def _select_c(
    train_features: object,
    train_labels: np.ndarray,
    validation_features: object,
    validation_labels: np.ndarray,
    c_values: tuple[float, ...],
    max_iter: int,
    seed: int,
) -> float:
    ranked = []
    for c_value in c_values:
        model = _fit_probe(train_features, train_labels, c_value, max_iter, seed)
        scores = _scores(model, validation_features)
        ranked.append((roc_auc_score(validation_labels, scores), -c_value, c_value))
    return float(max(ranked)[2])


def _ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def _paired_accuracy(metadata: pd.DataFrame, scores: np.ndarray) -> float:
    frame = metadata[["pair_id", "label"]].copy()
    frame["score"] = scores
    wide = frame.pivot(index="pair_id", columns="label", values="score")
    differences = wide[1] - wide[0]
    return float((differences.gt(0).sum() + 0.5 * differences.eq(0).sum()) / len(wide))


def _control_result(
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    metadata: pd.DataFrame,
    train: np.ndarray,
    validation: np.ndarray,
    test: np.ndarray,
    c_values: tuple[float, ...],
    max_iter: int,
    seed: int,
) -> dict[str, object]:
    return _control_result_from_splits(
        name,
        features[train],
        features[validation],
        features[test],
        labels[train],
        labels[validation],
        labels[test],
        metadata.loc[test],
        c_values,
        max_iter,
        seed,
    )


def _control_result_from_splits(
    name: str,
    train_features: object,
    validation_features: object,
    test_features: object,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    test_labels: np.ndarray,
    test_metadata: pd.DataFrame,
    c_values: tuple[float, ...],
    max_iter: int,
    seed: int,
) -> dict[str, object]:
    c_value = _select_c(
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        c_values,
        max_iter,
        seed,
    )
    if hasattr(train_features, "tocsr"):
        from scipy.sparse import vstack

        fit_features = vstack([train_features, validation_features])
    else:
        fit_features = np.concatenate([train_features, validation_features])
    fit_labels = np.concatenate([train_labels, validation_labels])
    model = _fit_probe(fit_features, fit_labels, c_value, max_iter, seed)
    scores = _scores(model, test_features)
    return {
        "control": name,
        "c_value": c_value,
        **_ranking_metrics(test_labels, scores),
        "paired_accuracy": _paired_accuracy(test_metadata, scores),
    }


def _transition_bootstrap(
    predictions: pd.DataFrame, *, samples: int, confidence_level: float, seed: int
) -> pd.DataFrame:
    groups = {pair_id: group for pair_id, group in predictions.groupby("pair_id", sort=False)}
    pair_ids = np.asarray(list(groups))
    placebo_by_pair = {
        pair_id: str(group["placebo_trace_id"].iloc[0]) for pair_id, group in groups.items()
    }
    placebo_ids = np.asarray(sorted(set(placebo_by_pair.values())))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        pair_counts = pd.Series(
            rng.choice(pair_ids, len(pair_ids), replace=True)
        ).value_counts()
        placebo_counts = pd.Series(
            rng.choice(placebo_ids, len(placebo_ids), replace=True)
        ).value_counts()
        pieces = []
        draw_index = 0
        for pair_id, pair_count in pair_counts.items():
            weight = int(pair_count) * int(
                placebo_counts.get(placebo_by_pair[str(pair_id)], 0)
            )
            for _repeat in range(weight):
                pieces.append(groups[str(pair_id)].assign(pair_id=f"draw-{draw_index}"))
                draw_index += 1
        if not pieces:
            continue
        sample = pd.concat(pieces, ignore_index=True)
        draws.append(
            {
                **_ranking_metrics(sample["label"].to_numpy(), sample["score"].to_numpy()),
                "paired_accuracy": _paired_accuracy(sample, sample["score"].to_numpy()),
            }
        )
    draw_frame = pd.DataFrame(draws)
    if draw_frame.empty:
        raise RuntimeError("Two-way transition bootstrap produced no valid resamples")
    tail = (1 - confidence_level) / 2
    point = {
        **_ranking_metrics(
            predictions["label"].to_numpy(), predictions["score"].to_numpy()
        ),
        "paired_accuracy": _paired_accuracy(predictions, predictions["score"].to_numpy()),
    }
    return pd.DataFrame(
        {
            "metric": list(point),
            "estimate": list(point.values()),
            "ci_low": [float(draw_frame[name].quantile(tail)) for name in point],
            "ci_high": [float(draw_frame[name].quantile(1 - tail)) for name in point],
        }
    )


def _boundary_difference_bootstrap(
    predictions: pd.DataFrame,
    *,
    locations: tuple[str, ...],
    samples: int,
    confidence_level: float,
    seed: int,
) -> pd.DataFrame:
    if len(locations) != 2:
        raise ValueError("The paired boundary comparison expects exactly two locations")
    groups = {trace_id: group for trace_id, group in predictions.groupby("trace_id", sort=False)}
    trace_ids = np.asarray(list(groups))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        pieces = []
        for draw_index, trace_id in enumerate(rng.choice(trace_ids, len(trace_ids), replace=True)):
            pieces.append(groups[trace_id].assign(trace_id=f"draw-{draw_index}"))
        sample = pd.concat(pieces, ignore_index=True)
        values = {}
        for location in locations:
            scoped = sample[sample["location"] == location]
            threshold = float(scoped["threshold"].iloc[0])
            values[location] = {
                "auroc": roc_auc_score(scoped["label"], scoped["score"]),
                "process_f1": change_point_metrics(
                    scoped, scoped["score"].to_numpy(), threshold
                )["process_f1"],
            }
        draws.append(
            {
                metric: values[locations[0]][metric] - values[locations[1]][metric]
                for metric in ("auroc", "process_f1")
            }
        )
    frame = pd.DataFrame(draws)
    tail = (1 - confidence_level) / 2
    return pd.DataFrame(
        [
            {
                "contrast": f"{locations[0]}_minus_{locations[1]}",
                "metric": metric,
                "estimate": float(frame[metric].mean()),
                "ci_low": float(frame[metric].quantile(tail)),
                "ci_high": float(frame[metric].quantile(1 - tail)),
            }
            for metric in frame
        ]
    )
