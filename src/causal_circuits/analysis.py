"""Leakage-resistant linear probes, controls, and change-point metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from causal_circuits.config import ProbeConfig


@dataclass
class ProbeResults:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    controls: pd.DataFrame
    transfer: pd.DataFrame
    pca_curve: pd.DataFrame
    directions: np.ndarray
    projection_stds: np.ndarray
    thresholds: np.ndarray
    c_values: np.ndarray
    selected_layer: int
    selected_intervention_layer: int


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "average_precision": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def change_point_metrics(
    metadata: pd.DataFrame, scores: np.ndarray, threshold: float
) -> dict[str, float]:
    """ProcessBench exact first-error score induced by the first threshold crossing."""
    frame = metadata[["trace_id", "step_index", "first_error"]].copy()
    frame["score"] = np.asarray(scores, dtype=float)
    expected: list[int] = []
    predicted: list[int] = []
    for _, trace_rows in frame.groupby("trace_id", sort=False):
        trace_rows = trace_rows.sort_values("step_index")
        expected.append(int(trace_rows["first_error"].iloc[0]))
        crossings = trace_rows.loc[trace_rows["score"] >= threshold, "step_index"]
        predicted.append(-1 if crossings.empty else int(crossings.iloc[0]))
    expected_array = np.asarray(expected)
    predicted_array = np.asarray(predicted)
    error_mask = expected_array >= 0
    correct_mask = ~error_mask
    error_accuracy = _safe_mean(predicted_array[error_mask] == expected_array[error_mask])
    correct_accuracy = _safe_mean(predicted_array[correct_mask] == -1)
    denominator = error_accuracy + correct_accuracy
    process_f1 = 2 * error_accuracy * correct_accuracy / denominator if denominator > 0 else 0.0
    exact_accuracy = _safe_mean(predicted_array == expected_array)
    return {
        "error_accuracy": error_accuracy,
        "correct_accuracy": correct_accuracy,
        "process_f1": process_f1,
        "first_error_exact": exact_accuracy,
    }


def choose_threshold(metadata: pd.DataFrame, scores: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 181)
    ranked = [
        (
            change_point_metrics(metadata, scores, float(threshold))["process_f1"],
            -abs(float(threshold) - 0.5),
            float(threshold),
        )
        for threshold in candidates
    ]
    return max(ranked)[2]


def fit_layer_probes(
    activations: np.ndarray,
    metadata: pd.DataFrame,
    config: ProbeConfig,
    *,
    seed: int,
) -> ProbeResults:
    """Fit one regularized logistic probe per hidden-state index."""
    _validate_inputs(activations, metadata, config.target)
    labels = metadata[config.target].to_numpy(dtype=int)
    masks = {
        split: metadata["partition"].eq(split).to_numpy()
        for split in ("train", "validation", "test")
    }
    if any(labels[mask].min() == labels[mask].max() for mask in masks.values()):
        raise ValueError("Every data partition must contain both target classes")

    n_layers, hidden_size = activations.shape[1:]
    directions = np.zeros((n_layers, hidden_size), dtype=np.float32)
    projection_stds = np.zeros(n_layers, dtype=np.float32)
    thresholds = np.zeros(n_layers, dtype=np.float32)
    selected_cs = np.zeros(n_layers, dtype=np.float32)
    metrics_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for layer in range(n_layers):
        x = np.asarray(activations[:, layer, :], dtype=np.float32)
        best_c, validation_scores = _select_c(
            x[masks["train"]],
            labels[masks["train"]],
            x[masks["validation"]],
            labels[masks["validation"]],
            config.c_values,
            config.max_iter,
            seed,
        )
        threshold = choose_threshold(metadata.loc[masks["validation"]], validation_scores)
        selected_cs[layer] = best_c
        thresholds[layer] = threshold
        metrics_rows.append(
            _metric_row(
                layer,
                "validation",
                metadata.loc[masks["validation"]],
                labels[masks["validation"]],
                validation_scores,
                threshold,
                best_c,
            )
        )

        fit_mask = masks["train"] | masks["validation"]
        scaler, classifier = _fit_logistic(
            x[fit_mask], labels[fit_mask], best_c, config.max_iter, seed
        )
        test_scores = classifier.predict_proba(scaler.transform(x[masks["test"]]))[:, 1]
        direction = classifier.coef_[0] / scaler.scale_
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise RuntimeError(f"Layer {layer} produced a zero probe direction")
        directions[layer] = direction / norm
        projection_stds[layer] = max(float(np.std(x[fit_mask] @ directions[layer], ddof=1)), 1e-8)
        metrics_rows.append(
            _metric_row(
                layer,
                "test",
                metadata.loc[masks["test"]],
                labels[masks["test"]],
                test_scores,
                threshold,
                best_c,
            )
        )
        layer_predictions = metadata.loc[
            masks["test"], ["trace_id", "source", "step_index", "first_error"]
        ].copy()
        layer_predictions["layer"] = layer
        layer_predictions["label"] = labels[masks["test"]]
        layer_predictions["score"] = test_scores
        prediction_rows.append(layer_predictions)

    metrics = pd.DataFrame(metrics_rows)
    validation = metrics[metrics["split"] == "validation"].sort_values(
        ["process_f1", "auroc", "layer"], ascending=[False, False, True]
    )
    selected_layer = int(validation.iloc[0]["layer"])
    eligible = validation[validation["layer"] < n_layers - 1]
    selected_intervention_layer = int(eligible.iloc[0]["layer"])

    controls = evaluate_controls(
        np.asarray(activations[:, selected_layer, :], dtype=np.float32),
        metadata,
        labels,
        max_iter=config.max_iter,
        seed=seed,
    )
    transfer = domain_transfer(
        np.asarray(activations[:, selected_layer, :], dtype=np.float32),
        metadata,
        labels,
        c_value=float(selected_cs[selected_layer]),
        max_iter=config.max_iter,
        seed=seed,
    )
    pca_curve = pca_subspace_curve(
        np.asarray(activations[:, selected_layer, :], dtype=np.float32),
        metadata,
        labels,
        config.pca_dimensions,
        c_value=float(selected_cs[selected_layer]),
        max_iter=config.max_iter,
        seed=seed,
    )
    return ProbeResults(
        metrics=metrics,
        predictions=pd.concat(prediction_rows, ignore_index=True),
        controls=controls,
        transfer=transfer,
        pca_curve=pca_curve,
        directions=directions,
        projection_stds=projection_stds,
        thresholds=thresholds,
        c_values=selected_cs,
        selected_layer=selected_layer,
        selected_intervention_layer=selected_intervention_layer,
    )


def evaluate_controls(
    hidden: np.ndarray,
    metadata: pd.DataFrame,
    labels: np.ndarray,
    *,
    max_iter: int,
    seed: int,
) -> pd.DataFrame:
    train = metadata["partition"].eq("train").to_numpy()
    validation = metadata["partition"].eq("validation").to_numpy()
    test = metadata["partition"].eq("test").to_numpy()
    rows: list[dict[str, object]] = []

    position = metadata[["step_index", "step_fraction"]].to_numpy(dtype=np.float32)
    rows.append(
        _fit_control(
            "position", position, metadata, labels, train, validation, test, max_iter, seed
        )
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20_000)
    train_text = metadata.loc[train, "step_text"].astype(str)
    x_train = vectorizer.fit_transform(train_text)
    x_validation = vectorizer.transform(metadata.loc[validation, "step_text"].astype(str))
    x_test = vectorizer.transform(metadata.loc[test, "step_text"].astype(str))
    classifier = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=max_iter, solver="liblinear", random_state=seed
    ).fit(x_train, labels[train])
    validation_scores = classifier.predict_proba(x_validation)[:, 1]
    threshold = choose_threshold(metadata.loc[validation], validation_scores)
    scores = classifier.predict_proba(x_test)[:, 1]
    rows.append(
        _control_row("current-step TF-IDF", metadata.loc[test], labels[test], scores, threshold)
    )

    rng = np.random.default_rng(seed)
    shuffled = labels[train].copy()
    rng.shuffle(shuffled)
    scaler, classifier = _fit_logistic(hidden[train], shuffled, 1.0, max_iter, seed)
    validation_scores = classifier.predict_proba(scaler.transform(hidden[validation]))[:, 1]
    threshold = choose_threshold(metadata.loc[validation], validation_scores)
    scores = classifier.predict_proba(scaler.transform(hidden[test]))[:, 1]
    rows.append(
        _control_row("shuffled-label hidden", metadata.loc[test], labels[test], scores, threshold)
    )
    return pd.DataFrame(rows)


def domain_transfer(
    hidden: np.ndarray,
    metadata: pd.DataFrame,
    labels: np.ndarray,
    *,
    c_value: float,
    max_iter: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    sources = sorted(metadata["source"].unique())
    for train_source in sources:
        train = metadata["source"].eq(train_source) & metadata["partition"].eq("train")
        validation = metadata["source"].eq(train_source) & metadata["partition"].eq("validation")
        if labels[train].min() == labels[train].max():
            continue
        scaler, classifier = _fit_logistic(hidden[train], labels[train], c_value, max_iter, seed)
        validation_scores = classifier.predict_proba(scaler.transform(hidden[validation]))[:, 1]
        threshold = choose_threshold(metadata.loc[validation], validation_scores)
        for test_source in sources:
            test = metadata["source"].eq(test_source) & metadata["partition"].eq("test")
            scores = classifier.predict_proba(scaler.transform(hidden[test]))[:, 1]
            row = _control_row(
                "domain transfer", metadata.loc[test], labels[test], scores, threshold
            )
            row.update({"train_source": train_source, "test_source": test_source})
            rows.append(row)
    return pd.DataFrame(rows)


def pca_subspace_curve(
    hidden: np.ndarray,
    metadata: pd.DataFrame,
    labels: np.ndarray,
    dimensions: tuple[int, ...],
    *,
    c_value: float,
    max_iter: int,
    seed: int,
) -> pd.DataFrame:
    """Test accessibility in top-variance PCA subspaces (not intrinsic dimension)."""
    train = metadata["partition"].eq("train").to_numpy()
    validation = metadata["partition"].eq("validation").to_numpy()
    test = metadata["partition"].eq("test").to_numpy()
    maximum = min(max(dimensions, default=0), hidden.shape[1], int(train.sum()))
    if maximum < 1:
        return pd.DataFrame()
    pca = PCA(n_components=maximum, svd_solver="randomized", random_state=seed).fit(hidden[train])
    transformed = pca.transform(hidden)
    rows = []
    for dimension in sorted({value for value in dimensions if value <= maximum}):
        scaler, classifier = _fit_logistic(
            transformed[train, :dimension], labels[train], c_value, max_iter, seed
        )
        validation_scores = classifier.predict_proba(
            scaler.transform(transformed[validation, :dimension])
        )[:, 1]
        threshold = choose_threshold(metadata.loc[validation], validation_scores)
        scores = classifier.predict_proba(scaler.transform(transformed[test, :dimension]))[:, 1]
        row = _control_row("PCA subspace", metadata.loc[test], labels[test], scores, threshold)
        row["dimensions"] = dimension
        row["variance_explained"] = float(pca.explained_variance_ratio_[:dimension].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _fit_control(name, features, metadata, labels, train, validation, test, max_iter, seed):
    scaler, classifier = _fit_logistic(features[train], labels[train], 1.0, max_iter, seed)
    validation_scores = classifier.predict_proba(scaler.transform(features[validation]))[:, 1]
    threshold = choose_threshold(metadata.loc[validation], validation_scores)
    scores = classifier.predict_proba(scaler.transform(features[test]))[:, 1]
    return _control_row(name, metadata.loc[test], labels[test], scores, threshold)


def _control_row(name, metadata, labels, scores, threshold):
    return {
        "control": name,
        **binary_metrics(labels, scores),
        **change_point_metrics(metadata, scores, threshold),
        "threshold": threshold,
    }


def _select_c(x_train, y_train, x_validation, y_validation, c_values, max_iter, seed):
    candidates = []
    for c_value in c_values:
        scaler, classifier = _fit_logistic(x_train, y_train, c_value, max_iter, seed)
        scores = classifier.predict_proba(scaler.transform(x_validation))[:, 1]
        auroc = binary_metrics(y_validation, scores)["auroc"]
        candidates.append((auroc, -abs(np.log10(c_value)), c_value, scores))
    _, _, best_c, scores = max(candidates, key=lambda item: item[:2])
    return float(best_c), scores


def _fit_logistic(features, labels, c_value, max_iter, seed):
    scaler = StandardScaler().fit(features)
    classifier = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=max_iter,
        solver="liblinear",
        random_state=seed,
    ).fit(scaler.transform(features), labels)
    return scaler, classifier


def _metric_row(layer, split, metadata, labels, scores, threshold, c_value):
    return {
        "layer": layer,
        "split": split,
        "c_value": c_value,
        "threshold": threshold,
        **binary_metrics(labels, scores),
        **change_point_metrics(metadata, scores, threshold),
    }


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def _validate_inputs(activations: np.ndarray, metadata: pd.DataFrame, target: str) -> None:
    if activations.ndim != 3:
        raise ValueError("activations must have shape [examples, layers, hidden]")
    required = {
        target,
        "partition",
        "trace_id",
        "source",
        "step_index",
        "step_fraction",
        "step_text",
        "first_error",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    if len(metadata) != len(activations):
        raise ValueError("Activation and metadata row counts differ")
