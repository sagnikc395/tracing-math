"""Predictive robustness analyses for Experiment 2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from causal_circuits.analysis import (
    ProbeResults,
    _fit_logistic,
    _paired_comparison_bootstrap,
    _select_c,
    binary_metrics,
    change_point_metrics,
    choose_threshold,
    fit_layer_probes,
    group_bootstrap_metrics,
    summarize_bootstrap,
)
from causal_circuits.config import AnalysisConfig, ProbeConfig, ProbeFamilyConfig
from causal_circuits.experiment2_config import Experiment2Config
from causal_circuits.pipeline import load_activation_shards


def expanded_analysis_configs(
    config: Experiment2Config,
) -> tuple[ProbeConfig, AnalysisConfig]:
    """Build the expanded, explicitly exploratory probe configuration."""
    probe = ProbeConfig(
        target="invalid_so_far",
        primary_family="l2",
        families=(ProbeFamilyConfig(name="l2", penalty="l2"),),
        diagnostic_targets=("error_onset",),
        train_fraction=config.data.train_fraction,
        validation_fraction=config.data.validation_fraction,
        test_fraction=config.data.test_fraction,
        c_values=config.analysis.c_values,
        max_iter=config.analysis.max_iter,
        bootstrap_samples=config.analysis.bootstrap_samples,
        pca_dimensions=config.analysis.pca_dimensions,
    )
    analysis = AnalysisConfig(
        threshold_min=config.analysis.threshold_min,
        threshold_max=config.analysis.threshold_max,
        threshold_points=config.analysis.threshold_points,
        localization_tolerances=(0, 1, 2),
        calibration_bins=10,
        confidence_level=config.analysis.confidence_level,
        subgroup_min_traces=config.analysis.subgroup_min_traces,
        exploratory_bootstrap_samples=config.analysis.bootstrap_samples,
    )
    return probe, analysis


def run_marker_robustness(config: Experiment2Config) -> dict[str, object]:
    """Run expanded and shortcut-resistant analyses on frozen Experiment 1 states."""
    activations, metadata = load_activation_shards(config.experiment1_dir)
    probe_config, analysis_config = expanded_analysis_configs(config)
    results = fit_layer_probes(
        activations,
        metadata,
        probe_config,
        seed=config.seed,
        analysis_config=analysis_config,
    )
    output = config.output_dir / "marker_robustness"
    save_probe_results(results, output)

    source_artifact = np.load(config.experiment1_dir / "probes" / "directions.npz")
    frozen_layer = int(source_artifact["selected_layer"])
    hidden = np.asarray(activations[:, frozen_layer, :], dtype=np.float32)

    error_only_metrics = []
    error_only_predictions = []
    raw_metrics, raw_predictions = fit_error_only_probe(
        hidden,
        metadata,
        config,
        centered=False,
    )
    error_only_metrics.append(raw_metrics)
    error_only_predictions.append(raw_predictions)
    centered_metrics, centered_predictions = fit_error_only_probe(
        hidden,
        metadata,
        config,
        centered=True,
    )
    error_only_metrics.append(centered_metrics)
    error_only_predictions.append(centered_predictions)
    error_only_metric_frame = pd.concat(error_only_metrics, ignore_index=True)
    error_only_metric_frame.to_csv(output / "error_only_metrics.csv", index=False)
    pd.concat(error_only_predictions, ignore_index=True).to_csv(
        output / "error_only_predictions.csv", index=False
    )

    generator_holdout(hidden, metadata, config).to_csv(
        output / "leave_one_generator_out.csv", index=False
    )
    domain_calibration, direction_cosines = domain_calibration_analysis(
        hidden,
        metadata,
        config,
    )
    domain_calibration.to_csv(output / "domain_calibration.csv", index=False)
    direction_cosines.to_csv(output / "source_direction_cosines.csv", index=False)
    primary_predictions = results.predictions[
        results.predictions["layer"] == results.selected_layer
    ].copy()
    onset_jump = bootstrap_onset_jump(primary_predictions, config)
    onset_jump.to_csv(output / "onset_jump_bootstrap.csv", index=False)
    surface_predictions, surface_metrics = fit_surface_metadata_control(metadata, config)
    surface_predictions.to_csv(output / "surface_metadata_predictions.csv", index=False)
    surface_metrics.to_csv(output / "surface_metadata_metrics.csv", index=False)
    paired_control = paired_control_comparison(
        primary_predictions,
        surface_predictions,
        hidden_threshold=float(results.thresholds[results.selected_layer]),
        surface_threshold=float(surface_metrics.loc[0, "threshold"]),
        config=config,
    )
    paired_control.to_csv(output / "hidden_vs_surface_paired.csv", index=False)

    error_lookup = error_only_metric_frame.set_index("variant")
    control_lookup = paired_control.set_index("metric")
    decision = {
        "raw_error_only_decodable": bool(
            error_lookup.loc["raw_error_traces", "auroc_ci_low"] > 0.5
        ),
        "centered_error_only_decodable": bool(
            error_lookup.loc[
                "first_step_centered_error_traces", "auroc_ci_low"
            ]
            > 0.5
        ),
        "positive_onset_jump": bool(onset_jump.loc[0, "ci_low"] > 0),
        "beats_combined_surface_control": bool(
            control_lookup.loc["auroc", "ci_low"] > 0
            and control_lookup.loc["process_f1", "ci_low"] > 0
        ),
        "criterion": (
            "raw and first-step-centered error-only AUROC intervals above 0.5, onset-jump "
            "interval above zero, and hidden-minus-combined-surface AUROC and Process F1 "
            "intervals above zero"
        ),
    }
    decision["change_point_supported"] = bool(
        decision["raw_error_only_decodable"]
        and decision["centered_error_only_decodable"]
        and decision["positive_onset_jump"]
        and decision["beats_combined_surface_control"]
    )
    (output / "decision.json").write_text(json.dumps(decision, indent=2))

    return {
        "selected_layer": results.selected_layer,
        "frozen_experiment1_layer": frozen_layer,
        "activation_rows": len(metadata),
        "traces": int(metadata["trace_id"].nunique()),
        **decision,
        "output_dir": str(output),
    }


def save_probe_results(results: ProbeResults, output: Path) -> None:
    """Persist every populated probe result table without changing Experiment 1."""
    output.mkdir(parents=True, exist_ok=True)
    tables = {
        "layer_metrics.csv": results.metrics,
        "test_predictions.csv": results.predictions[
            results.predictions["layer"] == results.selected_layer
        ],
        "controls.csv": results.controls,
        "domain_transfer.csv": results.transfer,
        "test_group_bootstrap.csv": results.bootstrap,
        "test_group_bootstrap_summary.csv": results.bootstrap_summary,
        "pca_subspace.csv": results.pca_curve,
        "diagnostic_target_metrics.csv": results.diagnostic_metrics,
        "calibration.csv": results.calibration,
        "threshold_sensitivity.csv": results.threshold_sensitivity,
        "score_trajectories.csv": results.trajectories,
        "subgroup_metrics.csv": results.subgroups,
    }
    for name, frame in tables.items():
        if not frame.empty:
            frame.to_csv(output / name, index=False)
    np.savez(
        output / "directions.npz",
        directions=results.directions,
        projection_stds=results.projection_stds,
        thresholds=results.thresholds,
        c_values=results.c_values,
        selected_layer=results.selected_layer,
        selected_intervention_layer=results.selected_intervention_layer,
    )


def fit_error_only_probe(
    hidden: np.ndarray,
    metadata: pd.DataFrame,
    config: Experiment2Config,
    *,
    centered: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train only within erroneous traces, optionally removing each trace's first-step state."""
    eligible = metadata["has_error_trace"].eq(1).to_numpy()
    variant = "raw_error_traces"
    values = np.asarray(hidden, dtype=np.float32).copy()
    if centered:
        # A first-step reference is meaningful only when the annotated error occurs later.
        eligible &= metadata["first_error"].gt(0).to_numpy()
        variant = "first_step_centered_error_traces"
        for _, indices in metadata.loc[eligible].groupby("trace_id", sort=False).groups.items():
            positions = np.asarray(list(indices), dtype=int)
            ordered = positions[np.argsort(metadata.loc[positions, "step_index"].to_numpy())]
            values[ordered] -= values[ordered[0]]

    frame = metadata.loc[eligible].copy()
    features = values[eligible]
    labels = frame["invalid_so_far"].to_numpy(dtype=int)
    masks = {
        split: frame["partition"].eq(split).to_numpy()
        for split in ("train", "validation", "test")
    }
    _require_binary_partitions(labels, masks, variant)
    best_c, validation_scores = _select_c(
        features[masks["train"]],
        labels[masks["train"]],
        features[masks["validation"]],
        labels[masks["validation"]],
        config.analysis.c_values,
        config.analysis.max_iter,
        config.seed,
    )
    threshold = choose_binary_threshold(
        labels[masks["validation"]],
        validation_scores,
        config,
    )
    fit = masks["train"] | masks["validation"]
    scaler, classifier = _fit_logistic(
        features[fit],
        labels[fit],
        best_c,
        config.analysis.max_iter,
        config.seed,
    )
    test_scores = classifier.predict_proba(scaler.transform(features[masks["test"]]))[:, 1]
    test = frame.loc[masks["test"]].copy()
    metrics = {
        "variant": variant,
        "c_value": best_c,
        "threshold": threshold,
        "n_test_traces": test["trace_id"].nunique(),
        "n_test_steps": len(test),
        **binary_metrics(labels[masks["test"]], test_scores, threshold=threshold),
        **change_point_metrics(test, test_scores, threshold),
    }
    predictions = test[
        [
            "trace_id",
            "source",
            "generator",
            "step_index",
            "first_error",
            "invalid_so_far",
        ]
    ].copy()
    predictions["variant"] = variant
    predictions["score"] = test_scores
    predictions["threshold"] = threshold
    bootstrap = group_bootstrap_metrics(
        test,
        labels[masks["test"]],
        test_scores,
        threshold,
        samples=config.analysis.bootstrap_samples,
        seed=config.seed,
    )
    for row in summarize_bootstrap(bootstrap, config.analysis.confidence_level).itertuples():
        if row.metric in {"auroc", "average_precision", "first_error_exact"}:
            metrics[f"{row.metric}_ci_low"] = row.ci_low
            metrics[f"{row.metric}_ci_high"] = row.ci_high
    return pd.DataFrame([metrics]), predictions


def choose_binary_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    config: Experiment2Config,
) -> float:
    """Select a threshold by validation balanced accuracy without correct-trace assumptions."""
    candidates = np.linspace(
        config.analysis.threshold_min,
        config.analysis.threshold_max,
        config.analysis.threshold_points,
    )
    ranked = []
    for threshold in candidates:
        metric = binary_metrics(labels, scores, threshold=float(threshold))["balanced_accuracy"]
        ranked.append((metric, -abs(float(threshold) - 0.5), float(threshold)))
    return max(ranked)[2]


def generator_holdout(
    hidden: np.ndarray,
    metadata: pd.DataFrame,
    config: Experiment2Config,
) -> pd.DataFrame:
    """Train without one generator and evaluate only that generator's held-out traces."""
    labels = metadata["invalid_so_far"].to_numpy(dtype=int)
    rows = []
    test_frame = metadata[metadata["partition"] == "test"]
    for generator, group in test_frame.groupby("generator", sort=True):
        n_test_traces = group["trace_id"].nunique()
        if n_test_traces < config.analysis.generator_min_test_traces:
            continue
        train = metadata["partition"].eq("train") & metadata["generator"].ne(generator)
        validation = metadata["partition"].eq("validation") & metadata["generator"].ne(generator)
        test = metadata["partition"].eq("test") & metadata["generator"].eq(generator)
        if any(len(np.unique(labels[mask])) < 2 for mask in (train, validation, test)):
            continue
        best_c, validation_scores = _select_c(
            hidden[train],
            labels[train],
            hidden[validation],
            labels[validation],
            config.analysis.c_values,
            config.analysis.max_iter,
            config.seed,
        )
        validation_metadata = metadata.loc[validation]
        if _has_error_and_correct_traces(validation_metadata):
            threshold = choose_threshold(
                validation_metadata,
                validation_scores,
                threshold_min=config.analysis.threshold_min,
                threshold_max=config.analysis.threshold_max,
                threshold_points=config.analysis.threshold_points,
            )
        else:
            threshold = choose_binary_threshold(labels[validation], validation_scores, config)
        fit = train | validation
        scaler, classifier = _fit_logistic(
            hidden[fit],
            labels[fit],
            best_c,
            config.analysis.max_iter,
            config.seed,
        )
        scores = classifier.predict_proba(scaler.transform(hidden[test]))[:, 1]
        rows.append(
            {
                "held_out_generator": generator,
                "n_test_traces": n_test_traces,
                "n_test_steps": int(test.sum()),
                "c_value": best_c,
                "threshold": threshold,
                **binary_metrics(labels[test], scores, threshold=threshold),
                **change_point_metrics(metadata.loc[test], scores, threshold),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_onset_jump(
    predictions: pd.DataFrame,
    config: Experiment2Config,
) -> pd.DataFrame:
    """Bootstrap the paired score change from the last pre-error step to the error step."""
    jumps = []
    for trace_id, trace in predictions.groupby("trace_id", sort=False):
        first_error = int(trace["first_error"].iloc[0])
        if first_error <= 0:
            continue
        onset = trace.loc[trace["step_index"] == first_error, "score"]
        previous = trace.loc[trace["step_index"] == first_error - 1, "score"]
        if not onset.empty and not previous.empty:
            jumps.append(
                {
                    "trace_id": trace_id,
                    "jump": float(onset.iloc[0] - previous.iloc[0]),
                }
            )
    frame = pd.DataFrame(jumps)
    rng = np.random.default_rng(config.seed)
    means = np.asarray(
        [
            rng.choice(frame["jump"].to_numpy(), size=len(frame), replace=True).mean()
            for _ in range(config.analysis.bootstrap_samples)
        ]
    )
    tail = (1 - config.analysis.confidence_level) / 2
    return pd.DataFrame(
        [
            {
                "n_traces": len(frame),
                "mean_onset_jump": frame["jump"].mean(),
                "median_onset_jump": frame["jump"].median(),
                "positive_jump_rate": (frame["jump"] > 0).mean(),
                "ci_low": np.quantile(means, tail),
                "ci_high": np.quantile(means, 1 - tail),
            }
        ]
    )


def domain_calibration_analysis(
    hidden: np.ndarray,
    metadata: pd.DataFrame,
    config: Experiment2Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate cross-source ranking transfer from target-threshold calibration."""
    labels = metadata["invalid_so_far"].to_numpy(dtype=int)
    sources = sorted(metadata["source"].unique())
    rows = []
    directions = {}
    for train_source in sources:
        train = metadata["partition"].eq("train") & metadata["source"].eq(train_source)
        validation_source = metadata["partition"].eq("validation") & metadata["source"].eq(
            train_source
        )
        if any(len(np.unique(labels[mask])) < 2 for mask in (train, validation_source)):
            continue
        best_c, validation_scores = _select_c(
            hidden[train],
            labels[train],
            hidden[validation_source],
            labels[validation_source],
            config.analysis.c_values,
            config.analysis.max_iter,
            config.seed,
        )
        source_threshold = choose_threshold(
            metadata.loc[validation_source],
            validation_scores,
            threshold_min=config.analysis.threshold_min,
            threshold_max=config.analysis.threshold_max,
            threshold_points=config.analysis.threshold_points,
        )
        scaler, classifier = _fit_logistic(
            hidden[train],
            labels[train],
            best_c,
            config.analysis.max_iter,
            config.seed,
        )
        raw_direction = classifier.coef_[0] / scaler.scale_
        directions[train_source] = raw_direction / np.linalg.norm(raw_direction)
        for test_source in sources:
            target_validation = metadata["partition"].eq("validation") & metadata["source"].eq(
                test_source
            )
            test = metadata["partition"].eq("test") & metadata["source"].eq(test_source)
            if any(len(np.unique(labels[mask])) < 2 for mask in (target_validation, test)):
                continue
            target_validation_scores = classifier.predict_proba(
                scaler.transform(hidden[target_validation])
            )[:, 1]
            target_threshold = choose_threshold(
                metadata.loc[target_validation],
                target_validation_scores,
                threshold_min=config.analysis.threshold_min,
                threshold_max=config.analysis.threshold_max,
                threshold_points=config.analysis.threshold_points,
            )
            scores = classifier.predict_proba(scaler.transform(hidden[test]))[:, 1]
            binary = binary_metrics(labels[test], scores, threshold=source_threshold)
            source_change = change_point_metrics(
                metadata.loc[test],
                scores,
                source_threshold,
            )
            target_change = change_point_metrics(
                metadata.loc[test],
                scores,
                target_threshold,
            )
            bootstrap_source = group_bootstrap_metrics(
                metadata.loc[test],
                labels[test],
                scores,
                source_threshold,
                samples=config.analysis.bootstrap_samples,
                seed=config.seed,
            )
            bootstrap_target = group_bootstrap_metrics(
                metadata.loc[test],
                labels[test],
                scores,
                target_threshold,
                samples=config.analysis.bootstrap_samples,
                seed=config.seed + 1,
            )
            tail = (1 - config.analysis.confidence_level) / 2
            rows.append(
                {
                    "train_source": train_source,
                    "test_source": test_source,
                    "c_value": best_c,
                    "source_threshold": source_threshold,
                    "target_threshold": target_threshold,
                    "auroc": binary["auroc"],
                    "auroc_ci_low": bootstrap_source["auroc"].quantile(tail),
                    "auroc_ci_high": bootstrap_source["auroc"].quantile(1 - tail),
                    "source_threshold_process_f1": source_change["process_f1"],
                    "source_process_f1_ci_low": bootstrap_source["process_f1"].quantile(tail),
                    "source_process_f1_ci_high": bootstrap_source["process_f1"].quantile(
                        1 - tail
                    ),
                    "target_threshold_process_f1": target_change["process_f1"],
                    "target_process_f1_ci_low": bootstrap_target["process_f1"].quantile(tail),
                    "target_process_f1_ci_high": bootstrap_target["process_f1"].quantile(
                        1 - tail
                    ),
                    "calibration_gain": (
                        target_change["process_f1"] - source_change["process_f1"]
                    ),
                    "n_test_traces": metadata.loc[test, "trace_id"].nunique(),
                }
            )
    cosine_rows = []
    for first_source, first_direction in directions.items():
        for second_source, second_direction in directions.items():
            cosine_rows.append(
                {
                    "source_a": first_source,
                    "source_b": second_source,
                    "cosine": float(np.dot(first_direction, second_direction)),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(cosine_rows)


def fit_surface_metadata_control(
    metadata: pd.DataFrame,
    config: Experiment2Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a combined lexical, position, length, source, and generator control."""
    masks = {
        split: metadata["partition"].eq(split).to_numpy()
        for split in ("train", "validation", "test")
    }
    labels = metadata["invalid_so_far"].to_numpy(dtype=int)
    text = metadata["step_text"].fillna("").astype(str)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000)
    tfidf.fit(text[masks["train"]])
    text_features = tfidf.transform(text)

    numeric = ["step_index", "step_fraction", "n_steps", "token_count"]
    categorical = ["source", "generator"]
    transformer = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                categorical,
            ),
        ],
        sparse_threshold=1.0,
    )
    transformer.fit(metadata.loc[masks["train"], numeric + categorical])
    metadata_features = transformer.transform(metadata[numeric + categorical])
    features = hstack([text_features, csr_matrix(metadata_features)], format="csr")

    candidates = []
    for c_value in config.analysis.c_values:
        classifier = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=config.analysis.max_iter,
            solver="liblinear",
            random_state=config.seed,
        ).fit(features[masks["train"]], labels[masks["train"]])
        scores = classifier.predict_proba(features[masks["validation"]])[:, 1]
        candidates.append(
            (
                roc_auc_score(labels[masks["validation"]], scores),
                -abs(np.log10(c_value)),
                c_value,
                scores,
            )
        )
    _, _, best_c, validation_scores = max(candidates, key=lambda item: item[:2])
    threshold = choose_threshold(
        metadata.loc[masks["validation"]],
        validation_scores,
        threshold_min=config.analysis.threshold_min,
        threshold_max=config.analysis.threshold_max,
        threshold_points=config.analysis.threshold_points,
    )
    fit = masks["train"] | masks["validation"]
    classifier = LogisticRegression(
        C=best_c,
        class_weight="balanced",
        max_iter=config.analysis.max_iter,
        solver="liblinear",
        random_state=config.seed,
    ).fit(features[fit], labels[fit])
    test_scores = classifier.predict_proba(features[masks["test"]])[:, 1]
    test = metadata.loc[masks["test"]].copy()
    predictions = test[
        ["trace_id", "step_index", "first_error", "invalid_so_far"]
    ].copy()
    predictions["label"] = labels[masks["test"]]
    predictions["score"] = test_scores
    metrics = pd.DataFrame(
        [
            {
                "control": "TF-IDF + position + length + source + generator",
                "c_value": best_c,
                "threshold": threshold,
                **binary_metrics(labels[masks["test"]], test_scores, threshold=threshold),
                **change_point_metrics(test, test_scores, threshold),
            }
        ]
    )
    return predictions, metrics


def paired_control_comparison(
    hidden_predictions: pd.DataFrame,
    surface_predictions: pd.DataFrame,
    *,
    hidden_threshold: float,
    surface_threshold: float,
    config: Experiment2Config,
) -> pd.DataFrame:
    """Calculate paired whole-trace intervals for hidden-minus-surface performance."""
    merged = hidden_predictions.merge(
        surface_predictions[["trace_id", "step_index", "score"]],
        on=["trace_id", "step_index"],
        suffixes=("_primary", "_candidate"),
        validate="one_to_one",
    )
    # The shared utility returns candidate minus primary. Here candidate is surface and primary is
    # hidden, so negate every draw to report the more intuitive hidden-minus-surface difference.
    bootstrap = -_paired_comparison_bootstrap(
        merged,
        hidden_threshold,
        surface_threshold,
        AnalysisConfig(
            confidence_level=config.analysis.confidence_level,
            localization_tolerances=(0, 1, 2),
        ),
        samples=config.analysis.bootstrap_samples,
        seed=config.seed,
    )
    tail = (1 - config.analysis.confidence_level) / 2
    rows = []
    for metric in bootstrap:
        values = bootstrap[metric].dropna()
        rows.append(
            {
                "metric": metric,
                "hidden_minus_surface_estimate": values.mean(),
                "ci_low": values.quantile(tail),
                "ci_high": values.quantile(1 - tail),
                "bootstrap_samples": len(values),
            }
        )
    return pd.DataFrame(rows)


def _require_binary_partitions(
    labels: np.ndarray,
    masks: dict[str, np.ndarray],
    name: str,
) -> None:
    if any(len(np.unique(labels[mask])) < 2 for mask in masks.values()):
        raise ValueError(f"{name} requires both labels in train, validation, and test")


def _has_error_and_correct_traces(metadata: pd.DataFrame) -> bool:
    traces = metadata.drop_duplicates("trace_id")
    return traces["has_error_trace"].nunique() == 2
