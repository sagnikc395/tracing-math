"""CPU-only follow-up analyses over frozen Experiment 1 artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class CPUFollowupConfig:
    """Paths and frozen resampling settings for the CPU follow-up."""

    experiment1_dir: Path
    data_path: Path
    output_dir: Path
    seed: int = 42
    permutation_samples: int = 5_000
    bootstrap_samples: int = 2_000
    confidence_level: float = 0.95
    subgroup_min_traces: int = 20
    audit_examples_per_category: int = 12

    @classmethod
    def from_yaml(cls, path: str | Path) -> CPUFollowupConfig:
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError("CPU follow-up configuration must be a YAML mapping")
        config = cls(
            experiment1_dir=Path(raw["experiment1_dir"]),
            data_path=Path(raw["data_path"]),
            output_dir=Path(raw["output_dir"]),
            seed=int(raw.get("seed", 42)),
            permutation_samples=int(raw.get("permutation_samples", 5_000)),
            bootstrap_samples=int(raw.get("bootstrap_samples", 2_000)),
            confidence_level=float(raw.get("confidence_level", 0.95)),
            subgroup_min_traces=int(raw.get("subgroup_min_traces", 20)),
            audit_examples_per_category=int(raw.get("audit_examples_per_category", 12)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.permutation_samples < 1 or self.bootstrap_samples < 1:
            raise ValueError("resampling counts must be positive")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between zero and one")
        if self.subgroup_min_traces < 1 or self.audit_examples_per_category < 1:
            raise ValueError("trace and audit sample minima must be positive")


@dataclass(frozen=True)
class TraceSeries:
    trace_id: str
    source: str
    generator: str
    first_error: int
    n_steps: int
    token_count: int
    final_answer_correct: int
    scores: np.ndarray


def run_cpu_followup(config: CPUFollowupConfig) -> dict[str, Any]:
    """Run the complete follow-up without loading a language model or activation shard."""
    predictions = _load_predictions(config.experiment1_dir)
    threshold = _load_threshold(config.experiment1_dir, predictions)
    traces = _trace_series(predictions)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)

    permutation_summary, permutation_draws = temporal_randomization_test(
        traces,
        threshold=threshold,
        samples=config.permutation_samples,
        seed=config.seed,
        confidence_level=config.confidence_level,
    )
    permutation_summary.to_csv(output / "temporal_randomization_summary.csv", index=False)
    permutation_draws.to_csv(output / "temporal_randomization_draws.csv", index=False)

    trajectory = event_study(
        traces,
        samples=config.bootstrap_samples,
        seed=config.seed,
        confidence_level=config.confidence_level,
    )
    trajectory.to_csv(output / "error_aligned_trajectory.csv", index=False)

    placebo = matched_placebo_analysis(
        traces,
        samples=config.bootstrap_samples,
        seed=config.seed,
        confidence_level=config.confidence_level,
    )
    placebo.to_csv(output / "matched_placebo_onset.csv", index=False)

    centered = centered_discrimination(
        predictions,
        samples=config.bootstrap_samples,
        seed=config.seed,
        confidence_level=config.confidence_level,
    )
    centered.to_csv(output / "within_trace_discrimination.csv", index=False)

    subgroups = subgroup_outcomes(
        traces,
        threshold=threshold,
        samples=config.bootstrap_samples,
        seed=config.seed,
        confidence_level=config.confidence_level,
        min_traces=config.subgroup_min_traces,
    )
    subgroups.to_csv(output / "subgroup_outcomes.csv", index=False)

    sensitivity = causal_sensitivity_analysis(
        config.experiment1_dir,
        confidence_level=config.confidence_level,
    )
    sensitivity.to_csv(output / "causal_assay_sensitivity.csv", index=False)

    audit_path = write_failure_audit_sample(
        traces,
        threshold=threshold,
        data_path=config.data_path,
        output_path=output / "failure_audit_sample.jsonl",
        examples_per_category=config.audit_examples_per_category,
        seed=config.seed,
    )
    _plot_followup(trajectory, permutation_draws, permutation_summary, output)

    summary = _build_summary(
        predictions=predictions,
        threshold=threshold,
        permutation=permutation_summary,
        placebo=placebo,
        centered=centered,
        sensitivity=sensitivity,
        audit_path=audit_path,
        config=config,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    (output / "results.md").write_text(_results_markdown(summary, placebo, centered, sensitivity))
    return summary


def temporal_randomization_test(
    traces: list[TraceSeries],
    *,
    threshold: float,
    samples: int,
    seed: int,
    confidence_level: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare observed timing with within-trace circular shifts of the frozen scores."""
    expected = np.asarray([trace.first_error for trace in traces], dtype=int)
    observed_predicted = np.asarray(
        [_first_crossing(trace.scores, threshold) for trace in traces], dtype=int
    )
    observed = _trace_metrics(expected, observed_predicted)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for sample in range(samples):
        predicted = []
        for trace in traces:
            offset = int(rng.integers(0, len(trace.scores))) if len(trace.scores) > 1 else 0
            predicted.append(_first_crossing(np.roll(trace.scores, offset), threshold))
        rows.append({"sample": sample, **_trace_metrics(expected, np.asarray(predicted))})
    draws = pd.DataFrame(rows)
    tail = (1 - confidence_level) / 2
    specifications = {
        "first_error_exact": "higher",
        "error_within_1_accuracy": "higher",
        "error_within_2_accuracy": "higher",
        "process_f1": "higher",
        "mean_absolute_localization_error": "lower",
        "mean_signed_localization_error": "two_sided",
    }
    summary_rows = []
    for metric, direction in specifications.items():
        values = draws[metric].dropna().to_numpy(dtype=float)
        estimate = float(observed[metric])
        if direction == "higher":
            exceedances = int(np.sum(values >= estimate))
        elif direction == "lower":
            exceedances = int(np.sum(values <= estimate))
        else:
            center = float(np.mean(values))
            exceedances = int(np.sum(np.abs(values - center) >= abs(estimate - center)))
        summary_rows.append(
            {
                "metric": metric,
                "observed": estimate,
                "null_mean": float(np.mean(values)),
                "null_ci_low": float(np.quantile(values, tail)),
                "null_ci_high": float(np.quantile(values, 1 - tail)),
                "p_value": (exceedances + 1) / (len(values) + 1),
                "direction": direction,
                "permutations": len(values),
            }
        )
    return pd.DataFrame(summary_rows), draws


def event_study(
    traces: list[TraceSeries],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
    window: int = 3,
) -> pd.DataFrame:
    """Estimate the frozen score trajectory around the human-annotated first error."""
    erroneous = [trace for trace in traces if trace.first_error >= 0]
    rng = np.random.default_rng(seed)
    tail = (1 - confidence_level) / 2
    rows = []
    for relative_step in range(-window, window + 1):
        values = np.asarray(
            [
                trace.scores[trace.first_error + relative_step]
                for trace in erroneous
                if 0 <= trace.first_error + relative_step < len(trace.scores)
            ],
            dtype=float,
        )
        boot = np.asarray(
            [rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)]
        )
        rows.append(
            {
                "relative_step": relative_step,
                "mean_score": float(values.mean()),
                "median_score": float(np.median(values)),
                "ci_low": float(np.quantile(boot, tail)),
                "ci_high": float(np.quantile(boot, 1 - tail)),
                "n_traces": len(values),
            }
        )
    return pd.DataFrame(rows)


def matched_placebo_analysis(
    traces: list[TraceSeries],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> pd.DataFrame:
    """Compare error-onset score jumps with metadata-matched transitions on correct traces."""
    erroneous = [trace for trace in traces if trace.first_error > 0]
    correct = [trace for trace in traces if trace.first_error < 0 and len(trace.scores) > 1]
    pairs = []
    for trace in erroneous:
        match, tier, placebo_step = _match_placebo(trace, correct)
        real_jump = float(trace.scores[trace.first_error] - trace.scores[trace.first_error - 1])
        placebo_jump = float(match.scores[placebo_step] - match.scores[placebo_step - 1])
        prior_jumps = np.diff(trace.scores[: trace.first_error])
        prior_mean = float(prior_jumps.mean()) if len(prior_jumps) else float("nan")
        pairs.append(
            {
                "trace_id": trace.trace_id,
                "placebo_trace_id": match.trace_id,
                "match_tier": tier,
                "real_jump": real_jump,
                "placebo_jump": placebo_jump,
                "paired_difference": real_jump - placebo_jump,
                "prior_mean_jump": prior_mean,
                "onset_minus_prior": real_jump - prior_mean,
            }
        )
    frame = pd.DataFrame(pairs)
    rng = np.random.default_rng(seed)
    tail = (1 - confidence_level) / 2
    rows = []
    for metric in ("real_jump", "placebo_jump", "paired_difference", "onset_minus_prior"):
        values = frame[metric].dropna().to_numpy(dtype=float)
        boot = np.asarray(
            [rng.choice(values, size=len(values), replace=True).mean() for _ in range(samples)]
        )
        rows.append(
            {
                "metric": metric,
                "estimate": float(values.mean()),
                "median": float(np.median(values)),
                "ci_low": float(np.quantile(boot, tail)),
                "ci_high": float(np.quantile(boot, 1 - tail)),
                "n_traces": len(values),
                "exact_source_generator_matches": int(
                    (frame["match_tier"] == "source_generator").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def centered_discrimination(
    predictions: pd.DataFrame,
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> pd.DataFrame:
    """Measure discrimination after removing each erroneous trace's first-step score."""
    frame = predictions[
        predictions["has_error_trace"].eq(1) & predictions["first_error"].gt(0)
    ].copy()
    first_scores = frame.sort_values("step_index").groupby("trace_id")["score"].first()
    frame["centered_score"] = frame["score"] - frame["trace_id"].map(first_scores)
    trace_ids = frame["trace_id"].drop_duplicates().to_numpy()
    raw = roc_auc_score(frame["label"], frame["score"])
    centered = roc_auc_score(frame["label"], frame["centered_score"])
    macro_values = []
    for _, trace in frame.groupby("trace_id", sort=False):
        if trace["label"].nunique() == 2:
            macro_values.append(roc_auc_score(trace["label"], trace["score"]))
    rng = np.random.default_rng(seed)
    raw_draws = []
    centered_draws = []
    for _ in range(samples):
        selected = rng.choice(trace_ids, size=len(trace_ids), replace=True)
        sampled = pd.concat([frame[frame["trace_id"] == trace_id] for trace_id in selected])
        raw_draws.append(roc_auc_score(sampled["label"], sampled["score"]))
        centered_draws.append(roc_auc_score(sampled["label"], sampled["centered_score"]))
    tail = (1 - confidence_level) / 2
    return pd.DataFrame(
        [
            _interval_row("pooled_raw_auroc", raw, raw_draws, tail, len(trace_ids)),
            _interval_row(
                "pooled_first_step_centered_auroc",
                centered,
                centered_draws,
                tail,
                len(trace_ids),
            ),
            {
                "metric": "mean_within_trace_auroc",
                "estimate": float(np.mean(macro_values)),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "n_traces": len(macro_values),
            },
        ]
    )


def subgroup_outcomes(
    traces: list[TraceSeries],
    *,
    threshold: float,
    samples: int,
    seed: int,
    confidence_level: float,
    min_traces: int,
) -> pd.DataFrame:
    """Report trace-level localization outcomes across frozen metadata groups."""
    frame = _trace_outcome_frame(traces, threshold)
    frame["error_position"] = np.select(
        [
            frame["first_error"].lt(0),
            frame["first_error"] / frame["n_steps"].clip(lower=1) <= 1 / 3,
            frame["first_error"] / frame["n_steps"].clip(lower=1) <= 2 / 3,
        ],
        ["correct", "early", "middle"],
        default="late",
    )
    frame["trace_length_bin"] = pd.qcut(
        frame["n_steps"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    ).astype(str)
    frame["token_count_bin"] = pd.qcut(
        frame["token_count"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
    ).astype(str)
    rng = np.random.default_rng(seed)
    tail = (1 - confidence_level) / 2
    rows = []
    for column in (
        "source",
        "generator",
        "final_answer_correct",
        "error_position",
        "trace_length_bin",
        "token_count_bin",
    ):
        for value, group in frame.groupby(column, dropna=False, sort=True):
            if len(group) < min_traces:
                continue
            draws = []
            for _ in range(samples):
                sampled = group.iloc[rng.integers(0, len(group), len(group))]
                draws.append(_outcome_summary(sampled))
            draw_frame = pd.DataFrame(draws)
            point = _outcome_summary(group)
            row: dict[str, Any] = {
                "subgroup": column,
                "value": str(value),
                "n_traces": len(group),
                **point,
            }
            for metric in ("complete_trace_accuracy", "error_exact", "correct_rejection"):
                values = draw_frame[metric].dropna()
                row[f"{metric}_ci_low"] = (
                    float(values.quantile(tail)) if not values.empty else float("nan")
                )
                row[f"{metric}_ci_high"] = (
                    float(values.quantile(1 - tail)) if not values.empty else float("nan")
                )
            rows.append(row)
    return pd.DataFrame(rows)


def causal_sensitivity_analysis(
    experiment1_dir: Path,
    *,
    confidence_level: float,
    power: float = 0.8,
) -> pd.DataFrame:
    """Estimate paired mean effects and approximate minimum detectable effects."""
    path = experiment1_dir / "interventions" / "individual.csv"
    frame = pd.read_csv(path)
    learned = frame[
        frame["direction_type"].eq("learned") & frame["alpha"].ne(0)
    ].copy()
    baseline_scale = float(
        frame[frame["direction_type"].eq("learned")]
        .drop_duplicates(["trace_id", "step_index"])["baseline_verdict_score"]
        .std(ddof=1)
    )
    z_alpha = norm.ppf(0.5 + confidence_level / 2)
    z_power = norm.ppf(power)
    rows = []
    for alpha, group in learned.groupby("alpha", sort=True):
        values = group["delta_verdict_score"].to_numpy(dtype=float)
        standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
        estimate = float(values.mean())
        half_width = z_alpha * standard_error
        detectable = (z_alpha + z_power) * standard_error
        rows.append(
            {
                "alpha": float(alpha),
                "estimate": estimate,
                "ci_low": estimate - half_width,
                "ci_high": estimate + half_width,
                "standard_error": standard_error,
                "minimum_detectable_effect_80pct": detectable,
                "minimum_detectable_standardized_effect_80pct": detectable / baseline_scale,
                "baseline_score_standard_deviation": baseline_scale,
                "n": len(values),
            }
        )
    return pd.DataFrame(rows)


def write_failure_audit_sample(
    traces: list[TraceSeries],
    *,
    threshold: float,
    data_path: Path,
    output_path: Path,
    examples_per_category: int,
    seed: int,
) -> Path:
    """Freeze a stratified qualitative sample before manual inspection."""
    outcomes = _trace_outcome_frame(traces, threshold)
    rng = np.random.default_rng(seed)
    selected_ids = []
    for category, group in outcomes.groupby("outcome", sort=True):
        count = min(examples_per_category, len(group))
        positions = rng.choice(len(group), size=count, replace=False)
        selected_ids.extend((category, trace_id) for trace_id in group.iloc[positions].trace_id)
    lookup = {trace_id: category for category, trace_id in selected_ids}
    details = {
        trace.trace_id: trace
        for trace in traces
        if trace.trace_id in lookup
    }
    records = []
    with data_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            trace_id = str(row["trace_id"])
            if trace_id not in lookup:
                continue
            trace = details[trace_id]
            records.append(
                {
                    "trace_id": trace_id,
                    "sampling_category": lookup[trace_id],
                    "source": trace.source,
                    "generator": trace.generator,
                    "first_error": trace.first_error,
                    "predicted_first_error": _first_crossing(trace.scores, threshold),
                    "problem": row["problem"],
                    "steps": row["steps"],
                    "manual_error_type": "",
                    "manual_notes": "",
                }
            )
    records.sort(key=lambda row: (row["sampling_category"], row["trace_id"]))
    with output_path.open("w") as handle:
        for row in records:
            handle.write(json.dumps(row) + "\n")
    return output_path


def _load_predictions(experiment1_dir: Path) -> pd.DataFrame:
    path = experiment1_dir / "probes" / "test_predictions.csv"
    frame = pd.read_csv(path)
    required = {
        "trace_id",
        "source",
        "generator",
        "step_index",
        "first_error",
        "n_steps",
        "token_count",
        "final_answer_correct",
        "has_error_trace",
        "label",
        "score",
        "layer",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if frame["layer"].nunique() != 1:
        raise ValueError("CPU follow-up expects predictions from exactly one frozen layer")
    if frame.duplicated(["trace_id", "step_index"]).any():
        raise ValueError("Prediction rows must be unique by trace and step")
    return frame.sort_values(["trace_id", "step_index"]).reset_index(drop=True)


def _load_threshold(experiment1_dir: Path, predictions: pd.DataFrame) -> float:
    artifact = np.load(experiment1_dir / "probes" / "directions.npz")
    layer = int(predictions["layer"].iloc[0])
    if layer != int(artifact["selected_layer"]):
        raise ValueError("Prediction layer does not match the frozen selected layer")
    return float(artifact["thresholds"][layer])


def _trace_series(predictions: pd.DataFrame) -> list[TraceSeries]:
    traces = []
    for trace_id, frame in predictions.groupby("trace_id", sort=False):
        ordered = frame.sort_values("step_index")
        expected_steps = np.arange(len(ordered))
        if not np.array_equal(ordered["step_index"].to_numpy(), expected_steps):
            raise ValueError(f"Non-contiguous step indices for {trace_id}")
        first = ordered.iloc[0]
        traces.append(
            TraceSeries(
                trace_id=str(trace_id),
                source=str(first["source"]),
                generator=str(first["generator"]),
                first_error=int(first["first_error"]),
                n_steps=int(first["n_steps"]),
                token_count=int(first["token_count"]),
                final_answer_correct=int(first["final_answer_correct"]),
                scores=ordered["score"].to_numpy(dtype=float),
            )
        )
    return traces


def _first_crossing(scores: np.ndarray, threshold: float) -> int:
    crossings = np.flatnonzero(scores >= threshold)
    return int(crossings[0]) if crossings.size else -1


def _trace_metrics(expected: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    erroneous = expected >= 0
    correct = ~erroneous
    detected = erroneous & (predicted >= 0)
    error_exact = float(np.mean(predicted[erroneous] == expected[erroneous]))
    correct_accuracy = float(np.mean(predicted[correct] == -1))
    denominator = error_exact + correct_accuracy
    signed = predicted[detected] - expected[detected]
    return {
        "first_error_exact": error_exact,
        "correct_rejection": correct_accuracy,
        "process_f1": 2 * error_exact * correct_accuracy / denominator if denominator else 0.0,
        "complete_trace_accuracy": float(np.mean(predicted == expected)),
        "error_detection_rate": float(np.mean(predicted[erroneous] >= 0)),
        "error_within_1_accuracy": float(
            np.mean(
                (predicted[erroneous] >= 0)
                & (np.abs(predicted[erroneous] - expected[erroneous]) <= 1)
            )
        ),
        "error_within_2_accuracy": float(
            np.mean(
                (predicted[erroneous] >= 0)
                & (np.abs(predicted[erroneous] - expected[erroneous]) <= 2)
            )
        ),
        "mean_signed_localization_error": float(np.mean(signed)) if signed.size else float("nan"),
        "mean_absolute_localization_error": (
            float(np.mean(np.abs(signed))) if signed.size else float("nan")
        ),
    }


def _match_placebo(
    target: TraceSeries,
    correct: list[TraceSeries],
) -> tuple[TraceSeries, str, int]:
    exact = [
        trace
        for trace in correct
        if trace.source == target.source and trace.generator == target.generator
    ]
    candidates = exact
    tier = "source_generator"
    if not candidates:
        candidates = [trace for trace in correct if trace.source == target.source]
        tier = "source"
    if not candidates:
        candidates = correct
        tier = "global"
    target_fraction = (target.first_error + 1) / target.n_steps
    ranked = []
    for trace in candidates:
        step = int(np.clip(round(target_fraction * trace.n_steps) - 1, 1, len(trace.scores) - 1))
        fraction = (step + 1) / trace.n_steps
        distance = (
            abs(np.log(trace.n_steps / target.n_steps))
            + abs(np.log(max(trace.token_count, 1) / max(target.token_count, 1)))
            + 2 * abs(fraction - target_fraction)
        )
        ranked.append((distance, trace.trace_id, step, trace))
    _, _, step, match = min(ranked, key=lambda item: item[:2])
    return match, tier, step


def _interval_row(
    metric: str,
    estimate: float,
    values: list[float],
    tail: float,
    n_traces: int,
) -> dict[str, float | int | str]:
    array = np.asarray(values, dtype=float)
    return {
        "metric": metric,
        "estimate": estimate,
        "ci_low": float(np.quantile(array, tail)),
        "ci_high": float(np.quantile(array, 1 - tail)),
        "n_traces": n_traces,
    }


def _trace_outcome_frame(traces: list[TraceSeries], threshold: float) -> pd.DataFrame:
    rows = []
    for trace in traces:
        predicted = _first_crossing(trace.scores, threshold)
        if trace.first_error < 0:
            outcome = "correct_rejection" if predicted < 0 else "false_alarm"
        elif predicted < 0:
            outcome = "miss"
        elif predicted < trace.first_error:
            outcome = "early"
        elif predicted == trace.first_error:
            outcome = "on_time"
        else:
            outcome = "late"
        rows.append(
            {
                "trace_id": trace.trace_id,
                "source": trace.source,
                "generator": trace.generator,
                "first_error": trace.first_error,
                "predicted_first_error": predicted,
                "n_steps": trace.n_steps,
                "token_count": trace.token_count,
                "final_answer_correct": trace.final_answer_correct,
                "outcome": outcome,
            }
        )
    return pd.DataFrame(rows)


def _outcome_summary(frame: pd.DataFrame) -> dict[str, float]:
    erroneous = frame["first_error"].ge(0)
    correct = ~erroneous
    complete = frame["predicted_first_error"].eq(frame["first_error"])
    return {
        "complete_trace_accuracy": float(complete.mean()),
        "error_exact": (
            float(complete[erroneous].mean()) if erroneous.any() else float("nan")
        ),
        "correct_rejection": (
            float(frame.loc[correct, "predicted_first_error"].lt(0).mean())
            if correct.any()
            else float("nan")
        ),
    }


def _plot_followup(
    trajectory: pd.DataFrame,
    permutation_draws: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    output: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    axes[0].plot(trajectory["relative_step"], trajectory["mean_score"], marker="o")
    axes[0].fill_between(
        trajectory["relative_step"], trajectory["ci_low"], trajectory["ci_high"], alpha=0.2
    )
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set(xlabel="Step relative to first error", ylabel="Mean frozen probe score")

    observed = float(
        permutation_summary.loc[
            permutation_summary["metric"].eq("first_error_exact"), "observed"
        ].iloc[0]
    )
    axes[1].hist(permutation_draws["first_error_exact"], bins=30, color="#777777")
    axes[1].axvline(observed, color="#b2182b", linewidth=2, label="Observed")
    axes[1].set(xlabel="Exact localization under circular shifts", ylabel="Count")
    axes[1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "temporal_validity.png", dpi=200)
    figure.savefig(output / "temporal_validity.pdf")
    plt.close(figure)


def _build_summary(
    *,
    predictions: pd.DataFrame,
    threshold: float,
    permutation: pd.DataFrame,
    placebo: pd.DataFrame,
    centered: pd.DataFrame,
    sensitivity: pd.DataFrame,
    audit_path: Path,
    config: CPUFollowupConfig,
) -> dict[str, Any]:
    permutation_lookup = permutation.set_index("metric")
    placebo_lookup = placebo.set_index("metric")
    centered_lookup = centered.set_index("metric")
    verdict = json.loads(
        (config.experiment1_dir / "interventions" / "behavioral_verdict.json").read_text()
    )
    return {
        "status": "complete",
        "analysis_type": "post_hoc_cpu_followup_on_frozen_experiment1_outputs",
        "selected_layer": int(predictions["layer"].iloc[0]),
        "threshold": threshold,
        "test_rows": len(predictions),
        "test_traces": int(predictions["trace_id"].nunique()),
        "temporal_randomization": {
            "observed_exact": float(permutation_lookup.loc["first_error_exact", "observed"]),
            "null_exact_mean": float(
                permutation_lookup.loc["first_error_exact", "null_mean"]
            ),
            "exact_p_value": float(permutation_lookup.loc["first_error_exact", "p_value"]),
            "observed_within_1": float(
                permutation_lookup.loc["error_within_1_accuracy", "observed"]
            ),
            "null_within_1_mean": float(
                permutation_lookup.loc["error_within_1_accuracy", "null_mean"]
            ),
            "within_1_p_value": float(
                permutation_lookup.loc["error_within_1_accuracy", "p_value"]
            ),
        },
        "matched_placebo": {
            "onset_jump": float(placebo_lookup.loc["real_jump", "estimate"]),
            "placebo_jump": float(placebo_lookup.loc["placebo_jump", "estimate"]),
            "difference": float(placebo_lookup.loc["paired_difference", "estimate"]),
            "difference_ci_low": float(placebo_lookup.loc["paired_difference", "ci_low"]),
            "difference_ci_high": float(placebo_lookup.loc["paired_difference", "ci_high"]),
        },
        "within_trace_discrimination": {
            "raw_auroc": float(centered_lookup.loc["pooled_raw_auroc", "estimate"]),
            "centered_auroc": float(
                centered_lookup.loc["pooled_first_step_centered_auroc", "estimate"]
            ),
            "centered_ci_low": float(
                centered_lookup.loc["pooled_first_step_centered_auroc", "ci_low"]
            ),
            "centered_ci_high": float(
                centered_lookup.loc["pooled_first_step_centered_auroc", "ci_high"]
            ),
            "mean_within_trace_auroc": float(
                centered_lookup.loc["mean_within_trace_auroc", "estimate"]
            ),
        },
        "causal_assay": {
            "baseline_auroc": float(verdict["auroc"]),
            "baseline_specificity": float(verdict["specificity"]),
            "smallest_mde_80pct": float(
                sensitivity["minimum_detectable_effect_80pct"].min()
            ),
            "behaviorally_valid": bool(verdict["auroc"] > 0.5 and verdict["specificity"] > 0),
        },
        "decision": {
            "temporal_alignment_above_shift_null": bool(
                permutation_lookup.loc["first_error_exact", "p_value"] < 0.05
                and permutation_lookup.loc["first_error_exact", "observed"]
                > permutation_lookup.loc["first_error_exact", "null_mean"]
            ),
            "onset_jump_exceeds_placebo": bool(
                placebo_lookup.loc["paired_difference", "ci_low"] > 0
            ),
            "centered_discrimination_above_chance": bool(
                centered_lookup.loc["pooled_first_step_centered_auroc", "ci_low"] > 0.5
            ),
            "causal_assay_valid": bool(verdict["auroc"] > 0.5 and verdict["specificity"] > 0),
        },
        "failure_audit_sample": str(audit_path),
        "output_dir": str(config.output_dir),
    }


def _results_markdown(
    summary: dict[str, Any],
    placebo: pd.DataFrame,
    centered: pd.DataFrame,
    sensitivity: pd.DataFrame,
) -> str:
    temporal = summary["temporal_randomization"]
    matched = summary["matched_placebo"]
    within = summary["within_trace_discrimination"]
    causal = summary["causal_assay"]
    return f"""# CPU-only follow-up results

This analysis reuses the frozen Experiment 1 test predictions and intervention tables. It does not
load the language model, fit a new hidden-state probe, or inspect activation tensors. The tests are
post-hoc and are reported as robustness analyses rather than preregistered confirmation.

## Temporal alignment

The original detector localized {temporal['observed_exact']:.1%} of first errors exactly. Circularly
shifting each score trajectory within its trace reduced the null mean to
{temporal['null_exact_mean']:.1%} (permutation p = {temporal['exact_p_value']:.4g}). Within-one-step
localization was {temporal['observed_within_1']:.1%}, compared with a shifted null mean of
{temporal['null_within_1_mean']:.1%} (p = {temporal['within_1_p_value']:.4g}). The frozen score is
temporally related to the annotation, although exact localization remains low in absolute terms.

## Onset change and trace-level offsets

The mean score jump at the annotated onset was {matched['onset_jump']:.3f}. Metadata-matched
transitions from correct traces changed by {matched['placebo_jump']:.3f}; the paired difference was
{matched['difference']:.3f} with a {summary.get('confidence_level', 95)}% interval of
[{matched['difference_ci_low']:.3f}, {matched['difference_ci_high']:.3f}].

Among erroneous traces whose first error occurred after step 0, pooled AUROC was
{within['raw_auroc']:.3f}. Subtracting each trace's first-step score gave AUROC
{within['centered_auroc']:.3f} [{within['centered_ci_low']:.3f},
{within['centered_ci_high']:.3f}]. Mean AUROC calculated separately within each eligible trace was
{within['mean_within_trace_auroc']:.3f}. Stable trace-level offsets do not explain all of the frozen
probe's discrimination.

## Causal assay

The unmodified verdict score had AUROC {causal['baseline_auroc']:.3f} and specificity
{causal['baseline_specificity']:.3f}. The smallest approximate effect detectable with 80% power
across the tested doses was {causal['smallest_mde_80pct']:.5f} verdict-margin units. Statistical
sensitivity does not repair a readout that fails to distinguish valid from invalid boundaries.

## Files

The output directory contains the permutation draws, error-aligned trajectory, matched-placebo
results, centered discrimination, subgroup outcomes, causal sensitivity table, a frozen qualitative
audit sample, and a two-panel temporal-validity figure.
"""
