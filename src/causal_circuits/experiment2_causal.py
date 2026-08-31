"""Counterbalanced verdict audit and causally validated interventions for Experiment 2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from causal_circuits.analysis import binary_metrics
from causal_circuits.data import (
    ProcessTrace,
    format_user_content,
    load_traces,
)
from causal_circuits.experiment2_config import Experiment2Config
from causal_circuits.models import HuggingFaceMathModel, TraceTooLongError

SINGLE_TOKEN_SYSTEM_PROMPT = (
    "You are a mathematical reasoning verifier. Follow the user's label mapping and output "
    "exactly one requested label."
)


def audit_verdict_readout(config: Experiment2Config) -> dict[str, object]:
    """Audit a single-token verdict under fixed and reversed A/B mappings."""
    metadata = load_experiment1_metadata(config.experiment1_dir)
    traces_before_context_filter = metadata["trace_id"].nunique()
    metadata = metadata[metadata["token_count"] <= config.model.max_length - 64].copy()
    traces_after_context_filter = metadata["trace_id"].nunique()
    traces = load_traces(config.data.path)
    trace_lookup = {trace.trace_id: trace for trace in traces}
    adapter = HuggingFaceMathModel(
        config.model.name,
        device=config.model.device,
        dtype=config.model.dtype,
        max_length=config.model.max_length,
    )
    token_ids, token_text = resolve_single_token_labels(adapter, config.verdict.labels)
    mappings = verdict_mappings(config.verdict.labels)
    rows = []
    for partition, per_class in (
        ("validation", config.verdict.examples_per_class_validation),
        ("test", config.verdict.examples_per_class_test),
    ):
        sample = balanced_boundary_sample(
            metadata,
            partition=partition,
            per_class=per_class,
            seed=config.seed + (0 if partition == "validation" else 100),
        )
        for mapping_name, valid_label, invalid_label in mappings:
            prompts = []
            sample_rows = list(sample.itertuples(index=False))
            for row in sample_rows:
                trace = trace_lookup[str(row.trace_id)]
                prompt, _ = render_single_token_verdict(
                    adapter,
                    trace,
                    int(row.step_index),
                    valid_label=valid_label,
                    invalid_label=invalid_label,
                )
                prompts.append(prompt)
            scored = score_next_token_margins(
                adapter,
                prompts,
                valid_token_id=token_ids[valid_label],
                invalid_token_id=token_ids[invalid_label],
                batch_size=config.verdict.batch_size,
            )
            for row, result in zip(sample_rows, scored, strict=True):
                label = int(row.invalid_so_far)
                rows.append(
                    {
                        "partition": partition,
                        "trace_id": str(row.trace_id),
                        "source": str(row.source),
                        "generator": str(row.generator),
                        "step_index": int(row.step_index),
                        "first_error": int(row.first_error),
                        "invalid_so_far": label,
                        "mapping": mapping_name,
                        "valid_label": valid_label,
                        "invalid_label": invalid_label,
                        "margin": result["margin"],
                        "probability_invalid": float(expit(result["margin"])),
                        "margin_prediction_correct": int(
                            (result["margin"] >= 0) == bool(label)
                        ),
                        "greedy_token_id": result["greedy_token_id"],
                        "greedy_matches_expected": int(
                            result["greedy_token_id"]
                            == token_ids[invalid_label if label else valid_label]
                        ),
                    }
                )
    individual = pd.DataFrame(rows)
    counterbalanced = counterbalance_verdict_rows(individual)
    summary = summarize_verdict_audit(individual, counterbalanced, config)
    output = config.output_dir / "verdict_audit"
    output.mkdir(parents=True, exist_ok=True)
    individual.to_csv(output / "individual.csv", index=False)
    counterbalanced.to_csv(output / "counterbalanced.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    (output / "label_tokens.json").write_text(
        json.dumps(
            {
                label: {"token_id": token_ids[label], "token_text": token_text[label]}
                for label in config.verdict.labels
            },
            indent=2,
        )
    )
    validation = summary[
        (summary["partition"] == "validation")
        & (summary["mapping"] == "counterbalanced")
    ].iloc[0]
    competent = bool(
        validation["auroc"] >= 0.60
        and validation["recall"] >= 0.55
        and validation["specificity"] >= 0.55
    )
    decision = {
        "validation_competent": competent,
        "criterion": (
            "counterbalanced validation AUROC >= 0.60, recall >= 0.55, "
            "and specificity >= 0.55"
        ),
        "validation_auroc": float(validation["auroc"]),
        "validation_recall": float(validation["recall"]),
        "validation_specificity": float(validation["specificity"]),
        "causal_stage_policy": (
            "run with explicit readout-failure flag" if not competent else "run as interpretable"
        ),
        "context_filter_excluded_traces": int(
            traces_before_context_filter - traces_after_context_filter
        ),
    }
    (output / "decision.json").write_text(json.dumps(decision, indent=2))
    return {
        **decision,
        "rows": len(individual),
        "output_dir": str(output),
    }


def run_causal_validation(config: Experiment2Config) -> dict[str, object]:
    """Compare the learned direction with a gradient-aligned positive causal control."""
    metadata = load_experiment1_metadata(config.experiment1_dir)
    traces_before_context_filter = metadata["trace_id"].nunique()
    metadata = metadata[metadata["token_count"] <= config.model.max_length - 64].copy()
    traces_after_context_filter = metadata["trace_id"].nunique()
    traces = load_traces(config.data.path)
    trace_lookup = {trace.trace_id: trace for trace in traces}
    sample = balanced_boundary_sample(
        metadata,
        partition="test",
        per_class=config.causal.examples_per_class,
        seed=config.seed + 200,
    )
    artifact = np.load(config.experiment1_dir / "probes" / "directions.npz")
    selected_layer = int(artifact["selected_intervention_layer"])
    directions = np.asarray(artifact["directions"], dtype=np.float32)
    projection_stds = np.asarray(artifact["projection_stds"], dtype=np.float32)
    selected_direction = directions[selected_layer]
    selected_scale = float(projection_stds[selected_layer])

    adapter = HuggingFaceMathModel(
        config.model.name,
        device=config.model.device,
        dtype=config.model.dtype,
        max_length=config.model.max_length,
    )
    token_ids, _ = resolve_single_token_labels(adapter, config.verdict.labels)
    mappings = verdict_mappings(config.verdict.labels)
    alignment_rows = []
    intervention_rows = []
    for row in sample.itertuples(index=False):
        trace = trace_lookup[str(row.trace_id)]
        for mapping_name, valid_label, invalid_label in mappings:
            prompt, boundary = render_single_token_verdict(
                adapter,
                trace,
                int(row.step_index),
                valid_label=valid_label,
                invalid_label=invalid_label,
            )
            baseline, gradients = gradients_at_boundary(
                adapter,
                prompt,
                boundary,
                valid_token_id=token_ids[valid_label],
                invalid_token_id=token_ids[invalid_label],
            )
            for layer, gradient in enumerate(gradients):
                direction = directions[layer]
                gradient_norm = float(np.linalg.norm(gradient))
                direction_norm = float(np.linalg.norm(direction))
                dot = float(np.dot(gradient, direction))
                denominator = max(gradient_norm * direction_norm, 1e-12)
                alignment_rows.append(
                    {
                        "trace_id": str(row.trace_id),
                        "source": str(row.source),
                        "step_index": int(row.step_index),
                        "invalid_so_far": int(row.invalid_so_far),
                        "mapping": mapping_name,
                        "layer": layer,
                        "gradient_norm": gradient_norm,
                        "probe_gradient_dot": dot,
                        "probe_gradient_cosine": dot / denominator,
                        "one_sigma_local_derivative": dot * float(projection_stds[layer]),
                    }
                )
            selected_gradient = gradients[selected_layer]
            selected_gradient /= max(float(np.linalg.norm(selected_gradient)), 1e-12)
            for direction_type, direction in (
                ("learned_probe", selected_direction),
                ("gradient_positive_control", selected_gradient),
            ):
                for alpha in config.causal.alphas:
                    if alpha == 0:
                        margin = baseline
                    else:
                        margin, _ = intervened_next_token_margin(
                            adapter,
                            prompt,
                            boundary,
                            layer=selected_layer,
                            direction=direction,
                            magnitude=float(alpha) * selected_scale,
                            valid_token_id=token_ids[valid_label],
                            invalid_token_id=token_ids[invalid_label],
                        )
                    intervention_rows.append(
                        {
                            "trace_id": str(row.trace_id),
                            "source": str(row.source),
                            "generator": str(row.generator),
                            "step_index": int(row.step_index),
                            "first_error": int(row.first_error),
                            "invalid_so_far": int(row.invalid_so_far),
                            "mapping": mapping_name,
                            "direction_type": direction_type,
                            "layer": selected_layer,
                            "alpha": float(alpha),
                            "margin": margin,
                            "baseline_margin": baseline,
                            "delta_margin": margin - baseline,
                        }
                    )
    alignment = pd.DataFrame(alignment_rows)
    interventions = pd.DataFrame(intervention_rows)
    counterbalanced = counterbalance_interventions(interventions)
    intervention_summary = summarize_causal_interventions(counterbalanced, config)
    alignment_summary = summarize_gradient_alignment(alignment, config)
    output = config.output_dir / "causal_validation"
    output.mkdir(parents=True, exist_ok=True)
    alignment.to_csv(output / "gradient_alignment_individual.csv", index=False)
    alignment_summary.to_csv(output / "gradient_alignment_summary.csv", index=False)
    interventions.to_csv(output / "interventions_individual.csv", index=False)
    counterbalanced.to_csv(output / "interventions_counterbalanced.csv", index=False)
    intervention_summary.to_csv(output / "intervention_summary.csv", index=False)

    readout_decision_path = config.output_dir / "verdict_audit" / "decision.json"
    readout_decision = (
        json.loads(readout_decision_path.read_text())
        if readout_decision_path.exists()
        else {"validation_competent": False, "criterion": "verdict audit not found"}
    )
    positive = intervention_summary[
        intervention_summary["direction_type"] == "gradient_positive_control"
    ]
    minimum_alpha = min(value for value in config.causal.alphas if value < 0)
    maximum_alpha = max(value for value in config.causal.alphas if value > 0)
    negative = positive[positive["alpha"] == minimum_alpha].iloc[0]
    positive_row = positive[positive["alpha"] == maximum_alpha].iloc[0]
    positive_control_passed = bool(
        negative["ci_high"] < 0 and positive_row["ci_low"] > 0
    )
    decision = {
        "selected_layer": selected_layer,
        "readout_validation_competent": bool(readout_decision["validation_competent"]),
        "positive_control_passed": positive_control_passed,
        "positive_control_criterion": (
            "the most negative gradient dose has CI entirely below zero and the most positive "
            "dose has CI entirely above zero"
        ),
        "learned_direction_interpretable": bool(
            readout_decision["validation_competent"] and positive_control_passed
        ),
        "n_traces": int(sample["trace_id"].nunique()),
        "context_filter_excluded_traces": int(
            traces_before_context_filter - traces_after_context_filter
        ),
    }
    (output / "decision.json").write_text(json.dumps(decision, indent=2))
    return {**decision, "output_dir": str(output)}


def verdict_mappings(labels: tuple[str, str]) -> list[tuple[str, str, str]]:
    first, second = labels
    return [
        ("fixed", first, second),
        ("reversed", second, first),
    ]


def resolve_single_token_labels(
    adapter: HuggingFaceMathModel,
    labels: tuple[str, str],
) -> tuple[dict[str, int], dict[str, str]]:
    """Resolve two configured labels to distinct, exactly one-token continuations."""
    token_ids = {}
    token_text = {}
    for label in labels:
        candidates = (label, f" {label}")
        resolved = None
        for candidate in candidates:
            ids = adapter.tokenizer(candidate, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                resolved = (int(ids[0]), candidate)
                break
        if resolved is None:
            raise ValueError(f"Verdict label {label!r} is not a single token")
        token_ids[label], token_text[label] = resolved
    if len(set(token_ids.values())) != 2:
        raise ValueError("Verdict labels resolved to the same token id")
    return token_ids, token_text


def render_single_token_verdict(
    adapter: HuggingFaceMathModel,
    trace: ProcessTrace,
    step_index: int,
    *,
    valid_label: str,
    invalid_label: str,
) -> tuple[str, int]:
    """Render a prefix verifier prompt with an explicit counterbalanced label mapping."""
    if not 0 <= step_index < len(trace.steps):
        raise IndexError(f"Step {step_index} is outside trace {trace.trace_id}")
    content, markers = format_user_content(
        trace.problem,
        trace.steps[: step_index + 1],
        include_question=False,
    )
    content += (
        "\n\nQuestion: Is the reasoning valid up to and including the last displayed step?\n"
        f"Answer {valid_label} if it is valid. Answer {invalid_label} if an error has occurred.\n"
        "Output exactly one label and nothing else."
    )
    messages = [
        {"role": "system", "content": SINGLE_TOKEN_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    rendered = adapter.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = adapter.tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if len(encoded["input_ids"]) > adapter.max_length:
        raise TraceTooLongError("Experiment 2 verdict prompt exceeds the context limit")
    boundary = adapter._marker_token_index(
        rendered,
        markers[-1],
        encoded["offset_mapping"],
    )
    return rendered, boundary


def score_next_token_margins(
    adapter: HuggingFaceMathModel,
    prompts: list[str],
    *,
    valid_token_id: int,
    invalid_token_id: int,
    batch_size: int,
) -> list[dict[str, float | int]]:
    results = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        encoded = adapter.tokenizer(
            chunk,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        if int(encoded["attention_mask"].sum(dim=1).max()) > adapter.max_length:
            raise TraceTooLongError("Experiment 2 verdict batch exceeds the context limit")
        inputs = {key: value.to(adapter.device) for key, value in encoded.items()}
        with adapter._torch.inference_mode():
            logits = adapter.model(**inputs, use_cache=False, return_dict=True).logits
        for batch_index in range(len(chunk)):
            positions = adapter._torch.nonzero(
                inputs["attention_mask"][batch_index],
                as_tuple=False,
            ).squeeze(1)
            final_position = int(positions[-1].item())
            next_logits = logits[batch_index, final_position]
            results.append(
                {
                    "margin": float(
                        (next_logits[invalid_token_id] - next_logits[valid_token_id]).item()
                    ),
                    "greedy_token_id": int(next_logits.argmax().item()),
                }
            )
    return results


def gradients_at_boundary(
    adapter: HuggingFaceMathModel,
    prompt: str,
    boundary: int,
    *,
    valid_token_id: int,
    invalid_token_id: int,
) -> tuple[float, list[np.ndarray]]:
    """Return the verdict margin and its gradient at every decoder-layer input."""
    encoded = adapter.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    inputs = {key: value.to(adapter.device) for key, value in encoded.items()}
    captured = {}
    handles = []
    for layer, module in enumerate(adapter.decoder_layers):
        def capture(_module, args, *, layer_index=layer):
            captured[layer_index] = args[0]

        handles.append(module.register_forward_pre_hook(capture))
    try:
        with adapter._torch.enable_grad():
            logits = adapter.model(**inputs, use_cache=False, return_dict=True).logits
            positions = adapter._torch.nonzero(
                inputs["attention_mask"][0],
                as_tuple=False,
            ).squeeze(1)
            next_logits = logits[0, int(positions[-1].item())]
            margin = next_logits[invalid_token_id] - next_logits[valid_token_id]
            tensors = [captured[index] for index in range(len(adapter.decoder_layers))]
            gradients = adapter._torch.autograd.grad(margin, tensors)
        boundary_gradients = [
            gradient[0, boundary].detach().float().cpu().numpy().copy()
            for gradient in gradients
        ]
        return float(margin.detach().float().cpu().item()), boundary_gradients
    finally:
        for handle in handles:
            handle.remove()


def intervened_next_token_margin(
    adapter: HuggingFaceMathModel,
    prompt: str,
    boundary: int,
    *,
    layer: int,
    direction: np.ndarray,
    magnitude: float,
    valid_token_id: int,
    invalid_token_id: int,
) -> tuple[float, int]:
    encoded = adapter.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    inputs = {key: value.to(adapter.device) for key, value in encoded.items()}
    vector = adapter._torch.as_tensor(direction, device=adapter.device)

    def inject(_module, args):
        hidden = args[0].clone()
        hidden[0, boundary] += magnitude * vector.to(hidden.dtype)
        return (hidden, *args[1:])

    handle = adapter.decoder_layers[layer].register_forward_pre_hook(inject)
    try:
        with adapter._torch.inference_mode():
            logits = adapter.model(**inputs, use_cache=False, return_dict=True).logits
        positions = adapter._torch.nonzero(
            inputs["attention_mask"][0],
            as_tuple=False,
        ).squeeze(1)
        next_logits = logits[0, int(positions[-1].item())]
        margin = float((next_logits[invalid_token_id] - next_logits[valid_token_id]).item())
        return margin, int(next_logits.argmax().item())
    finally:
        handle.remove()


def balanced_boundary_sample(
    metadata: pd.DataFrame,
    *,
    partition: str,
    per_class: int,
    seed: int,
) -> pd.DataFrame:
    frame = metadata[metadata["partition"] == partition]
    groups = []
    used_traces = set()
    for label in (1, 0):
        candidates = frame[
            frame["invalid_so_far"].eq(label) & ~frame["trace_id"].isin(used_traces)
        ]
        candidates = candidates.sample(frac=1, random_state=seed + label).drop_duplicates(
            "trace_id"
        )
        if len(candidates) < per_class:
            raise ValueError(
                f"Requested {per_class} class-{label} {partition} traces but found "
                f"{len(candidates)}"
            )
        chosen = candidates.sample(n=per_class, random_state=seed + 10 + label)
        groups.append(chosen)
        used_traces.update(chosen["trace_id"].astype(str))
    return pd.concat(groups, ignore_index=True).sample(frac=1, random_state=seed)


def counterbalance_verdict_rows(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "partition",
        "trace_id",
        "source",
        "generator",
        "step_index",
        "first_error",
        "invalid_so_far",
    ]
    grouped = frame.groupby(keys, as_index=False).agg(
        margin=("margin", "mean"),
        mappings_correct=("margin_prediction_correct", "sum"),
        greedy_mappings_correct=("greedy_matches_expected", "sum"),
    )
    grouped["probability_invalid"] = expit(grouped["margin"])
    grouped["margin_prediction_correct"] = (
        (grouped["margin"] >= 0) == grouped["invalid_so_far"].astype(bool)
    ).astype(int)
    grouped["both_mappings_correct"] = (grouped["mappings_correct"] == 2).astype(int)
    return grouped


def summarize_verdict_audit(
    individual: pd.DataFrame,
    counterbalanced: pd.DataFrame,
    config: Experiment2Config,
) -> pd.DataFrame:
    rows = []
    for (partition, mapping), group in individual.groupby(["partition", "mapping"]):
        rows.append(
            _verdict_summary_row(group, partition, mapping, config)
        )
    for partition, group in counterbalanced.groupby("partition"):
        rows.append(
            _verdict_summary_row(group, partition, "counterbalanced", config)
        )
    return pd.DataFrame(rows)


def _verdict_summary_row(
    frame: pd.DataFrame,
    partition: str,
    mapping: str,
    config: Experiment2Config,
) -> dict[str, object]:
    labels = frame["invalid_so_far"].to_numpy(dtype=int)
    scores = frame["probability_invalid"].to_numpy(dtype=float)
    metrics = binary_metrics(labels, scores, threshold=0.5)
    rng = np.random.default_rng(config.seed)
    bootstraps = []
    for _ in range(config.analysis.bootstrap_samples):
        indices = rng.choice(len(frame), len(frame), replace=True)
        if len(np.unique(labels[indices])) < 2:
            continue
        bootstraps.append(binary_metrics(labels[indices], scores[indices], threshold=0.5)["auroc"])
    tail = (1 - config.analysis.confidence_level) / 2
    return {
        "partition": partition,
        "mapping": mapping,
        "n": len(frame),
        **metrics,
        "auroc_ci_low": float(np.quantile(bootstraps, tail)),
        "auroc_ci_high": float(np.quantile(bootstraps, 1 - tail)),
        "margin_accuracy": float(frame["margin_prediction_correct"].mean()),
        "greedy_exact_accuracy": (
            float(frame["greedy_matches_expected"].mean())
            if "greedy_matches_expected" in frame
            else float("nan")
        ),
        "both_mappings_correct_rate": (
            float(frame["both_mappings_correct"].mean())
            if "both_mappings_correct" in frame
            else float("nan")
        ),
    }


def counterbalance_interventions(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "trace_id",
        "source",
        "generator",
        "step_index",
        "first_error",
        "invalid_so_far",
        "direction_type",
        "layer",
        "alpha",
    ]
    return frame.groupby(keys, as_index=False).agg(
        margin=("margin", "mean"),
        baseline_margin=("baseline_margin", "mean"),
        delta_margin=("delta_margin", "mean"),
    )


def summarize_causal_interventions(
    frame: pd.DataFrame,
    config: Experiment2Config,
) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(config.seed)
    tail = (1 - config.analysis.confidence_level) / 2
    for (direction_type, alpha), group in frame.groupby(["direction_type", "alpha"]):
        values = group["delta_margin"].to_numpy(dtype=float)
        means = np.asarray(
            [rng.choice(values, size=len(values), replace=True).mean() for _ in range(
                config.analysis.bootstrap_samples
            )]
        )
        rows.append(
            {
                "direction_type": direction_type,
                "alpha": float(alpha),
                "n_traces": group["trace_id"].nunique(),
                "mean_margin": group["margin"].mean(),
                "mean_delta": values.mean(),
                "standard_error": values.std(ddof=1) / np.sqrt(len(values)),
                "ci_low": np.quantile(means, tail),
                "ci_high": np.quantile(means, 1 - tail),
                "signed_consistency": (
                    float(((values * float(alpha)) > 0).mean())
                    if alpha != 0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_gradient_alignment(
    frame: pd.DataFrame,
    config: Experiment2Config,
) -> pd.DataFrame:
    counterbalanced = (
        frame.groupby(["trace_id", "layer"], as_index=False)
        .agg(
            probe_gradient_dot=("probe_gradient_dot", "mean"),
            probe_gradient_cosine=("probe_gradient_cosine", "mean"),
            one_sigma_local_derivative=("one_sigma_local_derivative", "mean"),
            gradient_norm=("gradient_norm", "mean"),
        )
    )
    rng = np.random.default_rng(config.seed)
    tail = (1 - config.analysis.confidence_level) / 2
    rows = []
    for layer, group in counterbalanced.groupby("layer"):
        values = group["one_sigma_local_derivative"].to_numpy(dtype=float)
        means = np.asarray(
            [rng.choice(values, len(values), replace=True).mean() for _ in range(
                config.analysis.bootstrap_samples
            )]
        )
        rows.append(
            {
                "layer": int(layer),
                "n_traces": group["trace_id"].nunique(),
                "mean_probe_gradient_dot": group["probe_gradient_dot"].mean(),
                "mean_probe_gradient_cosine": group["probe_gradient_cosine"].mean(),
                "mean_one_sigma_local_derivative": values.mean(),
                "derivative_ci_low": np.quantile(means, tail),
                "derivative_ci_high": np.quantile(means, 1 - tail),
                "mean_gradient_norm": group["gradient_norm"].mean(),
            }
        )
    return pd.DataFrame(rows)


def load_experiment1_metadata(experiment1_dir: str | Path) -> pd.DataFrame:
    shard_dir = Path(experiment1_dir) / "activation_shards"
    paths = sorted(shard_dir.glob("shard_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No Experiment 1 activation metadata found under {shard_dir}. "
            "Use the Drive-backed Experiment 1 run, not only the published compact artifacts."
        )
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
