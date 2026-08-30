# Tracing mathematical error detection in language models

This repository tests a focused mechanistic question:

> When a mathematical solution first becomes invalid, does a small math language model
> internally register that change, and does the representation causally affect its verdict?

The target is a five-page submission to the NeurIPS 2026
[Interpretability for Discovery](https://interpretability4discovery.github.io/) workshop. The
deadline is **September 2, 2026 at 11:59:59 PM AoE**. The experiment is sized for one Google
Colab T4 and uses existing model weights and human annotations; it does not fine-tune the LLM
or generate a new dataset.

## Study in one minute

- **Data:** the official 3,400-example
  [Qwen/ProcessBench](https://huggingface.co/datasets/Qwen/ProcessBench), with human labels for
  the zero-indexed first erroneous step (`-1` means that every step is correct).
- **Model:** `Qwen/Qwen2.5-Math-1.5B-Instruct` in FP16.
- **Representation:** residual-stream hidden state at the end marker for every reasoning step.
- **Predictive test:** a regularized logistic probe at each hidden-state index.
- **Change-point test:** convert probe scores into the first threshold-crossing step and compare
  that step with the human annotation.
- **Generality test:** train the direction on one ProcessBench source and evaluate it on each
  other source.
- **Causal test:** add or subtract the held-out probe direction and measure the change in the
  model's `INCORRECT` versus `CORRECT` answer score.
- **Controls:** problem-grouped splits, validation-only selection, position, current-step TF-IDF,
  shuffled labels, and matched random orthogonal intervention directions.
- **Exploratory breadth:** compare L2, L1, and elastic-net linear probes and measure calibration,
  near-miss localization, detection lead/lag, threshold sensitivity, error-aligned trajectories,
  subgroup robustness, and paired causal-effect uncertainty.

The main claim must be conditional on the results. Decodability alone is not called a mechanism.
The causal claim is retained only if a signed dose response beats matched random directions on
held-out examples.

## Why this version is tighter than the original idea

The primary outcome is not merely layer-wise AUROC. It is whether a probe trained on
`invalid_so_far` produces the right **first threshold crossing**. That directly tests “when does
the model know?” and prevents a high score on later, obviously corrupted steps from hiding poor
localization at the actual error.

All steps from the same normalized problem remain in one partition. Layers, regularization, and
the crossing threshold are selected on validation data only. The test set is opened once for
reported metrics and interventions.

The PCA experiment is deliberately described as a **top-variance subspace accessibility** test,
not as intrinsic dimensionality. A binary linear probe always compresses its decision to one
scalar, so probe performance by itself cannot establish that the underlying computation is
one-dimensional.

## Colab quick start

The easiest route is to use the all-in-one notebook:

- [Open the complete experiment in Colab](https://colab.research.google.com/github/sagnikc395/tracing-mathematical-error-detection-in-language-models/blob/main/notebooks/colab_experiment.ipynb)

It has a `RUN_MODE` switch for the cheap smoke test and the full preregistered run. See
[`notebooks/README.md`](notebooks/README.md) for the workflow, persistent Drive layout, and output
locations. The notebook supports token-authenticated cloning and can commit the final result package
back into the repository's `artifacts/` subtree after the run.

Select `Runtime → Change runtime type → T4 GPU`, clone or upload this repository, then run:

```bash
pip install -e .
python -m causal_circuits --config configs/experiment.yaml validate-config
python -m causal_circuits --config configs/experiment.yaml download-data
python -m causal_circuits --config configs/experiment.yaml extract-activations
python -m causal_circuits --config configs/experiment.yaml fit-probes
python -m causal_circuits --config configs/experiment.yaml run-interventions
python -m causal_circuits --config configs/experiment.yaml plot
```

Or run the full sequence:

```bash
python -m causal_circuits --config configs/experiment.yaml run-all
```

The global `--config` argument goes before the subcommand.

### Run a cheap smoke test first

Before the full run, copy the configuration and change:

```yaml
data:
  max_examples_per_split: 25

extraction:
  output_dir: artifacts/smoke

probe:
  bootstrap_samples: 0

analysis:
  exploratory_bootstrap_samples: 0

intervention:
  examples_per_class: 8
  random_directions: 2
```

Use a different output directory when changing the model, dataset sample, dtype, or context
length. The harness fingerprints the data and model settings and refuses to mix incompatible
activation shards.

### Colab reliability notes

- Extraction is sharded every 100 traces and safely resumes by skipping complete shards.
- Each trace is forwarded once to collect every step boundary; causal masking guarantees that a
  boundary state cannot see later steps.
- Complete over-length traces are logged in each shard manifest and excluded. They are never
  truncated because truncation would invalidate the human error location.
- FP16 weights fit on a 16 GB T4 without 4-bit quantization. Quantization is avoided in the
  primary run because it changes the activations being interpreted.
- Save the repository or at least `data/processed` and `artifacts` to Google Drive before the
  Colab runtime expires. Causal interventions are the slowest stage.

## Artifact layout

```text
data/processed/processbench.jsonl
artifacts/qwen2.5-math-1.5b/
├── extraction_identity.json
├── activation_shards/
│   ├── shard_00000_00099.npy
│   ├── shard_00000_00099.csv
│   └── shard_00000_00099.json
├── probes/
│   ├── layer_metrics.csv
│   ├── test_predictions.csv
│   ├── controls.csv
│   ├── domain_transfer.csv
│   ├── pca_subspace.csv
│   ├── test_group_bootstrap.csv
│   ├── test_group_bootstrap_summary.csv
│   ├── probe_family_metrics.csv
│   ├── probe_family_predictions.csv
│   ├── probe_family_comparisons.csv
│   ├── diagnostic_target_metrics.csv
│   ├── calibration.csv
│   ├── threshold_sensitivity.csv
│   ├── score_trajectories.csv
│   ├── subgroup_metrics.csv
│   └── directions.npz
├── interventions/
│   ├── behavioral_verdict.json
│   ├── individual.csv
│   ├── effect_statistics.csv
│   └── summary.csv
└── figures/
    ├── layerwise_probe.pdf
    ├── pca_subspace.pdf
    ├── probe_calibration.pdf
    ├── error_aligned_trajectory.pdf
    └── causal_dose_response.pdf
```

## Labels and metrics

For trace $i$, step $k$, and human first-error label $e_i$:

\[
y_{ik}=\mathbb{1}[e_i \ge 0 \land k \ge e_i].
\]

The primary probe predicts $y_{ik}$. Step-level AUROC and average precision are secondary
metrics. The primary localization metric follows ProcessBench: predict the first step whose
score exceeds a validation-selected threshold, or `-1` if no step crosses it. Report erroneous
trace accuracy, fully-correct trace accuracy, and their harmonic mean.

The confirmatory probe remains class-balanced L2 logistic regression. L1 and elastic-net probes
are exploratory capacity-matched comparisons and cannot select the primary layer or causal
direction. Expanded diagnostics report step classification and calibration, exact and ±1/±2-step
localization, miss/false-alarm and early/late rates, score trajectories around the annotated
onset, and source/generator/difficulty subgroups. All thresholds and hyperparameters remain
validation-selected; test sweeps are descriptive and never feed selection.

For the intervention at layer $\ell$, the learned raw-coordinate probe direction is normalized
to $v_\ell$, and the boundary state is changed by

\[
h' = h + \alpha\,\sigma_\ell v_\ell,
\]

where $\sigma_\ell$ is the training-set standard deviation of projections onto $v_\ell$.
This makes $\alpha$ comparable across layers. The behavioral outcome is the length-normalized
log-probability difference between `INCORRECT` and `CORRECT`.

## Minimum result package

Freeze the submission around three panels:

1. held-out layer-wise AUROC and first-error F1, with position and lexical controls;
2. the four-by-four cross-domain transfer matrix;
3. the held-out causal dose response with random-direction controls.

A strong positive result is a localized, cross-domain signal with a monotonic causal effect. A
useful negative result is a decodable signal that fails localization, transfer, or causal
validation. The workshop explicitly welcomes careful failure analyses; do not overclaim a weak
intervention.

## Local development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
```

See [experiment.md](experiment.md) for the preregistered hypotheses, analysis decisions,
ablation priority, paper plan, and deadline schedule.

## Primary sources

- [ProcessBench paper and official code](https://github.com/QwenLM/ProcessBench)
- [ProcessBench dataset](https://huggingface.co/datasets/Qwen/ProcessBench)
- [Qwen2.5-Math-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct)
- [Interpretability for Discovery call for papers](https://interpretability4discovery.github.io/cfp.html)
