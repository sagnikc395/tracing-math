# Tracing mathematical error detection in language models

This project asks a focused mechanistic question:

> When a mathematical solution first becomes invalid, does a small math language model
> internally register that change, and does the representation causally affect its verdict?

The intended output is a five-page submission to the NeurIPS 2026
[Interpretability for Discovery](https://interpretability4discovery.github.io/) workshop. The
submission deadline is **September 2, 2026 at 11:59:59 PM AoE**.

## Current status — August 30, 2026

**Stage: implementation complete; experimental results pending.**

The research question and analysis plan are preregistered, the ProcessBench data has been
downloaded locally, and the command-line and Colab pipelines are implemented. The unit test suite
currently passes (32 tests), and the maintained Python sources pass Ruff. However, no activation shards,
probe results, causal-intervention results, or paper-ready figures are present in the repository
yet. The paper directory still contains the unedited NeurIPS template.

| Milestone | Status |
| --- | --- |
| Freeze hypotheses, splits, metrics, controls, and claim criteria | Complete |
| Implement data, activation, probe, intervention, and plotting stages | Complete |
| Provide a resumable Colab workflow | Complete |
| Download all 3,400 ProcessBench traces locally | Complete |
| Run the full end-to-end experiment on an A100 | Next |
| Freeze the completed experiment and controls | Pending |
| Interpret results and select the supported claim | Pending |
| Replace the paper template with the five-page manuscript | Pending |

Accordingly, this README describes the **planned and implemented experiment**, not empirical
findings. No mechanistic claim should be inferred until the held-out results and controls have
been generated.

## Planned study

- **Data:** the official 3,400-example
  [Qwen/ProcessBench](https://huggingface.co/datasets/Qwen/ProcessBench), with a zero-indexed human
  label for the first erroneous step (`-1` means that every step is correct).
- **Model:** `Qwen/Qwen2.5-Math-1.5B-Instruct` in BF16 for the A100 paper run.
- **Representation:** residual-stream hidden states at the end marker of each reasoning step.
- **Predictive test:** a class-balanced L2 logistic probe at every hidden-state index.
- **Localization test:** the first validation-selected threshold crossing, compared with the
  human first-error annotation.
- **Generality test:** train the direction on one ProcessBench source and evaluate it on the other
  sources.
- **Causal test:** add or subtract the held-out probe direction and measure the change in the
  model's `INCORRECT` versus `CORRECT` verdict score.
- **Controls:** problem-grouped splits, validation-only selection, position, current-step TF-IDF,
  shuffled labels, and matched random orthogonal intervention directions.

The primary outcome is not simply layer-wise AUROC. It is whether a probe trained on
`invalid_so_far` crosses its threshold at the annotated error step. This tests *when* the model
detects an error and prevents strong performance on later, obviously corrupted steps from hiding
poor localization.

All steps from the same normalized problem remain in one partition. Layers, regularization, and
the crossing threshold are selected on validation data only. The test set is reserved for final
metrics and interventions.

## Immediate next steps

1. Open the full Colab workflow, confirm the configuration, and inspect the downloaded data and
   partition balance.
2. Start the resumable activation extraction and persist it to Google Drive; inspect exclusions
   and activation shapes after the first shard completes.
3. Fit probes and controls, freeze the validation-selected layer, and run causal
   interventions.
4. Generate the result package and write only the conclusion supported by the decision criteria
   in [experiment.md](experiment.md).

The detailed day-by-day submission schedule is in
[Section 16 of experiment.md](experiment.md#16-deadline-schedule).

## Run the experiment

### Experiment 2 follow-up

The robustness and causally validated follow-up is specified in
[results/experiment2.md](results/experiment2.md) and runs from
[notebooks/experiment2.ipynb](notebooks/experiment2.ipynb). It requires the complete Experiment 1
activation directory on Drive. Its expensive paths are optimized for an A100 without changing the
analysis: semantic prompts are length-bucketed, only requested boundary/final states are moved or
projected, and the eight non-zero causal variants for a trace/mapping are evaluated in one batch.
See [notebooks/README.md](notebooks/README.md#experiment-2-runtime-behavior) for checkpoint and memory
guidance.

### Recommended: Colab

Use the all-in-one notebook:

- [Open the experiment in Colab](https://colab.research.google.com/github/sagnikc395/tracing-mathematical-error-detection-in-language-models/blob/main/notebooks/experiment.ipynb)

Select **Runtime → Change runtime type → A100 GPU** and run the notebook from the top. It uses the
fixed paper configuration with BF16 A100 batches and Drive-backed checkpoints/logs, runs
Experiments A--C, creates the essential paper figures, and publishes the checked result package. See
[notebooks/README.md](notebooks/README.md) for authentication, workflow, and output details.

### Command line

Install the package, validate the configuration, and run each stage in order:

```bash
pip install -e .
python -m causal_circuits --config configs/experiment.yaml validate-config
python -m causal_circuits --config configs/experiment.yaml download-data
python -m causal_circuits --config configs/experiment.yaml extract-activations
python -m causal_circuits --config configs/experiment.yaml fit-probes
python -m causal_circuits --config configs/experiment.yaml run-interventions
python -m causal_circuits --config configs/experiment.yaml plot
```

Or run the complete sequence:

```bash
python -m causal_circuits --config configs/experiment.yaml run-all
```

The global `--config` argument must appear before the subcommand.

The harness fingerprints the data and model settings and refuses to mix incompatible activation
shards. Do not reuse an output directory after changing the model, dataset sample, dtype, or
context length.

### Runtime behavior

- Extraction is sharded every 100 traces and resumes by skipping complete shards. Configurable
  batches collect every trace's exact step boundaries without changing causal masking.
- Intervention examples and both candidate answers are batched. Group-level checkpoints resume
  learned-alpha and random-direction work after a disconnect.
- Over-length traces are logged and excluded rather than truncated, because truncation could
  invalidate the annotated error location.
- The paper notebook uses BF16 on an A100; the default batch size remains one for compatibility.
  Quantization is excluded from the primary run because it changes the activations under study.
- The paper notebook persists activation shards, intervention checkpoints, stage status, logs, and
  final artifacts to Google Drive.

## Analysis and claim criteria

For trace $i$, step $k$, and human first-error label $e_i$, the primary target is

$$
y_{ik}=\mathbb{1}[e_i \ge 0 \land k \ge e_i].
$$

Step-level AUROC and average precision are secondary metrics. The primary localization metric
predicts the first step whose score exceeds a validation-selected threshold, or `-1` if no step
crosses it. The report includes erroneous-trace accuracy, fully-correct-trace accuracy, and their
harmonic mean.

At the selected intervention layer $\ell$, the raw-coordinate probe direction is normalized to
$v_\ell$ and the boundary state is changed by

$$
h' = h + \alpha\,\sigma_\ell v_\ell,
$$

where $\sigma_\ell$ is the training-set standard deviation of projections onto $v_\ell$. The
behavioral outcome is the length-normalized log-probability difference between `INCORRECT` and
`CORRECT`.

The minimum result package has three parts:

1. held-out layer-wise AUROC and first-error F1 with position and lexical controls;
2. the four-by-four cross-domain transfer matrix;
3. the held-out causal dose response with random-direction controls.

Decodability alone is not evidence of a mechanism. A causal claim is retained only if the signed
dose response beats matched random directions on held-out examples. A negative or mixed result is
reported as such; the preregistered interpretation table is in
[experiment.md](experiment.md#12-decision-table-for-the-paper-narrative).

## Expected outputs

The full run will create the following untracked result tree. These files are **expected outputs**;
they have not yet been generated in the current repository state.

```text
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
│   ├── test_group_bootstrap.csv
│   ├── test_group_bootstrap_summary.csv
│   └── directions.npz
├── interventions/
│   ├── behavioral_verdict.json
│   ├── individual.csv
│   ├── effect_statistics.csv
│   └── summary.csv
└── figures/
    ├── method_and_trajectory.pdf
    ├── predictive_results.pdf
    └── transfer_and_causal.pdf
```

Activation shards are resumable caches and are excluded from Git. The notebook publishes only the
fixed configuration, essential tables and directions, intervention outputs, and the three figures.

## Repository guide

| Path | Purpose | Current state |
| --- | --- | --- |
| `configs/experiment.yaml` | Frozen paper-only configuration | Ready |
| `src/causal_circuits/` | Data, model, probe, intervention, and plotting code | Implemented |
| `tests/` | Unit tests for configuration, data, analysis, circuits, and figures | 32 passing |
| `notebooks/experiment.ipynb` | Fixed Experiments A--C Colab workflow | Ready to run |
| `experiment.md` | Preregistered hypotheses, decisions, claim table, and schedule | Complete |
| `paper/` | NeurIPS style, checklist, and manuscript source | Template only |
| `artifacts/` | Generated results and figures | Not generated |

## Local development

```bash
uv sync --extra dev
uv run ruff check .
uv run pytest
```

## Primary sources

- [ProcessBench paper and official code](https://github.com/QwenLM/ProcessBench)
- [ProcessBench dataset](https://huggingface.co/datasets/Qwen/ProcessBench)
- [Qwen2.5-Math-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct)
- [Interpretability for Discovery call for papers](https://interpretability4discovery.github.io/cfp.html)
