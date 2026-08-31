# Tracing mathematical error detection in language models

This repository studies when a mathematics-specialized language model internally registers that a
written solution has become invalid, how precisely that signal localizes the first error, and
whether the decoded direction causally affects the model's own verdict.

The project uses `Qwen/Qwen2.5-Math-1.5B-Instruct` and the 3,400-example
[ProcessBench](https://huggingface.co/datasets/Qwen/ProcessBench) dataset. It is being developed for
the NeurIPS 2026 [Interpretability for Discovery](https://interpretability4discovery.github.io/)
workshop.

## Current status — August 31, 2026

**Experiment 1 is complete. Experiment 2 is implemented and preregistered, with results pending.**

The Experiment 1 analysis, compact result package, three paper figures, detailed result report, and
compiled manuscript are complete. Experiment 2 adds stronger shortcut controls, natural step-end
representations, a counterbalanced single-token verdict, gradient alignment, and a positive causal
control. Its full and fast-preview configurations, resumable pipeline, tests, and Colab workflow
are ready, but no `artifacts/experiment2` result directory is present yet.

| Component | State |
| --- | --- |
| Experiment 1 data, extraction, probes, controls, and interventions | Complete |
| Experiment 1 report and frozen result tables | Complete |
| NeurIPS manuscript, PDF, and three figures | Complete |
| Experiment 2 design and decision rules | Frozen before results |
| Experiment 2 implementation and resumable Colab workflow | Complete |
| Experiment 2 A100 run and result interpretation | Pending |
| Tests | 32 passing |
| Ruff on maintained Python sources and tests | Passing |

The compiled PDF is [paper/neurips_2026.pdf](paper/neurips_2026.pdf). The full Experiment 1 report
is [results/experiment1.md](results/experiment1.md), and the preregistered Experiment 2 design is
[results/experiment2.md](results/experiment2.md).

## Experiment 1 results

At the end marker of each reasoning step, the experiment recorded all 29 residual-stream hidden
states. A class-balanced L2 logistic probe predicted

\[
y_{ik}=\mathbb{1}[e_i\geq 0 \text{ and } k\geq e_i],
\]

where `e_i` is the human-annotated first-error step and `y_ik` denotes whether trace `i` is invalid
so far at step `k`. Model selection used validation data only; final metrics use 669 held-out traces
and 4,985 step boundaries.

### Main findings

| Result | Held-out value |
| --- | ---: |
| Selected hidden-state index | 23 |
| Step AUROC | **0.866** [0.849, 0.884] |
| Average precision | **0.843** [0.813, 0.870] |
| Step F1 | **0.743** [0.716, 0.769] |
| AUROC within erroneous traces | **0.873** |
| Exact first-error step among erroneous traces | **0.289** |
| Within one step of the error | **0.593** |
| Within two steps of the error | **0.727** |
| Correct rejection of fully correct traces | **0.612** |
| Process F1 | **0.393** [0.351, 0.436] |
| Exact complete-trace outcome | **0.404** [0.368, 0.442] |

The signal is strongly linearly decodable and remains discriminative within traces known to
contain an error. Its temporal localization is useful but imperfect: it detects an error somewhere
on 88.7% of erroneous traces, while exact detection is 28.9% and predictions are late by about 0.59
steps on average among detected traces.

The hidden-state probe outperformed current-step TF-IDF (AUROC 0.733), position alone (0.730), and
a shuffled-label hidden-state control (0.518). Source-specific probes transferred across all four
ProcessBench sources: every off-diagonal AUROC exceeded 0.739, with mean off-diagonal AUROC 0.805.
Threshold-dependent localization transferred less reliably, with mean off-diagonal Process F1
0.261.

### Causal result

Adding or subtracting the learned direction at hidden-state index 23 did not produce the predicted
monotonic change in the teacher-forced `INCORRECT`-versus-`CORRECT` margin. Every tested nonzero-dose
95% interval included zero, and the learned direction did not beat matched random orthogonal
directions.

This is **not evidence of causal non-use**. The unmodified verdict assay was itself invalid: all 256
sampled boundaries were classified as `INCORRECT`, specificity was zero, and its AUROC was 0.283.
Experiment 1 therefore establishes decodability, partial localization, and cross-domain ranking
transfer, but it does not establish that the decoded direction participates in a functioning native
verdict mechanism.

## Experiment 2 follow-up

Experiment 2 is designed to resolve the main ambiguities left by Experiment 1 without changing its
confirmatory result. It has four stages:

1. **Stronger predictive robustness:** error-only and first-step-centered probes, onset-jump tests,
   combined surface/metadata controls, generator holdouts, subgroup analyses, calibration, and
   source-direction geometry.
2. **Natural step-end replication:** fresh activations from the last non-whitespace token of each
   written step, before the artificial end marker is visible.
3. **Counterbalanced verdict audit:** single-token `A`/`B` verdict labels under fixed and reversed
   semantic mappings, averaged to cancel stable token preference.
4. **Causal validation:** layer-wise alignment with local verdict gradients, learned-direction
   interventions, and gradient-aligned positive-control interventions at matched token, layer,
   norm, and dose.

The decision rules and complete output schema are frozen in
[results/experiment2.md](results/experiment2.md). Experiment 2 requires the complete Experiment 1
activation-shard directory; the compact result package alone is insufficient.

## Repository structure

```text
.
├── configs/
│   ├── experiment.yaml          # Experiment 1 scientific configuration
│   ├── experiment2.yaml         # full Experiment 2 publication configuration
│   └── experiment2_fast.yaml    # non-publication preview configuration
├── data/processed/              # downloaded ProcessBench JSONL; ignored by Git
├── src/causal_circuits/
│   ├── data.py                  # loading, normalization, and grouped splits
│   ├── models.py                # Qwen loading, prompts, and hidden states
│   ├── analysis.py              # probes, metrics, controls, and bootstrap analysis
│   ├── circuits.py              # verdict scoring and Experiment 1 interventions
│   ├── pipeline.py / cli.py     # Experiment 1 orchestration and CLI
│   └── experiment2_*            # Experiment 2 analysis, extraction, causality, runtime, and CLI
├── notebooks/
│   ├── experiment.ipynb         # complete Experiment 1 A100/Drive workflow
│   ├── experiment2.ipynb        # complete Experiment 2 A100/Drive workflow
│   └── README.md                # notebook authentication, resume, and runtime notes
├── results/
│   ├── experiment1.md           # complete empirical report
│   └── experiment2.md           # preregistered follow-up design and decision table
├── paper/
│   ├── neurips_2026.tex / .pdf  # current manuscript and compiled output
│   ├── figures/                 # PNG versions of the three Experiment 1 figures
│   └── starter/                 # untouched NeurIPS template and style files
├── artifacts/                   # generated results and caches; ignored by Git
├── tests/                       # 32 unit tests
├── experiment.md                # original hypotheses, protocol, and claim criteria
└── pyproject.toml               # package metadata, dependencies, and console scripts
```

Large activation shards, downloaded data, and runtime artifacts are intentionally excluded from
Git. This workspace currently contains a compact Experiment 1 snapshot under
`artifacts/experiment1/qwen2.5-math-1.5b`; the pipeline's canonical Experiment 1 output path is
`artifacts/qwen2.5-math-1.5b`, as set in `configs/experiment.yaml`.

## Installation and verification

Python 3.10 or newer is required. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests main.py
```

The equivalent editable pip installation is:

```bash
pip install -e '.[dev]'
pytest
ruff check src tests main.py
```

## Run Experiment 1

The recommended reproducible paper workflow is
[notebooks/experiment.ipynb](notebooks/experiment.ipynb) on an A100 runtime. It stores the dataset,
activation shards, logs, checkpoints, and results in Google Drive. Extraction is sharded and
resumable; incompatible dataset/model/configuration identities are rejected.

For a local run:

```bash
math-error-tracing --config configs/experiment.yaml validate-config
math-error-tracing --config configs/experiment.yaml download-data
math-error-tracing --config configs/experiment.yaml extract-activations
math-error-tracing --config configs/experiment.yaml fit-probes
math-error-tracing --config configs/experiment.yaml run-interventions
math-error-tracing --config configs/experiment.yaml plot
```

Or run the sequence with:

```bash
math-error-tracing --config configs/experiment.yaml run-all
```

The global `--config` option must precede the subcommand. The completed paper run used BF16 and an
activation-extraction batch size of 16 on an A100; the notebook applies those runtime settings and
records the resolved configuration with the artifacts.

## Run Experiment 2

The recommended workflow is [notebooks/experiment2.ipynb](notebooks/experiment2.ipynb), which points
to the complete Drive-backed Experiment 1 cache and writes all new outputs separately. Locally, run:

```bash
math-error-experiment2 --config configs/experiment2.yaml validate-config
math-error-experiment2 --config configs/experiment2.yaml analyze-robustness
math-error-experiment2 --config configs/experiment2.yaml extract-semantic
math-error-experiment2 --config configs/experiment2.yaml fit-semantic
math-error-experiment2 --config configs/experiment2.yaml audit-verdict
math-error-experiment2 --config configs/experiment2.yaml causal-validation
math-error-experiment2 --config configs/experiment2.yaml plot
```

Use `run-all` for the full sequence and `status` for a lightweight checkpoint report:

```bash
math-error-experiment2 --config configs/experiment2.yaml run-all
math-error-experiment2 --config configs/experiment2.yaml status
```

Semantic extraction resumes by shard, verdict scoring by completed inference batch, and causal
validation by completed trace/mapping job. The pipeline writes atomic checkpoints and refuses to
mix incompatible runs. `configs/experiment2_fast.yaml` is for iteration only; publication claims
must use `configs/experiment2.yaml`.

## Expected Experiment 2 outputs

```text
artifacts/experiment2/
├── experiment_config.yaml
├── stage_status.json
├── run_summary.json
├── logs/
├── marker_robustness/            # stronger controls and Stage A decision
├── semantic_activation_shards/   # resumable cache
├── semantic_boundary/            # natural-step probes and Stage B decision
├── verdict_audit/                # counterbalanced readout and Stage C decision
├── causal_validation/            # gradients, interventions, and Stage D decision
└── figures/
    ├── experiment2_summary.pdf
    └── experiment2_summary.png
```

## Primary references

- [ProcessBench paper and official code](https://github.com/QwenLM/ProcessBench)
- [ProcessBench dataset](https://huggingface.co/datasets/Qwen/ProcessBench)
- [Qwen2.5-Math-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct)
- [Interpretability for Discovery call for papers](https://interpretability4discovery.github.io/cfp.html)
