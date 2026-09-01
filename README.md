# Tracing mathematical error detection in language models

This repository studies when a mathematics-specialized language model internally registers that a
written solution has become invalid, how precisely that signal localizes the first error, and
whether the decoded direction causally affects the model's own verdict.

The project uses `Qwen/Qwen2.5-Math-1.5B-Instruct` and the 3,400-example
[ProcessBench](https://huggingface.co/datasets/Qwen/ProcessBench) dataset. It is being developed for
the NeurIPS 2026 [Interpretability for Discovery](https://interpretability4discovery.github.io/)
workshop.

## Current status — August 31, 2026

**Experiment 1 and the CPU-only post-hoc follow-up are complete.**

The Experiment 1 analysis, compact result package, three paper figures, detailed result report, and
compiled manuscript are complete. The follow-up reuses frozen Experiment 1 predictions and causal
outputs. It tests temporal randomization, error-aligned trajectories, matched placebo onsets,
within-trace discrimination, subgroup outcomes, causal-assay sensitivity, and sampled failure cases.
It does not load the language model or extract new activations.

| Component | State |
| --- | --- |
| Experiment 1 data, extraction, probes, controls, and interventions | Complete |
| Experiment 1 report and frozen result tables | Complete |
| NeurIPS manuscript, PDF, and three figures | Complete |
| CPU-only follow-up implementation | Complete |
| CPU-only follow-up run and interpretation | Complete |

The compiled PDF is [paper/neurips_2026.pdf](paper/neurips_2026.pdf). The full Experiment 1 report
is [results/experiment1.md](results/experiment1.md), and the CPU follow-up report is
[results/experiment2_cpu.md](results/experiment2_cpu.md).

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

## CPU-only follow-up

The post-hoc follow-up asks whether the frozen Experiment 1 score trajectories contain temporal
information beyond trace-level offsets and ordinary within-trace drift. It also reports subgroup
outcomes and checks whether the failed causal assay could have detected effects of plausible size.
All resampling uses a fixed seed and trace-level bootstrap or permutation units.

The analysis reads the compact Experiment 1 predictions, threshold, intervention table, and the
processed ProcessBench JSONL. It writes to `artifacts/experiment2_cpu` and leaves the frozen inputs
unchanged. Because this analysis was designed after Experiment 1, its results are post-hoc rather
than confirmatory.

The frozen detector localized 28.9% of first errors exactly, compared with 16.4% after circularly
shifting its scores within each trace (permutation p = 0.0002). The score rose by 0.257 at the
annotated onset. Metadata-matched transitions from correct traces rose by 0.113, for a difference
of 0.144 with a two-way trace-clustered 95% interval of [0.096, 0.193]. Subtracting each erroneous
trace's first-step score left pooled AUROC unchanged at 0.881 [0.862, 0.900]. These tests support a
real but gradual temporal transition. They do not turn the probe into a precise detector: false
alarms increase sharply with trace length, and the original causal verdict remains behaviorally
invalid.

## Repository structure

```text
.
├── configs/
│   ├── experiment.yaml          # Experiment 1 scientific configuration
│   └── experiment2_cpu.yaml     # CPU follow-up resampling and output settings
├── data/processed/              # downloaded ProcessBench JSONL; ignored by Git
├── src/causal_circuits/
│   ├── data.py                  # loading, normalization, and grouped splits
│   ├── models.py                # Qwen loading, prompts, and hidden states
│   ├── analysis.py              # probes, metrics, controls, and bootstrap analysis
│   ├── circuits.py              # verdict scoring and Experiment 1 interventions
│   ├── pipeline.py / cli.py     # Experiment 1 orchestration and CLI
│   └── cpu_followup*.py         # post-hoc analysis and CLI
├── notebooks/
│   ├── experiment.ipynb         # complete Experiment 1 A100/Drive workflow
│   └── README.md                # notebook authentication, resume, and runtime notes
├── results/
│   ├── experiment1.md           # Experiment 1 empirical report
│   └── experiment2_cpu.md       # CPU follow-up methods and results
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

## Run the CPU follow-up

After the Experiment 1 compact artifacts and processed ProcessBench data are present, run:

```bash
math-error-cpu-followup --config configs/experiment2_cpu.yaml
```

## Expected CPU follow-up outputs

```text
artifacts/experiment2_cpu/
├── temporal_randomization_summary.csv
├── temporal_randomization_draws.csv
├── error_aligned_trajectory.csv
├── matched_placebo_onset.csv
├── within_trace_discrimination.csv
├── subgroup_outcomes.csv
├── causal_assay_sensitivity.csv
├── failure_audit_sample.jsonl
├── temporal_validity.pdf / .png
├── summary.json
└── results.md
```

## Primary references

- [ProcessBench paper and official code](https://github.com/QwenLM/ProcessBench)
- [ProcessBench dataset](https://huggingface.co/datasets/Qwen/ProcessBench)
- [Qwen2.5-Math-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-Math-1.5B-Instruct)
- [Interpretability for Discovery call for papers](https://interpretability4discovery.github.io/cfp.html)
