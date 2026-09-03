# Tracing mathematical error detection in language models

When a language model reads a worked mathematical solution, something has to change internally at
the moment the argument stops being correct. A model that can be prompted to say "step four is
wrong" presumably had that information available somewhere before it said so. This repository is an
attempt to look for it directly, and to be careful about what finding it would and would not mean.

## Why this question

Most evaluation of mathematical reasoning treats the model as a black box: you ask for a verdict and
score the verdict. That tells you whether the model gets the answer right, but not whether it
noticed. The two come apart in ways that matter. A model might carry a perfectly good internal
signal that an argument has gone off the rails and still produce a confident wrong answer, because
nothing in its training connected the two. Or it might have no such signal at all and simply be
pattern-matching on surface features of broken-looking arithmetic.

The interesting version of the question is not "can a probe predict the label" — probes are good at
predicting labels — but "is there something in the residual stream that tracks *invalidity itself*,
as opposed to the many things that correlate with it?" Erroneous solutions tend to be longer. They
tend to use different vocabulary. Errors tend to appear later in a trace than earlier. Any of these
can carry a probe to a respectable AUROC without the model representing anything about correctness.

So the project is built around separating the signal from its confounds, and around three questions
that get progressively harder to answer:

1. **Is the signal there?** Can a linear read of hidden states predict, at each step boundary,
   whether the solution has already become invalid — and does it beat matched nuisance baselines
   built from position, surface text, and trace length?
2. **Is it localized?** Does the signal turn on *at* the first error, or does it drift up gradually
   across the trace in a way that would make "detection" mostly an artifact of thresholding?
3. **Does the model use it?** If you push activations along the decoded direction, does the model's
   own verdict move — and does it move more than it does along random directions of the same
   magnitude?

The third question is the one that would license the word "represents" in a strong sense. The first
two are prerequisites, and they are where careful work is actually needed.

## What this repository is

It is a small research codebase built around `Qwen/Qwen2.5-Math-1.5B-Instruct` and
[ProcessBench](https://huggingface.co/datasets/Qwen/ProcessBench), a dataset of 3,400 mathematical
solutions with human annotations marking the first erroneous step. The pipeline records
residual-stream activations at every reasoning-step boundary, fits probes against the
`invalid_so_far` label, runs the control conditions that make the probe result interpretable, and
attempts causal interventions on the model's verdict.

Concretely, the moving parts are:

- **A step-boundary extraction pass.** One causal forward pass per trace, capturing hidden states at
  each step boundary across all layers. Sharded and resumable, with dataset and model identity
  checks so partial runs cannot be silently mixed with incompatible ones.
- **Probes and their controls.** Class-balanced L2 logistic probes with problem-grouped splits, so
  that no problem appears in both training and evaluation. Layer, regularization, and threshold are
  all selected on validation data only. Alongside them: position-only, TF-IDF, and shuffled-label
  controls, plus a contextual baseline fit on the visible prefix text.
- **Localization machinery.** First-crossing analysis, exact and within-one-step accuracy, lateness,
  and matched-transition probes that compare the activation *difference* at a real error onset
  against the same difference at an arbitrary transition in a correct trace.
- **Interventions.** Additive steering along the probe direction, gated on a behavioral
  precondition — if the model's unmodified verdict readout does not separate correct from erroneous
  prefixes, the intervention stops rather than producing an uninterpretable dose-response curve.
  Random orthogonal directions serve as the control. Counterfactual activation patching is
  implemented and gated on human-verified corrected steps.

The manuscript in [paper/](paper/) is written for the NeurIPS 2026
[Interpretability for Discovery](https://interpretability4discovery.github.io/) workshop.
[ANALYSIS.md](ANALYSIS.md) is the running project report, and [results/](results/) holds the
experiment records.

## How the code is organized

```text
src/tracing_math/
├── config.py          # typed project configuration and validation
├── data.py            # ProcessBench I/O, prompts, and grouped partitions
├── model.py           # Hugging Face model adapter and state extraction
├── probes.py          # predictive probes, controls, and metrics
├── localization.py    # shared first-error localization calculations
├── conditional.py     # nuisance-versus-hidden conditional comparisons
├── contextual.py      # visible-prefix text baseline
├── transitions.py     # matched transitions and boundary controls
├── interventions.py   # gated verdict assays and random-direction controls
├── pipeline.py        # core extraction, fitting, intervention, and figures
├── analysis.py        # CPU analysis over frozen prediction artifacts
├── operations.py      # resumable activation-backed analyses
├── parallel.py        # deterministic parallel-execution helpers
└── cli.py             # the single Click command group
```

Generated data and artifacts are not source code. Downloaded datasets live under `data/processed/`
and runtime outputs under `artifacts/`; both are excluded from Git.

## Getting set up

Python 3.10 or newer, with `uv`:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
```

Validate the configuration before touching the model or the dataset:

```bash
uv run math-error --config configs/project.yaml validate-config
```

## Running things

Everything goes through one entry point and one configuration file. The global `--config` option
comes before the command.

```bash
# Core workflow
uv run math-error --config configs/project.yaml download-data
uv run math-error --config configs/project.yaml extract-activations
uv run math-error --config configs/project.yaml fit-probes
uv run math-error --config configs/project.yaml run-interventions
uv run math-error --config configs/project.yaml render-figures

# CPU analysis over frozen predictions
uv run math-error --config configs/project.yaml analyze

# Activation-backed follow-ups
uv run math-error --config configs/project.yaml fit-conditional
uv run math-error --config configs/project.yaml fit-contextual-baseline
uv run math-error --config configs/project.yaml fit-transition
uv run math-error --config configs/project.yaml transition-diagnostics
uv run math-error --config configs/project.yaml transition-sensitivity
uv run math-error --config configs/project.yaml extract-boundary-controls
uv run math-error --config configs/project.yaml analyze-boundary-controls
uv run math-error --config configs/project.yaml prepare-counterfactuals
uv run math-error --config configs/project.yaml run-counterfactuals
```

`run-all` runs the core workflow in order; it needs model downloads, a fair amount of storage, and
enough GPU memory to hold the model plus activations:

```bash
uv run math-error --config configs/project.yaml run-all
```

Pass `--workers 4` or `--workers -1` to `fit-probes` and `run-all` to override CPU parallelism.
Extraction shards and intervention groups resume where they left off and refuse to continue against
a different dataset or model.

## Configuration

`configs/project.yaml` is the only maintained configuration. It groups settings by capability
(`model`, `data`, `extraction`, `probe`, `intervention`, `analysis`, `artifacts`) and keeps every
path explicit. Some result directories keep historical names because they are frozen on-disk data
contracts; those names are not Python packages or console commands.

There is also an A100/Google Drive workflow for people without local GPUs, documented in
[notebooks/README.md](notebooks/README.md) and implemented in
[notebooks/experiment.ipynb](notebooks/experiment.ipynb). It runs the same operations against the
same configuration.

## Safeguards

The easiest way to get an exciting result here is to cheat by accident, so a few things are enforced
rather than left to discipline:

- Partitions are deterministic and grouped by problem, so no problem leaks across splits.
- Model and threshold selection touch training and validation data only.
- CPU uncertainty estimates are seeded and computed at the trace level, not the step level.
- Counterfactual patching refuses to run on annotations that have not been explicitly verified.
- The verdict intervention halts before dose-response interpretation if its baseline gate fails.
- Activation shards, downloaded datasets, caches, and secrets stay out of the repository.
