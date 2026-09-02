# Tracing mathematical error detection in language models

This repository studies whether a mathematics-specialized language model represents when a written
solution becomes invalid, how precisely that signal localizes the first error, and whether the
measured direction affects the model's own verdict.

The pipeline uses `Qwen/Qwen2.5-Math-1.5B-Instruct` and the 3,400-example
[ProcessBench](https://huggingface.co/datasets/Qwen/ProcessBench) dataset. The manuscript and frozen
results are intended for the NeurIPS 2026
[Interpretability for Discovery](https://interpretability4discovery.github.io/) workshop.

## Status

The empirical package and manuscript are complete. The central claim is deliberately narrow:
hidden states provide incremental held-out predictive information beyond matched nuisance features
on one inspected split. The results do not establish precise localization, calibration,
self-monitoring, generalization, or causal use.

See:

- [paper/neurips_2026.pdf](paper/neurips_2026.pdf) for the compiled manuscript;
- [results/experiment1.md](results/experiment1.md) for the primary result record;
- [results/experiment2_cpu.md](results/experiment2_cpu.md) for CPU diagnostics;
- [results/experiment3_extended.md](results/experiment3_extended.md) for transition and patching
  protocols; and
- [CRITIQUE.md](CRITIQUE.md) and [IMPROVEMENTS.md](IMPROVEMENTS.md) for the internal audit trail.

## Package layout

```text
src/tracing_math/
├── config.py          # typed project configuration and validation
├── data.py            # ProcessBench I/O, prompts, and grouped partitions
├── model.py           # Hugging Face model adapter and state extraction
├── probes.py          # predictive probes, controls, and localization metrics
├── interventions.py   # gated verdict assays and random-direction controls
├── pipeline.py        # core extraction, fitting, intervention, and figures
├── analysis.py        # CPU analysis over frozen prediction artifacts
├── conditional.py     # nuisance-versus-hidden conditional comparisons
├── transitions.py     # matched transitions, boundary controls, and summaries
├── operations.py      # resumable activation-backed analyses
└── cli.py             # the single Click command group
```

Generated data and artifacts are not source code. Keep downloaded datasets under `data/processed/`
and runtime outputs under `artifacts/`; both are excluded from Git.

## Installation and checks

Python 3.10 or newer is required. Use `uv`:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
```

Validate configuration before model or dataset work:

```bash
uv run math-error --config configs/project.yaml validate-config
```

## Unified CLI

All workflows use the same entry point and configuration. The global `--config` option precedes the
command.

```bash
# Core model workflow
uv run math-error --config configs/project.yaml download-data
uv run math-error --config configs/project.yaml extract-activations
uv run math-error --config configs/project.yaml fit-probes
uv run math-error --config configs/project.yaml run-interventions
uv run math-error --config configs/project.yaml render-figures

# CPU analysis over frozen predictions
uv run math-error --config configs/project.yaml analyze

# Activation-backed analyses
uv run math-error --config configs/project.yaml fit-conditional
uv run math-error --config configs/project.yaml fit-transition
uv run math-error --config configs/project.yaml transition-diagnostics
uv run math-error --config configs/project.yaml transition-sensitivity
uv run math-error --config configs/project.yaml extract-boundary-controls
uv run math-error --config configs/project.yaml analyze-boundary-controls
uv run math-error --config configs/project.yaml prepare-counterfactuals
uv run math-error --config configs/project.yaml run-counterfactuals
```

`run-all` executes the core workflow in order. It requires model downloads, substantial storage,
and suitable GPU memory:

```bash
uv run math-error --config configs/project.yaml run-all
```

Use `--workers 4` or `--workers -1` with `fit-probes` and `run-all` to override CPU parallelism.
Extraction shards and intervention groups are resumable and reject incompatible dataset/model
identities.

## Configuration

`configs/project.yaml` is the only maintained project configuration. It groups settings by capability
(`model`, `data`, `extraction`, `probe`, `intervention`, `analysis`, and `artifacts`) and keeps paths
explicit. Existing result directories may retain historical names because they are frozen on-disk
data contracts; those names are not used as Python packages or console commands.

The optional A100/Google Drive workflow is documented in [notebooks/README.md](notebooks/README.md)
and implemented in [notebooks/experiment.ipynb](notebooks/experiment.ipynb). Local CLI runs use the
same configuration and operations.

## Research safeguards

- Partitioning is deterministic and problem-grouped.
- Model and threshold selection use training/validation data only.
- CPU uncertainty is seeded and trace-level.
- Counterfactual patching requires explicitly verified annotations.
- The verdict intervention stops before dose-response interpretation when its baseline gate fails.
- No activation shards, downloaded datasets, caches, or secrets should be committed.
