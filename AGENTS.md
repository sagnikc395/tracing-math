# Repository Guidelines

## Project structure and module organization

The installable Python package lives in `src/tracing_math/`. Experiment 1 code belongs under `experiment1/`: use `data.py` for ProcessBench I/O and prompts, `model.py` for the Hugging Face adapter, `probes.py` for predictive analysis, `interventions.py` for causal tests, and `pipeline.py` for orchestration and artifact I/O. The post-hoc CPU analysis belongs under `followup/`. Declare commands and options with Click in each area's `cli.py`; keep scientific work in the pipeline and analysis modules. Tests mirror these packages under `tests/experiment1/` and `tests/followup/`.

Scientific settings belong in `configs/`. The CPU follow-up uses `experiment2_cpu.yaml` and reads frozen Experiment 1 outputs. Notebooks contain the A100/Google Drive workflow, `results/` holds written findings, and `paper/` contains the LaTeX manuscript and figures. Generated datasets, activation shards, and run outputs belong in `data/processed/` or `artifacts/`, not Git.

## Build, test, and development commands

Use Python 3.10 or newer and `uv`:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
```

The first command creates the development environment; the others run the full test suite and lint maintained Python code. Validate configuration before costly model runs:

```bash
uv run math-error-tracing --config configs/experiment1.yaml validate-config
```

The Experiment 1 CLI also provides `run-all`, but a full run requires model downloads, substantial storage, and suitable GPU memory. The CPU follow-up runs with `uv run math-error-cpu-followup --config configs/experiment2.yaml`; it does not load the model or activation shards.

## Coding style and naming conventions

Ruff enforces Python 3.10 syntax, import sorting, and a 100-character line limit. Use four-space indentation, type hints for public interfaces, `snake_case` for functions and modules, and `PascalCase` for classes. Keep configuration and artifact schemas explicit; do not silently change frozen experiment defaults or decision rules.

For paper and results prose, read and follow `.agents/skills/SKILL.md`; prefer concrete claims, plain verbs, and restrained language.

## Testing guidelines

Tests use pytest and should remain deterministic, CPU-friendly, and independent of network or GPU access. Name files `test_<area>.py` and test functions `test_<behavior>`. Add regression tests for configuration validation, checkpoint/resume behavior, metric calculations, and artifact compatibility. Run pytest and Ruff before opening a pull request.

## Commits and pull requests

Recent history favors short, imperative subjects, sometimes with prefixes such as `feat:` or `docs:`. Use a focused subject that names the change, for example `feat: add semantic shard validation`, and avoid mixing manuscript, pipeline, and generated-output changes unnecessarily.

Pull requests should explain the scientific or engineering intent, list commands run, and note any configuration or output-schema changes. Link the relevant issue. Include updated plots or result tables when reported findings change, but do not commit secrets, downloaded datasets, caches, or large activation artifacts. Copy `.env.sample` for local credentials and keep `.env` private.
