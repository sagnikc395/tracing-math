# Repository Guidelines

## Project structure and module organization

The installable Python package lives in `src/tracing_math/`. Keep it capability-oriented: `data.py` handles ProcessBench I/O and prompts, `model.py` owns the Hugging Face adapter, `probes.py` handles predictive analysis, `interventions.py` handles causal assays, `pipeline.py` owns core orchestration, `analysis.py` owns CPU diagnostics, and `operations.py` owns optional activation-backed analyses. `cli.py` is the only command surface; keep scientific work out of CLI declarations. Tests live directly under `tests/` and use the same neutral capability names.

Scientific and runtime settings belong in `configs/project.yaml`. The unified CLI reads frozen predictions or activation artifacts according to the selected command. Notebooks contain the A100/Google Drive workflow, `results/` holds written findings, and `paper/` contains the LaTeX manuscript and figures. Generated datasets, activation shards, and run outputs belong in `data/processed/` or `artifacts/`, not Git.

## Build, test, and development commands

Use Python 3.10 or newer and `uv`:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
```

The first command creates the development environment; the others run the full test suite and lint maintained Python code. Validate configuration before costly model runs:

```bash
uv run math-error --config configs/project.yaml validate-config
```

The unified CLI also provides `run-all`, but a full run requires model downloads, substantial storage, and suitable GPU memory. The `analyze` command reads frozen prediction artifacts and does not load the model or activation shards.

## Coding style and naming conventions

Ruff enforces Python 3.10 syntax, import sorting, and a 100-character line limit. Use four-space indentation, type hints for public interfaces, `snake_case` for functions and modules, and `PascalCase` for classes. Keep configuration and artifact schemas explicit; do not silently change frozen experiment defaults or decision rules.

For paper and results prose, read and follow `.agents/skills/SKILL.md`; prefer concrete claims, plain verbs, and restrained language.

## Testing guidelines

Tests use pytest and should remain deterministic, CPU-friendly, and independent of network or GPU access. Name files `test_<area>.py` and test functions `test_<behavior>`. Add regression tests for configuration validation, checkpoint/resume behavior, metric calculations, and artifact compatibility. Run pytest and Ruff before opening a pull request.

## Commits and pull requests

Recent history favors short, imperative subjects, sometimes with prefixes such as `feat:` or `docs:`. Use a focused subject that names the change, for example `feat: add semantic shard validation`, and avoid mixing manuscript, pipeline, and generated-output changes unnecessarily.

Pull requests should explain the scientific or engineering intent, list commands run, and note any configuration or output-schema changes. Link the relevant issue. Include updated plots or result tables when reported findings change, but do not commit secrets, downloaded datasets, caches, or large activation artifacts. Copy `.env.sample` for local credentials and keep `.env` private.
