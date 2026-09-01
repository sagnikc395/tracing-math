# Improvement Suggestions

This document tracks concrete improvements for the repository. All 33 tests pass
and ruff is clean, so these are additive enhancements rather than bug fixes.

## High-priority

### 1. Missing test coverage for shared utilities

`localization.py` has zero dedicated tests. It is imported by both `probes.py` and
`analysis.py`, making it a critical shared dependency. Edge cases like empty arrays,
single-element arrays, all-erroneous traces, and all-correct traces are untested.

### 2. Untested follow-up analysis functions

`event_study`, `subgroup_outcomes`, `causal_sensitivity_analysis`, and
`write_failure_audit_sample` in `analysis.py` have no unit tests. Only 5 of the
10+ public functions in that 1171-line module are tested.

### 3. Missing type hints on several interfaces

`_balanced_sample` and `_intervention_row` in `interventions.py` lack return type
annotations. The project guideline requires type hints for public interfaces; even
internal helpers should be annotated since they are used across modules.

### 4. Probes module is too large (1 459 lines)

`probes.py` contains fitting, evaluation, controls, calibration, trajectory
analysis, transfer tests, subgroup analysis, and plotting helpers. Splitting it
into `fitting.py`, `evaluation.py`, and `controls.py` would improve navigation
and make test isolation easier.

## Medium-priority

### 5. No logging framework

The project uses `click.echo` and `tqdm` but no `logging` module. Adding
structured logging — especially for long-running stages like extraction and
intervention — would help diagnose failures and enable configurable verbosity
via `--verbose` / `--quiet` flags.

### 6. No CI/CD configuration

There is no `.github/workflows/` directory or equivalent. A minimal workflow that
runs `uv sync --extra dev`, `pytest`, and `ruff check` on every push and pull
request would catch regressions early.

### 7. No pre-commit hooks

Adding `.pre-commit-config.yaml` with ruff hooks would prevent lint drift before
commits land. This pairs naturally with the CI workflow above.

### 8. No `--dry-run` for expensive stages

`extract-activations` and `run-interventions` are the most costly commands. A
`--dry-run` flag that validates inputs, reports how many traces or shards would be
processed, and exits would let researchers sanity-check before committing GPU time.

### 9. Config validation messages lack actual values

Many `validate()` methods raise `ValueError("...must be positive")` without
printing the value that failed. Including `f"; got {value}"` in each message
would shorten debugging sessions.

### 10. Ruff config is minimal

Currently only `E`, `F`, `I`, `UP` are selected. Adding additional rule sets
such as `B` (bugbear), `SIM` (simplify), and `S` (security) would catch more
issues before they reach review.

## Lower-priority

### 11. No `py.typed` marker

Adding `src/tracing_math/py.typed` would signal to downstream type checkers
(mypy, pyright) that this package ships inline types.

### 12. Coverage reporting

Running `pytest --cov=tracing_math --cov-report=term-missing` would quantify
actual line coverage rather than relying on file-count heuristics. Integrating
coverage into CI with a minimum threshold would prevent regressions.

### 13. No retry logic for HuggingFace downloads

`load_huggingface_traces` will fail on transient network errors. A simple retry
wrapper would make the `download-data` stage more robust on flaky connections.

### 14. Experiment 1 CLI lacks `--verbose`

The `fit-probes` command silently prints a short JSON summary. A `--verbose` flag
could stream per-layer AUROC progress as layers are evaluated, which is helpful
during long runs.

### 15. Follow-up CLI has no subcommands

The Experiment 1 CLI provides `validate-config`, `fit-probes`,
`run-interventions`, etc. The follow-up `math-error-cpu-followup` runs everything
in one shot with no way to run only the temporal test or only the subgroup
analysis. Splitting into subcommands would mirror the Experiment 1 pattern and
enable selective reruns.

### 16. Prompt constants are duplicated

`SYSTEM_PROMPT` and `VERDICT_QUESTION` are defined in both `data.py` and
`model.py` with slightly different text. Consolidating these into a single
location would prevent silent drift.

### 17. No `.editorconfig`

Adding a `.editorconfig` with `indent_style = space`, `indent_size = 4`, and
`max_line_length = 100` would ensure consistent formatting across editors and
IDEs regardless of individual settings.
