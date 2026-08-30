# Colab notebooks

[`colab_experiment.ipynb`](colab_experiment.ipynb) is the single, reproducible front end to the
experiment code in `src/`. It does not reimplement the analysis, so command-line and notebook runs
produce the same artifacts.

## Workflow

1. Open the notebook on a T4 runtime and leave `RUN_MODE = "smoke"`. This runs the unit tests and
   uses 25 traces per ProcessBench source. The causal stage is off by default in smoke mode but can
   be enabled with `RUN_INTERVENTIONS = True`.
2. After the smoke run succeeds, change the same notebook to `RUN_MODE = "full"` and restart from
   the configuration cell. Full mode mounts Google Drive, restores all preregistered sample sizes,
   enables the causal stage, and summarizes the paper-facing metrics and figures.
3. After the completion checklist passes, run the final publishing cell to copy the result package
   into the repository's `artifacts/` subtree, commit it, and push it to GitHub.

Use **Runtime → Change runtime type → T4 GPU** before running the notebook. A fresh runtime must
rerun the setup cell, but completed activation shards in Drive are detected and skipped.

## Outputs

Smoke mode writes to `artifacts/smoke`. By default, full mode writes to
`MyDrive/math-error-tracing/artifacts/qwen2.5-math-1.5b` and keeps its ProcessBench JSONL beside
that artifact directory. Change `DRIVE_PROJECT_DIR` in the notebook if desired.

Do not point smoke and full runs at the same artifact directory. Extraction identity checks reject
incompatible model/data settings, but separate directories also make accidental result mixing
obvious.

## GitHub token

Create a fine-grained personal access token with **Contents: read and write** access to this
repository and add it to Colab's Secrets panel as `GITHUB_TOKEN`. The notebook uses it both for the
initial clone and the final push through `GIT_ASKPASS`; it never embeds the token in a URL, Git
remote, source cell, commit, or printed command.

The final publishing cell commits result tables, learned directions, intervention outputs, figures,
the extraction identity, and the generated experiment config. Activation shards are excluded by
default because they are large resumable caches. Files above 95 MiB are rejected; publishing those
caches requires a separate Git LFS-aware workflow.
