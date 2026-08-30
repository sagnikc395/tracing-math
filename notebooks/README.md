# Colab notebooks

[`experiment.ipynb`](experiment.ipynb) is the single, reproducible front end to the
experiment code in `src/`. It does not reimplement the analysis, so command-line and notebook runs
produce the same artifacts.

## Workflow

1. Open the notebook on a T4 runtime and run the setup cells. The notebook uses the full
   preregistered configuration from `configs/experiment.yaml`, mounts Google Drive, and enables all
   experiment stages by default.
2. Review the configuration cell before starting extraction. Storage paths and stages are direct
   notebook variables. `CONFIG_OVERRIDES` accepts nested YAML overrides for an intentional
   alternative run; keep it empty for the preregistered experiment and use a distinct
   `ARTIFACT_NAME` when changing identity-sensitive settings.
3. Run the remaining cells in order. Probe fitting produces the confirmatory results plus
   exploratory L1/elastic-net comparisons, calibration, localization tolerance, trajectory,
   subgroup, and paired causal-effect tables from the same cached activations.
4. After the completion checklist passes, run the final publishing cell to copy the result package
   into the repository's `artifacts/` subtree, commit it, and push it to GitHub.

Use **Runtime → Change runtime type → T4 GPU** before running the notebook. A fresh runtime must
rerun the setup cell, but completed activation shards in Drive are detected and skipped.

## Outputs

By default, the notebook writes artifacts to
`MyDrive/math-error-tracing/artifacts/qwen2.5-math-1.5b` and keeps its ProcessBench JSONL under
`MyDrive/math-error-tracing/data/`. Change `DRIVE_PROJECT_DIR`, `DATA_FILENAME`, or
`ARTIFACT_NAME` in the notebook if desired.

Do not point configurations with different model, dataset sample, dtype, or context length at the
same artifact directory. Extraction identity checks reject incompatible cached shards, and a
distinct artifact name keeps alternative result packages separate.

## Configuration

`CONFIG_OVERRIDES = {}` preserves the full settings in `configs/experiment.yaml`. Overrides merge
recursively. For example, a limited diagnostic run can use:

```python
DATA_FILENAME = "processbench-25-per-source.jsonl"
ARTIFACT_NAME = "diagnostic-25-per-source"
CONFIG_OVERRIDES = {
    "data": {"max_examples_per_split": 25},
    "probe": {"bootstrap_samples": 0},
    "analysis": {"exploratory_bootstrap_samples": 0},
    "intervention": {"examples_per_class": 8, "random_directions": 2},
}
```

The stage flags `RUN_DOWNLOAD`, `RUN_EXTRACTION`, `RUN_PROBES_AND_CONTROLS`,
`RUN_INTERVENTIONS`, and `RUN_PLOTS` are all `True` by default and can be disabled independently
when resuming or splitting a run across sessions.

## GitHub token

Create a fine-grained personal access token with **Contents: read and write** access to this
repository and add it to Colab's Secrets panel as `GITHUB_TOKEN`. The notebook uses it both for the
initial clone and the final push through `GIT_ASKPASS`; it never embeds the token in a URL, Git
remote, source cell, commit, or printed command.

The final publishing cell commits result tables, learned directions, intervention outputs, figures,
the extraction identity, and the generated experiment config. Activation shards are excluded by
default because they are large resumable caches. Files above 95 MiB are rejected; publishing those
caches requires a separate Git LFS-aware workflow.
