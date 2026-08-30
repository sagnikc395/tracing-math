# Paper experiment notebook

[`experiment.ipynb`](experiment.ipynb) is the single Colab entry point for the experiments in the
paper. It runs, in order:

1. ProcessBench download and resumable step-boundary activation extraction;
2. Experiment A: layer-wise decoding, localization, grouped-bootstrap uncertainty, and the required
   position, lexical, shuffled-label, embedding-state, and within-error-trace controls;
3. Experiment B: the complete cross-domain transfer matrix; and
4. Experiment C: the signed causal dose response and matched random-orthogonal controls;
5. the three essential paper figures; and
6. publication of the final result package to GitHub.

Use **Runtime → Change runtime type → T4 GPU**, then run every cell from top to bottom. The notebook
mounts Google Drive and writes the fixed run to:

```text
MyDrive/math-error-tracing/
├── data/processbench.jsonl
└── artifacts/qwen2.5-math-1.5b/
```

Before starting, create a fine-grained GitHub token with **Contents: read and write** permission and
store it in Colab Secrets as `GITHUB_TOKEN`. Authentication uses a temporary `GIT_ASKPASS` script;
the token is not embedded in the clone URL, Git remote, notebook source, or output.

All scientific settings come from `configs/experiment.yaml`, which contains only the confirmatory L2
probe and Experiments A--C settings. The notebook changes only the data and artifact paths so results
survive Colab disconnects. It intentionally has no sampled-run overrides, stage switches, smoke
tests, exploratory probe families, or PCA analysis.

Activation extraction is resumable. Re-running the notebook reuses complete shards and regenerates
the downstream paper artifacts from them. The publishing cell commits only the fixed generated
configuration, essential tables, learned directions, intervention outputs, and these figures:

- `method_and_trajectory.pdf`;
- `predictive_results.pdf`; and
- `transfer_and_causal.pdf`.

The dataset and activation shards remain in Drive and are never published by the notebook.
