# All experiments notebook

[`experiment.ipynb`](experiment.ipynb) is the single Colab entry point for Experiments 1--3. It
runs, in order:

1. ProcessBench download and resumable step-boundary activation extraction;
2. Experiment A: layer-wise decoding, localization, grouped-bootstrap uncertainty, and the required
   position, lexical, shuffled-label, embedding-state, and within-error-trace controls;
3. Experiment B: the complete cross-domain transfer matrix; and
4. Experiment C: a gated Yes/No verdict baseline, then the signed causal dose response and matched
   random-orthogonal controls when baseline specificity is nonzero;
5. the three essential paper figures;
6. publication of the frozen result package to GitHub;
7. Experiment 2, the CPU follow-up with temporal tests, length-aware thresholds, and paired control
   intervals;
8. Experiment 3's matched first-error transition probe over the original activation shards;
9. resumable GPU extraction at natural step endings and artificial marker endings;
10. annotation-gated counterfactual activation patching; and
11. an optional, independently checkpointed 7B same-family replication.

Use **Runtime → Change runtime type → A100 GPU**, then run every cell from top to bottom. The notebook
mounts Google Drive and writes the fixed run to:

```text
MyDrive/math-error-tracing/
├── data/processbench.jsonl
├── run_status.json
├── logs/
└── artifacts/qwen2.5-math-1.5b-a100-bf16/
```

Before starting, create a fine-grained GitHub token with **Contents: read and write** permission and
store it in Colab Secrets as `GITHUB_TOKEN`. Authentication uses a temporary `GIT_ASKPASS` script;
the token is not embedded in the clone URL, Git remote, notebook source, or output.

Sections 1--7 take their scientific settings from `configs/experiment1.yaml`, which contains only the
confirmatory L2 probe and Experiment 1 settings. Those sections change only persistence paths and
runtime settings (BF16 and batch sizes) for the A100. Section 9 builds Experiment 2's configuration;
Sections 10--12 use `configs/experiment3.yaml`. Both write separate artifact trees and label
their analyses post-hoc. The 7B replication is off by default so it cannot delay the required stages.

Activation extraction and causal interventions are resumable. Re-running the notebook reuses
complete activation shards and intervention groups. `run_status.json`, per-stage logs,
`extraction_progress.json`, and `interventions/progress.json` record what is running and what has
finished. Extended stages use separate status keys and write to `artifacts/experiment3-extended`.
The counterfactual JSONL template lives in Drive so manual verification survives a runtime reset.
The publishing cell commits only the fixed generated configuration, essential tables,
learned directions, intervention outputs, and these figures:

- `method_and_trajectory.pdf`;
- `predictive_results.pdf`; and
- `transfer_and_causal.pdf`.

A final publication cell copies compact follow-up tables and manifests. It excludes activation
shards, checkpoint files, the counterfactual annotation template, and the downloaded dataset.

The dataset and activation shards remain in Drive and are never published by the notebook.
