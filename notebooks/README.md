# Paper experiment notebook

## Experiment 2 follow-up

[`experiment2.ipynb`](experiment2.ipynb) runs the robustness and causal-validation follow-up
specified in [`results/experiment2.md`](../results/experiment2.md). It requires the complete
Drive-backed Experiment 1 directory, including activation shards, and writes every new result to:

```text
MyDrive/math-error-tracing/artifacts/experiment2/
```

The notebook runs stronger within-trace and combined-surface controls, extracts fresh activations
at natural step-final tokens, audits a counterbalanced single-token verdict, and compares the
learned intervention direction with a gradient-aligned positive causal control. Its final cell
copies compact tables and figures to the repository's `artifacts/experiment2` while excluding the
large semantic activation shards. It does not commit or push.

All long GPU stages are disconnect-safe. Semantic extraction resumes by shard, verdict scoring by
completed batch, and causal validation by completed trace/mapping job. During a run, inspect
`stage_status.json`, `semantic_extraction_progress.json`, the stage-local `progress.json`, and the
append-only files under `logs/`. The verdict and causal directories also contain readable atomic
CSV checkpoints, so partial numerical results can be inspected before the stage finishes. Re-run a
cell after reconnecting to continue rather than restart it.

### Experiment 2 runtime behavior

The A100 configuration preserves all preregistered traces, layers, counterbalanced mappings,
intervention doses, bootstrap samples, and decision rules. Runtime reductions come only from
removing redundant computation:

- semantic-extraction batches are grouped by approximate prompt length within each 100-trace
  shard, reducing padding, and all requested layer/step states cross from GPU to CPU once per
  batch;
- verdict and gradient stages run the decoder directly and project only the final non-padding
  hidden state, instead of constructing vocabulary logits at every prompt position; and
- the eight non-zero learned/gradient direction-dose variants for one causal job share a single
  batched forward pass. The two zero-dose rows reuse the already-computed baseline exactly.

`causal.batch_size: 8` is tuned for the requested A100 runtime. If a smaller GPU runs out of
memory, lower only this batch size and re-run the cell. Batch size is operational rather than
scientific, so existing verdict and causal checkpoints remain compatible and completed jobs are
reused. Do not reduce sample counts, layers, mappings, doses, or bootstrap counts for the final
study.

## Experiment 1 paper run

[`experiment.ipynb`](experiment.ipynb) is the single Colab entry point for the experiments in the
paper. It runs, in order:

1. ProcessBench download and resumable step-boundary activation extraction;
2. Experiment A: layer-wise decoding, localization, grouped-bootstrap uncertainty, and the required
   position, lexical, shuffled-label, embedding-state, and within-error-trace controls;
3. Experiment B: the complete cross-domain transfer matrix; and
4. Experiment C: the signed causal dose response and matched random-orthogonal controls;
5. the three essential paper figures; and
6. publication of the final result package to GitHub.

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

All scientific settings come from `configs/experiment.yaml`, which contains only the confirmatory L2
probe and Experiments A--C settings. The notebook changes only persistence paths and runtime settings
(BF16 and batch sizes) for the A100. It intentionally has no sampled-run overrides, stage switches,
smoke tests, exploratory probe families, or PCA analysis.

Activation extraction and causal interventions are resumable. Re-running the notebook reuses
complete activation shards and intervention groups. `run_status.json`, per-stage logs,
`extraction_progress.json`, and `interventions/progress.json` record what is running and what has
finished. The publishing cell commits only the fixed generated configuration, essential tables,
learned directions, intervention outputs, and these figures:

- `method_and_trajectory.pdf`;
- `predictive_results.pdf`; and
- `transfer_and_causal.pdf`.

The dataset and activation shards remain in Drive and are never published by the notebook.
