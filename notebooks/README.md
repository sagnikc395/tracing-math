# Colab workflow

[`experiment.ipynb`](experiment.ipynb) is the single notebook for the tracing pipeline. It uses the
same `math-error` command and `configs/project.yaml` as local runs.

[`contextual_baseline_colab.ipynb`](contextual_baseline_colab.ipynb) is the separate E3 workflow. It
clones `github.com/sagnikc395/tracing-math` with a `GITHUB_TOKEN`, authenticates model downloads with
an `HF_TOKEN`, runs `fit-contextual-baseline`, and can push only its generated Markdown result record.
Add both tokens as Colab secrets and grant the notebook access. It never stages data, activation
shards, embeddings, or other Drive artifacts.

[`e4_e5_e6_colab.ipynb`](e4_e5_e6_colab.ipynb) runs five trace-equal group-split refits for E4,
the three transition-matching variants for E5, and calibration metrics plus a reliability diagram for
E6. It uses the same private-clone and optional result-record push flow.

The notebook assumes that the repository is already available in the Colab runtime or on Google
Drive. It mounts Drive for data, checkpoints, logs, and artifacts. It does not clone a repository,
read credentials, change Git state, or contact remote services.

## Runtime

Use a GPU runtime with BF16 support for the model-backed stages. The notebook uses one Drive-backed
run root:

```text
MyDrive/math-error-tracing/
├── data/processbench.jsonl
├── data/counterfactual_pairs.jsonl
├── run_status.json
├── logs/
└── artifacts/
```

Place the repository checkout in one of these locations before running the first cell:

- the current Colab working directory;
- `/content/tracing-mathematical-error-detection-in-language-models`;
- `MyDrive/math-error-tracing/tracing-mathematical-error-detection-in-language-models`; or
- `MyDrive/math-error-tracing/repository`.

The first cell installs the local package in editable mode. Dataset and model downloads use their
public package identifiers from the project configuration.

## Workflow

Run cells from top to bottom. The stages are:

1. mount Drive and resolve runtime paths;
2. validate the unified project configuration;
3. download ProcessBench and extract resumable step-boundary activations;
4. fit hidden-state probes and visible controls;
5. run the gated verdict assay and render figures;
6. run CPU analysis over frozen predictions;
7. fit the conditional nuisance-plus-hidden comparison;
8. fit the matched transition probe and write matching diagnostics;
9. compare natural-token and marker-token boundary states;
10. create the counterfactual annotation template; and
11. check the compact result inventory.

The CPU analysis does not load the model or activation shards. Counterfactual patching remains gated:
reviewers must complete the correction fields and set `verified` to `true` before the patching
command can be run. No patching result is reported by this notebook.

All commands use one entry point:

```bash
python -m tracing_math --config /content/tracing_math_project.yaml validate-config
python -m tracing_math --config /content/tracing_math_project.yaml extract-activations
python -m tracing_math --config /content/tracing_math_project.yaml fit-probes
python -m tracing_math --config /content/tracing_math_project.yaml analyze
python -m tracing_math --config /content/tracing_math_project.yaml fit-conditional
python -m tracing_math --config /content/tracing_math_project.yaml fit-transition
```

Existing shards and analysis outputs are reused by the pipeline. Status and logs are written under the
Drive run root so a disconnected runtime can resume from the last completed stage.
