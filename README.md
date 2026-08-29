# Causal circuits underlying zero-shot variant-effect prediction in ESM-2

This repository implements the experiment proposed in [experiment.md](experiment.md): trace the
sparse computation behind ESM-2 masked-marginal mutation scores, test its causal sufficiency and
necessity, and compare residue-level circuit importance with independent deep mutational scanning
(DMS) measurements.

The scaffold deliberately separates the stable experiment code from two external assets:

- ProteinGym assay data, which belongs under `data/raw/` and `data/processed/`.
- A pinned ProtoMech checkout and pretrained ESM2-8M CLT checkpoint, which belong under
  `checkpoints/protomech/` and are not committed.

## Experiment map

| Stage | Input | Output | Status |
| --- | --- | --- | --- |
| DMS normalization | ProteinGym CSV + WT FASTA | canonical single-mutant CSV | implemented |
| Full-model baseline | normalized CSV + ESM2-8M | masked-marginal scores | implemented |
| Circuit attribution | ESM-2 + pretrained CLT | ranked latent-token nodes | adapter contract ready |
| Sufficiency/necessity | ranked nodes + CLT | recovery and ablation curves | adapter contract ready |
| Biological validation | circuit scores + DMS | residue constraint metrics | core metrics implemented |

The circuit adapter is intentionally not a fake implementation. ProtoMech is research code rather
than a versioned Python package; pin its repository revision and checkpoint hash before connecting
its model-specific hooks in `ProtoMechBackend`.

## Setup

Core data processing and tests:

```bash
uv sync --extra dev
uv run pytest
uv run causal-circuits validate-config
```

Add ESM-2 inference dependencies on the machine or Colab runtime used for scoring:

```bash
uv sync --extra dev --extra model
```

The default model is `facebook/esm2_t6_8M_UR50D`. Its weights are downloaded by Transformers on
first use, so scoring needs network access once and enough space for the local model cache.

## Data contract

Download one single-mutant ProteinGym assay and its exact wild-type sequence. Keep the original CSV
under `data/raw/`. A source table must contain a mutation column such as `A42G` and a numeric fitness
column. Convert it to the internal schema with:

```bash
uv run causal-circuits prepare-dms \
  --input data/raw/tem1_firnberg.csv \
  --output data/processed/tem1_firnberg.csv \
  --mutation-column mutant \
  --fitness-column DMS_score \
  --directionality 1 \
  --wild-type-fasta data/raw/tem1.fasta
```

`--directionality` must make larger normalized `fitness` values mean better function. Supplying the
FASTA is strongly recommended: the command then fails early on coordinate or wild-type mismatches.
The normalized file retains source columns and adds:

| Column | Meaning |
| --- | --- |
| `mutation` | canonical `WT-position-mutant` string, e.g. `A42G` |
| `wild_type` | wild-type amino acid |
| `position` | one-indexed biological residue position |
| `mutant` | mutant amino acid |
| `fitness` | directionality-normalized experimental score |

Only single substitutions with the 20 canonical amino acids are accepted in the first experiment.
This avoids silently mixing epistasis, indels, or assay-specific coordinate conventions.

## Run the full-model baseline

Review [configs/experiment.yaml](configs/experiment.yaml), especially the assay paths and fitness
directionality, then run:

```bash
uv run causal-circuits score \
  --config configs/experiment.yaml \
  --wild-type-fasta data/raw/tem1.fasta
```

The scorer masks each distinct position once, caches its 20 amino-acid log probabilities, and
computes

```text
log P(mutant | masked sequence) - log P(wild type | masked sequence)
```

It writes the configured score CSV and prints its Spearman correlation with DMS fitness. Existing
score tables can be checked with:

```bash
uv run causal-circuits analyze \
  --scores artifacts/tem1_firnberg/zero_shot_scores.csv
```

## Connect ProtoMech

1. Clone ProtoMech outside this repository or as an ignored checkout under `checkpoints/`.
2. Record the exact Git commit, checkpoint URL/version, SHA-256 hash, ESM model name, latent width,
   and license in `checkpoints/protomech/README.md`.
3. Implement the three methods specified by `SparseCircuitBackend` in
   `src/causal_circuits/circuits.py`: attribution, score with retained nodes, and score with ablated
   nodes.
4. Confirm the CLT replacement model reproduces the native mutant-minus-WT logit margin before
   interpreting any circuit result.
5. Run top-k fractions from the YAML and compare every targeted intervention with matched random
   node sets using the same number of trials and random seed.

The unit of a node is `(layer, latent, token)`. Global ranking uses mean absolute
activation-times-gradient attribution across mutations. Report both model fidelity (correlation
with full ESM-2) and experimental validity (correlation with DMS); they answer different questions.

## Reproducibility checklist

- Keep DMS labels out of circuit selection; use them only for validation.
- Preserve raw assay scores and document the directionality transform.
- Pin dataset version, model revision, ProtoMech commit, and checkpoint hash.
- Save per-mutation attributions before aggregation.
- Use fixed seeds and matched-size random ablations.
- Report confidence intervals and the exact number of variants/residues retained after filtering.
- Start with ESM2-8M and one assay; add ESM2-35M or a second assay only after the full pipeline runs.
- Do not commit downloaded data, weights, caches, or potentially identifying manuscript metadata.

## Repository layout

```text
configs/                 experiment parameters
data/raw/                immutable downloaded inputs (gitignored)
data/processed/          normalized assay tables (gitignored)
checkpoints/             external model metadata and weights (weights gitignored)
artifacts/               scores, attributions, tables, and figures (gitignored)
src/causal_circuits/     data, scoring, circuit, analysis, and CLI code
tests/                   fast tests that do not download models
experiment.md            research question and proposed analyses
```

## Immediate next milestone

The first go/no-go checkpoint is a single command producing ESM2-8M scores for the primary assay
with the expected score direction and a plausible DMS Spearman correlation. Only then should the
ProtoMech adapter be wired in. This makes data/coordinate bugs visible before model interventions
make debugging harder.
