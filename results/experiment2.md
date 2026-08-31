# Experiment 2 design: Is the mathematical error signal a genuine change point, and is the causal assay sensitive?

## Status and purpose

This document specifies Experiment 2 before its results are inspected. Experiment 2 is a follow-up
to the completed study reported in `results/experiment1.md`. It does not replace or retroactively
alter Experiment 1. All Experiment 2 results must be described as follow-up robustness or
mechanistic-validation evidence.

Experiment 1 found that a linear probe of `Qwen/Qwen2.5-Math-1.5B-Instruct` hidden states predicted
whether a mathematical solution was invalid so far. The selected probe achieved held-out AUROC
0.866 and outperformed position and current-step TF-IDF controls. The score transferred across the
four ProcessBench sources, but exact first-error localization was imperfect. Additive intervention
along the learned direction did not affect the tested verdict score. That causal null was
inconclusive because the original `CORRECT`/`INCORRECT` readout classified every sampled boundary
as incorrect and had AUROC 0.283.

Experiment 2 addresses the following specific limitations:

1. the probe might exploit trace-wide difficulty, generator identity, final-answer outcome, or a
   combination of textual and positional features;
2. the `invalid_so_far` target might capture accumulated post-error corruption rather than a change
   at the first erroneous step;
3. the representation was measured at an artificial `<<END_STEP_k>>` marker;
4. cross-domain AUROC transferred more reliably than the threshold used for localization;
5. the original verdict labels had different strings and potentially different tokenization;
6. the original intervention had no positive control demonstrating that the hook, token, layer,
   scale, and outcome could detect a real causal perturbation;
7. the original study did not measure geometric alignment between the probe direction and the
   directions to which the verdict was locally sensitive.

All Experiment 2 outputs are written under `artifacts/experiment2`. The full Experiment 1 artifact
directory is read-only input.

## 1. Research questions

### RQ1: Is the score a within-trace error change point?

Does invalidity remain linearly decodable after training only on erroneous traces and removing each
trace's first-step activation as a trace-specific reference? Does the frozen score increase between
the last valid step and the annotated first-error step?

### RQ2: Does the hidden probe add information beyond stronger surface controls?

Does the hidden-state probe outperform a single combined control containing current-step TF-IDF,
absolute and fractional position, trace length, token length, source, and generator identity, using
paired whole-trace bootstrap intervals?

### RQ3: Is the result tied to the artificial step marker?

If hidden states are instead taken from the last non-whitespace token of each written reasoning
step—before the end marker is causally visible—does the result remain above chance and close to the
marker-boundary result?

### RQ4: Can a tokenization-balanced native verdict be measured?

Can the model distinguish valid from invalid prefixes when the verdict is expressed with two
single-token labels whose semantic mapping is reversed in a second prompt and then averaged?

### RQ5: If a learned-direction intervention is null, is the causal assay capable of detecting an
effect?

At the same token and layer, does a gradient-aligned positive-control direction change the verdict
margin in the expected signed direction? How closely is the learned probe direction aligned with
the local verdict gradient at every layer?

## 2. Fixed resources and partitions

- **Model:** `Qwen/Qwen2.5-Math-1.5B-Instruct` in bfloat16.
- **Dataset:** the same ProcessBench file and SHA-identified records used by Experiment 1.
- **Partition seed:** 42.
- **Grouping:** all traces and steps from the same normalized problem remain in one partition.
- **Split fractions:** 60% training, 20% validation, and 20% test.
- **Primary target:** `invalid_so_far`, equal to 1 at and after the first annotated error.
- **Diagnostic target:** `error_onset`, equal to 1 only at the first annotated error.
- **Frozen Experiment 1 layer:** hidden-state index 23.
- **Regularization grid:** `C` in {0.01, 0.1, 1, 10}.
- **Uncertainty:** 1,000 bootstrap resamples of complete traces.
- **Confidence level:** 95%.

The test partition is not used to select regularization, probability thresholds, layers, verdict
prompts, or label mappings. Experiment 2 decision rules are defined below before execution.

## 3. Stage A: Stronger predictive robustness

### A1. Expanded frozen-split analysis

The original marker-boundary activations are reanalyzed using the same split. The analysis produces:

- full layer-wise `invalid_so_far` performance;
- an `error_onset` probe at every layer;
- PCA probes using 1, 2, 4, 8, 16, 32, 64, and 128 components;
- calibration and validation-versus-test threshold curves;
- error-aligned score trajectories;
- source, generator, final-answer-correctness, error-position, trace-length, and token-length
  subgroups.

These outputs are exploratory. The original Experiment 1 estimate remains the confirmatory result.

### A2. Error-only probe

An L2 probe is trained using only traces with an annotated error. Within each such trace, steps
before the first error are negative and the first-error-and-later steps are positive. Validation
AUROC selects `C`; validation balanced accuracy selects the threshold. The final probe is refitted
on training plus validation and evaluated on held-out erroneous traces.

This prevents the probe from succeeding by merely distinguishing wholly correct solutions from
solutions that contain an error.

### A3. First-step-centered error-only probe

For erroneous traces whose first error occurs after step 0, the first-step activation is subtracted
from every activation in the same trace. The error-only probe is then fitted to these centered
states. This removes any additive trace-wide offset already present at the beginning of the
solution.

This is a diagnostic representation analysis, not a deployable online detector: the first step is
used as a within-trace reference, and traces erroneous from step 0 cannot be included.

### A4. Paired onset jump

For every held-out erroneous trace with first error `e > 0`, calculate

\[
J_i=s_{i,e}-s_{i,e-1},
\]

where `s` is the frozen probe score. Whole traces are bootstrapped to obtain a confidence interval
for the mean jump. A positive interval means the score increases at the human-annotated transition.

### A5. Combined surface and metadata control

A single sparse classifier combines:

- current-step TF-IDF unigrams and bigrams;
- absolute and fractional step position;
- number of steps;
- prompt token count;
- source identity;
- generator identity.

The hidden-state and combined-control predictions are compared on the same test traces. Paired
whole-trace bootstrap intervals are calculated for AUROC, average precision, Process F1, and exact
first-error outcome.

### A6. Leave-one-generator-out tests

For every generator with at least 20 held-out traces and both step labels, remove that generator
from probe training and validation, then evaluate only on its test traces. This tests whether the
representation generalizes to unseen solution-writing models.

### A7. Cross-source uncertainty, calibration, and direction geometry

At frozen index 23, source-specific probes are trained on one source's training partition. For each
train/test source pair, report a whole-trace AUROC interval and Process F1 under two thresholds:

- the threshold learned from the training source's validation traces;
- a recalibrated threshold learned from the target source's validation traces without changing the
  probe direction.

The analysis also reports the cosine matrix among the four raw-coordinate source-specific probe
directions. This distinguishes three possibilities: shared ranking and aligned directions, shared
ranking with different directions, or a shared direction whose probability calibration changes by
source.

### Stage A decision rule

The strong change-point claim is supported only if all four conditions hold:

1. error-only AUROC has a 95% interval entirely above 0.5;
2. first-step-centered error-only AUROC has a 95% interval entirely above 0.5;
3. the paired onset-jump interval is entirely above zero;
4. hidden-minus-combined-surface intervals are entirely above zero for both AUROC and Process F1.

If only the raw error-only result succeeds, the narrower conclusion is that a within-erroneous-trace
signal exists but may reflect stable trace differences. If centered decoding succeeds but the onset
jump does not, the signal distinguishes broad pre-error and post-error regions without sharply
changing at the annotated step.

## 4. Stage B: Natural step-end replication

Experiment 1 recorded the hidden state at the final token of `<<END_STEP_k>>`. Experiment 2 performs
a new model pass and records the state at the last token containing non-whitespace step text before
that marker. Under causal attention, this token cannot see the marker, the later steps, or the
verdict question.

All 29 hidden-state indices are extracted. Probes, controls, source transfer, grouped uncertainty,
and diagnostics are fitted independently with validation-only selection. A second comparison fixes
the layer to Experiment 1 index 23 and uses paired test traces to calculate
semantic-boundary-minus-marker differences.

### Stage B decision rule

The signal is considered marker-invariant if:

1. the validation-selected semantic-boundary probe has a test AUROC interval entirely above 0.5;
2. at frozen index 23, the lower bound for semantic-minus-marker AUROC exceeds -0.05;
3. at frozen index 23, the lower bound for semantic-minus-marker Process F1 exceeds -0.10.

These non-inferiority bounds permit small losses because the semantic token is not an explicit
summary marker. If semantic decoding collapses, Experiment 1 must be described as marker-dependent.

## 5. Stage C: Counterbalanced single-token verdict audit

The original multi-token `CORRECT`/`INCORRECT` sequence score is replaced by the next-token logit
difference between labels `A` and `B`.

Every sampled prefix is evaluated twice:

1. fixed mapping: `A = valid`, `B = invalid`;
2. reversed mapping: `B = valid`, `A = invalid`.

For each mapping, the canonical margin is

\[
m=\operatorname{logit}(\text{invalid label})-
  \operatorname{logit}(\text{valid label}).
\]

The two canonical margins are averaged for each example. This cancels a stable preference for one
letter. The code verifies at runtime that both labels are distinct single tokens.

One boundary per trace is sampled, balanced between valid-so-far and invalid-so-far. Validation
uses at most 64 examples per class; test uses at most 128 per class. Report AUROC, average precision,
balanced accuracy, recall, specificity, margin accuracy, greedy first-token exact accuracy, and the
fraction correct under both mappings.

To ensure the longer counterbalanced instruction cannot exceed the 2,048-token context limit,
verdict and causal sampling is restricted to traces whose recorded Experiment 1 complete-prompt
length is at most 1,984 tokens, reserving 64 tokens for the replacement instruction. This exclusion
is applied before class-balanced sampling and must be reported with the final sample counts.

### Stage C decision rule

Before test results are examined, the verdict readout is declared competent only if its
counterbalanced validation results satisfy:

- AUROC at least 0.60;
- recall at least 0.55;
- specificity at least 0.55.

The causal stage still runs if this rule fails so that implementation sensitivity can be diagnosed,
but learned-direction results are then labeled behaviorally uninterpretable.

## 6. Stage D: Gradient alignment and positive-control interventions

### D1. Local verdict gradients

For each causal test example and each mapping, calculate the gradient of the canonical next-token
verdict margin with respect to the marked boundary activation at every decoder-layer input:

\[
g_{i,\ell}=\nabla_{h_{i,\ell}}m_i.
\]

Compare this gradient with the frozen probe direction `v_l` using:

- dot product `g^T v`;
- cosine similarity;
- one-standard-deviation local derivative `sigma_v g^T v`.

The two label mappings are averaged before layer-level conclusions are drawn.

### D2. Learned-direction intervention

At frozen index 23, intervene using the Experiment 1 probe direction:

\[
h'=h+\alpha\sigma_vv,
\qquad
\alpha\in\{-2,-1,0,1,2\}.
\]

The outcome is the counterbalanced single-token invalid-minus-valid logit margin.

### D3. Gradient positive control

For the same example, token, layer, norm, and doses, replace `v` with the normalized local gradient
direction. A positive gradient dose must locally increase the outcome by construction; observing
this in finite differences demonstrates that the hook and behavioral assay are capable of detecting
a causal change of the chosen size.

The causal sample contains 32 valid and 32 invalid held-out traces, with at most one boundary per
trace. Paired whole-trace bootstrap intervals are reported at every dose.

### Stage D decision rule

The positive control passes only if:

- the most negative gradient-direction dose has an interval entirely below zero; and
- the most positive gradient-direction dose has an interval entirely above zero.

A learned-direction null is interpretable as local causal non-use only if both the Stage C verdict
competence rule and the Stage D positive-control rule pass. Even then, the conclusion is restricted
to the tested boundary, layer, additive intervention, model, and local verdict outcome.

If the gradient control fails, no learned-direction null is interpreted: the assay is insensitive
at the tested scale or contains an implementation problem. If the gradient control passes but the
native verdict is incompetent, the hook is technically validated but no claim about useful native
monitoring is made.

## 7. Complete decision table

| Observation | Supported conclusion |
|---|---|
| Stages A and B pass | A marker-independent within-trace change-point signal is linearly decodable |
| A passes, B fails | The signal depends substantially on the artificial end marker |
| Raw error-only succeeds, centered probe fails | Decoding relies partly on trace-wide offsets |
| Hidden probe does not beat combined surface control | A surface/generator shortcut remains plausible |
| C and gradient control pass; learned intervention succeeds | The direction locally participates in the validated verdict |
| C and gradient control pass; learned intervention is null | Decodable invalidity is locally misaligned with verdict-sensitive directions |
| C fails; gradient control passes | Technical causality is measurable, but useful native verdict behavior is absent |
| Gradient control fails | The causal experiment is not sensitive enough to interpret |

No nonsignificant result is described as proof of an exactly zero effect. An equivalence claim would
require a separately justified smallest effect size of interest.

## 8. Artifact layout

```text
artifacts/experiment2/
├── experiment_config.yaml
├── stage_status.json                         # attempt counts, timings, and current progress
├── run_summary.json
├── logs/
│   ├── experiment2.log                       # aggregate append-only event log
│   └── <stage>.log                           # per-command log
├── semantic_extraction_identity.json
├── semantic_extraction_progress.json
├── semantic_activation_shards/             # resumable, not intended for publication
├── marker_robustness/
│   ├── layer_metrics.csv
│   ├── diagnostic_target_metrics.csv
│   ├── pca_subspace.csv
│   ├── subgroup_metrics.csv
│   ├── error_only_metrics.csv
│   ├── onset_jump_bootstrap.csv
│   ├── leave_one_generator_out.csv
│   ├── domain_calibration.csv
│   ├── source_direction_cosines.csv
│   ├── surface_metadata_metrics.csv
│   ├── hidden_vs_surface_paired.csv
│   └── decision.json
├── semantic_boundary/
│   ├── layer_metrics.csv
│   ├── test_predictions.csv
│   ├── semantic_vs_marker_paired.csv
│   └── decision.json
├── verdict_audit/
│   ├── checkpoint_identity.json
│   ├── individual.checkpoint.csv             # atomic, resumable scoring checkpoint
│   ├── progress.json
│   ├── individual.csv
│   ├── counterbalanced.csv
│   ├── summary.csv
│   ├── label_tokens.json
│   └── decision.json
├── causal_validation/
│   ├── checkpoint_identity.json
│   ├── gradient_alignment.checkpoint.csv     # atomic, one trace/mapping job at a time
│   ├── interventions.checkpoint.csv
│   ├── progress.json
│   ├── gradient_alignment_individual.csv
│   ├── gradient_alignment_summary.csv
│   ├── interventions_individual.csv
│   ├── interventions_counterbalanced.csv
│   ├── intervention_summary.csv
│   └── decision.json
└── figures/
    ├── experiment2_summary.pdf
    └── experiment2_summary.png
```

## 9. Running the experiment

The full Experiment 1 activation shards are required. The compact published artifacts in this
repository contain the directions and result tables but not those shards. The Colab notebook points
to the Drive-backed Experiment 1 cache.

Run every stage locally with:

```bash
python -m causal_circuits.experiment2_cli --config configs/experiment2.yaml validate-config
python -m causal_circuits.experiment2_cli --config configs/experiment2.yaml analyze-robustness
python -m causal_circuits.experiment2_cli --config configs/experiment2.yaml extract-semantic
python -m causal_circuits.experiment2_cli --config configs/experiment2.yaml fit-semantic
python -m causal_circuits.experiment2_cli --config configs/experiment2.yaml audit-verdict
python -m causal_circuits.experiment2_cli --config configs/experiment2.yaml causal-validation
python -m causal_circuits.experiment2_cli --config configs/experiment2.yaml plot
```

At any point, print the compact live report without loading model weights or result tables:

```bash
python -m causal_circuits.experiment2_cli --config configs/experiment2.yaml status
```

Alternatively:

```bash
python -m causal_circuits.experiment2_cli --config configs/experiment2.yaml run-all
```

The semantic extraction resumes at 100-trace shards, the verdict audit resumes at completed
inference batches, and causal validation resumes at completed trace/mapping jobs. Checkpoint files
are replaced atomically and guarded by dataset/model/config identities, so incompatible partial
runs are rejected instead of mixed. `run-all` skips stages already marked complete for the same
resolved configuration; pass global `--force` before `run-all` to re-enter them (stage-local raw
checkpoints are still reused).

While a run is active, inspect `stage_status.json`, the current stage's `progress.json`, and
`logs/experiment2.log`. Partial raw results are directly readable from the `*.checkpoint.csv`
files. Re-running the same command after a disconnect continues from those checkpoints. Every
numerical result and decision file is stored under `artifacts/experiment2`; Experiment 1 files are
never modified.

### Runtime implementation notes

The optimized implementation does not omit any stage or alter the fixed scientific design. In
particular, it retains all 29 hidden-state indices, both verdict mappings, 32 causal examples per
class, both causal directions, all five doses, and 1,000 bootstrap resamples.

The GPU path avoids work that is not part of any estimand. Semantic traces are length-bucketed
within each resumable shard to limit padding, and their requested boundary states are transferred
from GPU to CPU in one operation per batch. Verdict and causal scoring project only the final
non-padding decoder state; full vocabulary logits for earlier prompt positions were never read and
are no longer materialized. For each trace/mapping causal job, the eight non-zero combinations of
two directions and four doses are evaluated in batches of up to `causal.batch_size`; the two
zero-dose records reuse the gradient pass's baseline margin.

The A100 configuration uses `causal.batch_size: 8`. This value affects only scheduling and memory,
not samples or numerical definitions, and may be lowered after an out-of-memory error without
invalidating existing checkpoints. Resume guards intentionally ignore batch-size-only changes but
still reject changes to the model, data, labels, sample counts, directions, doses, dtype, or context
length.

For the fixed 64-trace causal sample and two mappings, this changes the intervention portion from
eight decoder invocations per trace/mapping job to one, reducing total causal decoder invocations
from 1,152 (128 gradient plus 1,024 intervention passes) to 256 (128 gradient plus 128 batched
intervention passes). This count describes model invocations, not an assumed wall-clock speedup;
batch compute and prompt length still determine actual Colab runtime.
