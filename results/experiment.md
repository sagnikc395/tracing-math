# Experiment record: tracing mathematical error detection

**Status:** primary GPU run complete; CPU follow-up complete; selected post-hoc extension analyses
complete; counterfactual patching pending human verification.

**Run date:** the primary artifacts were produced before the CPU analysis recorded on September 2,
2026. The repository does not preserve the primary GPU wall-clock timestamp or host details.

This file is the experiment record. It distinguishes the settings used to produce the frozen
artifacts from later analyses that reused those artifacts. Results are reported only when the
corresponding artifact is present.

## 1. Run ledger

| Run | Scope | Inputs | Status | Main output |
|---|---|---|---|---|
| Experiment 1 | Model extraction, hidden-state probes, controls, transfer, and gated verdict intervention | ProcessBench, Qwen2.5-Math-1.5B-Instruct, GPU | Complete | `artifacts/qwen2.5-math-1.5b/` |
| Experiment 2 | CPU analyses of frozen predictions and intervention records | Experiment 1 artifacts only | Complete | `artifacts/experiment2_cpu/` |
| Experiment 3A | Matched error-onset transition probe | Experiment 1 activation shards | Complete; post-hoc | `artifacts/experiment3_extended/transition_probe/` |
| Experiment 3B | Natural-token versus artificial-marker boundary control | New boundary activation extraction plus frozen selection | Complete; post-hoc | `artifacts/experiment3_extended/boundary_control/` |
| Experiment 3C | Conditional hidden-state versus nuisance-feature comparison | Experiment 1 activation shards and frozen partitions | Complete; post-hoc | `artifacts/experiment3_extended/conditional_hidden_state/` |
| Experiment 3D | Counterfactual activation patching | Human-verified corrected steps required | Pending | No patching artifact |

Experiment 2 and Experiment 3 were designed after inspecting the primary results. They are robustness
or diagnostic analyses, not untouched confirmations of the primary hypothesis.

## 2. Questions and claim boundary

The primary run asked:

1. Can a linear probe decode whether a solution prefix has become invalid?
2. Does the score help locate the annotated first error rather than only identify a trace as
   erroneous?
3. Does a direction learned from one ProcessBench source rank boundaries from other sources?
4. Does moving the hidden state along that direction change the model's own error verdict?

The evidence supports the first three questions only in a limited sense. Hidden states contain
linearly decodable information about `invalid_so_far`, the score has partial temporal alignment with
the annotation, and source-specific rankings transfer above chance. The primary causal assay did not
show the predicted effect on its recorded verdict score, and its native verdict baseline was weak.
The run therefore does not establish that the model uses the decoded signal for a reliable internal
verdict.

## 3. Canonical setup for the primary run

### 3.1 Resolved configuration

The frozen primary artifacts were produced from a notebook-resolved configuration, not directly from
the checked-in `configs/project.yaml`. The notebook kept the scientific settings from that file but
changed runtime paths, GPU batch sizes, and dtype:

| Setting | Resolved primary-run value |
|---|---|
| Seed | `42` |
| Model | `Qwen/Qwen2.5-Math-1.5B-Instruct` |
| Device | CUDA GPU, selected by `device: auto` |
| Model dtype | `bfloat16` |
| Maximum complete prompt length | `2,048` tokens |
| Dataset | `Qwen/ProcessBench` |
| Dataset splits | `gsm8k`, `math`, `olympiadbench`, `omnimath` |
| Examples per split | All available examples |
| Activation output | `/content/drive/MyDrive/math-error-tracing/artifacts/qwen2.5-math-1.5b-a100-bf16` in the resolved run |
| Activation shard size | `100` traces |
| Extraction batch size | `16` |
| Intervention batch size | `8` |
| Probe target | `invalid_so_far` |
| Partition fractions | `60%` train, `20%` validation, `20%` test |
| Probe regularization grid | `C in {0.01, 0.1, 1.0, 10.0}` |
| Probe optimizer iterations | `2,000` maximum |
| Bootstrap samples for primary probe summaries | `1,000` |
| Threshold grid | `0.05` through `0.95`, step `0.005` |
| Intervention doses | `alpha in {-4, -2, -1, 0, 1, 2, 4}` |
| Intervention examples | `128` valid-so-far and `128` invalid-so-far boundaries |
| Random directions | `20`, orthogonal to the learned direction |

The checked-in config currently says `float16`, extraction batch size `1`, intervention batch size
`1`, and `exploratory_bootstrap_samples: 2000`. The frozen primary run instead records `bfloat16`,
batch sizes `16` and `8`, and `exploratory_bootstrap_samples: 0` in
`artifacts/qwen2.5-math-1.5b/experiment_config.yaml`. The latter file is the authority for the
primary GPU run. The CPU follow-up has its own recorded settings in its output summary.

The resolved primary config and extraction identity record the dataset fingerprint
`447f0a4b35c5747a9f9a3dab1e70d43f71efd501497b8cec668b71337099784a` and `bfloat16` dtype. The
repository does not preserve the exact GPU model name, runtime duration, or primary extraction
progress file.

### 3.2 Data and labels

The downloaded dataset contained 3,400 human-annotated traces from four sources:

- GSM8K;
- MATH;
- OlympiadBench;
- Omni-MATH.

A trace is a problem, a generated step-by-step solution, its generator, a `first_error` annotation,
and a final-answer-correctness flag. `first_error = -1` means that all displayed steps are correct;
`first_error = e >= 0` means that zero-indexed step `e` is the first erroneous step.

For trace `i`, step `k`, and first-error index `e_i`, the primary label was

\[
y_{ik}=\mathbb{1}[e_i\geq 0 \text{ and } k\geq e_i].
\]

This label is called `invalid_so_far`. It is positive at the first annotated error and remains
positive for later steps. It does not claim that every later step is locally wrong.

### 3.3 Leakage-resistant partitions

The split seed was `42`. Problems were normalized by collapsing whitespace and case-folding, then
grouped by the first 16 hexadecimal characters of a SHA-1 hash of the normalized problem text. All
traces and steps in one problem group were assigned to one partition. Groups were stratified by source
and whether any member trace contained an error. Within each stratum, groups were ordered by a seeded
SHA-256 digest and allocated approximately 60/20/20 to train, validation, and test.

The probe fit used training data for feature standardization and model fitting. `C`, the threshold,
and the reported layer were selected using validation data only. The selected probe was then refit on
train plus validation before one evaluation on the test partition. Test labels were not used for
selection.

### 3.4 Prompt construction and activation extraction

Each model prompt contained the problem and numbered reasoning blocks:

```text
[Step 0]
<step text>
<<END_STEP_0>>
```

After the displayed steps, the prompt asked whether the reasoning was valid through the last step
and requested exactly `CORRECT` or `INCORRECT`. The model wrapper used a causal forward pass over the
complete trace and recorded the residual-stream state at the final token of every end marker. Causal
attention prevents later tokens from changing an earlier boundary state.

The model exposed 29 hidden-state indices, numbered 0 through 28. Index 0 is the embedding output;
indices 1 through 28 are successive transformer states. Each state has 1,536 dimensions. Complete
prompts longer than 2,048 tokens were excluded rather than truncated, and the exclusion was recorded
in the shard manifest.

## 4. Primary run execution

The core workflow ran in this order:

```bash
uv run math-error --config <resolved-config> validate-config
uv run math-error --config <resolved-config> download-data
uv run math-error --config <resolved-config> extract-activations
uv run math-error --config <resolved-config> fit-probes
uv run math-error --config <resolved-config> run-interventions
uv run math-error --config <resolved-config> render-figures
```

The notebook wrapped the same CLI calls, mounted Google Drive, wrote stage logs and status files,
and resumed existing activation shards. The core CLI also supports the equivalent single command:

```bash
uv run math-error --config <resolved-config> run-all
```

`<resolved-config>` is a placeholder for the notebook-generated YAML file. It is not
`configs/project.yaml` for the frozen primary artifacts because the notebook changed runtime
settings listed above.

### 4.1 Probe fitting and selection

At each of the 29 hidden-state indices, the primary model was a class-balanced L2 logistic
regression on standardized hidden states:

\[
P(y_{ik}=1\mid h_{ik}^{(\ell)})
=\sigma(w_\ell^\top h_{ik}^{(\ell)}+b_\ell).
\]

`C` was selected from `{0.01, 0.1, 1, 10}` by validation AUROC. The probability threshold was
selected on validation Process F1 over the configured threshold grid. The layer was selected by
validation Process F1, with validation AUROC as the tie-breaker. This selected layer 23, `C = 0.01`,
and threshold `0.645`.

The primary reports include step AUROC, average precision, step F1, calibration metrics, and
trace-level localization metrics. Confidence intervals use 1,000 resamples of whole traces, which
preserves dependence among steps from the same solution.

### 4.2 Controls

The primary test used three controls with the same held-out traces:

1. **Position:** absolute step index and fractional position within the trace.
2. **Current-step TF-IDF:** unigram and bigram features from the current written step.
3. **Shuffled-label hidden state:** the hidden-state procedure after shuffling training labels.

The embedding output at index 0 is included in the layer-wise curve. The within-error analysis
restricts evaluation to traces known to contain an error, comparing pre-error with post-error
boundaries.

### 4.3 Cross-source transfer

At layer 23, a separate probe was trained on each source's training partition. Each source-specific
threshold was chosen on that source's validation partition. Each probe was then evaluated on the test
partition of all four sources, producing a 4-by-4 matrix. AUROC measures ranking transfer; Process F1
also depends on the source-specific threshold and therefore tests calibration and localization
transfer.

### 4.4 Gated verdict intervention

The probe's standardized weights were converted to raw hidden-state coordinates and normalized to a
unit direction `v`. The intervention scale was the standard deviation of train-plus-validation
projections onto `v`. For one boundary from each of 256 distinct held-out traces, the state entering
the selected decoder layer was changed to

\[
h'=h+\alpha\sigma_vv.
\]

The tested values were `alpha = -4, -2, -1, 0, 1, 2, 4`. The behavior readout used the separate
`step-error-yes-no-v1` prompt and measured

\[
m=P(\texttt{Yes})-P(\texttt{No}),
\]

where `Yes` means that the reasoning contains an error. The probabilities are conditional
single-token next-token probabilities under teacher forcing. Positive `m` favors `Yes`.

The intervention workflow first scored the unmodified boundaries. Its stored baseline was AUROC
`0.342`, specificity `0.203`, and zero-threshold accuracy `0.371` over 256 boundaries. Because this
readout was weak and below chance in ranking, the dose-response result is evidence only about this
readout under this intervention. It is not evidence of a functioning native correctness monitor.

Twenty random unit directions orthogonal to `v` were evaluated at `alpha = -4` and `+4` on the
matched 32-example subset. The intervention artifact contains 1,792 learned-direction rows and
1,280 random-direction rows.

## 5. Primary results

### 5.1 Held-out step decoding

At the validation-selected layer 23, the frozen test results were:

| Metric | Test estimate | Whole-trace 95% interval |
|---|---:|---:|
| AUROC | 0.866 | [0.849, 0.884] |
| Average precision | 0.843 | [0.813, 0.870] |
| Balanced accuracy | 0.780 | [0.762, 0.800] |
| Precision | 0.809 | [0.773, 0.843] |
| Recall | 0.686 | [0.652, 0.720] |
| Specificity | 0.874 | [0.852, 0.895] |
| Step F1 | 0.743 | [0.716, 0.769] |
| Brier score | 0.150 | [0.139, 0.160] |
| Log loss | 0.458 | [0.429, 0.486] |
| Expected calibration error | 0.046 | [0.030, 0.070] |

The test artifact contains 669 retained traces and 4,985 retained step boundaries:

| Quantity | Count |
|---|---:|
| Erroneous traces | 432 |
| Fully correct traces | 237 |
| `invalid_so_far = 1` boundaries | 2,181 |
| `invalid_so_far = 0` boundaries | 2,804 |
| Final answer correct | 350 |
| Final answer incorrect | 319 |

The test sources contributed:

| Source | Retained traces | Retained boundaries |
|---|---:|---:|
| GSM8K | 84 | 439 |
| MATH | 204 | 1,387 |
| OlympiadBench | 188 | 1,661 |
| Omni-MATH | 193 | 1,498 |

The numerically highest test AUROC was 0.868 at layer 22. It was not selected because layer choice
was locked from validation before test evaluation.

### 5.2 Localization

Using the layer-23 threshold, the first threshold crossing produced:

| Metric | Estimate |
|---|---:|
| Exact first-error accuracy among erroneous traces | 0.289 |
| Correct rejection of fully correct traces | 0.612 |
| Process F1 | 0.393 |
| Exact complete-trace outcome | 0.404 |
| Error detected somewhere | 0.887 |
| Within one step | 0.593 |
| Within two steps | 0.727 |
| Mean signed localization error among detections | +0.587 steps |
| Mean absolute localization error among detections | 1.397 steps |

The positive signed error indicates that detected errors were late on average. Whole-trace bootstrap
intervals for the main localization quantities are:

| Metric | Estimate | 95% interval |
|---|---:|---:|
| Correct-trace accuracy | 0.611 | [0.554, 0.673] |
| Error-step accuracy | 0.291 | [0.246, 0.332] |
| Process F1 | 0.393 | [0.351, 0.436] |
| Exact complete-trace outcome | 0.404 | [0.368, 0.442] |
| Detection rate | 0.887 | [0.854, 0.917] |
| Within one step | 0.594 | [0.547, 0.638] |
| Within two steps | 0.728 | [0.687, 0.770] |
| Mean signed error | +0.583 | [+0.362, +0.796] steps |
| Mean absolute error | 1.393 | [1.241, 1.557] steps |

### 5.3 Controls

| Predictor | AUROC | Average precision | Step F1 | Process F1 | Exact complete-trace outcome |
|---|---:|---:|---:|---:|---:|
| Hidden state, layer 23 | **0.866** | **0.843** | **0.743** | **0.393** | **0.404** |
| Current-step TF-IDF | 0.733 | 0.667 | 0.490 | 0.254 | 0.260 |
| Position | 0.730 | 0.621 | 0.359 | 0.054 | 0.108 |
| Shuffled-label hidden state | 0.518 | 0.436 | 0.275 | 0.179 | 0.184 |

These are descriptive held-out differences. The later CPU analysis computed paired whole-trace
intervals for the differences, but those intervals are reported separately from this primary table.

Among the 432 erroneous test traces, the hidden-state score had AUROC 0.873, average precision
0.925, balanced accuracy 0.782, and step F1 0.785 when pre-error and post-error boundaries were
compared within erroneous traces.

### 5.4 Cross-source transfer

#### Step AUROC

| Train source \\ Test source | GSM8K | MATH | OlympiadBench | Omni-MATH |
|---|---:|---:|---:|---:|
| GSM8K | 0.817 | 0.802 | 0.753 | 0.791 |
| MATH | 0.775 | 0.877 | 0.789 | 0.858 |
| OlympiadBench | 0.739 | 0.852 | 0.804 | 0.868 |
| Omni-MATH | 0.768 | 0.859 | 0.810 | 0.882 |

All 12 off-diagonal AUROCs were above 0.739. Their mean was 0.805; the mean diagonal AUROC was
0.845. The ranking signal transferred to every source pair, with lower average performance across
sources than within the training source.

#### Process F1

| Train source \\ Test source | GSM8K | MATH | OlympiadBench | Omni-MATH |
|---|---:|---:|---:|---:|
| GSM8K | 0.253 | 0.245 | 0.055 | 0.134 |
| MATH | 0.209 | 0.433 | 0.267 | 0.297 |
| OlympiadBench | 0.279 | 0.363 | 0.309 | 0.351 |
| Omni-MATH | 0.275 | 0.371 | 0.280 | 0.349 |

Mean off-diagonal Process F1 was 0.261, compared with mean diagonal Process F1 of 0.336. The
AUROC matrix is consistently above chance, while threshold-dependent localization is less stable.

### 5.5 Primary intervention result

The learned-direction paired changes in `P(Yes) - P(No)` were:

| `alpha` | Mean paired change | 95% interval |
|---:|---:|---:|
| -4 | +0.000207 | [-0.001941, +0.002227] |
| -2 | +0.000922 | [-0.001233, +0.002976] |
| -1 | -0.001605 | [-0.003552, +0.000497] |
| 0 | 0 | [0, 0] |
| +1 | -0.001201 | [-0.003416, +0.000777] |
| +2 | -0.000447 | [-0.002403, +0.001535] |
| +4 | -0.000466 | [-0.002607, +0.001679] |

Every nonzero-dose interval includes zero. The estimated dose slope was `-0.000120`; rank
monotonicity across the seven means was `-0.500`; and the hypothesized sign occurred for 23.4% of
individual nonzero interventions. At the extreme doses, learned-direction changes did not show a
reliable advantage over the 20 random orthogonal directions. The empirical one-sided values were
`1.0` at `alpha = -4` and `0.0476` at `alpha = +4` under the stored comparison procedure.

These values do not support the predicted monotonic causal effect. Since the unmodified readout had
AUROC 0.342 and specificity 0.203, they also do not establish that the learned direction is
causally irrelevant to a competent native verdict mechanism.

## 6. Post-hoc CPU follow-up

### 6.1 Protocol

The CPU follow-up reused the 4,985 frozen test predictions and intervention records. It loaded no
language model and no activation tensor and fit no new hidden-state probe. It used seed 42, 5,000
within-trace circular-shift randomizations, 2,000 grouped bootstrap draws, and 95% intervals.

### 6.2 Temporal alignment and onset change

Circularly shifting each score trajectory within its trace reduced exact localization from 0.289 to
a null mean of 0.164 (`p = 0.0002`). Within-one-step localization was 0.593 versus a null mean of
0.436 (`p = 0.0002`). The observed score is therefore aligned with the annotation beyond this
within-trace shifted null. This test does not by itself remove absolute-position explanations.

The mean score jump at the annotated first error was 0.257. A metadata-matched transition from a
fully correct trace changed by 0.113. The paired difference was 0.144, with a 95% interval of
[0.096, 0.193]. The analysis used 380 erroneous traces whose first error occurred after step 0 and
139 unique correct placebo traces. Correct traces also showed positive score drift.

Subtracting each eligible erroneous trace's first-step score left pooled AUROC at 0.881
[0.862, 0.900]. Mean AUROC calculated separately within traces was 0.968. The score therefore
retained pre-error versus post-error ordering after removing a stable trace-level offset.

### 6.3 Failure patterns and threshold sensitivity

The strongest failure pattern was false alarms on long traces. Complete-trace accuracy fell from
0.491 in the shortest trace-length quartile to 0.264 in the longest; correct-trace rejection fell
from 0.719 to 0.350. By token-count quartile, correct rejection fell from 0.792 to 0.231. Across
sources, complete-trace accuracy was 0.512 on GSM8K, 0.500 on MATH, 0.298 on OlympiadBench, and
0.358 on Omni-MATH. These comparisons are descriptive and post-hoc.

Thresholds fitted separately by trace-length bin did not improve the frozen test result: Process F1
fell from 0.393 to 0.377, and correct rejection fell from 0.612 to 0.540. Trace-equal weighting
also changed AUROC only from 0.866 to 0.867, a difference of +0.001 with 95% interval
[-0.007, +0.010].

### 6.4 Equally tuned shortcut controls

The CPU analysis gave the shortcut models the same `C` grid and validation selection protocol as the
hidden probe, refit each selected model on train plus validation, and used inverse boundary-count
weights so each trace contributed equal total training loss.

| Control | `C` | AUROC | Process F1 | Exact error | Correct rejection |
|---|---:|---:|---:|---:|---:|
| Hidden state, frozen | 0.01 | 0.866 | 0.393 | 0.289 | 0.612 |
| Prefix TF-IDF | 10.0 | 0.751 | 0.274 | 0.192 | 0.477 |
| Structural metadata | 10.0 | 0.776 | 0.168 | 0.148 | 0.194 |
| Metadata + final outcome | 1.0 | 0.854 | 0.262 | 0.164 | 0.646 |
| Joint text + metadata | 1.0 | 0.810 | 0.210 | 0.169 | 0.278 |
| Joint text + metadata + outcome | 1.0 | 0.874 | 0.294 | 0.176 | 0.895 |

Final-answer correctness is a benchmark annotation and is unavailable to an online detector. The
last row is therefore an oracle-assisted diagnostic control, not a deployable baseline. The joint
text-plus-metadata control remains below the hidden probe on AUROC and Process F1. The oracle-assisted
control has slightly higher AUROC, with hidden-minus-control difference `-0.008`
[−0.028, +0.013], while the hidden probe has higher exact error localization by `0.113`
[0.064, 0.163] and lower correct rejection by `-0.283` [−0.347, −0.218].

The conditional hidden-state analysis gave an additional post-hoc comparison. Adding hidden states
to prefix text increased AUROC by `+0.059` [0.041, 0.075] and reduced log loss by `-0.080`
[−0.104, −0.055] relative to prefix text alone, under its resolved trace-equal protocol. This is
an analysis result, not a new untouched test.

## 7. Post-hoc extended analyses

### 7.1 Matched error-onset transition probe

For each eligible erroneous trace, the activation difference from the last valid boundary to the
annotated first-error boundary was paired with a transition from a fully correct trace. Matching
preferred source and generator, then minimized differences in relative position, trace length, and
token count. The original problem-grouped partitions remained fixed.

There were 1,919 pairs across all partitions and 380 held-out test pairs. The 1,919 pairs used 687
unique placebo traces. Placebo reuse had median 2, 90th percentile 6, and maximum 46. Standardized
differences were -0.011 for relative position, 0.146 for trace length, and 0.285 for token count.

A validation-selected L2 transition probe used layer 21 and `C = 0.01`. On held-out pairs:

| Metric | Estimate | 95% interval |
|---|---:|---:|
| AUROC | 0.769 | [0.713, 0.820] |
| Average precision | 0.741 | [0.672, 0.816] |
| Paired accuracy | 0.753 | [0.676, 0.824] |

The controls were position AUROC 0.631, current-step TF-IDF AUROC 0.671, and shuffled-label hidden
AUROC 0.585. This supports a linearly distinguishable change at annotated onsets relative to the
matched transitions. It does not show that the change is abrupt, and reuse and residual matching
imbalance make this an exploratory result. One-to-one and inverse-reuse sensitivity refits were
implemented but no completed sensitivity artifact is recorded here.

### 7.2 Natural-token boundary control

A separate extraction recorded the hidden state at both the last non-whitespace token of each written
step and the final token of its artificial end marker. The comparison fixed layer 23 and `C = 0.01`,
then evaluated both locations on the original test partition.

| Location | AUROC | Process F1 | Exact first error | Within one step |
|---|---:|---:|---:|---:|
| Last natural step token | 0.868 | 0.420 | 0.329 | 0.572 |
| End-marker token | 0.866 | 0.393 | 0.289 | 0.593 |

Natural-token minus marker-token differences were +0.002 AUROC [−0.007, +0.011] and +0.026 Process
F1 [−0.020, +0.075] under the paired whole-trace bootstrap. The difference intervals include zero.
The decodable signal is not specific to the artificial marker location.

### 7.3 Conditional nuisance comparison

This post-hoc analysis compared prefix text (`N`), hidden state (`H`), and their combinations under
trace-equal training weights. At layer 23, adding hidden state to prefix text (`N+H`) produced
AUROC 0.869 and Process F1 0.397, compared with AUROC 0.811 and Process F1 0.211 for prefix text
alone. The paired `N+H - N` AUROC difference was +0.059 [0.041, 0.075]. The feature block also
included an oracle-assisted final-answer outcome; that outcome is not available to an online
monitor.

## 8. Counterfactual patching status

A 160-pair template was generated from selected erroneous traces. It contains the problem, the prefix
before the first error, the original erroneous step, a proposed corrected step, annotation notes, and
a verification flag. The current annotation inventory has 135 drafted corrections and 25 withheld
items. The withheld items require a full solution, a change to the overall solution route, or access
to a figure that could not be checked from text.

Patching requires a human reviewer to verify each correction and set `verified: true`. The loader also
requires a nonempty corrected step distinct from the original error step. No activation-patching
result is reported because the verification gate has not been completed.

## 9. Reproduction

### 9.1 Local checks

```bash
uv sync --extra dev
uv run math-error --config configs/project.yaml validate-config
uv run pytest
uv run ruff check src tests
```

### 9.2 Primary GPU workflow

Use the resolved notebook configuration to reproduce the frozen primary artifacts:

```bash
uv run math-error --config <resolved-config> download-data
uv run math-error --config <resolved-config> extract-activations
uv run math-error --config <resolved-config> fit-probes
uv run math-error --config <resolved-config> run-interventions
uv run math-error --config <resolved-config> render-figures
uv run math-error --config <resolved-config> analyze
```

The GPU stages require the model download, ProcessBench download, CUDA memory, and storage for
activation shards. The CPU `analyze` stage reads frozen predictions and intervention records; it does
not load the model or activation shards.

### 9.3 Extended analyses

The maintained CLI names for the extended analyses are:

```bash
uv run math-error --config <resolved-config> fit-transition
uv run math-error --config <resolved-config> transition-diagnostics
uv run math-error --config <resolved-config> transition-sensitivity
uv run math-error --config <resolved-config> extract-boundary-controls
uv run math-error --config <resolved-config> analyze-boundary-controls
uv run math-error --config <resolved-config> fit-conditional
uv run math-error --config <resolved-config> fit-contextual-baseline
uv run math-error --config <resolved-config> prepare-counterfactuals
uv run math-error --config <resolved-config> run-counterfactuals
```

`run-counterfactuals` remains gated on verified corrections and a usable verdict baseline.

## 10. Artifact map

All numerical claims in this record come from these files:

| Artifact | Use |
|---|---|
| `artifacts/qwen2.5-math-1.5b/experiment_config.yaml` | Resolved primary-run configuration |
| `artifacts/qwen2.5-math-1.5b/extraction_identity.json` | Dataset fingerprint, model, dtype, context limit |
| `artifacts/qwen2.5-math-1.5b/probes/layer_metrics.csv` | Layer-wise validation/test metrics |
| `artifacts/qwen2.5-math-1.5b/probes/test_predictions.csv` | Frozen test labels, metadata, and scores |
| `artifacts/qwen2.5-math-1.5b/probes/test_group_bootstrap_summary.csv` | Whole-trace primary intervals |
| `artifacts/qwen2.5-math-1.5b/probes/controls.csv` | Primary shortcut controls |
| `artifacts/qwen2.5-math-1.5b/probes/domain_transfer.csv` | Four-source transfer matrices |
| `artifacts/qwen2.5-math-1.5b/probes/directions.npz` | Directions, projection scales, thresholds, selected layers |
| `artifacts/qwen2.5-math-1.5b/interventions/behavioral_verdict.json` | Native verdict baseline and readout definition |
| `artifacts/qwen2.5-math-1.5b/interventions/summary.csv` | Learned and random intervention scores |
| `artifacts/qwen2.5-math-1.5b/interventions/effect_statistics.csv` | Paired dose intervals and random-direction comparisons |
| `artifacts/experiment2_cpu/summary.json` | CPU settings, environment, and headline diagnostics |
| `artifacts/experiment2_cpu/results.md` | CPU follow-up result record |
| `artifacts/experiment2_cpu/data_flow.csv` | Attempted, retained, and over-length counts |
| `artifacts/experiment2_cpu/onset_eligibility_audit.csv` | Step-0 exclusion audit for onset analyses |
| `artifacts/experiment3_extended/transition_probe/` | Matched transition probe and diagnostics |
| `artifacts/experiment3_extended/boundary_control/` | Natural-token versus marker control |
| `artifacts/experiment3_extended/conditional_hidden_state/` | Conditional nuisance comparison |
| `artifacts/experiment3_extended/contextual_text_baseline/` | Optional frozen visible-text semantic baseline (unrun) |

Generated datasets, activation shards, model caches, and run outputs are not source files and should
remain outside Git.

When run, `transition-sensitivity` writes `matching_sensitivity.csv` with each variant's point
estimate and 95% interval, `matching_sensitivity_bootstrap.csv` with the two-way pair/placebo-trace
bootstrap summaries, and `matching_sensitivity_diagnostics.csv` with reuse and covariate-balance
diagnostics. These artifacts are not present in the completed run record.

## 11. Conclusions and limitations

The completed primary and follow-up analyses support these statements:

1. At layer 23, hidden states linearly predict `invalid_so_far` on held-out ProcessBench traces with
   AUROC 0.866 [0.849, 0.884].
2. The score has partial temporal information: it detects 88.7% of erroneous traces somewhere and
   localizes 59.3% within one step, but exact localization is 28.9% and detections are late on
   average.
3. Ranking transfer is broader than threshold transfer: all cross-source off-diagonal AUROCs exceed
   0.739, while Process F1 varies from 0.055 to 0.371 off diagonal.
4. The frozen score changes near the annotated error beyond circularly shifted and matched-placebo
   comparisons, but these analyses were post-hoc.
5. The learned-direction intervention did not produce the predicted monotonic change in the stored
   `P(Yes) - P(No)` readout. The readout baseline was weak, so this is an inconclusive causal test,
   not proof that the representation is causally inert.
6. Counterfactual activation patching has not run because its corrections are not human-verified.

Limitations:

- One 1.5-billion-parameter model and one frozen dataset partition were tested.
- ProcessBench was repartitioned for representation analysis; these are not standard benchmark scores.
- The `invalid_so_far` target propagates the first-error label to all later steps.
- Prompt text, step structure, generator, trace length, and final-answer metadata can carry shortcuts.
- Long traces are overrepresented among false alarms, and 40 of 3,400 traces exceeded the context
  limit and were excluded: 27 from train, 4 from validation, and 9 from test.
- The transition analysis reuses placebo traces, with maximum reuse 46, and has residual matching
  imbalance in length and token count.
- The intervention changes one boundary state at one decoder layer and uses one teacher-forced
  single-token readout.
- The CPU and extended analyses were selected after inspecting primary results and are not
  independent confirmations.
- The primary artifacts do not preserve the exact GPU host, wall-clock duration, or full extraction
  progress log.
