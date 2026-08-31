# Experiment 1: Detecting the first mathematical error in Qwen2.5-Math-1.5B

## Plain-language summary

This experiment tested whether a small mathematics-specialized language model internally
distinguishes valid reasoning from reasoning that has already become invalid. The experiment used
human-annotated mathematical solutions from ProcessBench. Each solution contains multiple written
steps and identifies the first erroneous step, if any.

At the end of every reasoning step, the hidden state of
`Qwen/Qwen2.5-Math-1.5B-Instruct` was recorded. A linear classifier, called a **probe**, was trained
to predict whether the reasoning was still valid at that point or whether the first error had
already occurred. The classifier was trained on one part of the data, all model and threshold
choices were made on a separate validation part, and the final numbers below were calculated on a
held-out test part.

The main predictive result was positive. At the validation-selected hidden-state index 23, the
probe achieved a held-out step-level AUROC of **0.866** with a 95% whole-trace bootstrap interval of
**[0.849, 0.884]**. It outperformed classifiers using only step position or the words in the current
step. It also retained high discrimination when evaluated only within solutions known to contain
an error, and its ranking performance transferred across all four ProcessBench sources.

The temporal result was useful but imperfect. Across complete held-out traces, the probe predicted
the exact correct outcome—either the annotated first-error step or no error—on **40.4%** of traces.
Among erroneous traces, it identified the exact erroneous step on **29.0%**, a step within one
position of the error on **59.3%**, and a step within two positions on **72.7%**. When it detected an
error, its prediction was late by an average of about **0.59 steps**.

The causal intervention result was negative for the tested readout. Adding or subtracting the
learned probe direction at the step boundary did not produce the predicted monotonic change in the
model's `INCORRECT`-versus-`CORRECT` score, and no tested dose had a 95% interval excluding zero.
However, the unmodified behavioral readout was itself not competent: all 256 sampled boundaries
were classified as `INCORRECT`, and the score ranked invalid boundaries in the wrong direction
(AUROC **0.283**). Therefore this stage establishes decodability and partial localization, but it
does **not** establish that the representation is used—or not used—by a functioning native verdict
mechanism.

## 1. Research question

The experiment asked two related questions:

1. **Predictive question:** When a written mathematical solution first becomes invalid, does the
   model's hidden representation change in a way that a simple linear classifier can detect?
2. **Causal question:** If the detected representation is directly increased or decreased, does
   the model become more or less likely to judge the reasoning as incorrect?

The second question is deliberately stronger than the first. A classifier can decode information
that happens to be present in a hidden state even when the model does not use that information to
produce its output. Consequently, successful probing alone is evidence of an association, not of a
causal mechanism.

## 2. Model, data, and labels

### 2.1 Model

The evaluated model was `Qwen/Qwen2.5-Math-1.5B-Instruct`. It was run in bfloat16 precision with a
maximum complete prompt length of 2,048 tokens. No model weights were trained or changed. Only
external logistic-regression probes were fitted.

The model exposed 29 hidden-state indices, numbered 0 through 28. Index 0 is the embedding output;
the remaining indices correspond to successive states through the transformer. Each recorded
state had 1,536 dimensions.

### 2.2 Dataset

The source dataset was the official 3,400-example ProcessBench dataset. It contains mathematical
problems and model-generated, step-by-step solutions from four sources:

- GSM8K;
- MATH;
- OlympiadBench;
- Omni-MATH.

Each solution, called a **trace** in this report, has a human annotation `first_error`:

- `first_error = -1` means every displayed step is correct;
- `first_error = e >= 0` means step `e` is the first incorrect step.

The experiment converted this trace annotation into a label at every step boundary. For trace
\(i\), step \(k\), and annotated first error \(e_i\), the target was

\[
y_{ik}=\mathbb{1}[e_i\geq 0 \text{ and } k\geq e_i].
\]

Thus:

- every step of a fully correct trace has label 0;
- steps before the first error in an erroneous trace have label 0;
- the first erroneous step and every later step have label 1.

This target is named `invalid_so_far`. It asks whether the prefix of reasoning up to the current
step is invalid; it does not ask whether the current step alone is locally incorrect.

### 2.3 Held-out test-set composition

The published prediction artifact contains **669 test traces** and **4,985 step boundaries**:

| Test-set quantity | Count |
|---|---:|
| Erroneous traces | 432 |
| Fully correct traces | 237 |
| Label-1 boundaries (`invalid_so_far = 1`) | 2,181 |
| Label-0 boundaries | 2,804 |
| Traces whose recorded final answer was correct | 350 |
| Traces whose recorded final answer was incorrect | 319 |

The source composition was:

| Source | Test traces | Test step boundaries |
|---|---:|---:|
| GSM8K | 84 | 439 |
| MATH | 204 | 1,387 |
| OlympiadBench | 188 | 1,661 |
| Omni-MATH | 193 | 1,498 |

Test traces contained between 2 and 32 reasoning steps. Their recorded complete prompt lengths
ranged from 188 to 2,039 tokens.

These counts are recoverable from `probes/test_predictions.csv`. The published result package does
not contain the activation-shard metadata needed to reconstruct exact retained training and
validation counts or a source-by-source exclusion table. Those quantities must not be inferred
from the test counts above.

## 3. Experimental procedure

### 3.1 Leakage-resistant partitioning

The split seed was 42. Problems were normalized by collapsing whitespace and case-folding, then
grouped by a hash of the normalized problem text. Every solution and every step belonging to the
same normalized problem was assigned to the same partition. Groups were stratified by source and
whether any trace contained an error, then allocated approximately 60% to training, 20% to
validation, and 20% to testing.

This prevents steps from the same problem from appearing in both the probe-training and test data.
The test set was not used to select regularization, thresholds, or the reported layer.

### 3.2 Prompt and activation extraction

Each prompt contained the mathematical problem and numbered blocks of the following form:

```text
[Step 0]
<written reasoning step>
<<END_STEP_0>>
```

After all displayed steps, the prompt asked whether the reasoning was valid through the last
displayed step and instructed the model to answer exactly `CORRECT` or `INCORRECT`.

For each trace, one causal forward pass was used to record the residual-stream hidden state at the
last token of every `<<END_STEP_k>>` marker. Although the prompt contained later steps, causal
attention prevents a boundary state from seeing later tokens. All 29 hidden-state indices were
recorded and evaluated separately.

### 3.3 Linear probes and model selection

At each hidden-state index, a class-balanced L2-regularized logistic regression was trained after
standardizing each hidden dimension using training data. The regularization value was selected
from

\[
C\in\{0.01,0.1,1,10\}
\]

using validation AUROC. A probability threshold between 0.05 and 0.95, in increments of 0.005, was
chosen using validation Process F1. The displayed layer was selected by the highest validation
Process F1, with validation AUROC as a tie-breaker.

This procedure selected hidden-state index **23**, with \(C=0.01\) and threshold **0.645**. The
probe at that index was then refitted on the combined training and validation partitions and
evaluated once on the test partition. Index 23 was also eligible for intervention and was therefore
used in the causal experiment.

For reference, the numerically highest test AUROC occurred at index 22 (0.868), but that observation
did not change the preregistered selection: index 23 was chosen from validation results before test
evaluation.

### 3.4 Meaning of the reported metrics

- **Step AUROC:** the probability that a randomly selected invalid-so-far boundary receives a
  higher score than a randomly selected valid-so-far boundary. A value of 0.5 is chance ranking.
- **Average precision:** precision averaged across score thresholds, accounting for the positive
  class frequency.
- **Step F1:** ordinary binary F1 after applying the validation-selected threshold.
- **Predicted first error:** the first step whose probe score reaches the threshold. If no step
  reaches it, the prediction is `-1`, meaning no error.
- **Error accuracy / on-time detection:** the fraction of erroneous traces whose predicted first
  error exactly equals the human annotation.
- **Correct accuracy:** the fraction of fully correct traces for which the probe never crosses the
  threshold.
- **Process F1:** the harmonic mean of error accuracy and correct accuracy. It is high only if the
  method both localizes errors and avoids false alarms on correct traces.
- **First-error exact:** the fraction of all traces for which the predicted outcome exactly matches
  the annotation, including prediction `-1` for fully correct traces.
- **Detection rate:** the fraction of erroneous traces on which the probe crosses the threshold
  somewhere, whether or not the crossing is at the correct step.
- **Early, on-time, and late rates:** fractions of erroneous traces whose first threshold crossing
  occurs before, at, or after the annotated error. Missed traces are not included in any of these
  three categories, so these rates plus the miss rate sum to one.
- **Signed localization error:** predicted step minus annotated step among detected erroneous
  traces. Positive values indicate late detection.

Uncertainty was measured with 1,000 bootstrap samples that resampled whole traces, not individual
steps. This preserves the statistical dependence between steps from the same solution.

### 3.5 Controls

Three control classifiers used the same partitions:

1. **Position control:** absolute step index and fractional position within the trace.
2. **Current-step lexical control:** TF-IDF unigrams and bigrams from the current written step.
3. **Shuffled-label hidden-state control:** the hidden-state probe trained after shuffling training
   labels.

The position control tests whether errors simply tend to occur late. The lexical control tests
whether surface words in the current step explain the result. The shuffled-label control tests
whether the full probe procedure produces apparently strong performance without a real mapping
between activations and labels.

### 3.6 Cross-domain transfer

At index 23, four additional probes were trained, one per source. Each probe used only the training
partition from its source. Its threshold was selected only on validation traces from that source,
and it was then evaluated on test traces from all four sources. This produced a 4-by-4 matrix.

High off-diagonal AUROC indicates that a score learned from one source ranks valid and invalid
boundaries in another source. High threshold-dependent localization metrics additionally require
the source-selected probability scale and threshold to transfer.

### 3.7 Causal intervention

The standardized logistic-regression weight was converted back to raw hidden-state coordinates and
normalized to a unit vector \(v\). The standard deviation \(\sigma_v\) of training-plus-validation
projections onto that vector was used to define the intervention scale.

One held-out boundary was selected from each of 256 distinct traces: 128 valid-so-far boundaries
and 128 invalid-so-far boundaries. At the input to decoder layer 23, the boundary state was changed
to

\[
h'=h+\alpha\sigma_vv,
\]

for

\[
\alpha\in\{-4,-2,-1,0,1,2,4\}.
\]

Positive \(\alpha\) was intended to increase the internal invalidity feature. The expected result
was therefore an increasing, approximately monotonic dose-response curve.

The behavioral outcome was

\[
m=\operatorname{mean}\log P(\texttt{INCORRECT})-
  \operatorname{mean}\log P(\texttt{CORRECT}),
\]

where each candidate answer was teacher-forced after the prompt. A positive score favors
`INCORRECT`. Each intervention effect was paired to the same example's unmodified score.

As a control, 20 random unit directions orthogonal to \(v\) were evaluated at \(\alpha=-4\) and
\(+4\) on a matched subset of 32 examples. The complete intervention artifact contains 1,792
learned-direction rows and 1,280 random-direction rows, for 3,072 rows total.

## 4. Predictive results

### 4.1 Selected-layer step classification

The direct held-out metrics at index 23 were:

| Metric | Test result |
|---|---:|
| AUROC | 0.866 |
| Average precision | 0.843 |
| Balanced accuracy | 0.780 |
| Precision | 0.809 |
| Recall | 0.686 |
| Specificity | 0.874 |
| Step F1 | 0.743 |
| Brier score | 0.150 |
| Log loss | 0.458 |
| Expected calibration error | 0.046 |

The whole-trace bootstrap results were:

| Metric | Bootstrap estimate | 95% interval |
|---|---:|---:|
| AUROC | 0.866 | [0.849, 0.884] |
| Average precision | 0.843 | [0.813, 0.870] |
| Balanced accuracy | 0.780 | [0.762, 0.800] |
| Precision | 0.809 | [0.773, 0.843] |
| Recall | 0.687 | [0.652, 0.720] |
| Specificity | 0.874 | [0.852, 0.895] |
| Step F1 | 0.743 | [0.716, 0.769] |

The AUROC interval is entirely above 0.5, so invalid-so-far information was reliably linearly
decodable on held-out traces.

### 4.2 Within-erroneous-trace discrimination

When evaluation was restricted to the 432 traces known to contain an error, the same frozen probe
distinguished pre-error steps from the first-error-and-later steps with:

| Metric | Test result |
|---|---:|
| AUROC | 0.873 |
| Average precision | 0.925 |
| Balanced accuracy | 0.782 |
| Step F1 | 0.785 |

This result is important because it cannot be obtained merely by assigning one score to all steps
of an erroneous trace and another score to all steps of a correct trace. It shows substantial
within-trace discrimination between the portions before and after the annotated error. It does not,
by itself, eliminate every possible position or textual shortcut.

## 5. First-error localization results

The selected probe's direct test-set localization metrics were:

| Metric | Test result |
|---|---:|
| Exact first error among erroneous traces | 0.289 |
| Correct rejection of fully correct traces | 0.612 |
| Process F1 | 0.393 |
| Exact complete-trace outcome | 0.404 |
| Error detected somewhere | 0.887 |
| Error missed entirely | 0.113 |
| False alarm on a fully correct trace | 0.388 |
| Early detection | 0.171 |
| On-time detection | 0.289 |
| Late detection | 0.426 |
| Within 1 step of the error | 0.593 |
| Within 2 steps of the error | 0.727 |
| Mean signed error, detected traces | +0.587 steps |
| Mean absolute error, detected traces | 1.397 steps |

Whole-trace bootstrap uncertainty was:

| Metric | Bootstrap estimate | 95% interval |
|---|---:|---:|
| Correct-trace accuracy | 0.611 | [0.554, 0.673] |
| Error-step accuracy | 0.291 | [0.246, 0.332] |
| Process F1 | 0.393 | [0.351, 0.436] |
| Exact complete-trace outcome | 0.404 | [0.368, 0.442] |
| Detection rate | 0.887 | [0.854, 0.917] |
| Within 1 step | 0.594 | [0.547, 0.638] |
| Within 2 steps | 0.728 | [0.687, 0.770] |
| Mean signed error | +0.583 | [+0.362, +0.796] steps |
| Mean absolute error | 1.393 | [1.241, 1.557] steps |

The probe usually detected that an error existed, but exact timing was substantially harder than
step-level discrimination. Late predictions were more common than early predictions, and the
positive signed-error interval indicates a systematic delay of roughly half a step among detected
traces.

## 6. Control results

| Method | AUROC | Average precision | Step F1 | Process F1 | Exact complete-trace outcome |
|---|---:|---:|---:|---:|---:|
| Hidden-state probe, index 23 | **0.866** | **0.843** | **0.743** | **0.393** | **0.404** |
| Current-step TF-IDF | 0.733 | 0.667 | 0.490 | 0.254 | 0.260 |
| Position only | 0.730 | 0.621 | 0.359 | 0.054 | 0.108 |
| Shuffled-label hidden state | 0.518 | 0.436 | 0.275 | 0.179 | 0.184 |

Relative to the strongest surface control, current-step TF-IDF, the hidden-state probe was higher
by 0.133 AUROC, 0.176 average precision, 0.253 step F1, 0.139 Process F1, and 0.144 exact
complete-trace accuracy. These are descriptive differences; the present artifact package does not
include paired bootstrap intervals for probe-minus-control differences.

The position control detected an error somewhere on 84.5% of erroneous traces, but it usually
crossed much too late: its exact error-step accuracy was 3.0%, mean signed error was +4.02 steps,
and Process F1 was 0.054. Therefore high detection rate alone is not evidence of correct error
localization.

## 7. Cross-domain transfer results

Rows identify the source used to train the probe. Columns identify the held-out test source.

### 7.1 Step-level AUROC

| Train source \ Test source | GSM8K | MATH | OlympiadBench | Omni-MATH |
|---|---:|---:|---:|---:|
| GSM8K | 0.817 | 0.802 | 0.753 | 0.791 |
| MATH | 0.775 | 0.877 | 0.789 | 0.858 |
| OlympiadBench | 0.739 | 0.852 | 0.804 | 0.868 |
| Omni-MATH | 0.768 | 0.859 | 0.810 | 0.882 |

Every off-diagonal AUROC was above 0.739. The mean off-diagonal AUROC was 0.805, compared with a
mean diagonal AUROC of 0.845. Thus the ranking signal transferred to every source pair, although
within-source training remained somewhat better on average.

### 7.2 Process F1 using the training source's threshold

| Train source \ Test source | GSM8K | MATH | OlympiadBench | Omni-MATH |
|---|---:|---:|---:|---:|
| GSM8K | 0.253 | 0.245 | 0.055 | 0.134 |
| MATH | 0.209 | 0.433 | 0.267 | 0.297 |
| OlympiadBench | 0.279 | 0.363 | 0.309 | 0.351 |
| Omni-MATH | 0.275 | 0.371 | 0.280 | 0.349 |

The mean off-diagonal Process F1 was 0.261, compared with a mean diagonal value of 0.336. The
off-diagonal range was wide, from 0.055 to 0.371. This contrasts with the uniformly strong AUROC
matrix: cross-domain ordering transferred better than threshold calibration and exact localization.

### 7.3 Exact complete-trace outcome

| Train source \ Test source | GSM8K | MATH | OlympiadBench | Omni-MATH |
|---|---:|---:|---:|---:|
| GSM8K | 0.345 | 0.255 | 0.261 | 0.228 |
| MATH | 0.488 | 0.431 | 0.271 | 0.290 |
| OlympiadBench | 0.512 | 0.392 | 0.303 | 0.326 |
| Omni-MATH | 0.476 | 0.426 | 0.282 | 0.337 |

The mean off-diagonal exact outcome was 0.351, close to the mean diagonal value of 0.354. This
metric is affected by the source-specific balance of erroneous and fully correct traces and should
be interpreted together with Process F1 rather than alone.

## 8. Native verdict baseline

Before interpreting an intervention on a behavioral score, the unmodified score should itself
distinguish valid from invalid boundaries. It did not.

| Native verdict metric | Result |
|---|---:|
| AUROC | 0.283 |
| Average precision | 0.384 |
| Balanced accuracy | 0.500 |
| Recall for invalid boundaries | 1.000 |
| Specificity for valid boundaries | 0.000 |
| Zero-margin classification accuracy | 0.500 |

All 256 baseline margins were positive, so the zero-margin decision rule classified every sampled
boundary as `INCORRECT`. Because the sample was deliberately balanced with 128 examples per class,
this produced 50% accuracy, 100% recall, and 0% specificity.

Moreover, the mean margin was **5.860** for valid-so-far boundaries and **5.625** for invalid-so-far
boundaries. Therefore larger `INCORRECT`-minus-`CORRECT` scores were associated with the valid class,
which explains the below-chance AUROC of 0.283. The scalar verdict score used in this stage was not
a usable native error detector.

## 9. Causal intervention results

### 9.1 Learned-direction dose response

The table reports the mean paired change from each example's own unmodified margin. Positive values
mean increased preference for `INCORRECT`.

| Dose \(\alpha\) | Mean paired change | 95% interval |
|---:|---:|---:|
| -4 | +0.000399 | [-0.001075, +0.001845] |
| -2 | +0.000909 | [-0.000608, +0.002253] |
| -1 | +0.000402 | [-0.001117, +0.001973] |
| 0 | 0 | [0, 0] |
| +1 | +0.000475 | [-0.000919, +0.001977] |
| +2 | -0.000387 | [-0.002038, +0.001324] |
| +4 | -0.000110 | [-0.001690, +0.001380] |

Every nonzero-dose interval included zero. The mean dose slope was -0.000108 per unit \(\alpha\),
and Spearman rank monotonicity across the seven mean doses was -0.607. Both have the opposite sign
from the hypothesis that increasing the learned invalidity direction should monotonically increase
the `INCORRECT` margin. Across individual nonzero interventions, 47.1% had the hypothesized sign,
also below the 50% level expected from an uninformative symmetric sign pattern.

The same absence of a stable signed effect was present when the sample was divided by whether the
starting boundary was valid or invalid. The artifact contains one nominally positive source-and-dose
subgroup interval (GSM8K at \(\alpha=+1\)), but the dose curve was not monotonic and many subgroup
comparisons were inspected. It is not evidence for the preregistered overall causal hypothesis.

### 9.2 Comparison with random orthogonal directions

On the 32-example subset shared with the random-direction controls:

| Dose | Learned-direction mean | Mean across random directions | Empirical one-sided \(p\) |
|---:|---:|---:|---:|
| -4 | +0.002344 | +0.000905 | 0.810 |
| +4 | +0.001794 | +0.000533 | 0.286 |

Neither extreme learned-direction effect beat the distribution from 20 random orthogonal
directions. The minimum attainable conventional empirical value with 20 random directions and the
add-one calculation used here is 1/21, or approximately 0.0476.

### 9.3 What the causal result does and does not mean

The recorded intervention experiment supports the following narrow statement:

> At hidden-state index 23 and the marked reasoning boundary, additive movement along the learned
> probe direction, over the tested range of ±4 training projection standard deviations, did not
> produce the predicted change in the recorded teacher-forced verdict margin.

It does **not** establish any of the following stronger statements:

- that mathematical invalidity has no causal representation anywhere in the model;
- that no intervention location or intervention form could change behavior;
- that the model possesses a competent native verifier but ignores the decoded signal;
- that a statistically nonsignificant effect is exactly zero.

The most important reason for restraint is that the baseline verdict measure was degenerate and
ranked the two classes in the wrong direction. The experiment therefore failed the preregistered
precondition that the unmodified model exhibit measurable correctness-judgment behavior.

## 10. Conclusions supported by this stage

This stage supports four conclusions:

1. **Linear decodability:** Qwen2.5-Math-1.5B hidden states contain information that linearly
   separates valid-so-far from invalid-so-far mathematical reasoning prefixes.
2. **Partial temporal localization:** the signal usually detects that an error exists and often
   places it within one or two steps of the annotation, but exact first-error localization remains
   difficult and is systematically somewhat late.
3. **Partial domain generality:** source-specific probes rank boundaries above chance on every
   other ProcessBench source. Threshold-dependent Process F1 transfers less consistently, implying
   that ranking is more domain-stable than calibration.
4. **No observed effect in the tested intervention:** the preregistered additive intervention did
   not change the recorded verdict score in the predicted manner and did not beat random
   directions.

This stage does not support the claim that the learned direction participates in the model's own
correctness verdict. Because the baseline verdict readout was not behaviorally meaningful, it also
does not yet support the stronger negative claim that the decoded representation is causally inert.

## 11. Limitations of this stage

- Only one 1.5-billion-parameter model was tested, so the result does not establish universality
  across scales, training procedures, or architectures.
- ProcessBench was repartitioned for representation analysis. The numbers are not standard
  ProcessBench benchmark scores.
- `invalid_so_far` labels all steps after the first error as invalid, even if a later step is locally
  reasonable conditional on the earlier mistake.
- The current-step TF-IDF and position controls are useful but do not eliminate all textual,
  generator, trace-difficulty, or prompt-marker shortcuts.
- The activation was measured at an artificial end-of-step marker. Robustness to other token
  locations was not tested in this stage.
- Cross-domain cells do not have published grouped confidence intervals in this artifact package.
- Probe-minus-control differences do not have paired bootstrap intervals in this artifact package.
- The intervention altered one boundary token at one selected layer using one additive direction;
  it did not test counterfactual patching, removal, other token locations, or other layers.
- The teacher-forced verdict candidates have different text and potentially different tokenization.
  Given the degenerate baseline behavior, this outcome requires validation before another causal
  claim is attempted.
- The published package omits activation-shard exclusion logs, so exact extraction exclusions
  cannot be reported here.

## 12. Artifact map

All statements in this report are derived from the frozen configuration and the following files:

| File | Contents |
|---|---|
| `artifacts/qwen2.5-math-1.5b/experiment_config.yaml` | Exact completed-run configuration |
| `artifacts/qwen2.5-math-1.5b/extraction_identity.json` | Dataset fingerprint, model, dtype, and context limit |
| `artifacts/qwen2.5-math-1.5b/probes/layer_metrics.csv` | Validation and test metrics for all 29 hidden-state indices |
| `artifacts/qwen2.5-math-1.5b/probes/test_predictions.csv` | Held-out step labels, metadata, and scores |
| `artifacts/qwen2.5-math-1.5b/probes/test_group_bootstrap_summary.csv` | Whole-trace bootstrap intervals |
| `artifacts/qwen2.5-math-1.5b/probes/controls.csv` | Position, TF-IDF, and shuffled-label controls |
| `artifacts/qwen2.5-math-1.5b/probes/domain_transfer.csv` | Full four-source transfer results |
| `artifacts/qwen2.5-math-1.5b/probes/directions.npz` | Probe directions, scales, thresholds, and selected indices |
| `artifacts/qwen2.5-math-1.5b/interventions/individual.csv` | Per-example intervention outcomes |
| `artifacts/qwen2.5-math-1.5b/interventions/summary.csv` | Mean dose responses and standard errors |
| `artifacts/qwen2.5-math-1.5b/interventions/effect_statistics.csv` | Paired intervals, dose-shape statistics, and random controls |
| `artifacts/qwen2.5-math-1.5b/interventions/behavioral_verdict.json` | Unmodified verdict-readout performance |

The dataset fingerprint recorded for this run is
`447f0a4b35c5747a9f9a3dab1e70d43f71efd501497b8cec668b71337099784a`.
