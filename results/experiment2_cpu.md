# CPU-only follow-up: temporal validity of the frozen error score

## Status and scope

This follow-up was designed after the Experiment 1 results were known. It is a post-hoc robustness
analysis, not a replacement confirmatory experiment. It reuses the selected-layer test predictions
and intervention records without fitting another hidden-state probe. No language-model forward pass
or activation tensor is required.

The analysis uses 4,985 step predictions from 669 held-out traces. The selected hidden-state index
remains 23 and the validation-selected threshold remains 0.645. All resampling is grouped by trace.
The configuration fixes 5,000 temporal randomizations, 2,000 bootstrap draws, and seed 42.

The follow-up asks four questions:

1. Does the frozen score align with the annotated error more closely than the same trajectories at
   random within-trace offsets?
2. Is the score change at the first error larger than ordinary drift in matched correct traces?
3. Does discrimination survive removal of stable trace-level score offsets?
4. Which traces account for the detector's localization failures?

The causal records support a separate sensitivity calculation. That calculation does not validate
the failed verdict readout.

## Temporal randomization

Each score trajectory was circularly shifted by an independently sampled offset. This preserves
the trace's scores, threshold-crossing frequency, and most of its local shape while breaking the
observed alignment with the annotated first-error step. The original threshold and annotations were
not changed.

| Metric | Observed | Shifted null mean | 95% null interval | Permutation p |
| --- | ---: | ---: | ---: | ---: |
| Exact first-error localization | 0.289 | 0.164 | [0.137, 0.192] | 0.0002 |
| Within one step | 0.593 | 0.436 | [0.405, 0.468] | 0.0002 |
| Within two steps | 0.727 | 0.623 | [0.595, 0.650] | 0.0002 |
| Process F1 | 0.393 | 0.259 | [0.223, 0.292] | 0.0002 |
| Mean absolute error among detections | 1.397 | 2.075 | [1.956, 2.191] | 0.0002 |

The observed score contains temporal information beyond its within-trace score distribution and
shape. The result does not eliminate position as an explanation by itself because circular shifts
also break alignment with absolute step position. The matched placebo analysis addresses ordinary
position-dependent drift more directly.

## Error-aligned trajectory and matched placebo transitions

The mean frozen score rose from 0.341 on the last pre-error step to 0.559 at the annotated error.
It continued to rise after the error: 0.680 one step later and 0.731 two steps later. The score is
not a short impulse at the boundary. It looks like a transition into a persistent state.

Among the 380 erroneous traces whose first error occurred after step 0, the mean within-trace onset
jump was 0.257 [0.234, 0.280]. Each onset was matched to a transition from a fully correct trace
with the same source and generator and similar trace length, token count, and relative position.
The 380 pairs used 139 unique correct traces, so inference resampled erroneous and placebo trace IDs
as separate clusters.

| Comparison | Mean | 95% interval | Traces |
| --- | ---: | ---: | ---: |
| Error-onset jump | 0.257 | [0.234, 0.280] | 380 erroneous |
| Matched correct-trace jump | 0.113 | [0.093, 0.132] | 139 unique controls |
| Error onset minus matched transition | 0.144 | [0.096, 0.193] | 380 pairs |
| Error onset minus the trace's earlier mean jump | 0.066 | [0.033, 0.097] | 274 erroneous |

Correct solutions also show positive drift, but the change at a real annotated error is larger. The
onset jump also exceeds the same trace's average earlier change. This supports a genuine change near
the annotation without implying that the threshold localizes it reliably.

## Removing stable trace-level offsets

The analysis next subtracts each erroneous trace's first-step score from all later scores. This is
a fixed-score transformation; it does not refit the probe. Traces with an error at step 0 are
excluded because they have no valid first-step reference.

| Metric | Estimate | 95% interval | Traces |
| --- | ---: | ---: | ---: |
| Pooled raw AUROC | 0.881 | [0.862, 0.900] | 380 |
| Pooled first-step-centered AUROC | 0.881 | [0.862, 0.900] | 380 |
| Mean AUROC calculated separately within each trace | 0.968 | [0.958, 0.976] | 380 |

Stable differences between traces do not explain the selected probe's pre-error versus post-error
ranking. The high within-trace AUROC is compatible with imperfect first-error localization because
the score can order the two regions correctly while crossing a global threshold early or late.

## Where the detector fails

The detector's main weakness is false alarms on long correct traces. Complete-trace accuracy falls
from 0.491 in the shortest trace-length quartile to 0.264 in the longest. Exact localization among
erroneous traces falls from 0.320 to 0.230, while correct-trace rejection falls more sharply, from
0.719 to 0.350.

Token count shows the same pattern. Correct rejection is 0.792 in the lowest token-count quartile
and 0.231 in the highest. Exact localization among erroneous traces stays between 0.266 and 0.331.
This asymmetry points to accumulated score drift and global-threshold calibration, rather than a
complete absence of temporal information, as the main source of long-trace failures.

Results also differ across domains. Complete-trace accuracy is 0.512 on GSM8K and 0.500 on MATH,
compared with 0.298 on OlympiadBench and 0.358 on Omni-MATH. Most of the gap again comes from correct
rejection: 0.786 and 0.723 on the first two sources, versus 0.463 and 0.467 on the latter two.
Generator-level rows are reported only when at least 20 held-out traces are available.

These subgroup comparisons are descriptive and post-hoc. They should not be read as a set of
independent hypothesis tests.

## Equally tuned joint shortcut controls (2026-09-02 addition)

The workshop critique (C1) noted that the original shortcut controls were fit at a fixed
regularization value and on train data only, while the hidden probe selected C from four values on
validation and refit on train plus validation. The shortcut analysis now gives every control the
identical selection budget, refits the selected model on train plus validation, and fits each
control with sample weights inverse to boundary counts so every trace contributes equal total
training loss. Two joint baselines combine prefix text, structural metadata, and (in the stronger
variant) final-answer correctness.

| Control | C | AUROC | Process F1 | Exact | Correct rejection |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hidden state (frozen) | 0.01 | 0.866 | 0.393 | 0.289 | 0.612 |
| Prefix TF-IDF | 10.0 | 0.751 | 0.274 | 0.293 | 0.477 |
| Structural metadata | 10.0 | 0.776 | 0.168 | 0.164 | 0.194 |
| Metadata plus final outcome | 1.0 | 0.854 | 0.262 | 0.335 | 0.646 |
| Joint text + metadata | 1.0 | 0.810 | 0.210 | 0.208 | 0.278 |
| Joint text + metadata + outcome | 1.0 | 0.874 | 0.294 | 0.430 | 0.895 |

Paired whole-trace bootstrap intervals (hidden minus control, 2,000 draws):

| Control | Metric | Difference | 95% interval |
| --- | --- | ---: | ---: |
| Joint text + metadata | AUROC | 0.056 | [0.037, 0.073] |
| Joint text + metadata | Process F1 | 0.183 | [0.136, 0.229] |
| Joint text + metadata + outcome | AUROC | -0.008 | [-0.028, 0.013] |
| Joint text + metadata + outcome | Process F1 | 0.099 | [0.040, 0.158] |
| Joint text + metadata + outcome | Exact error | 0.113 | [0.064, 0.163] |
| Joint text + metadata + outcome | Correct rejection | -0.283 | [-0.347, -0.218] |

The substantive conclusion changed: under an equally tuned budget, the joint model that combines
prefix text, structural metadata, and final-answer correctness reaches AUROC 0.874, numerically
above the hidden state's 0.866, with a paired difference whose interval includes zero. The hidden
probe's surviving advantage is task-specific: trace-level localization (Process F1 and exact
first-error accuracy) and a lower false-alarm rate on correct traces than the joint outcome model,
which crosses threshold on almost every trace. Within final-answer-correctness strata, the hidden
AUROC advantage is 0.073 [0.039, 0.107] among correct-outcome traces and -0.001 [-0.022, 0.021]
among incorrect-outcome traces.

Final-answer correctness is a benchmark annotation computed against the reference solution, not an
online signal; the joint outcome models are diagnostic upper bounds on nuisance predictability,
not deployable baselines.

## Onset-task eligibility audit (2026-09-02 addition)

The transition analysis requires a noninitial first error. Of the 669 test traces, 52 erroneous
traces (7.8%) have `first_error == 0` and are excluded from every transition and matched-placebo
analysis; 380 erroneous traces with a later onset and 237 fully correct traces remain eligible.
The excluded traces span all four sources. The audit is written to
`onset_eligibility_audit.csv`.

## Environment record (2026-09-02 addition)

The summary now records the exact software environment of the analysis run (Python, NumPy,
pandas, SciPy, scikit-learn, matplotlib versions and platform) in `summary.json` and
`results.md`, addressing the reproducibility checklist gap noted by the critique.

## Sensitivity of the failed causal assay

Across the six nonzero learned-direction doses, the approximate minimum detectable mean effect at
80% power is 0.00208 to 0.00229 verdict-margin units. This is 0.0065 to 0.0071 standard deviations
of the unmodified score across examples. The observed mean changes range from -0.00039 to 0.00091.

The intervention experiment was statistically capable of resolving small mean shifts under the
normal paired-mean approximation. Its behavioral outcome was still invalid: baseline AUROC was
0.283 and specificity was zero. Power cannot make that readout a measure of useful error judgment.

## Interpretation

The combined evidence supports a narrower and more specific account than either "the model detects
its errors" or "the probe is only a shortcut." The selected hidden-state score changes near the
human-annotated error, survives removal of stable trace offsets, and aligns with the annotation more
closely than circularly shifted versions of the same trajectories. The change is gradual. The score
often continues upward after the error, and ordinary upward drift also occurs in correct solutions.
A single global threshold then produces late detections and false alarms, especially on long traces.

Experiment 1 and this follow-up establish accessible, partially localized information. They do not
show that the model uses that information in a functioning verdict computation. The causal assay
failed its behavioral prerequisite even though its mean-effect confidence intervals were narrow.

## Reproduction

Run the complete follow-up locally with:

```bash
math-error-cpu-followup --config configs/experiment2.yaml
```

On the development machine, the full run took about 37 seconds after the one-time Matplotlib font
cache was initialized. Outputs are written under `artifacts/experiment2_cpu/`; generated artifacts
remain outside Git.
