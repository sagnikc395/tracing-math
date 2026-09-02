# Extended workshop follow-ups

## Status

The matched first-error transition probe and the boundary-location control are complete; results
below. Counterfactual activation patching is implemented and the 160-pair template has been
generated and drafted for annotation: 135 pairs carry a proposed corrected step and 25 pairs are
withheld because a minimal correction could not be proposed responsibly. Every row still has
`verified: false`; verification is a human act, and no patching artifact exists. These analyses
were designed after inspecting Experiment 1 and must be described as post-hoc.

## Matched first-error transition probe

For every erroneous trace whose first error occurs after step 0, the analysis forms the activation
difference from the last valid boundary to the annotated first-error boundary. It pairs this with a
transition from a fully correct trace in the same partition. Matching first prefers source and
generator, then minimizes relative-position, trace-length, and token-count differences.

An L2 logistic probe distinguishes error-onset transitions from correct-trace placebos. The
original problem-grouped train, validation, and test partitions remain fixed. Regularization and
layer are selected on validation AUROC. The held-out report includes AUROC, average precision,
paired accuracy, two-way bootstrap intervals over error pairs and reused correct-trace IDs, and
position, current-step TF--IDF, and shuffled-label controls.

### Result

The run produced 380 held-out error/placebo pairs. Validation selected hidden-state index 21 with
C = 0.01. On the test pairs, the transition probe separates error onsets from matched correct
transitions with AUROC 0.769 (two-way bootstrap 95% CI [0.713, 0.820]), average precision 0.741
([0.672, 0.816]), and paired accuracy 0.753 ([0.676, 0.824]).

The matched controls stay below the transition probe: position AUROC 0.63, current-step TF--IDF
0.67, and shuffled-label 0.58. The difference between two boundaries of the same trace is therefore
not explained by absolute position, by the words of the destination step alone, or by probe-procedure
artifacts.

This supports a representation change at the annotated onset: the step-to-step difference at a real
first error is linearly distinguishable from a metadata-matched transition in a correct trace. It
does not show that the change is abrupt in time; the error-aligned trajectory from the CPU follow-up
remains gradual, and a linearly decodable transition is compatible with the score continuing to rise
after the onset.

## Boundary-location control

One forward pass records hidden states at the last natural token of each written step and at the
last token of its artificial `END_STEP` marker. The comparison fixes the Experiment 1 layer and
regularization value. Thresholds are fitted separately on validation data, then evaluated once on
the original test partition. A whole-trace paired bootstrap reports natural-token minus marker-token
differences in AUROC and Process F1.

### Result

At the frozen setting (index 23, C = 0.01, threshold 0.645), probing the natural last token of each
step reproduces the marker-based result almost exactly:

| Location | Step AUROC | Process F1 | Exact first error | Within 1 step |
| --- | ---: | ---: | ---: | ---: |
| Step content (natural token) | 0.868 | 0.419 | 0.329 | 0.572 |
| `END_STEP` marker | 0.866 | 0.404 | 0.289 | 0.593 |

The paired whole-trace bootstrap differences are AUROC +0.002 (95% CI [-0.007, +0.011]) and Process
F1 +0.026 ([-0.020, +0.075]); both intervals include zero. The invalidity signal does not require
the artificial step marker, so the paper's claim does not need to be narrowed to marked
step-summary states. Localization metrics remain moderate at either location, consistent with the
gradual-transition account.

## Counterfactual activation patching

The template contains a problem, the prefix before the first error, the original erroneous step,
and an empty corrected-step field. A pair enters the experiment only after the corrected step has
been checked and `verified` is set to `true`.

At the frozen intervention layer, the model records the real boundary state for each correct and
erroneous prefix. It then replaces the correct state with the erroneous state and performs the
reverse replacement. The outcome is the corrected single-token Yes-minus-No verdict score. The
report contains within-pair baseline separation and both directional patch effects with paired
bootstrap intervals.

Patching is interpretable only if the unmodified verdict separates the verified correct and error
prefixes. If it does not, the result is reported as another failed behavioral prerequisite rather
than evidence about causal use.

### Annotation state

The template generator selected 160 erroneous traces deterministically (35 `math`, 69 `omnimath`,
43 `olympiadbench`, 13 `gsm8k`). For each pair, the drafted `corrected_step` repairs the annotated
first error while leaving the prefix untouched; `annotation_notes` records what the original step
had wrong. The 25 withheld pairs fall into three groups: the error is a wrong final result whose
repair needs a full solution rather than a local edit (for example `omnimath-160`, `omnimath-980`),
the original step is methodologically broken in a way that changes the whole solution route (for
example `olympiadbench-635`, `omnimath-366`), or the item depends on a figure that could not be
checked from text (for example `math-394`, `omnimath-714`).

A pair enters the experiment only when a reviewer has checked the correction and set `verified` to
`true`. A verified pair must have a nonempty corrected step distinct from the error step, which the
loader enforces. Until then, patching stays unrun; no activation patching number may be reported.

**Status: pending.** Corrections are drafted, not verified. No artifact has been produced and no
result is reported.

## Commands

```bash
uv run math-error-extended-followup --config configs/experiment3.yaml prepare-counterfactual-template
uv run python data/processed/apply_corrections.py   # reapply drafts after re-generating the template
uv run math-error-extended-followup --config configs/experiment3.yaml run-counterfactual-patching
```
