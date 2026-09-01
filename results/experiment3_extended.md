# Extended workshop follow-ups

## Status

The code and Colab stages are implemented. No result is reported here until the corresponding
artifact exists. These analyses were designed after inspecting Experiment 1 and must be described
as post-hoc.

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

A positive result supports a representation change at the annotated onset. A null result favors
the narrower account that the frozen invalidity score changes gradually without a linearly sharp
transition.

## Boundary-location control

One forward pass records hidden states at the last natural token of each written step and at the
last token of its artificial `END_STEP` marker. The comparison fixes the Experiment 1 layer and
regularization value. Thresholds are fitted separately on validation data, then evaluated once on
the original test partition. A whole-trace paired bootstrap reports natural-token minus marker-token
differences in AUROC and Process F1.

If the natural-token result is similar, the signal does not require the marker. If it is much
weaker, the paper will narrow its claim to marked step-summary states.

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

## Commands

```bash
uv run math-error-extended-followup --config configs/experiment3.yaml validate-config
uv run math-error-extended-followup --config configs/experiment3.yaml fit-transition-probe
uv run math-error-extended-followup --config configs/experiment3.yaml extract-boundary-controls
uv run math-error-extended-followup --config configs/experiment3.yaml analyze-boundary-controls
uv run math-error-extended-followup --config configs/experiment3.yaml prepare-counterfactual-template
uv run math-error-extended-followup --config configs/experiment3.yaml run-counterfactual-patching
```
