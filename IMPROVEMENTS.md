# Experimental improvements needed for an accept-level paper

## Bottom line

No experiment can guarantee acceptance. The shortest credible path is to make the central negative
result correct, conditional, and independently checked. The paper does not need a larger collection
of loosely related analyses. It needs three things:

1. a clean metric and artifact audit;
2. a direct test of whether hidden states add value after nuisance features are known;
3. one confirmatory replication run after the complete audit protocol is frozen.

The repository already contains most of the supporting work: tuned joint nuisance controls,
trace-level paired intervals, calibration outputs, matching diagnostics, a boundary-location control,
and code for transition-matching sensitivity. Do not rerun these from scratch. Finish the missing
comparisons, repair the inconsistent reporting, and keep every claim tied to the experiment that
actually tests it.

The target should be a paper that can defend this sentence:

> Under a frozen evaluation protocol, strong invalid-prefix decodability does not by itself establish
> precise onset localization or behavioral use; the amount of predictive value left after visible
> nuisance features are included is measured directly and reported with paired uncertainty.

Whether the conditional hidden-state result is positive or null, the paper can work. The story must
follow the result.

## What is already done

These items should be treated as completed evidence, subject to the metric repair below:

- the persistent *invalid-so-far* target and first-crossing evaluation;
- problem-grouped train, validation, and test partitions;
- hidden-only probes across layers;
- tuned prefix TF-IDF, structural metadata, joint text-plus-metadata, and final-outcome oracle
  controls;
- whole-trace paired bootstrap intervals;
- within-trace centering and source-transfer analyses;
- hidden-score calibration outputs, including Brier score, log loss, and ECE;
- transition matching diagnostics, including reuse and covariate balance;
- the natural-token versus `END_STEP` boundary-location control;
- a validity gate that correctly prevents interpretation of the failed causal pilot.

Two implemented analyses remain unexecuted or unpublished:

- `run-transition-matching-sensitivity` is implemented, but
  `artifacts/experiment3_extended/transition_probe/matching_sensitivity.csv` is absent;
- calibration numbers exist in artifacts and results notes, but the PDF does not present enough
  evidence to support its repeated calibration claim.

## Priority order

| Priority | Work item | Main question | Why it changes the decision | Cost |
|---|---|---|---|---|
| **P0** | E0: metric and artifact audit | Are the headline localization values internally consistent? | A broken main table is independently rejectable. | Low; CPU only |
| **P0** | E1: conditional hidden-state comparison | Does the hidden representation add out-of-sample value after nuisance features? | This directly tests the paper's central incremental-value claim. | Moderate; cached activation shards required for the strongest version |
| **P0 if compute is available** | E2: frozen confirmatory replication | Does the audit survive on a model or dataset not used to design it? | The current headline analyses are post-hoc on an inspected test set. | One new extraction plus CPU analysis |
| **P1** | E3: contextual text-only baseline | Is the localization advantage more than semantic processing of visible text? | TF-IDF is too weak to isolate hidden-specific value. | Moderate |
| **P1** | E4: training-weight and split sensitivity | Is the result stable to trace weighting and group assignment? | Current uncertainty conditions on one split and an unmatched hidden-probe objective. | Moderate; no new labels |
| **P1** | E5: transition matching sensitivity | Is AUROC 0.769 driven by reused placebos or residual imbalance? | This is the paper's most onset-specific result. | Low once shards are available |
| **P2** | E6: calibration and paired error decomposition | What fails: probability calibration, threshold transfer, or error overlap? | It repairs claims already present in the abstract and conclusion. | Low; existing predictions suffice |
| **P3** | E7: self-generated traces | Does the model monitor its own generation rather than read fixed traces? | Needed only if the paper keeps the self-monitoring framing. | High annotation and GPU cost |

## E0. Repair the metric and artifact pipeline

**Status:** implemented on 2026-09-02. The audited CPU artifacts and manuscript now use the
canonical metric schema, generate the headline comparison from held-out predictions, assert the
Process F1 identity, and report paired correct-trace overlap.

This is a correctness repair, not a new scientific experiment. Section 3.2 defines Process F1 as the
harmonic mean of erroneous-trace exact localization and correct-trace rejection. Table 2 does not
obey that identity. For the joint outcome model, the displayed values `Exact = 0.430` and
`Correct rejection = 0.895` would give Process F1 about 0.581, not the reported 0.294. The reported
0.294 is compatible with erroneous-trace exact localization near 0.176. The current table appears to
mix complete-trace accuracy with erroneous-trace exact accuracy.

### Required work

1. Give the metrics unambiguous names in artifacts and code:

   - `error_exact`: exact onset among erroneous traces;
   - `correct_rejection`: no alarm among correct traces;
   - `process_f1`: harmonic mean of those two values;
   - `complete_accuracy`: exact trace outcome across all traces.

2. Generate every row of the main comparison table from the same schema. Do not hand-copy values
   from separate result files.
3. Add an assertion for every row:

   `process_f1 == 2 * error_exact * correct_rejection / (error_exact + correct_rejection)`.

4. Add a regression test that checks the generated table, not only the metric helper.
5. Recompute the paired hidden-minus-control differences from the corrected columns.
6. Produce a correct-trace overlap table with four counts: both models reject, hidden-only alarm,
   nuisance-only alarm, and both alarm. Keep the abstract's claim about complementary false alarms
   only if this paired table supports it.
7. Remove or correct the Section 4.1 sentence claiming that a model with correct rejection 0.895
   alarms on most correct traces.

### Completion criterion

One machine-readable summary must reproduce the abstract, Table 2, Section 4.1, and the conclusion.
`pytest`, Ruff, and `git diff --check` must pass after the regenerated paper is built.

## E1. Test incremental value with nested models

**Status:** implemented and covered by synthetic end-to-end tests on 2026-09-02. The real run is
pending because `artifacts/experiment1/qwen2.5-math-1.5b/activation_shards/` is not present in the
workspace. Do not substitute the in-sample `fit_predictions.csv` probe scores.

Hidden-only AUROC and nuisance-only AUROC do not measure incremental value. Two classifiers can have
the same AUROC and make different errors. The required experiment compares a nuisance model with the
same nuisance model after hidden features are added.

### Conditions

Fit these five conditions:

| ID | Features | Role |
|---|---|---|
| N | Prefix text plus contemporaneous structural metadata | Deployable nuisance baseline |
| N+H | N plus the selected hidden representation | Direct conditional test without future information |
| O | N plus final-answer correctness | Diagnostic oracle baseline |
| O+H | O plus the selected hidden representation | Conditional test given the oracle outcome |
| H | Hidden representation only | Link to the current paper |

Final-answer correctness must stay in the oracle branch. It is not available to an online boundary
detector.

### Primary protocol

- Use the existing problem groups. No row from the same normalized problem may cross partitions.
- Standardize hidden features using training data only.
- Use the same TF-IDF vocabulary construction, metadata encoding, regularization family, C grid,
  class balancing, and trace-equal sample weighting for N, N+H, O, and O+H.
- Select regularization on validation data. Select any first-crossing threshold on validation data.
- Do not use the current `fit_predictions.csv` scores as clean meta-training features: those scores
  were produced by a model refit on train plus validation and are in-sample for those rows.
- Prefer direct early fusion with raw hidden vectors. If score stacking is used as a sensitivity
  analysis, generate out-of-fold hidden scores for every meta-training row.
- Save all held-out predictions so every comparison can be paired by trace.

### Metrics

Treat the following as co-primary descriptions of different questions:

- AUROC and average precision for boundary ranking;
- log loss and Brier score for predictive information and calibration;
- `error_exact`, `correct_rejection`, Process F1, within-one, and within-two for localization;
- complete-trace accuracy for the combined trace outcome.

Report N+H minus N and O+H minus O with whole-trace paired intervals. If the paper keeps the word
"information," a proper score such as held-out log loss must support the claim. If only AUROC is
tested, say "incremental ranking value."

### Decision rule

Before running the confirmatory version, write down the smallest improvement that would matter for
the paper's intended use. A reasonable starting point is 0.02 AUROC, but the authors must justify and
freeze the margin before seeing the result. Then interpret the experiment as follows:

- **No useful increment:** the upper confidence bound for N+H minus N remains below the frozen
  practical margin, and proper scores show no material gain. This supports the negative claim.
- **Complementary signal:** N+H improves ranking, proper scores, or localization. Replace "no
  incremental information" with a precise account of where hidden features help.
- **Mixed result:** pooled ranking is unchanged but localization improves, or the reverse. Make that
  metric-specific contrast the result. Do not average it into a binary verdict.

### Required artifacts

- resolved configuration and a hash of it;
- selected feature blocks, regularization, layer, and threshold for each condition;
- validation selection table;
- held-out predictions;
- paired point estimates and confidence intervals;
- a short result record stating which claim is supported and which is not.

## E2. Run one frozen confirmatory replication

**Status:** the current paper has no untouched confirmatory evaluation for the post-hoc audit.

The joint controls, temporal tests, weighting checks, and subgroup analyses were designed after the
original test result was inspected. More bootstrap draws on that split will not solve this. Freeze the
complete protocol and evaluate it once on a new model or a genuinely untouched dataset.

### Recommended route

Use a second reader model on the same ProcessBench traces. A different model family is more
informative; a larger checkpoint from the same Qwen math family is still useful as a scale
replication, but it does not establish family transfer.

Before extraction, commit or otherwise time-stamp a resolved manifest containing:

- target model and revision;
- prompt and chat template;
- dtype, context length, and boundary location;
- group split rule and seed;
- layer-selection rule and C grid;
- all N, N+H, O, O+H, and H feature definitions;
- training weights;
- primary metrics and the practical increment margin;
- bootstrap unit and draw count;
- all stopping and exclusion rules.

Run the pipeline once. Do not add a new control because the first result is inconvenient. Any added
analysis belongs in an explicitly exploratory section.

### Minimum report

- full data flow and exclusions;
- hidden-only decodability;
- N versus N+H and O versus O+H;
- first-crossing localization components;
- calibration or threshold-transfer results under the definition adopted in E6;
- a compact comparison with the original model.

### Interpretation

- **Same qualitative pattern:** the paper can claim a replicated audit pattern across two reader
  models, while avoiding prevalence language.
- **Different pattern:** model dependence becomes the result. Report it rather than selecting the
  model that fits the original story.
- **Extraction or data failure:** do not use a partial run as evidence. Keep the original analysis
  exploratory and narrow the claims.

If a new extraction is impossible, repeated grouped splits on the current model are still useful for
robustness, but they are not an untouched replication.

## E3. Add one contextual text-only baseline

**Status:** missing; high value for the surviving localization claim.

Prefix TF-IDF tests lexical shortcuts. It does not test whether an ordinary semantic representation
of the visible text can localize the error. Choose one contextual text encoder before evaluating the
test set. Encode only the problem and prefix available at the current boundary, then fit the same
regularized linear head and threshold-selection procedure used elsewhere.

The baseline must not receive final-answer correctness, future steps, or target-model hidden states.
Report its boundary ranking and first-crossing metrics. If it matches the hidden probe, the result is
about visible-text semantics rather than a hidden-specific error variable. If it does not, the paper
has a stronger basis for saying that the target representation contributes something beyond TF-IDF.

Do not turn this into a model zoo. One frozen contextual baseline with a stated selection rule is
enough.

## E4. Match training weights and measure split sensitivity

**Status:** partly done.

The current follow-up gives nuisance controls inverse-boundary-count weights, while the original
hidden probe was class-balanced at the boundary level. The existing trace-equal sensitivity changes
evaluation weights for frozen predictions; it does not refit the hidden direction under a
trace-equal training objective.

### Required work

1. Refit H, N, N+H, O, and O+H with the same class and trace weighting.
2. Keep the original boundary-weighted hidden probe as a sensitivity row, not as the matched primary
   comparison.
3. Repeat the full group split, layer selection, C selection, refit, and threshold selection for a
   predetermined list of split seeds. Five is a minimum; ten is preferable if the cached activations
   make this cheap.
4. Report the distribution of each metric and conditional difference across seeds. Do not select the
   best seed.

This analysis measures split and selection sensitivity. It does not erase the post-hoc status of the
original dataset.

## E5. Finish the implemented transition-matching sensitivity

**Status:** implemented, not run in the current artifact tree.

Run:

```bash
uv run math-error-extended-followup \
  --config configs/experiment3.yaml \
  run-transition-matching-sensitivity
```

The command needs the activation shards, which are not present in the current local artifact
directory. Run it in the extraction environment. It should compare:

- the frozen matching-with-reuse protocol;
- inverse-reuse training weights;
- one-to-one matching without replacement.

The current implementation writes point estimates. Extend the report to include uncertainty and
matching balance for each variant. Record pair counts and unique placebo counts, because one-to-one
matching will change effective sample size.

Keep the transition result exploratory even if all variants agree. Agreement shows that the AUROC
0.769 estimate is not an obvious artifact of control reuse; it does not create an untouched test.

## E6. Define calibration and decompose paired errors

**Status:** most inputs already exist.

The paper currently uses "calibration" for several ideas: probability reliability, global-threshold
performance, and threshold transfer across sources and lengths. Pick one definition or report them
separately.

### If the claim is probability calibration

Report held-out Brier score, log loss, ECE with bin sensitivity, and a reliability diagram. Give
source- and length-stratified estimates only when the groups are large enough to interpret. State
that the estimates are post-hoc and conditional on the fitted model.

### If the claim is threshold stability

Report the validation-selected threshold's Process F1, erroneous-trace exact accuracy, and correct
rejection by source and length. Show whether source-specific or length-specific validation thresholds
improve the held-out result. Call this threshold transfer, not probability calibration.

Use the paired correct-trace overlap table from E0 to test any claim that the hidden and nuisance
models catch different cases. Aggregate rejection rates alone do not establish complementarity.

If space is tight, remove "calibration" from the abstract and keep the existing threshold-transfer
result. This is better than adding a crowded analysis that does not affect the main conclusion.

## E7. Decide between self-monitoring and verifier framing

**Status:** a framing decision comes before an expensive experiment.

The target checkpoint reads ProcessBench traces generated by other systems. That is a hidden-state
verifier experiment, not direct evidence that a model detects mistakes during its own generation.
The low-cost fix is to state this throughout the paper.

Run a self-generated-trace experiment only if self-monitoring is central to the desired claim. It
would require the exact target checkpoint to generate solutions under a frozen decoding protocol,
independent first-error annotation, and a repeat of E1. Automatic transfer of ProcessBench labels or
unverified synthetic error locations is not enough.

This is P3 because it introduces new data collection and annotation. A precise verifier paper can be
accepted without it; a self-awareness paper cannot.

## Work that should not consume the next cycle

- Do not add more weak metadata baselines. The missing comparison is conditional, not another
  separate classifier.
- Do not increase bootstrap draws on the inspected test split and call the result confirmatory.
- Do not run more random intervention directions. They cannot replace a positive control for assay
  sensitivity.
- Do not run counterfactual patching until corrections are human-verified and the unmodified readout
  separates valid and invalid prefixes.
- Do not test many contextual encoders and report the best one.
- Do not add a third model before E0 and E1 are correct.
- Do not hide a positive N+H result because the intended paper is a negative-results paper.

## Recommended execution sequence

1. Complete E0 and regenerate all existing headline numbers.
2. Freeze the E1 design, feature blocks, weighting, metrics, and practical margin.
3. Implement and test E1 on development partitions without looking at new confirmatory outputs.
4. Freeze the E2 manifest.
5. Run E2 once and lock its predictions.
6. Run E3 and E5 only after P0 artifacts are safe.
7. Use E4 and E6 to describe sensitivity and failure shape, not to rescue a preferred conclusion.
8. Rewrite the abstract and conclusion after the experiment ledger is complete.

## Definition of done

The experimental package is ready for the paper when all of the following are true:

- [ ] Every displayed Process F1 reproduces from displayed `error_exact` and
  `correct_rejection`.
- [ ] The abstract, table, prose, and machine-readable summary use the same values.
- [ ] N versus N+H and O versus O+H are evaluated with paired uncertainty.
- [ ] All compared models use documented and matched training weights and selection budgets.
- [ ] No in-sample fitted score is used as a meta-model training feature.
- [ ] A frozen confirmatory model or dataset has been evaluated once, or the entire central audit is
  labeled exploratory.
- [ ] One contextual text-only baseline tests the surviving hidden-specific localization claim.
- [ ] Split sensitivity is reported without selecting a favorable seed.
- [ ] Transition matching sensitivity has been run and includes pair counts and uncertainty.
- [ ] Calibration is defined and measured, or the term is removed.
- [ ] Reader-model and self-monitoring claims are not mixed.
- [ ] The causal section makes no use claim while the behavioral readout remains invalid.
- [ ] Resolved configs, data/model identities, selection records, predictions, metrics, and intervals
  are saved for every new experiment.
- [ ] `uv run pytest`, `uv run ruff check src tests`, the paper build, and a clean-environment
  reproduction check pass.

## Result record to fill after each run

```text
Experiment ID:
Concern addressed:
Status: completed / failed / partial
Protocol hash:
Model and data identity:
Partitions and exclusion counts:
Baselines and controls:
Training weights:
Selection rule:
Primary metrics:
Number of traces and boundaries:
Point estimates and uncertainty:
Unexpected findings:
Artifacts:
Claim supported:
Claim not supported:
Required manuscript change:
```

## Policy note

This is an internal experiment plan, not a submission-ready rebuttal. The authors must verify the
protocols and decide which work enters the manuscript. The
[NeurIPS 2026 Main Track Handbook](https://neurips.cc/Conferences/2026/MainTrackHandbook) makes
authors responsible for the correctness and originality of all text, figures, references, and
methods. It does not require disclosure for basic editing assistance, but important, original, or
non-standard use of agents or LLMs in the scientific method should be documented in the experimental
setup. The final checklist should describe any assistance that affected methodology rather than
copying this planning note into the paper.
