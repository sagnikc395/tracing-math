# Experimental improvements needed for an accept-level paper

## Bottom line

The paper has improved. The nested comparison (E1) now shows that hidden states add incremental
predictive information beyond tuned nuisances on the inspected split. The metric audit (E0) is
resolved. The self-monitoring language (E7) is fixed. The prose avoids significance inflation.

The paper can be completed without counterfactual activation patching. That analysis requires
human verification of mathematical corrections, and no automated check can establish that a drafted
correction is valid. Do not bypass that gate or report a patching result.

The remaining evidence gaps are optional strengthening work, not blockers for closing the current
paper:

1. a contextual semantic baseline to test whether the gain is more than a TF-IDF limitation;
2. a frozen replication or outer-fold evaluation to test whether the nested gain survives an
   untouched protocol.

The repository contains the completed primary run and several post-hoc diagnostics. Close the paper
by labeling the remaining analyses accurately, freezing the supported claim, and running the paper
checks. The story must follow the evidence already collected.

The target should be a paper that can defend this sentence:

> Under a frozen evaluation protocol, hidden states add held-out predictive information beyond
> equally tuned text and metadata nuisances; the gain is conditional on one model, one split, and
> one benchmark, and does not establish localization, calibration, or causal use.

Whether the contextual baseline matches or does not match the hidden score, the paper can work.
The story must follow the result.

## What is already done

These items are completed and reflected in the revised paper:

- E0: metric and artifact audit -- Table 2 regenerated from one schema, Process F1 identity
  verified, erroneous-trace exact accuracy separated from complete accuracy;
- E1: nested nuisance-plus-hidden comparison -- N+H improves AUROC by +0.0587 [0.0414, 0.0745]
  over N, log loss by -0.0798, Process F1 by +0.1866; O+H improves AUROC by +0.0348;
- E4 (partial): equally tuned controls with trace-equal training weights, refit on train plus
  validation, same C grid and threshold selection as the hidden probe;
- E7: self-monitoring language replaced with verifier framing throughout; model reads fixed traces,
  does not generate them;
- the persistent invalid-so-far target and first-crossing evaluation;
- problem-grouped train, validation, and test partitions;
- hidden-only probes across layers;
- whole-trace paired bootstrap intervals;
- within-trace centering and source-transfer analyses;
- transition matching diagnostics, including reuse and covariate balance;
- the natural-token versus END_STEP boundary-location control;
- a validity gate that correctly prevents interpretation of the failed causal pilot;
- onset-task eligibility audit (52 step-0 traces excluded, documented);
- environment record for reproducibility.

Two implemented analyses remain unexecuted:

- `run-transition-matching-sensitivity` is implemented but needs activation shards;
- calibration was claimed in the original paper but is not tested in the revised version.

Counterfactual activation patching is a separate optional analysis. It is intentionally closed as
unrun because its drafted corrections are not human-verified. The template and drafts may remain in
`data/processed/` as an audit trail, but they must not be counted as results or used to block the
paper.

## Priority order

| Priority | Work item | Main question | Why it changes the decision | Cost |
|---|---|---|---|---|
| **P0** | E3: contextual text-only baseline | Is the localization advantage more than semantic processing of visible text? | TF-IDF is too weak to isolate hidden-specific value. A contextual encoder matching the hidden score reduces the contribution to a TF-IDF limitation. | Moderate |
| **P0** | E2: frozen confirmatory replication | Does the nested gain survive on an untouched split or model? | The current headline analyses are post-hoc on an inspected test set. More bootstrap draws do not fix this. | One new extraction plus CPU analysis |
| **P1** | E5: transition matching sensitivity | Is AUROC 0.769 driven by reused placebos or residual imbalance? | This is the paper's most onset-specific result. | Low once shards are available |
| **P1** | E4 (complete): split sensitivity across seeds | Is the nested gain stable across group assignments? | Current uncertainty conditions on one split. | Moderate; no new labels |
| **P2** | E6: calibration or remove the term | What fails: probability calibration, threshold transfer, or error overlap? | The revised paper dropped the calibration claim, which is correct. But the term may need to be retracted from older versions of the manuscript, or replaced with proper scores. | Low; existing predictions suffice |
| **P3** | Self-generated traces | Does the model monitor its own generation rather than read fixed traces? | Needed only if the paper wants a self-monitoring claim. A precise verifier paper can be accepted without it. | High annotation and GPU cost |

## E2. Run one frozen confirmatory replication

**Status:** the current paper has no untouched confirmatory evaluation for the post-hoc audit.

The joint controls, temporal tests, weighting checks, nested comparisons, and subgroup analyses
were designed after the original test result was inspected. More bootstrap draws on that split will
not solve this. Freeze the complete protocol and evaluate it once on a new model or a genuinely
untouched dataset.

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

**Status:** partly done. Controls now use trace-equal training weights and the same selection
budget. The hidden probe's original boundary-equal fit is retained as the primary row.

### Remaining work

1. Refit H, N, N+H, O, and O+H with the same class and trace weighting.
2. Repeat the full group split, layer selection, C selection, refit, and threshold selection for a
   predetermined list of split seeds. Five is a minimum; ten is preferable if the cached activations
   make this cheap.
3. Report the distribution of each metric and conditional difference across seeds. Do not select the
   best seed.

This analysis measures split and selection sensitivity. It does not erase the post-hoc status of the
original dataset.

## E5. Finish the implemented transition-matching sensitivity

**Status:** implemented, not run in the current artifact tree.

Run:

```bash
uv run math-error \
  --config configs/project.yaml \
  transition-sensitivity
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

## E6. Calibration: define or remove

**Status:** the revised paper drops the calibration claim, which is correct. The term should also
be removed from any older manuscript versions.

If the authors want to keep a calibration-related analysis, pick one definition:

- **Probability calibration:** report held-out Brier score, log loss, ECE with bin sensitivity,
  and a reliability diagram. Give source- and length-stratified estimates only when the groups are
  large enough to interpret.
- **Threshold stability:** report the validation-selected threshold's Process F1, erroneous-trace
  exact accuracy, and correct rejection by source and length. Call this threshold transfer, not
  probability calibration.

If space is tight, leave calibration out. The existing threshold-transfer result is already reported
in the source-transfer section.

## Counterfactual patching decision

**Decision: close this loop without running patching.** The 160-pair template is a draft annotation
resource, not an experiment result. The 135 proposed corrections have not been established as
mathematically valid, and the 25 withheld items were intentionally not assigned local corrections.
There is no defensible route to a causal patching claim without a reviewer who can verify the
corrections.

For paper completion:

1. Keep the template and drafts archived as unrun exploratory material.
2. State in the methods or limitations that counterfactual patching was not run because corrections
   were not independently verified.
3. Remove patching from the required evidence list, abstract, conclusion, and definition of done.
4. Do not run `run-counterfactuals`, mark rows as `verified`, or infer a causal result from the
   template.
5. Report the completed learned-direction assay only as an inconclusive test of a weak verdict
   readout, as described in the experiment record.

This is a scope decision, not a failed attempt to manufacture a result. The paper's supported claim
is predictive and diagnostic: hidden states contain partially transferable information about
`invalid_so_far` under the tested setup. It is not a claim that the model uses that information
causally.

## Work that should not consume the next cycle

- Do not add more weak metadata baselines. The missing comparison is conditional, not another
  separate classifier.
- Do not increase bootstrap draws on the inspected test split and call the result confirmatory.
- Do not run more random intervention directions. They cannot replace a positive control for assay
  sensitivity.
- Do not run counterfactual patching for this paper. It is closed as optional and unrun unless a
  future project supplies independent mathematical verification.
- Do not test many contextual encoders and report the best one.
- Do not add a third model before E2 and E3 are done.
- Do not hide a positive N+H result because the intended paper is a negative-results paper.

## Recommended execution sequence

The minimum path to close the current paper is:

1. Keep counterfactual patching marked optional and unrun; do not wait for human verification.
2. Update the manuscript and result record so the supported claim is predictive/diagnostic and the
   causal assay is inconclusive because its verdict readout was weak.
3. Remove any remaining calibration, self-monitoring, precise-localization, or causal-use wording.
4. Freeze the artifact map, resolved configurations, exclusion counts, and environment record.
5. Build the paper and run the repository test, lint, and clean-environment checks.
6. Record any contextual baseline, replication, transition sensitivity, or split sensitivity as
   optional follow-up work unless it is actually completed and integrated.

The higher-evidence path remains available when time and compute permit:

1. Run E3 (contextual text baseline).
2. Freeze the complete E1+E3 protocol as the E2 manifest.
3. Run E2 (confirmatory replication) once and lock its predictions.
4. Run E5 (transition sensitivity) if activation shards become available.
5. Run E4 (split sensitivity) to characterize variability across seeds.

## Definition of done

The experimental package is ready for the paper when all of the following are true:

- [x] Every displayed Process F1 reproduces from displayed `error_exact` and `correct_rejection`.
- [x] The abstract, table, prose, and machine-readable summary use the same values.
- [x] N versus N+H and O versus O+H are evaluated with paired uncertainty.
- [x] All compared models use documented and matched training weights and selection budgets.
- [x] Reader-model and self-monitoring claims are not mixed.
- [x] The causal section makes no use claim while the behavioral readout remains invalid.
- [ ] One contextual text-only baseline tests the surviving hidden-specific localization claim
  (optional strengthening; not required to close this version).
- [ ] A frozen confirmatory model or dataset has been evaluated once, or the entire central audit is
  labeled exploratory (optional strengthening; the current audit is labeled exploratory where
  appropriate).
- [ ] Split sensitivity is reported without selecting a favorable seed (optional strengthening).
- [ ] Transition matching sensitivity has been run and includes pair counts and uncertainty (optional
  strengthening).
- [x] Calibration claims are removed unless a defined calibration analysis is completed.
- [x] Counterfactual patching is explicitly marked optional and unrun; no unverified correction is
  presented as an experimental result.
- [x] No in-sample fitted score is used as a meta-model training feature.
- [x] Resolved configs, data/model identities, selection records, predictions, metrics, and intervals
  are saved for the completed experiments.
- [ ] `uv run pytest`, `uv run ruff check src tests`, the paper build, and a clean-environment
  reproduction check pass before submission.

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
