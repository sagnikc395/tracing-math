# Workshop-specific critique of “Decodability Is Not Localization”

## Bottom line

This revision is substantially stronger than the earlier manuscript and is a good thematic match
for the NeurIPS 2026 **Interpretability for Discovery** workshop. The workshop explicitly welcomes
failure cases and negative results, including misleading interpretations, failed validation, and
practical limits on reliable discovery. That is now the paper’s central contribution: high
invalid-prefix AUROC can coexist with nuisance predictability, poor onset localization,
calibration failures, and no validated causal evidence.

The paper is now scientifically defensible as a careful audit or negative-result case study. Its
main remaining weakness is not overclaiming but identification: the added controls still do not
measure the hidden state’s incremental information over the strongest jointly available nuisance
predictor. Prefix text, structural metadata, and final-answer outcome are fitted separately, and
the new shortcut models use a fixed logistic regularization value while the hidden-state probe
receives validation-based regularization selection. Because outcome-aware metadata already reaches
AUROC 0.857 versus 0.866 for the hidden state, that comparison needs to be maximally fair.

The onset evidence remains exploratory on an already inspected split, and the study covers only
one small model and one benchmark construction. For this workshop, those limitations are less
damaging than they would be for a main-track universality claim because the call explicitly values
careful negative case studies. My provisional scientific recommendation is therefore **weak
accept**, with meaningful upside if the baseline comparison is made fully fair.

The exact PDF is not yet submission-compliant, however. The workshop allows **five pages of main
text**, while this manuscript’s main text continues onto page 6 before the references begin. That
is a formatting blocker independent of the scientific recommendation.

## Workshop fit and submission compliance

The official [workshop scope](https://interpretability4discovery.github.io/about.html) asks how
interpretability can turn internal representations into actionable, testable knowledge and lists
four topic areas. This paper fits most directly under **“Failure cases and negative results”**:
it shows that a compelling linear representation can be confounded by metadata, poorly localized,
miscalibrated, and unusable in a failed behavioral assay. It also contributes to the workshop’s
interest in evaluation frameworks that determine when interpretability supports reliable
discovery.

The fit is real but under-articulated. The paper is currently framed around hidden-state verifiers
for mathematical reasoning, not around the epistemic validation problem posed by the workshop. A
reviewer should be able to infer the connection, but the Introduction and Conclusion should state
it directly: *decodability is a candidate-discovery step, while nuisance controls, localization,
and causal or behavioral validation determine whether the decoded pattern can count as reliable
new knowledge.*

The official [call for papers](https://interpretability4discovery.github.io/cfp.html) establishes
the following submission checks:

- **Five-page main-text limit:** not satisfied by the current PDF; the final two main-text
  paragraphs appear on page 6. References and appendices do not count.
- **Main text must be self-contained:** largely satisfied. The acceptance-critical metrics,
  controls, and limitations are in the main text.
- **Double blind:** satisfied in the manuscript. The linked supplement must also remove names,
  usernames, and identifying repository metadata.
- **Responsible-use statement:** satisfied. The paper names false-alarm, late-detection, grading,
  and distribution-shift risks and recommends human or formal verification.
- **Reproducibility and code availability:** partly satisfied. Commands and artifact identity are
  reported, but the checklist acknowledges missing compute accounting and license details.
- **Non-archival/private review:** no issue is visible in the PDF, but authors should confirm that
  the OpenReview submission and any linked materials remain private during review.

The workshop site currently lists a submission deadline of **September 2, 2026, 11:59:59 PM AoE**
and permits an extra main-text page only after acceptance. The page-limit fix therefore belongs in
the submission version, not the camera-ready plan.

## Summary of the contribution

The paper analyzes residual-stream states from Qwen2.5-Math-1.5B-Instruct at written-step
boundaries in ProcessBench. A linear probe predicts a persistent *invalid-so-far* target and reaches
test AUROC 0.866. Prefix TF–IDF reaches 0.760, structural metadata reaches 0.780, and structural
metadata augmented with final-answer correctness reaches 0.857. The hidden-state probe retains a
larger advantage on thresholded trace localization: Process F1 is 0.393 versus 0.288 for the
outcome-aware metadata model.

Operational localization remains weak. Exact first-error accuracy is 28.9%, detections are late by
0.59 steps on average, and 38.8% of fully correct traces trigger a false alarm. A post-hoc matched
transition probe obtains AUROC 0.769, suggesting an onset-related representation change, while two
behavioral readouts rank valid and invalid prefixes below chance. The paper therefore concludes
that decodability, localization, calibration, and causal use are distinct.

## What the revision now does well

- **The claim matches the target.** The title, abstract, introduction, and conclusion consistently
  describe a model-specific audit rather than an internal “knowledge” result.
- **The shortcut result is not hidden.** The abstract foregrounds that outcome-aware metadata nearly
  matches hidden-state AUROC, including the paired confidence interval around the difference.
- **The paper separates estimands.** Boundary AUROC, trace-level localization, source transfer, and
  behavioral use are treated as different questions.
- **The onset protocol is now auditable.** Section 3.4 reports the transition construction, frozen
  partitions, class balance, pair counts, model-selection rule, controls, and clustered bootstrap.
- **Statistical reporting is much improved.** Probe-control differences are paired by trace, the
  different bootstrap draw counts are explained, and trace-equal evaluation is reported.
- **Data accounting is complete.** The paper provides attempted and retained trace counts,
  exclusions, boundary counts, error prevalence, per-source accounting, the split seed, and the
  dataset hash.
- **Negative evidence is handled correctly.** The causal pilot is explicitly uninterpretable rather
  than presented as evidence of causal non-use.
- **Generalization language is restrained.** ProcessBench components are called sources rather than
  independent mathematical domains, and the conclusion is limited to one model.
- **Presentation issues are fixed.** The PDF is anonymized, contains the checklist, has no hyperlink
  boxes, defines the embedding-index comparison, and reports the within-error Process F1 as `N/A`.

## Resolution of the previous major concerns

| Previous concern | Current status | Assessment |
|---|---|---|
| Persistent target does not identify onset | **Partially resolved** | The claim is narrowed and the transition protocol is explicit, but onset evidence remains post-hoc and lacks untouched replication. |
| Shortcut controls were too weak | **Partially resolved** | Prefix, structural, outcome, source, generator, and length audits were added. Joint nuisance prediction and equally tuned baselines are still missing. |
| Causal assay was invalid | **Conceded and narrowed** | The pilot is correctly gated and described as invalid. No causal claim remains. |
| One model and benchmark | **Conceded and narrowed** | The title and text name the model and call the work a case study. Scientific generalization remains untested. |
| Boundary weighting and incomplete reporting | **Mostly resolved** | Data flow and evaluation weighting are reported. The fitted probe is still boundary-weighted, and uncertainty conditions on one split and one selected pipeline. |
| Novelty relative to hidden-state verifiers | **Partially resolved** | A delta table and precise negative-result claim were added, but the comparison is too compressed to establish a sharp methodological delta. |

## Decision-critical concerns

### C0. The submission exceeds the workshop’s five-page main-text limit

**Class:** presentation / submission compliance  
**Severity:** fatal if enforced administratively  
**Decision impact:** high  
**Resolution confidence:** high

The workshop CFP permits up to five pages of main text and excludes only references and
appendices. In the current PDF, Section 5 continues for two paragraphs at the top of page 6, and the
references begin afterward on that page. The main text therefore occupies six pages. The CFP grants
an additional page only to accepted camera-ready papers.

**Needed revision:** Remove enough main-text material that the references begin no later than page
6 with no main-text carryover. The cleanest cut is to move Section 4.4’s failed causal pilot to the
appendix, shorten the generic related-work prose, and retain one sentence in the main results saying
that both readouts failed the validity gate. Do not move acceptance-critical shortcut or
localization evidence out of the self-contained main text.

### C1. The strongest nuisance comparison is absent and the added baselines are not equally tuned

**Class:** fair comparison / empirical support  
**Severity:** major  
**Decision impact:** high  
**Resolution confidence:** high

Table 3 compares the hidden probe separately against prefix TF–IDF, structural metadata, and
metadata plus final-answer correctness. It does not fit the natural strongest baseline: prefix text
plus structural metadata plus outcome. Separate baselines cannot reveal whether their errors are
complementary. This matters because the outcome-aware metadata model already reaches AUROC 0.857,
only 0.009 below the hidden-state probe, with a paired interval that includes zero.

The model-selection budgets are also asymmetric. Section 3.2 reports that the hidden-state probe
selects regularization from four values on validation data. Section 3.3 says the added controls
select their thresholds on validation, but does not report regularization selection. The
accompanying implementation fixes those logistic models at `C=1.0`. A nine-point AUROC gap is too
small to interpret under unequal tuning.

Final-answer correctness is appropriately called a diagnostic shortcut, but it is an oracle
benchmark field rather than information necessarily available to an online verifier. The current
result therefore answers “how predictable is the persistent label from benchmark metadata?” more
directly than “what validity information is uniquely present in the activation?”

**Needed revision:** Fit a joint sparse text-plus-metadata model, with and without final-answer
correctness, using the same validation regularization budget as the hidden probe. Report hidden
minus joint-control paired intervals for AUROC, average precision, Process F1, exact erroneous-trace
localization, and correct rejection. Also report the comparison within final-answer-correctness
strata. If the hidden-state advantage disappears, make that negative result central; if a
trace-localization advantage survives, state that narrower incremental claim.

### C2. The matched transition probe is still exploratory and reuses controls heavily

**Class:** empirical support / statistical reliability  
**Severity:** major  
**Decision impact:** high  
**Resolution confidence:** medium

Section 3.4 is now commendably explicit, and the manuscript clearly says the analysis was designed
after the primary result. That honesty prevents a false confirmatory claim, but it does not create
an untouched test. The transition AUROC of 0.769 is the most direct positive evidence that the
representation changes at the annotated onset, so its exploratory status materially limits the
paper’s strongest mechanistic interpretation.

Each error transition is matched independently to the closest correct transition, allowing the
same correct trace or boundary to be reused many times. The two-way bootstrap accounts for reused
correct-trace IDs in uncertainty estimation, but duplicated controls also affect model fitting and
the effective diversity of the classification task. The paper reports 380 held-out pairs but not
the number of unique control traces, reuse distribution, covariate balance after matching, or
whether a few controls dominate training.

**Needed revision:** Freeze the stated transition protocol before collecting an untouched test on a
second model, dataset, or reserved ProcessBench subset. Report unique controls, reuse counts, and
standardized covariate differences. Add a sensitivity analysis using one-to-one optimal matching,
matching with replacement plus inverse-reuse training weights, or both. The result should remain
explicitly exploratory until replicated.

### C3. The one-model scope limits generality, but does not defeat the workshop contribution

**Class:** generalization / significance  
**Severity:** moderate for this workshop  
**Decision impact:** medium  
**Resolution confidence:** medium

The paper handles scope correctly: it names Qwen2.5-Math-1.5B-Instruct in the title, calls the work
a case study, and says cross-source transfer is not cross-model transfer. A single small math-tuned
model could still have unusually accessible metadata, calibration drift, or a model-specific onset
geometry. However, the workshop explicitly welcomes careful failure analyses, so one well-audited
case can be sufficient if the paper presents it as a counterexample to an evaluation inference,
not as a universal model property.

**Needed revision:** For workshop submission, strengthen the wording that this is an existence proof
of a decodability/localization failure mode and avoid implying prevalence across models. A second
model would materially strengthen the paper but is better treated as a high-value archival
follow-up than a prerequisite for this venue. If run later, freeze the persistent, joint-control,
and transition protocols before evaluating it.

### C4. Trace-equal evaluation does not test trace-equal training, and uncertainty conditions on one split

**Class:** statistical reliability / robustness  
**Severity:** moderate to major  
**Decision impact:** medium  
**Resolution confidence:** high

The trace-equal sensitivity analysis reweights the already fitted probe’s evaluation rows. It shows
that the chosen score’s aggregate AUROC changes little when each trace receives equal test weight.
It does not test whether the learned direction changes when every trace contributes equal total
training loss. The primary logistic objective remains boundary-weighted, so long traces influence
the fitted coefficients more heavily.

All intervals also condition on one grouped split, one layer-selection outcome, and one threshold.
Whole-trace test bootstrapping correctly represents sampling uncertainty for the frozen predictor,
but not sensitivity to the problem-group assignment or full model-selection pipeline. This is
important because calibration varies sharply with trace length and source.

**Needed revision:** Refit the hidden and nuisance probes using inverse-boundary-count sample
weights and report paired test differences. Add repeated deterministic grouped splits or bootstrap
the complete training and selection pipeline. At minimum, label existing intervals as conditional
on the frozen split and fitted pipeline. Add uncertainty for the source-transfer cells if transfer
remains a main result.

### C5. The paper fits the workshop’s failure-case track, but does not yet explain its discovery lesson

**Class:** significance / venue fit  
**Severity:** moderate  
**Decision impact:** high  
**Resolution confidence:** high

The workshop is not simply a mechanistic-interpretability venue. Its stated goal is to turn learned
representations into knowledge that experts can test and validate, and it explicitly asks how to
distinguish reliable discovery from misleading interpretation. The paper supplies an excellent
negative example, but its Introduction motivates process verification and model error tracking
rather than interpretability-supported discovery. Its Conclusion calls the work an auditable
account of hidden-state verifier scores without extracting a general validation principle.

This is a framing problem, not a request to manufacture a scientific discovery. The workshop CFP
explicitly welcomes failed validation and practical limits. The paper’s venue-specific lesson is
that a linearly decodable pattern is not yet discovered knowledge: one must test nuisance
predictability, temporal correspondence, calibration, generalization, and behavioral or causal
validity before translating a representation into a claim about what a model has learned.

**Needed revision:** Add a short paragraph near the end of the Introduction mapping the four audit
levels to discovery validation. Recast the final paragraph as a concrete checklist for when an
internal pattern is safe to treat as a candidate discovery. Keep the mathematical case study
central; do not broaden it into unsupported claims about science. Table 1 can then remain compact,
although adding columns for controls, localization, and validation would still sharpen novelty.

### C6. Reproducibility is much better, but the checklist exposes easy unresolved gaps

**Class:** reproducibility / presentation  
**Severity:** moderate  
**Decision impact:** low to medium  
**Resolution confidence:** high

Appendix C gives commands, dependency ranges, a dataset hash, seed 42, and A100 usage. The checklist
nevertheless answers “No” for compute resources and existing-asset licenses. Exact package versions,
GPU memory, elapsed time, storage, and a total-compute estimate are absent. The paper also asserts
that an anonymized supplement contains code and resolved configurations; that claim is not
verifiable from the PDF alone.

These are not deep scientific flaws, but they are inexpensive to fix and matter for a paper whose
contribution is methodological auditability. The title page also retains generic “Affiliation / Address
/ email” placeholders. They do not reveal identity, but a submission-specific anonymous author
block would look more polished if the style permits it.

**Needed revision:** Freeze and report exact environment versions, A100 memory, approximate wall
time and storage by stage, and the licenses or terms for ProcessBench, Qwen, and redistributed
artifacts. Verify that the anonymous supplement executes the stated commands from a clean
environment. Remove generic title-page placeholders only if doing so remains compliant with the
official template.

## Secondary comments

- Section 4.4 still occupies a main-text subsection, and Figure 1 retains an arrow to the causal
  pilot. Because neither readout passes the baseline gate, the pilot could move entirely to the
  appendix unless failed-assay methodology is itself a contribution.
- Table 3 should show error-boundary accuracy and correct rejection separately. Process F1 can hide
  that the hidden and outcome-metadata models fail in different ways.
- “PB” in Table 1 is not defined. Use “ProcessBench” unless space is prohibitive.
- The main text says the outcome-aware metadata field may be unavailable online. It should also say
  exactly how final-answer correctness is obtained and whether it depends on a reference answer or
  benchmark annotation.
- The source and generator subgroup audit reports trace-level outcomes for the hidden probe, but
  not hidden-minus-nuisance gaps. The latter would better test whether the hidden-state advantage
  survives each subgroup.
- The paper should report how many first errors occur at step 0. Those traces are excluded from the
  transition task and may differ systematically from the 380 eligible test traces.
- The transition matching description should state the exact distance function and tier ordering,
  not only the variables used.
- Page 8 has substantial unused space. A compact table of exact environment versions, compute, and
  matching diagnostics would fit without affecting the main-text limit.

## Questions whose answers could change the assessment

1. Does a jointly tuned prefix-text plus structural-metadata baseline close the remaining AUROC or
   Process-F1 gap, with and without final-answer correctness?
2. How many unique correct transitions support the 380 test pairs, how often is each reused, and
   does one-to-one or inverse-reuse matching reproduce AUROC 0.769?
3. Does the transition result replicate under the frozen protocol on untouched data or a second
   model?
4. Does trace-equal-weighted probe fitting preserve the selected direction, layer, AUROC, and
   localization metrics?
5. How much variability arises from the grouped split and model-selection procedure rather than
   from resampling the frozen test predictions?
6. Do the hidden-minus-joint-control localization gains hold across source, generator, length, and
   final-answer-correctness strata?

## Prioritized revision plan

### R0 — Submission blockers and workshop positioning

1. Reduce the submission to five main-text pages. Move the failed causal pilot to the appendix and
   preserve the shortcut and localization results in the main text.
2. Add a concise discovery-validation paragraph to the Introduction and Conclusion, using the
   workshop’s own failure-case framing without broadening the empirical claim.
3. Verify that the manuscript and supplement contain no author names, usernames, repository owner
   paths, acknowledgments, or other identifying metadata.
4. Keep the responsible-use statement in the five-page main text; its omission is a stated
   desk-rejection condition.

### R1 — Highest-value work if it can be completed and verified before submission

1. Tune every shortcut model under the same validation budget and fit joint text-plus-metadata
   baselines, with and without final-answer correctness.
2. Report hidden-minus-joint-control intervals for exact error localization and correct rejection,
   not only AUROC and Process F1.
3. Quantify transition-control reuse and, if inexpensive, add inverse-reuse weighting or one-to-one
   matching sensitivity.
4. If these cannot be validated before the deadline, do not add rushed numbers. Narrow the
   incremental claim and list the comparison as the first camera-ready or follow-up analysis.

### R2 — Camera-ready improvements

1. Use the permitted sixth camera-ready page for joint-control results, matching diagnostics, and
   fuller reproducibility information rather than restoring generic background.
2. Add exact versions, compute time, memory, storage, licenses, matching balance, and step-0 counts.
3. Expand the closest-work table and identify the paper explicitly as an empirical audit of
   discovery validation.

### R3 — Archival follow-up, not required for workshop fit

1. Replicate persistent decoding and onset localization on a second model under frozen protocols.
2. Evaluate the transition probe on genuinely untouched data.
3. Refit with equal total training weight per trace and assess grouped-split/model-selection
   sensitivity.
4. Validate a behavioral assay with a positive control before revisiting causal use.

## Time-sensitive submission recommendation

Because the public CFP lists September 2, 2026 as the deadline, prioritize work by failure risk:

1. **First:** cut to five pages, compile, and verify the responsible-use statement and anonymity.
2. **Second:** add two or three sentences making the workshop failure-case contribution explicit.
3. **Third:** run the joint baseline only if the protocol and output can be checked before upload.
4. **Last:** verify the anonymous supplement and OpenReview PDF. Do not spend submission time on a
   second model or causal rerun.

## Provisional workshop assessment

The public workshop CFP does not publish a numerical review scale or acceptance borderline, so the
following uses qualitative NeurIPS-style language rather than claiming an official score mapping.

| Criterion | Score | Rationale |
|---|---:|---|
| Quality | 3/4 (good) | The design and reporting are careful, but the strongest nuisance baseline is absent, control tuning is asymmetric, and onset evidence remains post-hoc. |
| Clarity | 4/4 (excellent) | Claims, estimands, negative results, and limitations are unusually explicit; the paper is compact and easy to audit. |
| Workshop fit | 4/4 (excellent) | The CFP explicitly solicits misleading interpretations, failed validation, and negative results that delimit reliable discovery. |
| Significance for this workshop | 3/4 (good) | One model limits generality, but a carefully validated counterexample is valuable under the workshop’s stated scope. |
| Originality | 3/4 (good) | The individual tools are familiar, but the combined decodability/localization/nuisance/causal audit is a distinctive negative-result package. |
| Scientific recommendation | Weak accept | The paper offers a clear, relevant failure case with unusually honest validation boundaries. The joint-control comparison is the main residual scientific concern. |
| Submission readiness | Not ready until page limit is fixed | Main text currently spills onto page 6 despite a five-page submission limit. |
| Confidence | 4/5 | High confidence in the methodological and presentation assessment; novelty confidence is lower without a full independent audit of every related paper. |

The recommendation could move to a clearer accept if an equally tuned joint nuisance model leaves
a trace-localization advantage. If the joint control closes the gap, the workshop case can still
remain strong by making the absence of incremental hidden-state evidence the central negative
finding. Cross-model and untouched-onset replication would strengthen a later archival version but
are not necessary to establish thematic fit here.

## Use of this critique

This is AI-assisted author-review material, not a venue review or submission-ready rebuttal. The
author should verify every methodological statement and decide what to adopt. The public workshop
CFP specifies anonymity, reproducibility, and responsible use but does not state an additional
workshop-specific generative-assistance rule. Under the current
[NeurIPS 2026 author policy](https://neurips.cc/Conferences/2026/MainTrackHandbook), authors remain
responsible for correctness and originality; methodologically important or non-standard agent/LLM
use should be disclosed where required. Because agent assistance has contributed substantive
analysis, code, and manuscript revision in this project, the author should confirm whether the
paper’s current `N/A` checklist response accurately reflects the final workflow rather than assume
that writing-only treatment applies.
