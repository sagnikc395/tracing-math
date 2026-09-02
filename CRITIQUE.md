# Critique of "Decodability Is Not Localization"

## Scope of this critique

This is an independent critique of the revised `paper/neurips_2026.tex`, not a rebuttal to supplied reviews.
No reviewer comments, scores, confidence ratings, or rebuttal deadline were provided. I use the
concern-ranking and evidence-diagnosis structure from `SKILL2.md`, but I do not infer reviewer
intent or assess rebuttal viability.

The revised paper now incorporates the E1 nested comparison and the equally tuned shortcut
controls from E2. The main claim has shifted from "no incremental pooled information" to "hidden
states add incremental predictive information beyond nuisances, but the evidence is conditional on
one model, one split, and one benchmark." This is the correct framing for the available evidence.
The prose has been rewritten to avoid significance inflation, plain-verb violations, and AI-style
structural tics (SKILL1.md).

My assessment: **borderline, closer to accept-ready than before**. The paper now has a correct
main claim supported by a nested comparison, honest post-hoc labeling, and a useful audit
checklist. The remaining gaps are a contextual semantic baseline (C5), a frozen replication or
outer-fold evaluation (C3), and the stale Table 2 in the PDF that has not yet been regenerated from
the artifact.

## What changed in the revision

The revision addresses four decision-critical issues from the original critique:

1. **C0 (Table 2 correctness):** The revised paper regenerates Table 2 from one audited metric
   schema and adds the mechanical consistency check. The original estimand mismatch is gone.

2. **C1 (incremental information):** The revised paper adds Table 2a (nested comparison) showing
   N+H improves AUROC by +0.0587 [0.0414, 0.0745] over N. The stale "no incremental pooled
   information" claim is removed.

3. **C2 (self-monitoring language):** The revised paper replaces "a solving model's own
   representation" with "a model's hidden representation changes when it reads a reasoning step."
   The model reads traces; it does not generate them. The causal section is labeled empty by design.

4. **C4 (oracle vs deployable):** The revised paper separates deployable controls from oracle
   controls in Table 2 and states which models include final-answer correctness.

The prose throughout now follows SKILL1.md: plain verbs ("shows" not "underscores"), no
significance inflation ("useful" not "groundbreaking"), sentences that end at the fact, varied
rhythm, and earned adjectives.

## Remaining concerns

### C3. The nested gain is still split-conditional

**Class:** `STATISTICAL_RELIABILITY`
**Severity:** `MAJOR`
**Decision impact:** `HIGH`
**Resolution confidence:** `HIGH`
**Best response mode:** `NEW_ANALYSIS` or `CLAIM_NARROWING`

The nested comparison (N+H minus N) shows a substantial AUROC gain on the inspected split.
Bootstrap intervals capture resampling uncertainty, not split or selection sensitivity. The paper
acknowledges this in the limitations, but the abstract and Section 4.1 read as confirmatory for a
post-hoc analysis. Until the protocol is evaluated on an untouched grouped holdout, outer folds,
or a second reader model, the result is a credible empirical observation, not a confirmatory test.

**Evidence needed.** Freeze all choices, rerun on a fresh grouped holdout, outer folds, or a
second reader model, and keep evaluation untouched until code is locked.

### C5. The text controls are still too weak to isolate an internal error representation

**Class:** `MISSING_BASELINE`
**Severity:** `MODERATE` to `MAJOR`
**Decision impact:** `MEDIUM`
**Resolution confidence:** `MEDIUM`
**Best response mode:** `NEW_EXPERIMENT`

Prefix TF-IDF is a useful lexical baseline, but it is not a strong semantic verifier. The hidden
state is a contextual nonlinear representation of the same prefix. A localization advantage over
bag-of-words may reflect ordinary semantic processing rather than an internal variable that tracks
reasoning validity.

**Evidence needed.** Add at least one text-only semantic baseline that does not access the
target hidden state: a frozen sentence encoder with a linear head, a small external verifier, or
an equivalently budgeted model operating on the visible prefix. Also consider output-side signals
available online, such as token entropy or log-probability summaries. Evaluate the same
first-crossing metrics and use the same grouped partitions and tuning rules.

This baseline matters most for the surviving localization claim. If a semantic text model matches
the hidden score's Process F1 or onset jump, the result becomes a limitation of TF-IDF rather
than evidence for special information in the target model's residual stream.

### C6. "Calibration" is still claimed but not evaluated

**Class:** `EMPIRICAL_SUPPORT` / `CLARITY`
**Severity:** `MODERATE`
**Decision impact:** `MEDIUM`
**Resolution confidence:** `HIGH`
**Best response mode:** `NEW_ANALYSIS` or `CLAIM_NARROWING`

The abstract and conclusion do not mention calibration in the revised version, which is an
improvement. However, the original claim about calibration has not been explicitly retracted or
replaced with proper scores. The paper should either report Brier score, log loss, ECE with bin
sensitivity, and reliability plots overall and by source/length, or remove the calibration claim
entirely.

### C7. The transition result is still vulnerable to reuse and matching bias

**Class:** `ROBUSTNESS` / `STATISTICAL_RELIABILITY`
**Severity:** `MODERATE`
**Decision impact:** `MEDIUM`
**Resolution confidence:** `HIGH`
**Best response mode:** `NEW_ANALYSIS`

The transition analysis correctly labels itself post-hoc and documents placebo reuse (median 2,
maximum 46) and covariate imbalance (standardized differences 0.28 and 0.15). One-to-one matching
and inverse-reuse weighting sensitivity refits are implemented but have not yet run because
activation shards are needed. Until those refits complete, the transition AUROC of 0.769 is
exploratory.

### C8. The novelty case needs a sharper comparison with contemporaneous work

**Class:** `NOVELTY` / `RELATED_WORK`
**Severity:** `MODERATE`
**Decision impact:** `MEDIUM`
**Resolution confidence:** `MEDIUM`
**Best response mode:** `MANUSCRIPTRVISION`

The paper's strongest novelty is the full audit: human first-error labels, nuisance-plus-hidden
conditional testing, first-crossing metrics, matched transitions, source transfer, and a
failed-assay gate. Table 1 compares only three papers. Two contemporaneous 2026 papers should be
discussed:

- Where Does Reasoning Break? (arXiv 2605.13772) directly studies first-error localization through
  hidden-state transitions and reports cross-model and cross-dataset results.
- Hidden Error Awareness in Chain-of-Thought Reasoning (arXiv 2605.09502) contrasts hidden-state
  diagnosis with failed causal interventions across several model families.

Under the NeurIPS contemporaneous-work rule, papers posted after March 1, 2026 generally should not
defeat novelty, but the manuscript should state the difference rather than rely on chronology.

## Evidence-status table

The revised paper's conclusion now uses a status table rather than a binary pass/fail. The
appropriate status with current evidence is:

| Validation level | Status supported by the current paper |
|---|---|
| Persistent-prefix decodability | Supported on one frozen split |
| Incremental information over nuisances | Supported on the inspected split; replication needed |
| Exact localization | Improved by N+H but weak in absolute terms |
| Temporal onset correspondence | Positive but exploratory |
| Cross-source ranking | Supported within ProcessBench; threshold transfer is unstable |
| Probability calibration | Not tested in the revised paper |
| Behavioral or causal use | Not tested because the assay was invalid |

## Prioritized experiment and analysis plan

| Priority | Concerns | Underlying question | Minimum viable protocol | Decision value | Cost and negative-result implication |
|---|---|---|---|---|---|
| **P0 - must do now** | C5 | Is the gain specific to hidden states rather than contextual text semantics? | Add one frozen text-only semantic encoder and compare C versus C+H under the identical split, weights, tuning budget, and localization metrics. | Tests the strongest remaining alternative explanation. | Moderate compute. If C matches C+H, narrow the hidden-specific claim. |
| **P0 - must do now** | C3 | Does the nested gain survive a confirmatory evaluation? | Freeze all choices, rerun on a fresh grouped holdout, outer folds, or a second reader model, and keep evaluation untouched until code is locked. | Converts a post-hoc split result into a credible empirical claim. | Moderate to high GPU/CPU cost. Instability makes the result model- or split-conditional. |
| **P1 - high value** | C7 | Is the transition result driven by repeated placebos or residual imbalance? | Refit with one-to-one matching and inverse-reuse weights; report unique controls, balance, and paired metrics. | Tests the most onset-specific evidence with existing data. | Low CPU cost. Sensitivity implies the transition AUROC is exploratory. |
| **P2 - useful** | C6 | Are scores calibrated, or are thresholds merely unstable? | Report Brier score, log loss, ECE with bin sensitivity, calibration slope/intercept, and reliability plots overall and by source/length. | Supports or removes a repeated claim in the abstract and conclusion. | Low CPU cost. Poor calibration supports the warning; acceptable calibration requires rewriting the claim. |
| **P3 - defer** | C8 | Does the phenomenon generalize across model families? | Repeat the frozen protocol on at least one different family and one larger checkpoint, with a preregistered analysis manifest. | Raises the significance of an existence proof and tests prevalence. | High GPU cost. Heterogeneity should become the result rather than be averaged away. |
| **DO NOT RUN** | causal use | Can activation steering change behavior? | Do not interpret another directional intervention until a readout distinguishes valid and invalid baselines and a positive control shows assay sensitivity. | Prevents another uninterpretable causal section. | The failed gate already answers whether the current assay is usable. |

## Section-level revision status

The revision addresses the R0 items from the original critique:

- **Abstract:** The stale no-incremental claim is replaced with the E1 N+H-minus-N result. Oracle
  models are separated from deployable controls. The mixed conclusion is stated.
- **Table 2:** Regenerated from one audited schema. Erroneous-trace exact accuracy is distinguished
  from complete accuracy. The oracle models are labeled.
- **Section 3.4:** Added. The nested comparison protocol is now explicit.
- **Section 4.1:** Rewritten. The nested comparison leads. The stale claim about the joint model
  alarming on most correct traces is removed.
- **Conclusion:** The binary pass/fail is replaced with a status table. The checklist is updated
  to reflect that the nested test is now completed on the inspected split.
- **Introduction:** Self-monitoring language is replaced with "reads a reasoning trace." The model
  is stated to read fixed traces, not generate them.

The remaining R1 items:

- Add a semantic text-only baseline (C5).
- Add one-to-one and inverse-reuse transition sensitivities (C7).
- Expand related work to compare with 2026 onset-localization studies (C8).

## Provisional NeurIPS-style assessment

| Criterion | Assessment | Reason |
|---|---|---|
| Quality | Fair to good | The checked E1 artifact resolves the original metric defect. The nested comparison supports incremental information on the inspected split, but confirmation requires an untouched replication. |
| Clarity | Good | The paper is compact and candid. The reader-versus-solver protocol, metric names, and oracle separation are now correct. The prose avoids significance inflation and AI-style tics. |
| Significance | Borderline to good | The conditional gain could change how hidden-state verifier claims are audited. It needs a semantic baseline and replication to establish significance beyond one case study. |
| Originality | Good | The combination of human first-error labels, nested nuisance controls, temporal checks, and an explicit failed-assay gate is a useful audit contribution. |
| Overall | **Borderline, closer to accept-ready** | The result is stronger than before, but the semantic baseline and replication gaps are still acceptance-critical. |
| Confidence | **4/5** | High confidence in the internal arithmetic and design critique; lower confidence in the complete novelty landscape. |

The assessment could rise to 4/6 or higher if the contextual baseline does not explain away the
gain, and a frozen replication or outer-fold analysis preserves the direction of the N+H
improvement. If either experiment is null, the paper should narrow the contribution to a model-
and benchmark-conditional audit rather than claim a general hidden-state advantage.

## Policy and author-review note

This document is AI-assisted author-review material, not a venue review or a submission-ready
rebuttal. The authors should verify every number and interpretation against the underlying
artifacts. The NeurIPS 2026 Main Track Handbook allows authors to use tools in preparing and
writing a paper, while making the authors responsible for correctness and originality. Basic
editing assistance need not be documented; important, original, or non-standard agent/LLM use in
the methodology should be described in the experimental setup. The paper's final checklist
response should reflect what affected the scientific method, not merely the wording of this
critique.
