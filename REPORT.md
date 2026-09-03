# Decodability is not localization

A project report on probing Qwen2.5-Math-1.5B-Instruct for mathematical error detection

The question this project started with was whether a language model, while reading a worked
solution, internally registers the moment the argument stops being correct. If it does, a linear
read of the residual stream should be able to tell you. That much turned out to be true. Almost
everything I actually wanted to conclude from it turned out not to be.

The short version: a probe on the selected layer decodes "this prefix has already gone wrong" with
held-out AUROC 0.866, and that is a real result. But the same score names the first wrong step in
28.9% of erroneous traces, fires roughly half a step late, and fires on 38.8% of solutions that were
never wrong at all. Its ranking transfers across data sources; its threshold does not. And the causal
test never ran, because the model's own verdict about its own reasoning came out below chance on the
exact boundaries the intervention would have used. None of those four facts is visible in the number
0.866.

This report is the running account of what was run, what the artifacts say, and where the evidence
stops. The manuscript version is in [paper/](paper/); the frozen numbers all come from
[results/experiment.md](results/experiment.md) and the artifact files it maps.

## Setup

The model is `Qwen/Qwen2.5-Math-1.5B-Instruct` ([Yang et al., 2024](https://arxiv.org/abs/2409.12122))
reading [ProcessBench](https://aclanthology.org/2025.acl-long.50/)
([Zheng et al., 2025](https://aclanthology.org/2025.acl-long.50/)), which annotates the earliest
erroneous step in solutions drawn from GSM8K, MATH, OlympiadBench, and Omni-MATH
([Cobbe et al., 2021](https://arxiv.org/abs/2110.14168);
[Hendrycks et al., 2021](https://arxiv.org/abs/2103.03874);
[He et al., 2024](https://arxiv.org/abs/2402.14008);
[Gao et al., 2024](https://arxiv.org/abs/2410.07985)). The model never generates these traces. It
reads each fixed one in a single causal forward pass, and extraction records the 1,536-dimensional
residual-stream state at every step boundary across all 29 hidden-state indices.

Of 3,400 attempted traces, 3,360 were retained and 24,909 step boundaries recorded; the 40 exclusions
were prompts longer than the 2,048-token limit, dropped rather than truncated. Problems are
normalized and hashed so that every trace sharing a problem statement lands in one partition. Seed 42
gives a 60/20/20 split. The test partition holds 669 traces, 432 of them erroneous, and 4,985
boundaries.

The target at boundary *k* of trace *i* is `invalid_so_far`: 1 if the trace has an annotated error
at or before step *k*. It marks the first invalid step and every boundary after it. It does not claim
the current step is locally wrong, and that detail matters for how the results read later.

At each hidden-state index I fit a standardized, class-balanced L2 logistic probe over
`C ∈ {0.01, 0.1, 1, 10}`. Validation AUROC picked index 23 and `C = 0.01`; validation Process F1
picked threshold 0.645. The probe was refit on train plus validation and evaluated once on test. The
predicted first error is the first threshold crossing, or -1 if the score never crosses. Intervals
are whole-trace bootstraps, which keeps steps from the same solution together.

Every nuisance control gets the probe's selection budget: same `C` grid, validation-selected
threshold, refit on train plus validation, inverse boundary-count weights so each trace contributes
equal training loss. This is the part that is easy to skip and expensive to skip.

## Check 1: does the hidden state add anything the text does not?

Yes, and less than I expected.

The selected probe reaches test AUROC 0.866 [0.849, 0.884], average precision 0.843 [0.813, 0.870],
and step F1 0.743 [0.716, 0.769]. Reported alone, against nothing, that looks like a discovery.

Against equally tuned controls it looks smaller:

| Predictor | AUROC | Process F1 | Exact error | Correct rejection |
|---|---:|---:|---:|---:|
| Hidden state, layer 23 | .866 | .393 | .289 | .612 |
| Prefix TF-IDF | .751 | .274 | .192 | .477 |
| Prefix MiniLM embeddings (post-hoc) | .753 | .268 | not reported | not reported |
| Structural metadata | .776 | .168 | .148 | .194 |
| Text + metadata | .810 | .210 | .169 | .278 |
| Metadata + final answer † | .854 | .262 | .164 | .646 |
| Text + metadata + final answer † | .874 | .294 | .176 | .895 |

† These rows use final-answer correctness, which is a ProcessBench annotation and would not exist for
a detector running before the answer does. They are diagnostics, not baselines. I am reporting them
because a tuned bag of words plus six pieces of trace metadata plus one benchmark bookkeeping field
scores *above* the probe, which is worth knowing about the target.

The direct test is nested. Let `N` be prefix text plus structural metadata and `N+H` the same
features plus the full selected-layer hidden representation:

| Condition | AUROC | Log loss | Process F1 | Δ AUROC vs. nuisance only |
|---|---:|---:|---:|---|
| `N` | .811 | .533 | .211 | |
| `N+H` | .869 | .454 | .397 | +.0587 [.0414, .0745] |
| `O` (adds final answer) | .874 | .440 | .284 | |
| `O+H` | .909 | .385 | .444 | +.0348 [.0242, .0451] |
| `H` alone | .868 | .457 | .397 | |

Adding the hidden state improves AUROC by 0.0587 [0.0414, 0.0745] and log loss by -0.0798
[-0.1038, -0.0548]. Both intervals exclude zero on this split. The check passes. But notice the size
of the thing that survives: measured against prefix TF-IDF alone, the gap is 0.115, about twice what
survives the joint nuisance model, and measured against nothing at all the reader sees 0.866 versus
chance. Which comparison you print changes the apparent finding by a factor of two.

## Check 2: does the score localize the error?

No, and the failure has a specific shape.

![Decoding versus localization, the late crossing, and transfer across sources](results/figures/gap.png)

Within an erroneous trace, the probe orders boundaries nearly perfectly: mean within-trace AUROC is
0.968 [0.958, 0.976] across the 380 test traces whose first error is not at step 0. That number is
almost meaningless as evidence. Any score that rises monotonically with step index would post it,
and with a persistent label there is every reason for the score to rise. It sets a ceiling that the
decision rule built from the same scores does not come close to.

That rule names the exact first-error step in 28.9% of erroneous traces, lands within one step in
59.3%, within two in 72.7%. It crosses at all on 88.7% of erroneous traces, and on 38.8% of fully
correct ones. Among detected traces the mean signed error is +0.59 steps [+0.36, +0.80]. The crossing
comes after the annotation, never before it on average.

The center panel above explains it. The mean score is 0.341 one step before the annotated error,
0.559 at the onset, 0.680 one step after, 0.731 two steps after. The operating threshold is 0.645.
At the onset the average trace is still below it. The score does not step; it ramps, over several
boundaries around the error, and a fixed threshold placed on a ramp always crosses late.

The alignment to the annotation is real, though. Circularly shifting each frozen score trajectory
within its own trace 5,000 times preserves the values and destroys the alignment, and gives a null
exact rate of 0.164 [0.137, 0.192] against the observed 0.289 (p = 0.0002), and a null Process F1 of
0.259 against the observed 0.393 (p = 0.0002). Comparing the score jump at a non-initial first error
against the jump at a transition in a correct trace matched on source, generator, relative position,
trace length, and token count gives 0.257 versus 0.113, a paired difference of 0.144 [0.096, 0.193].

Correct traces drift upward too. Same ramp, no error to explain it.

Two robustness checks moved almost nothing. Fitting thresholds per trace-length bucket moved Process
F1 from 0.393 to 0.377 and correct-trace rejection from 0.612 to 0.540, and left exact localization
alone. Reading states at the last natural token of each step instead of at the artificial `<<END_STEP>>`
marker changed AUROC by +0.002 [-0.007, +0.011] and Process F1 by +0.026 [-0.020, +0.075], so the
decodable signal is not an artifact of the marker token I inserted.

Where the decision rule does break down is on long traces. Complete-trace accuracy falls from 0.491
in the shortest length quartile to 0.264 in the longest, and correct-trace rejection from 0.719 to
0.350. By token count the rejection rate falls from 0.792 to 0.231. Long correct solutions accumulate
enough drift to cross.

## Check 3: does the decision rule port across sources?

Partly. The ranking does; the operating point does not.

Training on one ProcessBench source and testing on another gives mean off-diagonal AUROC 0.805
against 0.845 on the diagonal, and every one of the twelve off-diagonal cells stays above 0.739.

| Train \ Test | GSM8K | MATH | OlympiadBench | Omni-MATH |
|---|---:|---:|---:|---:|
| GSM8K | .817 | .802 | .753 | .791 |
| MATH | .775 | .877 | .789 | .858 |
| OlympiadBench | .739 | .852 | .804 | .868 |
| Omni-MATH | .768 | .859 | .810 | .882 |

The thresholded metric moves much further. Mean off-diagonal Process F1 is 0.261 against 0.336 on the
diagonal, ranging from 0.055 to 0.371:

| Train \ Test | GSM8K | MATH | OlympiadBench | Omni-MATH |
|---|---:|---:|---:|---:|
| GSM8K | .253 | .245 | .055 | .134 |
| MATH | .209 | .433 | .267 | .297 |
| OlympiadBench | .279 | .363 | .309 | .351 |
| Omni-MATH | .275 | .371 | .280 | .349 |

The worst cell is GSM8K to OlympiadBench, where AUROC holds at 0.753 while Process F1 collapses to
0.055. A threshold fitted on short grade-school traces fires on 97% of correct OlympiadBench
solutions. The direction survives the move; the calibration does not, because trace length changed
underneath it.

These are four sources inside one benchmark construction, not four independent datasets, and they say
nothing about another model family or size.

## Check 4: does the model use the direction?

Unknown, and I want to be exact about why that is different from "no."

Testing use requires a behavioral readout that separates valid from invalid prefixes *before* any
intervention touches it. Without that, a flat dose-response curve is unreadable. So I asked the model
directly whether its reasoning contained an error and scored the difference between the conditional
next-token probabilities of "Yes" and "No", teacher-forced.

On 256 balanced held-out boundaries, that readout has AUROC 0.342 and specificity 0.203. It is worse
than chance at exactly the task the intervention was supposed to measure. An earlier single-token
CORRECT/INCORRECT readout was worse still, at AUROC 0.283 and specificity zero.

The pipeline gates the causal analysis on the readout clearing AUROC 0.5 with nonzero specificity, so
the intervention is recorded as an assay diagnostic. For completeness, it ran anyway: the boundary
state was moved along the probe direction at α ∈ {-4, -2, -1, 0, 1, 2, 4} projection standard
deviations, plus 20 random orthogonal directions at the extreme doses.

![Paired change in the verdict score at each dose, with every interval covering zero](results/figures/intervention.png)

Every learned-direction interval covers zero. The dose slope is -0.000120, rank monotonicity across
the seven means is -0.500, and the hypothesized sign occurs on 23.4% of individual nonzero
interventions. Against the random directions the one-sided values were 1.0 at α = -4 and 0.0476 at
α = +4. The smallest detectable effect at 80% power was 0.0029.

None of that is evidence about the model. I could have published the flat curves and written that the
direction is decoded but unused, and the sentence would have read cleanly and had nothing behind it.
Checking the readout first left a hole in the evidence rather than a false null, which is the worse
outcome to write up and the better one to have.

## The one exploratory positive

A probe fit on the onset *difference* `h[e] - h[e-1]`, rather than on the state itself, separates real
error onsets from matched correct transitions with AUROC 0.769 [0.713, 0.820] on 380 held-out pairs.
It beats position (0.631), destination-step TF-IDF (0.671), and shuffled labels (0.585). The selected
configuration was layer 21, `C = 0.01`.

The matching is imperfect and I do not want to oversell this. There are 1,919 pairs total drawn on 687
unique placebo traces; one placebo trace supports up to 46 pairs. Standardized differences after
matching are 0.146 for trace length and 0.285 for token count, which is not tight. The test split had
already been inspected before this analysis existed. The planned inverse-reuse and one-to-one refits
are implemented but have no completed artifact. It is a lead, not a finding.

## What did not get done

Counterfactual activation patching produced no result. The template holds 160 pairs, each with the
problem, the prefix before the first error, the original erroneous step, and a proposed correction.
135 corrections are drafted and 25 are withheld, because they need a full solution rewrite, a change
to the solution route, or access to a figure that cannot be checked from the text. The loader refuses
to run on any pair without `verified: true` set by a human reviewer, and that verification has not
happened. It also inherits the same failed behavioral gate as the steering experiment, so clearing
the annotation backlog alone would not make it interpretable.

The other missing piece is cheaper and more useful: the grouped split, layer and `C` selection, and
threshold selection have been run under one seed. Repeating that across seeds needs no new extraction
and is the first thing I would run next.

## What I take from this

Four things generalize past this model, and each costs almost nothing to check.

Report the decision rule, not the ranking. Within-trace AUROC 0.968 and exact localization 28.9% come
out of the same predictions. If the claim is that a model knows *when* its reasoning went wrong,
AUROC cannot answer it, because AUROC never touches the threshold.

Give the nuisance model the probe's tuning budget. An undertuned baseline makes the gap look twice
its size, and with no control at all the reader is just comparing 0.866 against chance.

Watch for benchmark bookkeeping in the feature set. Final-answer correctness lifts a nuisance model
to AUROC 0.874, past the probe, while being a field no online detector could read. It sits in the
metadata and is easy to include without noticing.

Gate causal claims on assay validity. Decodability does not imply use
([Hewitt and Liang, 2019](https://aclanthology.org/D19-1275/);
[Belinkov, 2022](https://doi.org/10.1162/coli_a_00422);
[Elazar et al., 2021](https://aclanthology.org/2021.tacl-1.10/)), and the standard steering methods
([Meng et al., 2022](https://arxiv.org/abs/2202.05262);
[Turner et al., 2023](https://arxiv.org/abs/2308.10248);
[Li et al., 2023](https://arxiv.org/abs/2306.03341)) only tell you something if the behavior you are
perturbing was measurable to begin with.

The related literature reaches compatible conclusions from other directions.
[Yuan et al. (2026)](https://arxiv.org/abs/2605.09502) separate diagnostic error awareness from causal
use across several models. [Bertolazzi et al. (2025)](https://aclanthology.org/2025.emnlp-main.1495/)
find models that compute arithmetic without validating it.
[Srivatsa et al. (2025)](https://aclanthology.org/2025.emnlp-main.553/) find that models cannot spot
math errors even when shown the solution. Meanwhile hidden-state step scorers are already being used
for test-time scaling and trace pruning ([Ni et al., 2026](https://aclanthology.org/2026.acl-long.536/);
[Liang et al., 2026](https://aclanthology.org/2026.findings-acl.1336/)), which is what makes the
distinction between ranking and thresholded localization a practical question rather than a
methodological one.

## Limitations

One model, one benchmark construction, one split seed. The nested gain is conditional on that
partition. The persistent `invalid_so_far` target labels every boundary after the annotated error, so
pooled metrics reward accumulated evidence and not necessarily error detection. The MiniLM control is
a single unpaired post-hoc run with an unpinned encoder revision, and it does not exhaust visible-text
representations. Calibration is reported as log loss and Brier score with no reliability analysis. The
temporal, boundary-location, transition, and semantic-baseline analyses were all designed after the
primary result was known, so their intervals describe resampling uncertainty inside a completed
pipeline rather than sensitivity to a fresh split. The primary artifacts do not preserve the GPU host
name, wall-clock duration, or full extraction progress log.

One more thing, which belongs in any writeup of a result like this: the probe is not a grader. At its
selected threshold it flags 38.8% of fully correct solutions as containing an error, and its
correct-trace rejection varies from 0.463 on OlympiadBench to 0.786 on GSM8K. Pointing it at student
work would produce frequent false accusations, distributed unevenly across problem difficulty. The
supported use is as a research diagnostic sitting next to human or formal verification.

## Reproducing it

Extraction needs one A100-class GPU; every analysis in this report runs on CPU against frozen
predictions.

```bash
uv sync --extra dev
uv run math-error --config configs/project.yaml validate-config
uv run math-error --config configs/project.yaml run-all
uv run math-error --config configs/project.yaml analyze
uv run math-error --config configs/project.yaml fit-conditional
uv run math-error --config configs/project.yaml fit-contextual-baseline
uv run math-error --config configs/project.yaml fit-transition
uv run math-error --config configs/project.yaml transition-diagnostics
uv run math-error --config configs/project.yaml analyze-boundary-controls
```

Note that the frozen primary artifacts came from a notebook-resolved configuration, not from
`configs/project.yaml` directly. The notebook kept the scientific settings and changed paths, batch
sizes, and dtype; `artifacts/qwen2.5-math-1.5b/experiment_config.yaml` is the authority on what
actually ran. The dataset fingerprint is
`447f0a4b35c5747a9f9a3dab1e70d43f71efd501497b8cec668b71337099784a` and the dtype was bfloat16.
`results/experiment.md` maps every number above to the artifact file it came from.
