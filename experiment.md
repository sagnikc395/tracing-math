# Experiment protocol: when does a math model know the reasoning went wrong?

Status: preregistered analysis plan, written before viewing experimental results.

Submission target: NeurIPS 2026 Workshop on Interpretability for Discovery. The current CFP
specifies a September 2, 2026, 11:59:59 PM AoE deadline, five pages of main text, double-blind
review, and a mandatory responsible-use statement. Requirements should be checked again on the
[official CFP](https://interpretability4discovery.github.io/cfp.html) before submission.

## 1. Research claim

The study asks whether a math-specialized causal language model contains a domain-general
internal change-point signal when a written solution first becomes mathematically invalid, and
whether manipulating that signal changes the model's own correctness judgment.

The intended discovery is about learned mathematical monitoring, not a new mathematical theorem:

> A model may represent the transition from valid to invalid reasoning at a reproducible layer,
> in a direction that transfers across problem sources and participates in its explicit verdict.

This is stronger than showing that hidden states correlate with erroneous text. Three kinds of
evidence are required: temporal localization at the annotated error, transfer across mathematical
domains, and a held-out causal intervention.

## 2. Hypotheses and falsifiers

### H1 — a localized internal change point

At one or more hidden-state indices, a linear probe will distinguish prefixes that remain valid
from prefixes at or after the first erroneous step. More importantly, the first validation-tuned
threshold crossing will align with the human first-error annotation.

Evidence for H1 requires all of the following:

- held-out step AUROC above chance with a trace-grouped 95% bootstrap interval;
- held-out ProcessBench first-error F1 above the position and lexical controls;
- above-chance AUROC when evaluation is restricted to erroneous traces, comparing their pre-error
  and post-error steps without relying on fully correct versus incorrect solution identity.

H1 is weakened if the signal appears only on late downstream steps, if the position baseline
matches it, or if within-erroneous-trace AUROC collapses.

### H2 — a partially shared mathematical-invalidity direction

A direction trained on one ProcessBench source will transfer to at least some other sources. The
four sources are GSM8K, MATH, OlympiadBench, and Omni-MATH.

Evidence for H2 is an off-diagonal transfer matrix consistently above chance, not merely strong
within-source performance. Report every cell. Do not hide a domain where transfer fails. A mixed
matrix supports a narrower conclusion: mathematical invalidity is partly domain-specific.

### H3 — causal influence on the model's verdict

Adding the invalidity direction at a held-out step boundary will increase

\[
\Delta_{\mathrm{verdict}}
= \overline{\log P(\texttt{INCORRECT})}
- \overline{\log P(\texttt{CORRECT})},
\]

while subtracting it will decrease this score. The bars denote mean token log probability, which
reduces bias from different answer token counts.

Evidence for H3 requires a signed, approximately monotonic dose response and an effect at the
largest preregistered magnitude that exceeds the matched random-orthogonal direction distribution.
The claim must be rejected or softened if the unmodified model has no measurable correctness
behavior, if positive and negative interventions do not have opposite effects, or if random
directions behave similarly.

## 3. Fixed resources

### Dataset

Use the official Apache-2.0 `Qwen/ProcessBench` Hugging Face dataset and all four released splits.
Each record contains:

- a problem;
- a model-generated list of reasoning steps;
- the generator name;
- whether the final answer is correct;
- `label`, the zero-indexed first erroneous step or `-1` when all steps are correct.

No synthetic corruption or LLM labeling is used. ProcessBench is normally an evaluation set; in
this study it is explicitly repartitioned into internal train, validation, and test groups for a
representation-analysis experiment. Consequently, the paper must not present the resulting probe
numbers as standard ProcessBench benchmark scores.

### Model

Primary model: `Qwen/Qwen2.5-Math-1.5B-Instruct`, loaded in FP16. This is small enough for a Colab
T4 while retaining math-specific instruction tuning. Do not substitute a newer model after seeing
results. A second model is optional only after every primary analysis and figure is frozen.

No LLM parameters are trained. Only scikit-learn logistic probes are fitted on cached states.

### Context policy

Set the maximum complete prompt length to 2,048 tokens. Exclude and log a trace when the complete
prompt is longer. Never truncate a problem or solution: doing so can move or remove the annotated
error and creates invalid labels. Report the number and source distribution of exclusions.

## 4. Unit of analysis and labels

Let trace $i$ contain steps $s_{i1},\ldots,s_{iK_i}$ and first-error index $e_i$. At each step
boundary $k$, define

\[
y_{ik}=\mathbb{1}[e_i \ge 0 \land k \ge e_i].
\]

Thus, a correct trace contributes only negative examples. An erroneous trace contributes negative
examples before its first error and positive examples from the first error onward. The primary
target is `invalid_so_far`; `error_onset`, which is positive only at $k=e_i$, is retained as a
diagnostic and must not replace the primary target after results are observed.

The statistical grouping unit is the normalized problem, not the step. All steps and duplicate
traces for the same normalized problem receive one partition.

## 5. Prompt and activation extraction

The user message contains the problem and blocks of the form:

```text
[Step 0]
<step text>
<<END_STEP_0>>
```

After the final displayed step, it asks whether the reasoning is valid up to and including that
step. The system asks for exactly `CORRECT` or `INCORRECT`.

For a trace, run the full prompt once with `output_hidden_states=True` and cache the residual state
at every `<<END_STEP_k>>` token. In a causal transformer, later reasoning steps and the verdict
question cannot influence an earlier boundary state. This reduces the full extraction from one
forward pass per prefix to one pass per trace without changing the prefix representation.

For hidden-state index $\ell$:

\[
h_{ik}^{(\ell)} \in \mathbb{R}^d.
\]

Index zero is the embedding output; subsequent indices are transformer block outputs. The final
hidden-state index can be probed but is ineligible for intervention because no later decoder block
would propagate a modified boundary state to verdict tokens.

## 6. Partitions and leakage prevention

Use seed 42. Form groups by SHA-1 of whitespace-normalized, case-folded problem text. Within each
`source × has_error` stratum, sort groups by a seeded SHA-256 digest and allocate 60% train, 20%
validation, and 20% test. This makes the partition deterministic, balanced by source/error status,
and invariant to step expansion.

Rules:

- never split individual steps;
- fit scalers and probes on training states only during hyperparameter selection;
- select $C \in \{0.01,0.1,1,10\}$ by validation AUROC;
- select the probability crossing threshold by validation ProcessBench F1;
- select the reported layer by validation ProcessBench F1, breaking ties with validation AUROC;
- after selection, refit that layer's scaler and probe on train plus validation;
- use test data only for final metrics, trajectories, and causal examples.

## 7. Experiment A: layer-wise decodability and localization

At every hidden-state index, fit a class-balanced logistic regression:

\[
P(y_{ik}=1\mid h_{ik}^{(\ell)})
=\sigma(w_\ell^\top h_{ik}^{(\ell)}+b_\ell).
\]

Report on the held-out test set:

- step AUROC and average precision;
- erroneous-trace first-error accuracy;
- correct-trace accuracy (predict no error);
- their harmonic mean, matching the ProcessBench balancing principle;
- overall exact first-error accuracy;
- step AUROC restricted to erroneous traces;
- 95% percentile intervals from 1,000 resamples of whole traces.

Plot AUROC and first-error F1 against hidden-state index. A rising curve is descriptive; the main
result is the validation-selected layer's held-out localization.

### Required controls

1. **Position:** logistic regression on absolute step index and fractional position.
2. **Lexical:** TF-IDF unigrams and bigrams from the current step only.
3. **Shuffled labels:** identical hidden-state probe with shuffled training labels.
4. **Embedding state:** hidden-state index zero, already present in the layer curve.
5. **Within-error traces:** compare pre-error and post-error steps only among traces known to have
   an error; this reduces generator and final-correctness shortcuts.

If the hidden-state probe does not beat position and lexical controls, stop causal-mechanism claims
and write the result as a shortcut/failure analysis.

## 8. Experiment B: cross-domain transfer

At the validation-selected layer, train separate probes on the training portion of one source.
Choose the threshold on that source's validation portion, then evaluate it on the test portion of
all four sources. Produce the full 4×4 AUROC and first-error-F1 matrices.

Interpretation:

- strong off-diagonal transfer suggests a shared invalidity representation;
- asymmetric transfer suggests that some domains learn a more general direction;
- diagonal-only success suggests domain- or style-specific error features;
- uniformly weak performance invalidates H2.

Generator-wise results may be reported as a diagnostic, but no generator subgroup becomes the
primary result after inspection.

## 9. Experiment C: causal probe-direction intervention

For the best intervention-eligible layer selected on validation data, convert the standardized
probe to raw hidden coordinates:

\[
\widetilde w_j = w_j / \mathrm{scale}_j,
\qquad
v = \widetilde w / \lVert\widetilde w\rVert_2.
\]

Let $\sigma_v$ be the standard deviation of $h^\top v$ on train plus validation states. Select at
most one boundary from each held-out trace, with disjoint traces between classes. For up to 128
valid and 128 invalid-so-far boundaries, intervene at the input to the corresponding decoder block:

\[
h' = h + \alpha\sigma_v v,
\qquad
\alpha\in\{-4,-2,-1,0,1,2,4\}.
\]

The intervention occurs at the reasoning boundary, before the later question tokens. Measure the
model's length-normalized teacher-forced log-probability margin for `INCORRECT` versus `CORRECT`.
First report the unmodified verdict AUROC and zero-threshold accuracy. If those are at chance, the
intervention can establish an effect on a readout but not a useful native error-monitoring behavior.

### Random controls

Sample 20 unit directions orthogonal to $v$. Match intervention norm using the same $\sigma_v$.
To fit the T4 budget, evaluate random directions on 16 examples per class and the two extreme
nonzero alphas. Compare learned-direction change from each example's own baseline with the random
distribution. The learned direction uses the full dose curve and full preregistered sample.

Do not interpret an intervention at a single alpha without the signed curve. Report both valid and
invalid starting states separately in the paper or appendix.

## 10. Secondary analysis: top-variance subspace accessibility

At the selected layer, fit PCA on training hidden states only. Fit probes using the first
1, 2, 4, 8, 16, 32, 64, and 128 components, omitting dimensions larger than the available rank.
Report variance explained and held-out AUROC.

This answers whether the invalidity signal is accessible in a small high-variance subspace. It
does **not** estimate intrinsic dimensionality and should remain an appendix result unless it is
needed to explain a core finding.

## 11. Computational plan for one Colab T4

Priority order:

1. smoke test with 25 traces per source and a separate `artifacts/smoke` directory;
2. full data download and sharded activation extraction;
3. layer probes, required controls, bootstrap, and transfer matrix;
4. causal verdict baseline and learned-direction dose response;
5. random causal controls;
6. PCA curve and any optional robustness checks.

The model is loaded in FP16, processes one trace at a time, and immediately moves selected
boundary states to CPU as FP16 arrays. Activation shards are written every 100 traces. Completed
shards are resumable. Do not spend the deadline on 7B models, SAE training, LoRA, generated
counterfactual data, or multiple prompt variants before the core three experiments are frozen.

## 12. Decision table for the paper narrative

| Observation | Defensible conclusion |
| --- | --- |
| Probe, localization, transfer, and causal test succeed | A partially general internal invalidity direction participates in the model's verdict. |
| Probe succeeds, localization fails | Error-related information exists, but not as an accurate internal change point. |
| Probe/localization succeed, transfer fails | Error monitoring is domain- or style-specific. |
| Probe succeeds, causal test matches random controls | Decodability is not evidence of a causal error-monitoring mechanism. |
| Causal effect exists but baseline verdict is chance | The direction controls the prompted readout, but useful native monitoring is unestablished. |
| Position or TF-IDF matches the probe | The apparent signal is plausibly a dataset/prompt shortcut. |

Negative outcomes remain relevant to the workshop because they identify when interpretability
does not support reliable discovery. The abstract must state the observed row of this table, not
the hoped-for row.

## 13. Figures and five-page allocation

Core figures:

1. method schematic plus an example error-score trajectory aligned to the human error;
2. layer-wise test AUROC/F1 with control table and bootstrap intervals;
3. cross-domain transfer heatmap and causal dose-response inset.

Suggested main-text allocation:

- 0.6 page: abstract and motivation;
- 0.7 page: related work and exact gap;
- 1.1 pages: data, representation, leakage-safe design, controls;
- 1.7 pages: three core results;
- 0.6 page: limitations, discovery implications, responsible use;
- 0.3 page: conclusion.

Keep PCA, subgroup tables, exclusion details, prompt text, and expanded intervention controls in
the appendix while making the main text self-contained.

## 14. Limitations to state regardless of outcome

- One 1.5B instruction-tuned model cannot establish universality across architectures or scales.
- ProcessBench contains model-generated written solutions; results need not transfer to latent
  reasoning without an explicit chain of thought.
- `invalid_so_far` labels all post-error steps positive, although some later steps may be locally
  valid conditional on an earlier mistake.
- A linear direction can causally affect a verdict without corresponding to a human-like concept
  of mathematical validity.
- Step markers and the verifier prompt may alter the model's computation.
- Excluding long traces can shift the evaluated difficulty distribution.
- Domain transfer may partly measure writing-style transfer rather than mathematical abstraction.
- Teacher-forced label margins are a narrow behavioral readout, not a complete critic evaluation.

## 15. Responsible-use statement draft

This work studies whether language-model internals can help identify errors in mathematical
reasoning. Such signals could support auditing and human review, but they may also encourage
overreliance on an imperfect automated verifier. A decodable or steerable direction does not prove
that a solution is mathematically correct. We therefore report shortcut controls, negative
results, uncertainty, and domain failures; release prompts and code; and recommend that internal
scores supplement rather than replace expert verification in high-stakes mathematical work.

## 16. Deadline schedule

### August 30

- run the 100-trace smoke configuration end to end;
- inspect label counts, partition balance, exclusions, and activation shapes;
- start the full sharded extraction and persist artifacts to Drive.

### August 31

- finish extraction;
- run probes, controls, grouped bootstrap, and transfer;
- freeze the selected layer and predictive results;
- begin learned-direction interventions.

### September 1

- finish random causal controls and freeze all experiments;
- create final figures and result tables;
- write Methods and Results first, using only claims licensed by the decision table.

### September 2

- finish Introduction, Limitations, and responsible-use statement;
- compile the five-page double-blind PDF;
- search PDF, metadata, repository links, GitHub usernames, Hugging Face usernames, author names,
  affiliations, and acknowledgments for identifying information;
- upload early and verify the OpenReview PDF before the deadline.

Do not plan new experiments on September 2. Optional analyses are dropped before required controls.

## 17. Artifact-to-claim map

| Claim/check | Artifact |
| --- | --- |
| Layer-wise decoding and first-error localization | `probes/layer_metrics.csv` |
| Trace-level uncertainty | `probes/test_group_bootstrap.csv` |
| Position, lexical, and shuffled-label controls | `probes/controls.csv` |
| Held-out trajectories and subgroup diagnostics | `probes/test_predictions.csv` |
| Cross-domain generality | `probes/domain_transfer.csv` |
| PCA accessibility | `probes/pca_subspace.csv` |
| Native verdict competence | `interventions/behavioral_verdict.json` |
| Individual paired causal effects | `interventions/individual.csv` |
| Dose response and random controls | `interventions/summary.csv` |
| Exclusion accounting | `activation_shards/shard_*.json` |

Record the Git commit, Colab GPU type, package versions, runtime duration, and all deviations from
this protocol before writing the final Results section.
