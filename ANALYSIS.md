# Project Report: Tracing Mathematical Error Detection in Language Models

**Goal:** Determine whether `Qwen2.5-Math-1.5B-Instruct` internally registers when a written math
solution first becomes invalid, how precisely that signal localizes the first error, and whether
the decoded direction causally affects the model's own verdict. Target: NeurIPS 2026
"Interpretability for Discovery" workshop (deadline Sept 2, 2026), using ProcessBench (3,400
human-annotated traces).

---

## 1. What was done

**Experiment 1 (complete, frozen).** Recorded all 29 residual-stream hidden states at every
reasoning-step boundary with one causal forward pass per trace. Trained class-balanced L2 logistic
probes to predict `invalid_so_far` on problem-grouped 60/20/20 splits with validation-only model
selection (selected layer 23, C=0.01, threshold 0.645). Ran position, TF-IDF, and shuffled-label
controls; a 4×4 cross-source transfer matrix; and a preregistered additive intervention along the
probe direction with 20 random-orthogonal-direction controls. Wrote the full report
(`results/experiment1.md`), three figures, and the compiled 5-page manuscript
(`paper/neurips_2026.pdf`).

**Experiment 2 — CPU follow-up (complete).** Post-hoc analyses on frozen Experiment 1 predictions,
no model loading: temporal randomization (5,000 circular shifts), error-aligned trajectories with
metadata-matched placebo onsets, first-step-centered within-trace discrimination, subgroup/failure
audits, and statistical power sensitivity for the failed causal assay.

**Experiment 3 — workshop follow-up (partially complete).**

- *Matched first-error transition probe (complete):* probes the step-to-step activation *difference*
  at real error onsets vs. matched transitions in correct traces.
- *Natural-token vs. marker-token control (complete):* re-probes at the last natural token of each
  step instead of the artificial `<<END_STEP>>` marker.
- *Counterfactual activation patching (implemented, not run):* pipeline built; 160 counterfactual
  pairs drafted (135 corrections, 25 responsibly withheld), with `annotation_notes` on every row.

**Infrastructure.** Resumable sharded extraction with dataset/model/config identity checks, gated
interventions (stops if specificity is zero), length-aware threshold fitting and paired
probe-control bootstrap intervals, two CLIs, and a deterministic CPU-only test suite (33 tests
passing, ruff clean).

---

## 2. What was observed

**The signal is real and decodable, but timing is the bottleneck.**

| Finding | Evidence |
| --- | --- |
| Strong linear decodability | Step AUROC **0.866** [0.849, 0.884] at layer 23, vs. 0.733 TF-IDF and 0.730 position |
| Real within-trace discrimination | AUROC 0.873 *restricted to erroneous traces* — not just a correct-vs-erroneous trace signature |
| Temporal alignment is genuine, not drift | Exact localization 0.289 vs. 0.164 under circular shift (p = 0.0002); onset jump 0.257 vs. 0.113 on matched correct-trace transitions |
| But localization is coarse and late | Exact first error 28.9%, within-1-step 59.3%, mean lateness +0.59 steps; detections continue rising after onset (gradual transition, not an impulse) |
| Ranking transfers across domains | Mean off-diagonal AUROC 0.805 across all 12 source pairs; but Process F1 transfers poorly (0.261) — ranking generalizes, calibration doesn't |
| A clean negative causal result | No dose had a 95% interval excluding zero; learned direction did not beat random orthogonal directions |
| The causal test was inconclusive, not negative | The unmodified verdict readout was degenerate: all 256 boundaries called `INCORRECT`, specificity 0, AUROC 0.283 — the behavioral precondition failed |
| Robustness to the marker | Probing natural step tokens reproduces marker results (AUROC 0.868 vs 0.866); the paper's claim doesn't hinge on the artificial marker |
| Main failure mode | False alarms on long traces: correct-trace rejection drops 0.792 → 0.231 across token-count quartiles; GSM8K/MATH ≫ OlympiadBench/Omni-MATH on Process F1 |

**Bottom line:** hidden states contain a transferable invalidity signal that tracks the human
annotation better than chance and better than surface controls, but it is a gradual, drift-prone
transition rather than a sharp change point, and the experiment does not show the model *uses* it.
The paper honestly reports this row of the preregistered decision table.

---

## 3. What is still left

| Item | State |
| --- | --- |
| Human verification of the 135 drafted counterfactual corrections (`verified: true`) | **Blocked on you** — patching cannot run until then |
| Counterfactual activation patching GPU run (swap erroneous ↔ correct boundary states, measure verdict-score change) | Pending; gated on the baseline verdict separating correct/error prefixes |
| Yes/No verdict baseline + gated intervention rerun (replaces the degenerate CORRECT/INCORRECT readout) | Implemented; GPU rerun pending |
| Length-aware thresholds + paired probe-control intervals on the paper's numbers | Implemented; frozen fit/control predictions pending — no new numeric claim added to the paper yet |
| Same-family 7B replication | Added to Colab notebook; optional GPU run |
| September 2 tasks: final PDF polish, double-blind identity sweep, early upload | Per protocol |

---

## 4. What could make it better

**Scientific (highest value for the submission):**

1. **Verify and run the patching experiment** — it's the only remaining path to a causal statement
   that isn't confounded by the broken verdict readout, and the code is ready.
2. **Fix the verdict readout before any new causal claim** — the Yes/No baseline is implemented;
   rerunning interventions on a readout that actually has specificity is the single most important
   pending GPU job.
3. **Address the drift/false-alarm problem directly** — the failure analysis shows a global
   threshold is the weakness on long traces. The implemented length-bin thresholds could be
   promoted to a headline result if they materially improve Process F1.
4. **Sharpen localization beyond first-crossing** — e.g., change-point detection on score
   trajectories (CUSUM-style) instead of a fixed threshold, which matches the observed
   gradual-transition structure.

**Engineering (tracked in IMPROVEMENTS.md):**

5. Test coverage gaps: `localization.py` and 5+ untested functions in `analysis.py` — both are
   shared, load-bearing modules.
6. CI (GitHub Actions running `uv sync`, `pytest`, `ruff`) and pre-commit hooks to lock in the
   currently clean state.
7. Split the 1,459-line `probes.py` into fitting/evaluation/controls modules; add a `--verbose`
   flag and `logging` for long GPU stages.
8. Small wins: retry logic for HuggingFace downloads, `--dry-run` for expensive stages, consolidate
   duplicated prompt constants, richer config-validation messages.

---

## 5. One-sentence status

Decodability, partial localization, and cross-domain transfer are established and written up;
causality is inconclusive due to a degenerate readout with the fix implemented but not rerun; and
the remaining work is human verification of counterfactual pairs plus two GPU runs ahead of the
September 2 deadline.
