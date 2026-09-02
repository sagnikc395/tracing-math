# CPU-only follow-up results

This analysis reuses the frozen Experiment 1 test predictions and intervention tables. It does not
load the language model, fit a new hidden-state probe, or inspect activation tensors. The tests are
post-hoc and are reported as robustness analyses rather than preregistered confirmation.

## Temporal alignment

The original detector localized 28.9% of first errors exactly. Circularly
shifting each score trajectory within its trace reduced the null mean to
16.4% (permutation p = 0.0002). Within-one-step
localization was 59.3%, compared with a shifted null mean of
43.6% (p = 0.0002). The frozen score is
temporally related to the annotation, although exact localization remains low in absolute terms.

## Onset change and trace-level offsets

The mean score jump at the annotated onset was 0.257. Metadata-matched
transitions from correct traces changed by 0.113; the paired difference was
0.144 with a
95% interval of
[0.096, 0.193].

Among erroneous traces whose first error occurred after step 0, pooled AUROC was
0.881. Subtracting each trace's first-step score gave AUROC
0.881 [0.862,
0.900]. Mean AUROC calculated separately within each eligible trace was
0.968. Stable trace-level offsets do not explain all of the frozen
probe's discrimination.

## Causal assay

The unmodified verdict score had AUROC 0.342 and specificity
0.203. The smallest approximate effect detectable with 80% power
across the tested doses was 0.00290 verdict-margin units. Statistical
sensitivity does not repair a readout that fails to distinguish valid from invalid boundaries.

## Length-aware thresholding

Thresholds fitted within train+validation trace-length bins changed test Process F1 from
0.393 to 0.377 and correct rejection
from 0.612 to
0.540.


## Probe-versus-control paired intervals

Whole-trace paired bootstrap intervals are in `probe_control_paired_intervals.csv`.


## Prefix and metadata shortcut controls

- prefix TF-IDF: AUROC 0.760, Process F1 0.281
- structural metadata: AUROC 0.780, Process F1 0.147
- metadata plus final outcome: AUROC 0.857, Process F1 0.288


## Trace-equal weighting

Boundary-weighted AUROC was 0.866; assigning every trace equal
total weight gave 0.867 (difference
+0.001, 95% interval [-0.007,
+0.010]).


## Files

The output directory contains the permutation draws, error-aligned trajectory, matched-placebo
results, centered discrimination, subgroup outcomes, causal sensitivity table, a frozen qualitative
audit sample, and a two-panel temporal-validity figure.
