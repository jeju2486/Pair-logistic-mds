# Validation status and release criteria

KOVAR 0.8.3 remains experimental. Unit tests establish software contracts, not
biological validity or calibrated type-I error.

The release suite checks canonical pair reversal invariance, rejection of
duplicate unordered pairs, inclusive 5% MAF handling, the one-cell default,
graph-count filtering before model scheduling, response-pattern reuse, SPA
policy, score-only checkpoint recovery, kinship validation, and single versus
multi-process agreement.

Statistical validation still requires simulation across realistic bacterial
tree shapes, sample sizes, locus frequencies, lineage concentration, linkage,
recombination, and null/associated pair generation. Compare normal-score and
automatic-SPA calibration, power, failure rates, and sensitivity to tree error.
Reference small-matrix comparisons should cover logistic regression at
`tau=0`, PQL null estimates, score variances, singular positive-semidefinite
covariances, separation, and variance-component boundaries.

Before confirmatory use, require acceptable null calibration at the intended
frequency range, reproducible seeds and dependency versions, explicit failure
statuses, exact hypothesis counts, and external biological validation. Resume
must also be tested on the target scheduler/filesystem before a multi-day run.
