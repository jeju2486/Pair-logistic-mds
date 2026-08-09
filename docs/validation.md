# Validation status and release criteria

## Current status

KOVAR 0.8.1 is experimental. The implementation uses a standalone PQL
logistic mixed model, prospective score test, optional saddlepoint calibration,
and directional output. Passing unit tests establishes software consistency; it
does not establish statistical calibration.

Do not use the current release as the sole evidence for a biological or
clinical conclusion.

## Required numerical validation

The GLMM implementation should be compared against independent reference
calculations for:

- intercept-only logistic regression when \(\tau=0\);
- PQL null-model estimates on fixed small matrices;
- prospective score statistics and variances;
- singular but positive-semidefinite kinship matrices;
- full alternative fits and confidence intervals;
- non-convergence, quasi-separation, and boundary variance estimates; and
- deterministic agreement between single- and multi-process execution.

The 0.8.1 exact optimizations additionally require tests for:

- equality of the legacy and spectral-coordinate variance-component objective;
- equality of the prepared-kinship and direct first-iteration eigensystems;
- separately fitted versus reused identical response patterns;
- separately fitted versus transformed complementary response patterns,
  including signed score and effect quantities;
- preservation of every original pair and directional hypothesis after cache
  expansion; and
- accurate cache, stage-timing, and BLAS/process diagnostics.

Where feasible, fixed test fixtures should be generated with GMMAT or another
trusted GLMM implementation and stored with the generation method and software
version.

## Required calibration simulations

At minimum, simulations should cross the following factors:

- sample size;
- minor-state frequency: 50%, 20%, 10%, and 5%;
- balanced and asymmetric predictor/response frequencies;
- weak, moderate, and strong phylogenetic covariance;
- positive-semidefinite matrices with different effective ranks;
- minimum joint-cell count;
- null phylogeny-only association;
- association concentrated in one lineage;
- repeated association across lineages;
- positive and negative effects; and
- homogeneous versus lineage-heterogeneous effects.

Primary evaluation metrics are:

- type-I error at ordinary and multiple-testing tail thresholds;
- QQ-plot calibration;
- power for repeated cross-lineage covariation;
- false-positive control for one-lineage co-occurrence;
- direction recovery;
- odds-ratio bias and confidence-interval coverage;
- null- and full-fit convergence rates; and
- normal versus SPA calibration.

Runtime and memory should be recorded, but they are secondary to calibration.

## SPA validation

The optional SPA code must be evaluated separately from the normal score test.
Validation should include:

- comparison with direct enumeration for small independent Bernoulli examples;
- comparison with SPAtest or SAIGE-compatible reference scenarios where
  assumptions overlap;
- outcome ratios around 1:1, 1:9, 1:19, and more extreme imbalance;
- predictors near the 5% frequency threshold;
- sparse but non-empty four-cell tables;
- central scores where SPA should agree with the normal approximation; and
- explicit failure and fallback behavior when the saddlepoint cannot be found.

Agreement with SAIGE is not expected because KOVAR does not implement SAIGE's
full variance-ratio calibration and large-cohort numerical machinery.

## Kinship validation

Tree covariance tests should verify:

- exact sample-label matching and ordering;
- finite, non-negative branch lengths;
- the expected root-to-MRCA covariance on hand-calculated trees;
- mean-diagonal normalization;
- rejection of materially asymmetric or indefinite matrices; and
- correction only of tiny negative eigenvalues attributable to roundoff.

GRM tests should verify marker filtering, standardization, proxy masking, and
the number of background loci retained. Calibration must compare tree-based and
same-matrix GRM analyses because the latter can suffer proximal contamination
or loss of structure after masking.

## Release gates

Before KOVAR is described as statistically validated rather than experimental:

1. null type-I error must be acceptably calibrated over the intended sample and
   frequency range;
2. reference score and null-model tests must pass;
3. every failed fit must produce an explicit non-success status;
4. directional multiple testing must use the documented hypothesis count;
5. tree and GRM sensitivity must be quantified;
6. the optional SPA policy must either pass calibration or remain clearly
   experimental and off by default; and
7. a reproducible validation report must record seeds, dependencies, and exact
   KOVAR version.

Until these gates are met, report findings as exploratory KOVAR covariation
candidates.

Long performance runs should currently be treated as non-resumable. A later
checkpoint implementation requires input/configuration fingerprints, atomic
response shards, and deterministic reconstruction before multiple-testing
correction; version 0.8.1 does not claim this capability.
