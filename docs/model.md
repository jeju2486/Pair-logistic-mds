# Statistical model in KOVAR 0.8.3

KOVAR tests one hypothesis per unordered binary-locus pair. The loci are
canonicalized to `u < v`; `u` is used as predictor and `v` as response only to
make the computation deterministic. This orientation is not biological
directionality.

For sample `i`, the alternative association parameter is defined by

\[
Y_{iv}\sim\mathrm{Bernoulli}(\mu_{iv}),\qquad
\mathrm{logit}(\mu_{iv})=\alpha_v+\beta_{uv}X_{iu}+b_{iv},
\]

\[
b_v\sim N(0,\tau_v K).
\]

`K` is calculated once from root-to-MRCA branch lengths in the supplied rooted
tree and normalized to mean diagonal one. The null model sets `beta=0`.

## PQL and null-model reuse

KOVAR estimates the null with penalized quasi-likelihood (PQL) and profiles the
response-specific variance component `tau`. PQL is approximate and can be
biased in sparse binary data. Identical and complementary response vectors are
bit-packed and share an exact canonical null fit. Predictor columns sharing a
response are scored in blocks.

The eigensystem of `K` is reused exactly in the first PQL iteration because its
initial weights are constant. Later PQL weights depend on the response, so the
weighted eigendecomposition remains response-specific.

## Score test and SPA

At the fitted null, the score is approximately

\[
U=x^T(y-\widehat\mu),\qquad
Q=U^2/\mathrm{Var}_0(U).
\]

`p_score` is the chi-square/normal-tail probability for `H0: beta=0`. It measures
statistical evidence, not covariation magnitude. KOVAR does not fit the full
alternative or report an odds ratio in 0.8.3.

SPA is `auto` by default. It is attempted when `p_score <= 0.05` and either
locus has minor-state frequency at most 0.10 or the minimum joint cell is below
10. A successful SPA result becomes `p_primary`; otherwise the normal score
p-value remains primary and the status records the SPA failure. This
variance-ratio SPA is motivated by SPAtest and SAIGE but is not SAIGE-equivalent.

## Pre-test filters

Filters run before null-model task scheduling:

1. If the pair table has a `count` column, require
   `count >= ceil(0.05 * number_of_samples)`.
2. Require minor-state frequency at both loci to be at least `--min-maf`
   (default 0.05, inclusive).
3. Require each of `n11`, `n10`, `n01`, and `n00` to be at least
   `--min-cell-count` (default 1).

Filtered pairs remain in the output. BH and Bonferroni corrections use only
finite `p_primary` values, once per unordered pair.

## Scope

The model adjusts for ancestry represented by the tree but cannot prove
independent evolutionary recurrence. KOVAR 0.8.3 does not perform CTMC
filtration, LD classification, gene annotation, full-model likelihood-ratio
testing, or effect-size estimation.
