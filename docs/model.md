# Statistical model in KOVAR 0.8.1

KOVAR 0.8.1 is an **experimental** directional binary-locus association
scanner. It asks whether the state of one locus predicts the state of another
after accounting for covariance among samples. The result is a
kinship-adjusted covariation candidate, not proof of a causal or biochemical
interaction.

## Directional hypotheses

For a candidate pair containing loci `u` and `v`, the default `both` mode fits
two distinct hypotheses:

1. `u_predicts_v`: `u` is the predictor and `v` is the binary response.
2. `v_predicts_u`: `v` is the predictor and `u` is the binary response.

These directions are statistical. A result labelled `u_predicts_v` does not
show that `u` evolved first, causes `v`, or regulates `v`.

KOVAR keeps the two directions as separate output rows and includes both in
directional multiple-testing correction. It does not collapse them into a
minimum or maximum pair score.

## Logistic mixed model

For response locus \(v\), sample \(i\), and predictor locus \(u\), the
alternative model is

\[
Y_{iv} \sim \operatorname{Bernoulli}(\mu_{iv}),
\]

\[
\operatorname{logit}(\mu_{iv}) = \alpha_v + \beta_{u,v}X_{iu} + b_{iv},
\]

\[
b_v \sim N(0, \tau_v K).
\]

Here:

- \(Y_{iv}\) is the presence/absence state of response locus \(v\);
- \(X_{iu}\) is the presence/absence state of predictor locus \(u\);
- \(K\) is the sample covariance obtained from a tree or background genomic
  relationship matrix;
- \(\tau_v\) is the response-specific phylogenetic variance component; and
- \(\beta_{u,v}\) is a directional log-odds effect.

KOVAR uses a standalone penalized quasi-likelihood (PQL) implementation with
pseudo-REML profiling of the single variance component. This follows the
general approximate-GLMM framework of Breslow and Clayton and the
null-model/score-test architecture used by GMMAT. It is not an exact numerical
reimplementation of GMMAT.

PQL is an approximation. It can be biased when sample size or binary-state
counts are small, and its calibration must be assessed for the phylogenetic
structures and frequency range used by KOVAR.

## One null model per exact response pattern

For each unique response locus, KOVAR first fits the null model

\[
\operatorname{logit}(\mu_{iv}) = \alpha_v + b_{iv}.
\]

Before fitting, KOVAR bit-packs each eligible binary response and
canonicalizes the response/complement pair. Identical response vectors share
the same null fit. For a complemented response,

\[
Y^*=1-Y,\qquad \mu^*=1-\mu,
\]

so the Bernoulli working weights, fitted \(\tau_v\), and mixed-model projection
are unchanged, while signed response quantities are transformed back to the
original presence-state orientation. KOVAR verifies exact packed-vector
equality rather than relying on a hash alone.

The fitted response probabilities, working weights, estimated \(\tau_v\), and
projection quantities are reused for all eligible predictors and all original
loci assigned to that exact canonical pattern. Every requested direction
remains in the output and in multiple-testing correction. This reuse is not a
frequency, lineage, or candidate-pair filter.

### PQL calculation reuse

The initial null probability is constant across samples, hence the first
working-weight matrix is \(W_0=w_0I\). Therefore

\[
W_0^{1/2}KW_0^{1/2}=w_0K,
\]

and the prepared eigensystem of \(K\) can be reused exactly for every first PQL
iteration. Subsequent weights vary by response and sample, so their weighted
eigenvectors cannot generally be reused as in a linear mixed model.

After each later weighted eigendecomposition, KOVAR rotates the working
response and fixed-effect design once. Trial values of \(\tau_v\) are profiled
using diagonal operations in that spectral coordinate system. This changes the
algebra and amount of computation, not the optimized pseudo-REML objective.

The output reports

\[
\rho_{v,\mathrm{latent}} =
\frac{\tau_v}{\tau_v + \pi^2/3}
\]

as a descriptive latent-scale phylogenetic fraction. It is not an LMM
heritability estimate and is not constrained by an `h2` cap.

## Prospective score test

For an eligible predictor, KOVAR calculates a prospective score statistic from
the fitted null model. In simplified notation,

\[
U = x^{T}(y-\widehat{\mu}),
\]

which is equivalent to using a fixed-effect-residualized predictor at a
converged null fit. The variance uses the mixed-model projection. The normal
score statistic is

\[
Z = \frac{U}{\sqrt{\operatorname{Var}_0(U)}}.
\]

The default primary p-value is the two-sided normal/chi-square score-test
p-value. `beta_score` and `se_score` are one-step score approximations; they are
not substitutes for a converged alternative-model estimate.

## Full alternative refit

Directions passing the configured refit threshold are fit again with the
predictor in the fixed-effect design. A successful refit provides:

- `beta_log_odds`;
- `se_log_odds`;
- `odds_ratio` and its confidence interval;
- `p_wald`; and
- `full_refit_status`.

Sparse or separated data can make an alternative fit unstable. KOVAR reports
that status and does not silently treat a failed refit as a reliable odds-ratio
estimate. The score p-value remains the primary screening result.

## Frequency and cell-count filters

The default marginal filter is a 5% minor-state frequency for both response and
predictor:

\[
\operatorname{MSF}(x) = \min(\bar{x}, 1-\bar{x}) \ge 0.05.
\]

The command and output retain the familiar term `MAF`, but the input is a
binary locus state rather than a diploid allele count. KOVAR models the raw
presence state; it does not recode the less frequent state to 1.

Marginal frequency does not guarantee an informative joint table. KOVAR also
requires a minimum count in all four predictor/response cells:

| Cell | Meaning |
|---|---|
| `n11` | predictor present, response present |
| `n10` | predictor present, response absent |
| `n01` | predictor absent, response present |
| `n00` | predictor absent, response absent |

The default minimum is 5 per cell. When direction is reversed, `n10` and `n01`
are reversed as well. This threshold is a safeguard against extremely sparse
tables, not a guarantee of adequate effective sample size or independent
evolutionary replication.

## Kinship covariance

### Preferred: external core-genome tree

An external branch-length Newick tree is preferred because it can describe
sample ancestry independently of the accessory loci being tested. KOVAR builds
the covariance

\[
K_{ij}=\text{root-to-MRCA branch length for samples }i\text{ and }j
\]

and scales it to mean diagonal 1. Tree tip labels must match input sample names.
Rooting and branch lengths affect the covariance and therefore the result.

### Fallback: pangenome-derived GRM

When no tree is supplied, KOVAR can derive a relationship matrix from eligible
background loci in the binary input. Proxy masking reduces direct leakage of a
tested response pattern into the relationship matrix, but the fallback remains
less clean than an independent tree:

- removing too few tested-locus proxies can make the random effect absorb the
  tested association;
- removing too many loci can weaken population-structure representation; and
- the same pangenome is being used both to construct background covariance and
  to test covariation.

The kinship source and construction diagnostics must therefore accompany any
reported result.

### No scientific shrinkage or rank truncation

Version 0.8.1 does not use the 0.7.0 `h2` cap, identity-shrinkage lambda, or
low-rank kernel truncation. A finite, symmetric covariance is normalized and
checked for positive semidefiniteness. Only tiny negative eigenvalues consistent
with floating-point roundoff are corrected; a materially indefinite matrix is
rejected.

## Optional saddlepoint approximation

The default p-value uses the normal score approximation. SPA is optional and
off by default.

KOVAR's SPA code applies a Lugannani-Rice approximation to a weighted centered
Bernoulli score and uses a variance-ratio adjustment between the conditional
Bernoulli variance and the mixed-model score variance. This is motivated by
SPAtest and SAIGE, but it is **not SAIGE-equivalent**. In particular, KOVAR does
not reproduce SAIGE's complete large-sample variance-ratio calibration,
conjugate-gradient implementation, or biobank-scale optimization strategy.

For SPA's Bernoulli cumulant calculation, the predictor is residualized against
the null fixed effects using the final working weights. This is distinct from
the mixed-model projection used for the target score variance.

SPA cannot create information in an empty or nearly empty joint cell and cannot
turn repeated observations from one clade into independent evolutionary
replication. Its KOVAR implementation remains experimental and requires
simulation calibration before routine use.

## Multiple testing

KOVAR treats each successfully tested direction as a hypothesis. It reports:

- `p_primary`, selected from the normal score or successful requested SPA;
- `q_bh`, Benjamini-Hochberg adjustment across finite directional tests;
- `bonferroni_significant`; and
- `n_directional_tests`, the number of finite primary p-values used.

Filtered and failed rows remain in the result with missing inferential values
and an explicit status.
