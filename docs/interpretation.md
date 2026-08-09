# Interpreting KOVAR results

## What a positive result means

A low KOVAR p-value means that the predictor locus provides evidence about the
binary state of the response locus after adjustment by the selected sample
covariance model.

Within this project, such a result may be called an **epistasis candidate**. The
precise interpretation is:

> a maximally inferred, kinship-adjusted directional covariation candidate.

It is not direct evidence of a fitness interaction, molecular mechanism,
evolutionary ordering, or causality.

## Direction is predictive, not causal

`u_predicts_v` and `v_predicts_u` answer different statistical questions because
the response prevalence, null variance component, logistic weights, and
convergence behavior can differ. They should be interpreted separately.

Possible patterns include:

- both directions significant: robust bidirectional statistical dependence;
- only one direction significant: asymmetric statistical evidence, which can
  arise from prevalence, information, model fit, or biology;
- neither direction significant: no supported adjusted covariation at the
  current power and filtering thresholds; or
- one direction unresolved: a filter or fitting failure prevents comparison.

Even when only `u_predicts_v` is significant, do not write that `u` causes `v`
without independent temporal, experimental, or mechanistic evidence.

## Positive and negative covariation

A positive fitted `beta_log_odds` or odds ratio greater than 1 indicates that
presence of the predictor is associated with increased odds of response
presence. A negative coefficient or odds ratio below 1 indicates dissociation.

Use the full-refit coefficient and interval when `full_refit_status` is `OK`.
The continuity-corrected descriptive odds ratio and `beta_score` are useful
diagnostics but are not equivalent to a stable full mixed-model estimate.

## Global and lineage-specific signals

The kinship random effect is intended to reduce associations explained by
shared lineage. This supports the search for covariation that recurs beyond a
single clonal background, but it does not prove global evolutionary replication.

Important consequences are:

- a pair confined to one lineage may be absorbed as population structure;
- true lineage-specific covariation may also be weakened because it is
  statistically aligned with that lineage;
- a heterogeneous effect can be diluted by a single global coefficient; and
- residual confounding can remain when the tree or GRM poorly represents the
  relevant ancestry.

KOVAR 0.8.1 does not perform ancestral reconstruction or CTMC filtering.
Cross-lineage stability analysis is a planned diagnostic rather than part of
the primary test. Claims of globally repeated covariation require lineage-aware
sensitivity analysis or independent evolutionary evidence.

## Non-epistatic explanations

Significant adjusted covariation may result from:

- functional dependency or compensation;
- physical linkage;
- carriage on the same mobile element;
- co-transfer or shared loss;
- adaptation to the same environment;
- annotation, assembly, or clustering artifacts; or
- population structure not fully represented by the covariance.

These alternatives should be assessed before assigning a mechanistic epistasis
interpretation.

## Reading filter and fit statuses

Common pre-test statuses include:

- `LOW_PREDICTOR_MAF`: predictor minor-state frequency is below the threshold;
- `LOW_RESPONSE_MAF`: response minor-state frequency is below the threshold;
- `LOW_CELL_COUNT`: at least one joint cell is below the threshold;
- `NEAR_REDUNDANT`: the direction was excluded as a near copy or complement; and
- `ELIGIBLE`: the direction passed pre-test filters.

After testing, `OK` means that the null model and score calculation succeeded.
Statuses containing `NULL_`, `SINGULAR_`, `SPA_FAILED`, or `REFIT_FAILED` require
inspection and should not be converted to significant results by downstream
code.

## Recommended reporting

For every reported candidate, include:

1. predictor and response locus identifiers;
2. direction;
3. predictor and response prevalence and minor-state frequency;
4. all four joint-cell counts;
5. kinship source and tree or GRM construction details;
6. `p_primary`, its method, and the multiple-testing result;
7. null-model convergence and estimated phylogenetic variance;
8. full-refit effect and interval, if successful; and
9. lineage sensitivity or independent validation when claiming global support.

Prefer the phrase **KOVAR covariation candidate** or **KOVAR epistasis candidate**
over confirmed epistasis.
