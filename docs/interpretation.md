# Interpreting KOVAR results

A low `p_primary` means that the two binary loci show stronger association than
expected under the fitted phylogenetic null model. It is evidence against
`H0: beta=0`, not the magnitude of covariation and not proof of epistasis.

KOVAR reports one row per unordered pair. `u` and `v` are sorted locus IDs. The
internal logistic response orientation is deterministic and should not be
reported as causal, temporal, or evolutionary direction.

`primary_method=score_normal` uses the normal score approximation.
`primary_method=score_spa` means the automatic SPA calculation succeeded and
replaced the normal-tail p-value. Inspect `p_score`, `p_spa`, `spa_status`, the
four joint cells, and response-model convergence when reviewing a hit.

Population correction can weaken a true signal confined to one lineage because
that signal is statistically aligned with ancestry. Conversely, residual
confounding may remain if the tree is inaccurate. A significant pair may also
reflect physical linkage, carriage on one mobile element, co-transfer, shared
environmental selection, or data artifacts.

Recommended reporting includes pair IDs, pair-source metadata, both MAFs, all
four joint counts, primary method and p-value, multiple-testing result, tree
construction, KOVAR version, and null-model convergence. Call results
"phylogeny-adjusted covariation candidates" unless independent evidence supports
a mechanistic epistasis claim.

The separate `ko-variation-annotation` workflow calculates raw descriptive table
odds ratios after explicit significance and physical-distance filters, and
exports annotated gene networks. These ORs are unadjusted; adjusted estimates
require original genotypes/tree and an alternative model. Gene-edge significance
minima are summaries, not gene-level tests. See [downstream.md](downstream.md).
