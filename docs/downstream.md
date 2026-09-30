# Distal signals and gene networks

Install the optional exporter dependencies with `pip install -e '.[network]'`.
The scanner and existing `ko-variation-plot` command retain their interfaces.

```bash
ko-variation-annotation --results kovar_results/ko_variation.tsv \
  --annotation loci.tsv --out downstream/hits --network \
  --significance-column q_bh --significance-threshold 0.05 --ld-distance 10000
```

The same command is available as `python -m ko_variation.annotation_cli`.
Without `--network`, selection/effect calculation needs only core dependencies.

## Selection contract

The helper reads the current scanner's canonical zero-based `u < v` schema,
four joint counts, `status`, and a chosen significance column. Only `OK` and
`OK_SPA_FAILED` rows can pass. The latter uses the valid normal fallback after
SPA failure; choosing `p_spa` excludes its missing SPA values. Missing
significance values are excluded; malformed or out-of-range values are errors.
Significance is **<=** the threshold; physical distance is **>** the LD cutoff.
BH q-values are taken from the original full scan, never recalculated after
filtering. Choosing raw p-values requires an appropriate multiplicity decision.

`--distance-column` defaults to `distance`. The chosen column must contain
nonnegative finite distances or missing values; missing distances are excluded.
Input metadata units must match `--ld-distance`; the helper does not infer units
or interpret locus-index differences as base pairs. For PAN-GWES distance ranges,
consider explicitly choosing `min_distance` to require distal separation across
all represented genomes. A physical cutoff is a proxy for linkage, not measured
LD (r-squared), an LD-decay estimate, or evidence against co-transfer.

If the chosen distance column is absent, supply annotation `contig` and
`position` columns. Positions must use one coordinate convention and reference;
absolute differences give linear physical distances. Circular chromosomes need
a precomputed shortest-path distance column; linear coordinates alone are not
sufficient. When contigs are known, cross-contig pairs are excluded by default;
`--cross-contig distal` retains them with missing physical distance and an
explicit `cross_contig` class. Missing contig annotations do not establish
cross-contig separation. An existing distance column takes priority over
annotation coordinates. A distance-only input cannot identify cross-contig pairs.

## Raw effects and adjusted inference

For presence/absence coding, `raw_odds_ratio = n11*n00/(n10*n01)`; OR > 1 means
co-presence/co-absence enrichment and OR < 1 means opposing states. Swapping
the two loci preserves OR; complementing one locus inverts it. These are
**unadjusted descriptive effects**, even when selected by phylogeny-adjusted
score p-values. They do not establish epistasis or a causal direction.

If any cell is zero, the default adds 0.5 to all four cells of that table;
otherwise no correction is applied. `raw_or_correction` records the addition.
Use `--zero-cell-correction 0` for uncorrected boundary values: OR can be zero,
infinite, or undefined. A two-sided Wald interval uses log(OR) +/- z*sqrt(sum(1/n)),
with `--confidence 0.95` by default. Uncorrected zero cells yield missing SE/CI.
These intervals assume independent observations; they do not account for
phylogeny, selection, or multiple testing and can be unreliable for sparse cells.

No phylogeny-adjusted effect is estimated by these helpers. Such an estimate
requires original sample genotypes, the aligned tree/covariance, and an
alternative logistic mixed model with the other locus as a predictor, including
convergence, separation, variance-component and uncertainty diagnostics. The
scanner's null-model p-value or z-statistic cannot supply that coefficient.

## Annotation and gene aggregation

`loci.tsv` is tab-separated with one row per zero-based `locus` and a nonempty
`gene`. Optional `label`, `product`, and `group` annotate nodes; optional
`contig`/`position` enable the distance fallback. Use stable gene IDs; repeating
a gene name for distinct genomic copies collapses them into one node. Duplicate
locus mappings are errors. One-to-many mappings must be resolved before use.

Missing gene mappings cause an error during network construction unless
`--missing-genes drop` is specified. Same-gene pairs are omitted unless
`--include-self` is set. Node size represents the number of distinct other gene
neighbours; color represents annotation group. Tables report distinct contributing
loci and supporting pair counts. Conflicting annotation attributes are joined
in sorted order, retaining all supplied text.

Each unordered gene edge aggregates unique locus pairs and records their IDs,
count, minimum selected significance, and the raw OR of the most significant
supporting pair (ties resolved by u/v). **The minimum is descriptive, not a
gene-level p-value or corrected gene test.** ORs are never pooled: pairs can use
the same isolates and be statistically dependent. Edge color records positive,
negative, mixed, neutral, or unknown raw directions; width represents pair count.
Representative ORs remain locus effects, not inferred gene interaction strengths.

Outputs are `PREFIX.distal.tsv`, `PREFIX.selection.json`, and, with `--network`,
`PREFIX.nodes.tsv`, `PREFIX.edges.tsv`, `PREFIX.png`, and `PREFIX.html`.
The JSON records version, selection settings, input paths and SHA-256 hashes,
counts and exporter settings.
PNG is 300 DPI by default (`--dpi`); PNG and HTML use the same deterministic
layout (`--seed 42`). The HTML works offline with search, pan, zoom, hover and
click details. Gene/annotation text is escaped, with no external scripts.
Large dense networks may require stricter selection or separate component plots
for readable publication panels. An empty selection still produces valid outputs.

## Python entry points and reproducible example

```python
from ko_variation.postprocess import SelectionConfig, select_distal_signals
from ko_variation.network import build_gene_network, export_gene_network

signals = select_distal_signals(results, SelectionConfig(ld_distance=10000), annotation)
nodes, edges = build_gene_network(signals)
export_gene_network(nodes, edges, "downstream/hits")
```

Here `results` and `annotation` are pandas DataFrames loaded from the TSVs.
Run `python examples/downstream/reproduce.py` from the installed repository.
It generates synthetic current-schema scanner output and all downstream files
under `examples/downstream/output/`. It uses zero relatedness as a statistical
oracle and makes no biological claims. Run the tests with
`python -m unittest discover -s tests -v`.
