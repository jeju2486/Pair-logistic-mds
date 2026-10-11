# Unitig matrix, tree heatmap and MIC ridge prediction

This optional downstream helper reads a completed covariation map. It changes
neither the main KOVAR runner nor the significance, distance, annotation or
effect-estimation workflow. It does not infer named resistance mutations.

Install in the environment used for downstream analysis:

```bash
python -m pip install -e '.[phenotype]'
```

## Inputs and identity contracts

- `--fasta`: the **sample-labelled** A/C fake FASTA used by KOVAR. One record per
  isolate; C/c = presence and A/a = absence. Duplicate IDs, unequal lengths or
  unexpected states are errors. Raw repeated `contig00001` headers are invalid.
- `--tree`: branch-length Newick tree with exactly the same uniquely named tips.
  A midpoint-rooted, ladderized copy is saved; the source is not edited.
- `--metadata`: TSV with `id`, `isolate`, a numeric MIC field and its sign field.
  Defaults are `daptomycin_mic` and `daptomycin_mic_sign`. The default join is
  `id + '_' + isolate.replace('.', '_')`, recorded in `phenotypes.tsv`. For other
  naming conventions use `--sample-map`, a one-to-one `metadata_id,sample` TSV.
- `--network`: completed network HTML with the embedded `data` JSON, or the same
  JSON object. All edges must retain `gene_a`, `gene_b` and `locus_pairs`.
- `--annotation`: locus-to-gene TSV containing `locus`, `gene`, `annotation_class`.
  Only coding-overlap mappings are included. Cluster IDs join records; biological
  annotation names label plots. Duplicate locus mappings are errors.
- `--target`: biological annotation name or exact cluster ID; default `mprF`.
  A name matching several distinct clusters requires an explicit cluster ID.

Exactly one column contract is required:

1. `--fasta-columns-are-locus-ids` declares that unfiltered column 0 is original
   zero-based unitig 0, column 1 is unitig 1, etc. Use only for the original
   sample-labelled matrix, not a filtered/reindexed SpydrPick FASTA.
2. `--locus-map FILE` supplies every `fasta_column,locus` pair explicitly, both
   zero based. The map must cover all columns once with unique original IDs.

The helper joins unitigs through annotation. **Sorted gene endpoint order does
not determine which unitig is u or v.** This avoids reversing rplX/lysP mappings.
It retains coding unitigs represented in the selected map for the target and its
direct/second-order neighbours, including support for those nodes' other edges.
It does not scan every variant across the full coding sequence of each gene.

## Running and outputs

```bash
ko-variation-phenotype \
  --fasta sample_ids.fasta --fasta-columns-are-locus-ids \
  --tree core.treefile --metadata collection.tsv \
  --network gene_covariation_functional.html \
  --annotation annotation/locus_to_gene.tsv --target mprF --out mprF_mic
```

`python -m ko_variation.phenotype` offers the same interface. `--stage tree`
exports the matrices, phenotypes, rooted tree and heatmap without fitting ridge.
`--no-full-tsv` suppresses the large full-genome text export while retaining the
full compressed binary cache and small neighbourhood TSV.

Outputs:

- `unitig_presence_absence.tsv.gz`: full isolate x original-unitig TSV.
- `matrix.npz`, `matrix.json`, `column_mapping.tsv`: binary cache, identity and
  original-column mapping.
- `features.tsv`, `neighbourhood_presence_absence.tsv`, `neighbourhood.json`:
  source unitigs, coding assignments, gene names and neighbourhood orders.
- `phenotypes.tsv`, `unmatched_metadata.tsv`: join and phenotype inclusion audit.
- `midpoint.treefile`, `tree_order.tsv`, `tree_presence_absence.png/.svg`.
- `predictions.tsv`, `performance.tsv`, `fold_performance.tsv`,
  `training_performance.tsv`, `model_features.tsv`.
- `ridge_performance.png/.svg`, `observed_predicted.png/.svg`.
- `ridge_checkpoints/`: completed outer folds; `run.json`: input SHA-256 values,
  package versions, settings, completion status and interpretation boundaries.
- `figure_captions.txt`: factual figure descriptions and control definitions.

SVG text remains editable. PNG defaults to 300 DPI. Unitig sequences themselves
are **not** present in the fake FASTA. Keep the original unitig sequence catalogue
with these outputs to interpret variants or call named substitutions later.

## Minimal modelling design

The response is log2(MIC), using only positive numbers with an explicit `=` sign.
Censored, missing-sign, conflicting-sign, invalid and missing records are
documented and excluded. Limits are never treated as exact values. Recorded
0.12 and 0.125 values remain distinct. Species/phenotype selection should precede
modelling: supply a species-specific collection table.

M0 contains map-represented coding unitigs assigned to the target. M1 adds direct
neighbour unitigs; M2 retains M1 and adds second-order unitigs. This is a
**target-unitig baseline**, not a curated resistance-substitution baseline.
Interpreting the result as information beyond all established resistance
determinants requires a separately defined and validated determinant baseline.

Maximum 10 complete-linkage clusters of pairwise tree patristic distances define
phylogenetic groups (`--groups`). They are computed without MIC. Outer grouped
cross-validation uses up to five folds (`--folds`), requiring at least three
phenotype-bearing groups. These partitions are shared across all models. The
clustering rule is a reproducible grouping choice, not a guarantee that every
related lineage is separated at a biologically established boundary.

Within each training split, predictors require at least five present AND five
absent isolates (`--min-count`). Exact duplicate columns collapse in input order,
with baseline columns first. Standardization uses training data only. Up to
three inner grouped folds choose alpha from 0.01, 0.1, 1, 10, 100, 1000. Inner
filters are fitted again inside each inner training fold. A model with no
retained variable columns predicts the training mean and is reported with zero
features, rather than pretending to have a genetic effect.

The default random background is coding-overlap unitigs represented by **other
nodes in the selected map**, excluding the entire target neighbourhood. This is
a map-background control, not unrestricted genome-wide random gene selection.
Each of 100 random replicates (`--random-sets`) uses the same baseline and matches
the additional retained predictor count separately to M1 and M2. Greedy nearest
training minor-frequency matching breaks ties randomly. Duplicate training
patterns are excluded. Actual mismatches are recorded; matching is approximate.
Random replicate IDs denote strategies across shared folds: their selected
unitigs can differ between folds because matching uses training data only.

Use `--random-pool` with a `locus` TSV for an independently justified background,
for example unitigs with verified distal placement relative to the target. The
helper never certifies physical distance to the target from a gene label or
from another gene's distal edge. Preserve and review upstream distance selection
provenance, particularly cross-contig handling.

Primary performance is RMSE from pooled out-of-fold predictions. Secondary
performance is the proportion within one twofold dilution and RMSE reduction
relative to M0. A training-mean-only benchmark is always included. Individual
fold scores are dependent diagnostics; random-set boxplots are control
distributions, **not confidence intervals or independent biological replicates**.

The supplied map was discovered using the cohort's genotypes and is frozen for
prediction. Held-out MIC values never tune feature selection or parameters, but
this fixed-map evaluation is not independent end-to-end validation of network
discovery. Additive ridge prediction does not establish phenotypic epistasis.

## Restart and resource use

Repeat the same command to reuse the binary matrix and completed outer folds.
Matrix identity uses the FASTA hash and indexing contract. Ridge identity also
includes network/annotation/metadata/tree hashes, modelling settings, algorithm
and dependency versions. Changed identities recompute results. Display-only DPI
changes do not invalidate completed fits. Incomplete folds are rerun; there is
no resumption halfway through an individual model fit.

The large full TSV is written progressively with low gzip compression. Fitting
loads only target/background columns, not a whole-genome copy for every model.
Numerical BLAS fitting uses one thread to avoid oversubscription; this helper
reports progress rather than hiding long runs. ARC-specific submission scripts
remain outside this public repository.
