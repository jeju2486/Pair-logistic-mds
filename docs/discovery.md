# Unitig discovery in KOVAR 0.9.0

`ko-variation discover` (also `ko-variation-discover`) accepts PAN-GWES unitig
presence/absence directly. SpydrPick, MI and ARACNE are not required. The old
`ko-variation --pairs ...` scanner remains available for existing workflows.

## Install and run

```bash
python -m pip install -e .
ko-variation --version

ko-variation discover \
  --fasta graph.sample_ids.fasta --tree core.treefile \
  --unitigs graph.unitigs --paths graph.paths \
  --unitig-distance /path/to/unitig_distance --kmer-length 61 \
  --out kovar_v090 --threads 4 --memory-gb 4
```

For older distance binaries that require global edges, also pass
`--edges graph.edges`. The adapter examines the installed binary's help and
passes the older `--edges-file`/`--run-sggs-only` options only when supported.
Relative entries in `.paths` are interpreted from the launch working directory;
absolute paths are preferable. The graph and distance tools are external
dependencies, not installed by KOVAR. The adapter expects PAN-GWES distance
columns including count and minimum distance (columns 6 and 8, one-based).

To defer graph-distance calculation and beta fitting, omit `--unitig-distance`.
Add it on a later `--resume`; no significance scan is repeated. Use `--no-beta`
for significance plus distance only. A subsequent resume can add beta.

## Inputs and identity

- Pseudo-FASTA has unique sample IDs and A=absence/C=presence. Repeated
  `contig00001` headers are rejected. Sequence columns are zero-based graph
  unitig IDs; filtering never renumbers them.
- `.unitigs` contains consecutive zero-based IDs followed by sequence, with
  exactly one row per original pseudo-FASTA column.
- Tree covariance is computed in retained FASTA sample order. Missing tree
  samples are an error unless `--tree-missing-samples drop` is explicit.
- `--exclude-samples ids.txt` removes specified isolates before filtering or
  tree covariance; excluded IDs are recorded in the run summary.
- MAF >=0.05 is the default. Exact pattern deduplication preserves all original
  mappings. Complementary patterns remain distinct. Within-identical-pattern
  pairs are deterministic redundancy and are not assigned score-test P-values.

## Search and statistical decisions

The default global hierarchy uses positive binary phi-squared >=0.8, followed
by nested 0.9, 0.95, 0.98 and 0.99 partitions and terminal blocks of 32.
Membership must satisfy the threshold directly against an observed anchor;
there is no transitive single-linkage grouping. Necessary presence-count bounds
eliminate impossible memberships before packed-bit parallel comparisons.

The default `single` policy uses a central observed pattern (maximum mean
correlation to members). This is computed from summed standardized vectors
in linear work, without a dense within-category matrix. `--rep-policy coverage`
adds farthest-first representatives, with ceil(1% of node size) representatives,
capped by `--max-reps 10`. Global membership still uses fixed anchors.

Representative comparisons use a relaxed `--screen-p 0.001` exploration gate
with no joint-cell restriction. Internal categories remain searchable even
when they have no external connections. Unresolved gates expand conservatively.
Negative representative gates can prune blocks: **this is heuristic and can
miss significant members**. Neither similarity nor representative significance
is copied to member pairs. `--search exhaustive` removes representative pruning
but still enumerates all unordered unique-pattern pairs.

Final eligibility defaults to `--min-cell-count 20` in each 2x2 cell, matching
the recent benchmark target. This intentionally excludes sparse patterns/pairs;
it is configurable and is not an intrinsic KOVAR requirement. It differs from
the legacy scanner's default of 1. The score response is the higher pattern ID
and the predictor the lower ID. IDs follow first occurrence by original locus.

Final significance is **Bonferroni adjusted P <=0.05**, not raw P <=0.05 or
P <=0.001. The family is all unordered unique-pattern pairs before hierarchy,
joint-cell or distance filtering. `--bonferroni-tests` can freeze a larger family
but cannot shrink it below that count. The family is at the pattern-test level;
expanded duplicate unitigs do not constitute additional independent tests.
For tiny panels, provide a screening gate at least as permissive as alpha/M.

The adaptive kernel bounds T=U^2/V after the fitted response null. Partial
weighted covariance projections bound the remaining variance; ranks
0/8/16/32/64/128 are added only where decisions are unresolved. Roundoff guards
widen intervals. Singular cases and intervals straddling either required
cutoff fall back to the optimized exact score core. `--exact` disables bounds
for a reference comparison. SPA is **off** in discovery; these intervals do not
certify SPA-adjusted tails. PQL/ordinary-score calibration retains the existing
model limitations; threshold certification is not experimental epistasis.

## Distance and post-hoc beta

Every significant pattern pair is expanded to its original unitig pairs.
Those IDs are supplied to `unitig_distance` in bounded zero-based query files;
unused MI/ARACNE fields are placeholders, not association measurements.

All significant candidates are retained:

| Minimum graph distance | Status | Post-hoc beta |
|---|---|---|
| >10,000 bp | DISTAL | fitted unless `--no-beta` |
| <=10,000 bp | LOCAL | skipped |
| Missing, negative sentinel, or no connected genomes | UNKNOWN | skipped |

Distance means **minimum shortest-path distance across connected single-genome
graphs**, not a reference chromosome coordinate. Unknown/disconnected pairs are
not presumed distal. The cutoff can be set with `--distal-bp` before the first
distance stage, then is frozen. Beta is an alternative phylogenetic logistic
mixed-model coefficient, fitted once per selected distal pattern pair and
reported only on distal original-unitig rows. Separation, nonconvergence,
clipped weights and invalid variance remain explicit effect statuses.

## Memory, CPUs, scratch and restart

`--threads` is a total CPU budget. Independent numerical tasks run in process
workers with one BLAS thread each; preparation's covariance decomposition and
the external distance executable use the same CPU budget in separate phases.
Parallel work submission is bounded, including on Python versions whose
Executor.map would eagerly consume all jobs. Response-major batches prevent
concurrent duplicate null fits within a frontier. Null RAM slots are limited
by `--memory-gb`; this is an allocation estimate, not an OS-enforced RSS limit.

Pseudo-FASTA is decoded one sample at a time to a temporary mapped matrix.
Locus blocks are packed/deduplicated through a disk index. Patterns and shared
covariance eigenvectors are mapped read-only. Category memberships, traversal
tasks, queried-pair decisions, distance results and beta statuses live in
SQLite, not full in-memory pair matrices. Initial category-pair queries are
generated incrementally; a billion-row initial task list is not preallocated.

Default work is `OUT/.kovar_work`; `--tmp-dir` can select persistent scratch.
**Keep it for restart and null reuse**. Do not use job-local temporary storage
that is deleted when Slurm ends. Startup estimates scratch preparation and
possible null storage. Full-pool nulls can require hundreds of GB or more;
disk-backed computation does not make storage free. SQLite transactions need
a filesystem with reliable file locking. An OS-released coordinator lock
prevents simultaneous writes to one work directory.

Repeat the same command with `--resume`. Committed sample rows, deduplication
blocks, category assignments, pair decisions, traversal batches, distance
batches and beta fits are reused. An unfinished worker job may repeat.
Input contents, fitting implementation, numerical environment and scientific
settings are checked. CPU count, memory budget and batch size may be adjusted
where they do not change scientific settings (currently batch size is frozen).
Changing scientific settings requires a new output. Distances/beta can be added
later, but changed distance inputs/cutoffs cannot overwrite earlier results.

### Reuse existing nulls

```bash
ko-variation discover ... --reuse-nulls-from /path/previous_output
```

Supply a v0.9 output or the former combined-benchmark output. Benchmark reuse
requires its original `settings.benchmark_dir/preparation.npz` and referenced
prior null directories to remain accessible. Both v1 and v2 provenance are
supported; v2's borrowed v1 cache is followed automatically. Sample order,
exact K, binary response patterns, fitting-source fingerprint, fitting settings,
array checksums and NumPy/SciPy versions must match. Patterns are matched by
bits, not by assuming old pattern IDs are unchanged. Arrays are read in place;
they are not copied or silently refitted. Only missing nulls are fitted.
`--require-cached-nulls` prohibits any missing-model fitting. Cached failed
fits stay unresolved. Old pair-decision checkpoints are not imported into the
new search implementation.

## Final files

- `unitigs.tsv.gz`: original ID, exact pattern ID, presence count, sample count
  and sequence-source path. Blank pattern IDs mark filtered loci.
- `associations.tsv.gz`: significant original-unitig pairs, canonical pattern
  orientation, optional exact P, labelled P bounds, minimum graph distance,
  connected-genome count, distance status and optional post-hoc beta.
- `run_summary.json`: family/cutoff, sample exclusions, decision counts,
  unqueried count, search interpretation, distance/effect statuses and runtime.
- `run_identity.json`: provenance for restart and cache compatibility.

No exact P is manufactured for a bound-certified edge. A small P remains genomic
association, not proof of a phenotype interaction, causality or mechanism.
The work database retains queried nonsignificant/ineligible/unresolved decisions
for restart, without exporting billions of unnecessary text rows. Final exports
are streamed and atomically replaced; an interrupted export can be regenerated.

An annotated distal map can use the existing network helper directly:

```bash
ko-variation-network --effects kovar_v090/associations.tsv.gz \
  --annotation locus_to_gene.tsv --gene-catalog gene_catalog.tsv \
  --out maps/covariation
```

Keep `run_summary.json` beside the association file. The helper validates its
Bonferroni cutoff, selects certified DISTAL rows and uses the explicitly labelled
`p_upper` as descriptive significance, without requiring exact P-values or
refitting beta. Gene names and functional colours retain the existing display
options. This optional visualization helper loads the significant edge table;
it is separate from the memory-bounded all-pattern discovery scan.

## Focused benchmark

`benchmarks/discovery_benchmark.py` runs exact, bounded exhaustive, and hierarchical
production paths on a supplied **small frozen pseudo-FASTA panel**. It reuses
compatible fitted nulls and reports query counts, recovery and invocation time.
The default cap is 2,000 original loci; it does not silently launch a full-pool
exhaustive scan. Exact invocation time includes fitting; the next arms reuse
models, so these are not independent cold-run timings. There are no full ARC
runtime or whole-pool recovery guarantees in this release.
