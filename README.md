# KOVAR

KOVAR (KO-Variation) screens candidate pairs of binary pangenome loci for
covariation after adjustment for shared ancestry. It uses a phylogeny-derived
sample covariance, a binary logistic mixed-model null fit, and a score test.
The output is evidence against no adjusted association; it is not an effect
size, a causal direction, or proof of biological epistasis.

Version 0.8.3 reports one result for each unordered pair. The separate
`ko-variation-annotation` helper selects distal significant signals, calculates
phylogeny-adjusted alternative-model odds ratios, and exports annotated gene networks.
See [downstream usage and interpretation](docs/downstream.md).

## Installation

```bash
git clone --branch feature/downstream-helpers https://github.com/jeju2486/Pair-logistic-mds.git
cd Pair-logistic-mds
pip install -e .
ko-variation --version
```

Python 3.10 or newer is required. KOVAR depends on NumPy, pandas, SciPy, and
threadpoolctl.

## Quick start

```bash
ko-variation \
  --fasta binary.fa \
  --pairs candidates.tsv \
  --tree rooted_tree.nwk \
  --out kovar_results \
  --threads 32
```

SPA is `auto` by default. To resume an interrupted run, repeat the identical
command and add `--resume`. To start a new run in a non-empty output directory,
use `--overwrite`.

## Inputs

`--fasta` is an aligned fake FASTA with identical locus order and length in all
samples. `A` means absence (0) and `C` means presence (1).

`--pairs` is a whitespace-, comma-, or tab-delimited file containing zero-based
`u` and `v` locus columns. A headerless nine-column PAN-GWES file is interpreted
as:

```text
u v distance ARACNE MI count M2 min_distance max_distance
```

Pairs are canonicalized to `u < v`. Duplicate unordered pairs, self-pairs, and
out-of-range loci stop the run. Candidate rows are not filtered by MI inside
KOVAR.

`--tree` must be a rooted Newick tree with branch lengths. Tip labels must match
FASTA sample names exactly. KOVAR does not build a fallback GRM in 0.8.3.

By default, a FASTA sample absent from the tree stops the run before candidate
pairs are loaded. To deliberately analyze the intersection, add
`--tree-missing-samples drop`. KOVAR retains matched isolates in FASTA order,
requires at least two, and recalculates locus frequencies, joint tables and
sample-count-dependent thresholds using those isolates. Locus IDs and candidate
rows retain their original meanings; input graph `count` metadata is preserved
and is not recomputed from the subset. Tree tips absent from FASTA are ignored.
This is an analysis of the retained cohort, not a reconstruction of missing
phylogenetic placements. Check for naming mismatches before choosing exclusion.

`sample_inclusion.tsv` lists every input isolate and its inclusion status;
`run_summary.txt` records input, retained and excluded counts and the policy.
The policy is included in checkpoint identity for subset runs. Repeat the same
policy on resume and when running `ko-variation-annotation` with the original
FASTA/tree so that selected effects use the same isolates as screening.

## How the calculation works

For the deterministic computational orientation `u -> v`, KOVAR fits the null
model

```text
logit P(v_i = 1) = alpha_v + b_iv
b_v ~ N(0, tau_v K)
```

where `K` is the root-to-MRCA covariance derived once from the tree. PQL updates
the fitted probabilities, working weights, and response-specific variance
component. Identical or complementary binary response patterns reuse one null
fit. Candidate predictors sharing a response are then scored in matrix blocks.

The normal score statistic tests `H0: beta_uv = 0`. With `--spa-mode auto`, an
experimental saddlepoint approximation is attempted only for a score-tail
result in sparse context; otherwise the normal score p-value remains primary.
This is motivated by SPAtest and SAIGE but is not a reproduction of SAIGE.

Because the reported hypothesis is unordered, reversing an input pair produces
the same canonical pair and p-value. The computational response orientation is
an implementation device, not a claim about causal or evolutionary direction.

## Filters

Filters are applied before expensive response-model scheduling:

- If a recognized `count` column is present, `count >= ceil(0.05 * N)` is
  required. If the column is absent, this route filter is not applicable.
- Both loci require minor-state frequency `>= 0.05` by default.
- All four joint cells require at least one observation by default.

Every input pair remains in `ko_variation.tsv`; filtered rows have an explicit
status and missing p-values. Use `--min-maf` or `--min-cell-count` to change the
last two thresholds. The graph-count fraction is fixed at 0.05 in this release.

## Outputs

- `ko_variation.tsv`: one row per unordered input pair, input metadata, filter
  status, joint counts, score statistic, normal/SPA p-values, primary p-value,
  BH q-value, and Bonferroni flag.
- `response_models.tsv`: response-null convergence and exact-pattern reuse
  diagnostics.
- `run_summary.txt`: short scientific run contract.
- `execution_metadata.tsv`: timing, memory, runtime, kinship, cache, and
  checkpoint diagnostics.
- `sample_inclusion.tsv`: input isolate names with `included` or
  `excluded_missing_tree` status.

`p_primary` is a significance measure, not the magnitude of covariation.
`primary_method` says whether it came from the normal score approximation or a
successful SPA calculation. `neglog10_p` is intended for plotting.

## Downstream helpers

```bash
pip install -e '.[network]'
ko-variation-annotation --results kovar_results/ko_variation.tsv \
  --fasta binary.fa --tree rooted_tree.nwk \
  --annotation loci.tsv --out downstream/hits --network \
  --ld-distance 0 --significance-threshold 0.05
```

Annotation requires zero-based `locus` and stable `gene` columns. The default
selects `p_primary <= 0.05 / n_tests` and physical `distance > 0` in matching units.
Both thresholds are configurable. Only selected pairs are refitted, using the
original FASTA and tree. Outputs include adjusted ORs and approximate PQL Wald
CIs, fit diagnostics, a 300-DPI PNG, editable SVG, offline D3 HTML, saved layout coordinates, and node/edge tables;
gene edges summarize covariation candidates, not proven functional interactions.
Run `python examples/downstream/reproduce.py` for a synthetic reproducible example.
Portable commands replace the former ARC submission wrappers:

- `ko-variation-select`: select a large full scan in chunks without changing n_tests.
- `ko-variation-map-loci`: map selected DNA unitigs to Bakta/Panaroo annotations.
- `ko-variation-network`: redraw completed effects without refitting.
- `ko-variation-amr-report`: link species-specific MIC availability, established
  presence/absence determinant labels, and representation in a selected distal map.

The documented [portable pipeline](docs/downstream.md#portable-workflow-for-large-pan-gwes-scans)
uses explicit paths and retains selection, annotation and refit recovery.
See [AMR reporting](docs/amr_report.md) for the curated catalogue, annotation-name
matching, and direct/second-order neighbour report. The tet-specific tracer has
been removed; its old commands are no longer supported.
After updating an existing checkout, rerun `pip install -e '.[network]'` to
register the new helper commands. Existing scanner commands are unchanged.

Full options, failed-fit handling, coordinate fallback, and aggregation rules are
documented in [docs/downstream.md](docs/downstream.md).

## Checkpoint and resume

Checkpointing is automatic. Atomic score-stage shards are written to
`OUT/.kovar_checkpoint` unless `--checkpoint-dir` is supplied. A resume is
accepted only when input fingerprints, version, numerical environment, and
scientific settings match. The checkpoint is removed after final outputs are
successfully written.

## Interpretation and limitations

A small p-value identifies a phylogeny-adjusted covariation candidate. It does
not distinguish functional epistasis from physical linkage, co-transfer,
shared ecological selection, annotation artifacts, or residual population
structure. The candidate generator also limits discovery: a biologically
important pair absent from the supplied PAN-GWES/SpydrPick list cannot be
recovered by KOVAR.

The PQL and SPA implementations remain experimental. Calibration should be
checked by simulation over the intended sample sizes, frequencies, tree shapes,
and population structures before confirmatory use.

## Research foundations

- Breslow and Clayton (1993), approximate inference in generalized linear mixed
  models, motivates the PQL framework.
- Chen et al. (2016), GMMAT, motivates fitting a logistic mixed-model null once
  per response and testing predictors with score statistics.
- Dey et al. (2017), SPAtest, and Zhou et al. (2018), SAIGE, motivate
  saddlepoint calibration for binary score-test tails.
- Lees et al. (2018), pyseer, motivates bacterial population-structure-aware
  association analysis and scalable null-model reuse. KOVAR differs by using a
  binary logistic mixed model for locus-pair screening.

See `docs/model.md`, `docs/interpretation.md`, and `docs/research_basis.md` for
more detail.

## Version history

### 0.8.3

- changed primary reporting from two directional rows to one canonical
  unordered pair;
- changed the default minimum joint-cell count from 5 to 1;
- made SPA `auto` the default;
- applies the PAN-GWES `count >= ceil(0.05*N)` route before GLMM scheduling when
  a `count` column is available;
- removed the built-in alternative full-refit/effect-size stage, directional
  options, GRM fallback, and low-level numerical tuning from the public CLI;
- reduced checkpointing to the only expensive production stage: score testing;
  and
- moved release notes after overview, installation, use, method, and output
  documentation.

### 0.8.2

Added atomic checkpoint/resume support, detailed execution metadata, and
response-pattern reuse diagnostics.

### 0.8.1

Added exact identical/complement response-pattern reuse and shared first-PQL
kinship eigensystem preparation.

### 0.8.0

Introduced the experimental logistic mixed-model score scanner.


Completed downstream results can be redrawn without annotation or refitting:

```bash
ko-variation-network --effects downstream/hits.distal.tsv --out downstream/hits
```

The compact map uses capped node radii, packed connected components, representative
locus-pair |beta| for edge width/spacing and association direction for colour. D3
is bundled for offline search, zoom, dragging, optional reflow and SVG/PNG export.
See [downstream display options](docs/downstream.md#compact-display-semantics) for
coefficient boundaries and fixed cross-species display settings.
