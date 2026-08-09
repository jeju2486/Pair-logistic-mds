# KOVAR 0.8.2

> **Experimental statistical release.** KOVAR 0.8.2 uses a binary
> mixed-model implementation that still requires broad type-I-error and power
> validation. Do not treat its output as sole evidence for a biological or
> clinical conclusion.

**KOVAR** - **K**inship-adjusted **O**mics **VAR**iation association - is a
directional pangenome covariation scanner. It tests whether one binary locus
predicts another after adjustment for covariance among samples.

A significant result is a maximally inferred **epistasis candidate** in the
terminology of this project. More precisely, it is a kinship-adjusted
directional covariation candidate. Statistical direction does not establish
causality, evolutionary order, or a molecular interaction.

## What changed in 0.8.2

Version 0.8.2 makes multi-day scans recoverable and separates operational
diagnostics from the scientific result table:

- writes atomic, versioned checkpoints for completed canonical response
  patterns and selected full alternative refits;
- resumes only after verifying input-content and scientific-configuration
  fingerprints;
- runs score testing and selected full refits as separately checkpointed stages;
- reports directional-table construction, response grouping, kinship
  eigendecomposition, weighted eigendecomposition, tau profiling, scoring,
  SPA, full-refit, multiple-testing, and checkpoint-I/O timing;
- estimates dense-worker memory, reports the multiprocessing start method, and
  warns when the requested worker count exceeds a conservative RAM budget; and
- writes detailed operational records to `execution_metadata.tsv` without
  adding checkpoint or timing columns to `ko_variation.tsv`.

Final-state rebuilding and alternative-model warm starts remain deferred: the
former adds another dense weighted solve, while prototype warm starts did not
reliably reduce PQL iterations.

## Performance changes introduced in 0.8.1

Version 0.8.1 accelerates the 0.8.0 model without removing candidate pairs or
changing the tested hypotheses:

- identifies identical and complementary binary response patterns and fits one
  null GLMM per exact canonical pattern;
- maps every cached fit back to every original response locus and preserves all
  requested directional result rows;
- reuses the tree/GRM eigensystem exactly in the first PQL iteration, when the
  initial working weight is constant;
- evaluates the variance-component profile in spectral coordinates, avoiding
  repeated sample-space transformations for each trial value of `tau`;
- skips SPA-specific predictor preparation when `--spa-mode off`;
- records response-cache, solver, stage-timing, BLAS-library, and process-pool
  diagnostics, with a warning for nested native-thread oversubscription.

These are calculation-reuse and algebraic optimizations. Version 0.8.1 does
not add a more restrictive pair filter, approximate the logistic weights with
an LMM, or add a GPU backend.

## Model introduced in 0.8.0

Version 0.8.0 replaces the 0.7.0 linear mixed-model scan with an experimental
directional logistic mixed model:

- fits a response-specific binary null model by penalized quasi-likelihood
  (PQL);
- screens predictors with a prospective mixed-model score test;
- tests `u_predicts_v` and `v_predicts_u` separately by default;
- uses a 5% minor-state-frequency filter plus a separate four-cell-count filter;
- uses the normal score p-value by default;
- provides optional experimental SPA calibration, without claiming SAIGE
  equivalence;
- refits promising directions to estimate conditional odds ratios;
- reports BH and Bonferroni results across finite directional tests; and
- removes the LMM `h2` cap, GRM identity-shrinkage lambda, and low-rank kernel
  truncation.

The production scanner still excludes CTMC filtration, ancestral-state/event
filtering, and automatic neighbor-joining tree construction.

## Model at a glance

For predictor locus \(u\) and response locus \(v\), KOVAR considers

\[
Y_{iv}\sim\operatorname{Bernoulli}(\mu_{iv}),\qquad
\operatorname{logit}(\mu_{iv})=\alpha_v+\beta_{u,v}X_{iu}+b_{iv},
\]

\[
b_v\sim N(0,\tau_vK).
\]

The sample covariance \(K\) preferably comes from an external core-genome tree.
A proxy-masked pangenome GRM is available when no tree is supplied.

KOVAR fits one null model per exact binary response pattern and reuses it for
eligible predictors and loci with an identical or complementary pattern. This
design is inspired by GMMAT, but KOVAR is a standalone NumPy/SciPy
implementation and not an exact numerical reproduction of GMMAT.

See [the statistical model](docs/model.md), [interpretation
guidance](docs/interpretation.md), and [research foundations](docs/research_basis.md).

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install .
```

Check the installed version and interface:

```bash
ko-variation --version
ko-variation --help
```

The core scanner requires NumPy, pandas, SciPy, and `threadpoolctl`. The latter
reports the native BLAS thread configuration used by the dense solver. The
optional plotting command also requires R and `data.table`; `ggplot2` is
optional.

## Inputs

### Binary locus matrix

`--fasta` accepts a PAN-GWES-style binary FASTA in which each record is a
sample and each sequence position is a locus. By default:

- `A` means absence (`0`); and
- `C` means presence (`1`).

All records must have equal length and unique sample names.

### Candidate pairs

`--pairs` accepts zero-based locus columns `u` and `v`. A headerless file is
interpreted as:

```text
u v distance ARACNE MI count M2 min_distance max_distance
```

Additional columns are preserved as result metadata. In the default
`--direction-mode both`, the input must not contain duplicate unordered pairs;
for example, including both `1 2` and `2 1` is an error.

### Sample covariance

Use `--tree` with a rooted, branch-length Newick tree when possible. Tip labels
must cover every FASTA sample. KOVAR calculates root-to-MRCA covariance and
checks the resulting matrix for symmetry and positive semidefiniteness.

Without `--tree`, KOVAR builds a background GRM from the binary matrix and
removes tested loci and close proxies according to the GRM proxy thresholds.
This is a fallback because deriving relatedness and testing covariation from the
same pangenome can introduce proximal-contamination tradeoffs.

## Run KOVAR

Recommended tree-based analysis:

```bash
ko-variation \
  --fasta matrix.fasta \
  --pairs candidate_pairs.tsv \
  --tree core_genome.tree \
  --out kovar_results \
  --direction-mode both \
  --min-maf 0.05 \
  --min-cell-count 5 \
  --spa-mode off \
  --threads 8
```

GRM fallback:

```bash
ko-variation \
  --fasta matrix.fasta \
  --pairs candidate_pairs.tsv \
  --out kovar_results \
  --direction-mode both \
  --min-maf 0.05 \
  --min-cell-count 5
```

Important options:

| Option | Default | Meaning |
|---|---:|---|
| `--direction-mode both\|input` | `both` | Test both directions, or only input `u_predicts_v` |
| `--min-maf` | `0.05` | Minimum minor-state frequency for both loci |
| `--min-cell-count` | `5` | Minimum count in each oriented 2x2 cell |
| `--spa-mode off\|auto\|always` | `off` | Experimental saddlepoint calibration policy |
| `--full-refit-p` | `0.05` | Full GLMM refit threshold; `0` disables |
| `--null-max-iter` | `100` | Maximum PQL null-model iterations |
| `--null-tolerance` | `1e-7` | PQL convergence tolerance |
| `--threads` | `1` | Response-wise worker processes |
| `--predictor-batch-size` | `256` | Predictors scored together per response |
| `--checkpoint-every` | `100` | Completed tasks per atomic checkpoint shard |
| `--resume` | off | Resume an exactly matching checkpoint |
| `--no-checkpoint` | off | Disable checkpoint creation |

`--spa-mode auto` is provisional: it considers SPA only for score-tail results
with low marginal frequencies or sparse cells. SPA remains experimental and
cannot compensate for insufficient joint counts or a lineage-confined pattern.

### Long-run performance diagnostics

KOVAR parallelizes unique response-pattern fits with `--threads`. Dense
eigendecomposition libraries may also create native threads. To avoid silently
running, for example, 32 worker processes each with 32 BLAS threads, KOVAR sets
missing BLAS environment settings to one before importing NumPy/SciPy. Explicit
user settings are preserved, inspected, written to `execution_metadata.tsv`, and
warned about when they create nested parallelism.

For a many-worker run, a conservative configuration is:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
ko-variation ... --threads 32
```

The fastest process/thread combination depends on sample count, RAM bandwidth,
and the installed BLAS library; benchmark a representative subset before a
multi-day analysis.

NumPy/SciPy already dispatch the dense eigendecomposition and matrix products
to compiled BLAS/LAPACK code, so wrapping the current calls in Cython would not
remove their dominant numerical cost. Version 0.8.1 instead avoids redundant
fits and transformations. Full alternative refits still run their own PQL fit
for every direction passing `--full-refit-p`; that optional effect-estimation
stage can remain expensive when many directions cross the threshold. Version
0.8.2 schedules that work after score testing so each stage has independent
progress and recovery.

### Checkpoint and resume

Checkpoint writing is enabled by default under `OUT/.kovar_checkpoint`. A new
run refuses to reuse an existing checkpoint directory. Resume must be explicit:

```bash
ko-variation ... --out kovar_results --resume
```

KOVAR hashes the FASTA, pair file, optional tree, and scientific settings before
loading checkpoint shards. A mismatch stops the run rather than combining
incompatible results. Checkpoint shards are deleted after successful atomic
final-output writes unless `--keep-checkpoints` is supplied. `--overwrite` does
not mean resume and does not bypass checkpoint validation.

With the default interval, an abrupt failure can require recomputing at most 99
completed tasks that had not yet been flushed. Global BH and Bonferroni values
are calculated only after every score task has been reconstructed.

## Filters

The 5% marginal filter is calculated as
`min(presence_frequency, 1 - presence_frequency)`. The raw presence state remains
the logistic response; KOVAR does not recode the minor state to 1.

The four directional cells are:

- `n11`: predictor present, response present;
- `n10`: predictor present, response absent;
- `n01`: predictor absent, response present; and
- `n00`: predictor absent, response absent.

When direction is reversed, `n10` and `n01` swap. A pair can pass the 5% marginal
filter and still fail `--min-cell-count`, so both safeguards are required.

## Outputs

KOVAR always writes:

- `ko_variation.tsv`: one row per requested direction, including oriented
  counts, frequencies, score test, primary p-value, optional SPA, optional full
  refit, multiple-testing values, statuses, and kinship diagnostics;
- `response_models.tsv`: one diagnostic record per original eligible response
  locus, including its canonical pattern, complement orientation, pattern
  multiplicity, and whether its null fit was reused;
- `run_summary.txt`: a concise run/model summary; and
- `execution_metadata.tsv`: configuration, cache reuse, wall-clock stage
  timings, worker-summed solver timings, checkpoint I/O, kinship, BLAS,
  process, and memory diagnostics.

Key `ko_variation.tsv` columns include:

```text
pair_id u v predictor_locus response_locus direction
predictor_prevalence response_prevalence predictor_maf response_maf
n11 n10 n01 n00 min_cell status
tau_phylogenetic latent_phylogenetic_fraction
score_u score_variance score_z p_score
spa_applied spa_status p_spa p_primary primary_method score_primary
beta_log_odds se_log_odds odds_ratio odds_ratio_ci_low odds_ratio_ci_high
full_refit_status q_bh bonferroni_significant n_directional_tests
```

Key response-cache diagnostics in `response_models.tsv` include:

```text
response_locus canonical_response_locus response_pattern_id
response_pattern_flipped response_pattern_size null_fit_reused
```

Filtered and failed directions remain in `ko_variation.tsv` with an explicit
status and missing inferential fields.

## Plot results

The optional plotter accepts any numeric score column. For 0.8.2 directional
results, use `score_primary`:

```bash
ko-variation-plot \
  --score kovar_results/ko_variation.tsv \
  --out kovar_results/ko_variation.png \
  --y-col score_primary
```

## Interpretation and limitations

Phylogenetic adjustment is intended to reduce lineage-confounded association.
It can also weaken a real relationship that occurs only within one lineage.
Conversely, a significant adjusted result does not prove that the pattern arose
independently in multiple lineages.

KOVAR 0.8.2 does not yet implement a validated globality classifier. Treat
cross-lineage stability analysis, functional annotation, physical linkage,
mobile-element carriage, shared ecology, and independent replication as
necessary follow-up work.

Other important limitations are:

- PQL and the score distribution are approximate;
- type-I-error calibration across KOVAR's intended phylogenies and frequencies
  is not yet established;
- an external tree is only as suitable as its rooting, branch lengths, and
  representation of sample ancestry;
- a same-matrix GRM can absorb tested signal or lose structure after masking;
- sparse data can cause separation or failed full refits;
- checkpoints cannot be reused after inputs, scientific settings, schema, or
  KOVAR version change; and
- covariation alone cannot distinguish functional interaction from linkage,
  co-transfer, shared selection, or technical artifacts.

Read [the validation plan](docs/validation.md) before interpreting experimental
results. The [package architecture](docs/architecture.md) distinguishes current
runtime modules from planned simulation, benchmark, and lineage-diagnostic
infrastructure.

## Primary research foundations

- [Breslow and Clayton (1993): approximate inference in GLMMs](https://doi.org/10.1080/01621459.1993.10594284)
- [Chen et al. (2016): GMMAT logistic mixed-model score tests](https://pmc.ncbi.nlm.nih.gov/articles/PMC4833218/)
- [Dey et al. (2017): SPAtest](https://pmc.ncbi.nlm.nih.gov/articles/PMC5501775/)
- [Zhou et al. (2018): SAIGE](https://pmc.ncbi.nlm.nih.gov/articles/PMC6119127/)
- [Lees et al. (2018): pyseer and microbial GWAS](https://pmc.ncbi.nlm.nih.gov/articles/PMC6289128/)

See [research_basis.md](docs/research_basis.md) for what KOVAR adopts from each
study and, equally importantly, what it does not claim to reproduce.
