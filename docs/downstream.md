# Selected adjusted effects and gene networks

The scanner CLI is unchanged. The add-on reuses its existing readers and PQL
mixed-model fitter; only selected pairs need an alternative fit. No raw effects
are calculated and no additional statistical dependencies are required.

```bash
ko-variation-annotation --results kovar_results/ko_variation.tsv \
  --fasta binary.fa --tree rooted_tree.nwk \
  --annotation loci.tsv --out downstream/hits --network \
  --significance-threshold 0.05 --ld-distance 0
```

The same command runs as `python -m ko_variation.annotation_cli`. Network exports
require `pip install -e '.[network]'`; omit `--network` for effects alone.

## Refit checkpoints and progress

The downstream helper now commits every completed pair to `OUT.refit.sqlite`
by default. Repeat the same command with `--resume` to load those results and
fit only unfinished pairs. A fit interrupted halfway is restarted, rather than
resumed from an internal PQL iteration. Saved unsuccessful fits are completed
attempts and are not retried automatically. The checkpoint is retained after
successful exports, so a resumed run can regenerate figures without refitting.

```bash
# First run: ordinary command above, with optional reporting intervals.
ko-variation-annotation --results kovar_results/ko_variation.tsv \
  --fasta binary.fa --tree rooted_tree.nwk --annotation loci.tsv \
  --out downstream/hits --network --progress-every 10 --progress-seconds 60

# Restart with identical inputs and scientific settings.
ko-variation-annotation --results kovar_results/ko_variation.tsv \
  --fasta binary.fa --tree rooted_tree.nwk --annotation loci.tsv \
  --out downstream/hits --network --resume
```

`--checkpoint-file` overrides the database path. `--no-checkpoint` disables
checkpointing; it cannot be combined with `--resume`. An existing checkpoint
requires `--resume` or a new output prefix. No checkpoint is silently replaced.
Genotype and prepared covariance values, selected pair order/counts, confidence,
selection settings, fitting-source hashes and numerical library versions must
match before saved effects are reused. Display settings and gene labels can
change because they do not change the saved locus-pair fits. SQLite commits
each completed pair with full synchronization; an OS lock prevents concurrent
writers and releases automatically when a process exits. Keep the database and
any SQLite journal together when recovering after a job termination.

Progress is enabled by default. Stage messages distinguish input reading,
covariance preparation, genotype checks and network export. Pair messages show
completed/total, percentage, recovered pairs, elapsed time, approximate remaining
time, current pair and fitting-status counts. A heartbeat prints every
`--progress-seconds` seconds even during a slow fit. `--progress-every` controls
additional completion messages; `--no-progress` suppresses reporting. Remaining
time is estimated from newly completed pairs in the current run and can change
as pair runtimes vary. Progress is written to stdout and appears in Slurm's
`.out` log when the job is submitted with `sbatch`.

The supplied `run_kovar_plot_saureus.sh` wrapper defaults to `RESUME=1`: it
checks stage caches, reuses matching screening/automatic annotation outputs,
and adds `--resume` when a refit database exists. Upstream cache identities use
source paths, sizes and nanosecond mtimes (including Bakta file inventories),
plus full SHA-256 checks of cached output files. Do not edit source contents
while preserving their timestamps. Refit reuse separately verifies the actual
genotypes, covariance and fitting configuration. Use a different `PREFIX` for
a fresh independent run; `RESUME=0` does not delete an existing refit database.
Only fully completed screening and annotation stages are reused. An interrupted
screening pass or pyseer annotation stage restarts that stage; per-pair recovery
applies to coefficient fitting.

Old runs that never wrote refit checkpoints cannot recover unsaved effects.
For an existing, already completed annotation from the older wrapper, set
`AUTO_ANNOTATE=0` explicitly to reuse that TSV on the first updated submission.
Otherwise, its missing cache manifest triggers annotation regeneration once.
Subsequent updated runs reuse the verified stage cache automatically. The
current running process cannot acquire these features retroactively.

## Selection parameters

`--significance-threshold` is the Bonferroni family-wise alpha, **0.05 by default**.
Keep `p_primary <= alpha / n_tests`, using `n_tests` from the full original scan.
If that column is absent, count nonmissing primary p-values before filtering:
use a complete result file in that case. Do not divide by the number of selected
hits or refits. Only scanner statuses `OK` and `OK_SPA_FAILED` are eligible.
The latter retains the valid normal-score fallback after SPA failure. Original
screening p-values and q-values remain unchanged; no new p-values are computed.

`--ld-distance` is **0 by default**, with the strict rule `distance > cutoff`.
Both defaults can be overridden. Zero imposes no positive linkage-exclusion
window and is not evidence that selected loci are biologically unlinked.
Distances must use units matching the cutoff. Physical distance is a proxy for
linkage, not a measured LD statistic such as r-squared.

`--distance-column` defaults to `distance`; missing distances are excluded.
For PAN-GWES ranges, choose `min_distance` to require separation across all
represented genomes. If the chosen column is absent, optional annotation
`contig` and `position` columns give absolute linear distances. Circular genomes
require a precomputed shortest-path distance column. Known cross-contig pairs
are excluded unless `--cross-contig distal` is supplied. They then have missing
physical distance and an explicit class. A distance-only file cannot identify
cross-contig pairs; supplied distances take priority over coordinates.

## Selected alternative fits

Supply the **same original FASTA and tree** used for screening. Tree tips must
match sample names; the covariance is built in FASTA order and prepared once.
Selected joint counts are checked against the supplied genotypes before refits.
This check cannot prove the supplied tree or sample-locus assignments are the
original ones; a TSV alone does not contain those provenance fingerprints.

If screening used `--tree-missing-samples drop`, repeat that option here with
the same original FASTA and tree. The helper applies the identical intersection
in FASTA order before checking joint counts or fitting effects. Its
`.selection.json` records the policy, retained count and excluded names. The
policy also participates in refit checkpoint identity. An empty selection skips
tree matching and fitting; its provenance marks `matching_performed` as false.

For each selected canonical pair `u < v`, the alternative is:

```text
logit P(v_i = 1) = alpha + beta * u_i + b_i
b ~ N(0, tau * K)
```

The existing PQL/pseudo-REML fitter estimates beta and tau. Reported columns are
`adjusted_beta`, `adjusted_beta_se`, `adjusted_odds_ratio = exp(beta)`,
`adjusted_or_ci_low`, `adjusted_or_ci_high`, `alternative_tau`, `effect_status`,
`effect_iterations`, `effect_message`, `effect_method`, and `effect_confidence`.
`--confidence` defaults to 0.95. SE comes from the final working-model covariance
`(C^T V^-1 C)^-1`; C contains the intercept and predictor and
V = diag(1/working_weights) + tau*K. CIs are exp(beta +/- z*SE).

These are **approximate PQL Wald intervals**, treating fitted tau as fixed.
They do not include variance-component uncertainty, selection correction, or
multiple-comparison coverage. Significant-pair selection can inflate effects.
PQL remains experimental, particularly with sparse binary data. The orientation
u -> v is computational, not causal; reversing a mixed-model fit need not give
an identical estimate. Report the original locus orientation alongside effects.

Zero joint cells indicate separation or monomorphic states and are skipped.
Failed convergence, variance-search boundaries, clipped working weights, and
nonfinite effects produce explicit statuses with missing effect/CI fields.
No raw OR or continuity correction is computed. Failed pairs stay in the TSV.
An empty selection skips covariance construction and alternative fitting.

## Gene networks

Annotation is TSV with one row per zero-based `locus` and stable nonempty `gene`.
Optional `label`, `product`, and `group` annotate nodes. Resolve one-to-many locus
mappings beforehand; duplicate locus mappings are rejected. Gene IDs shared
across distinct copies collapse those copies into one node.

Missing mappings are errors unless `--missing-genes drop` is set. Same-gene
pairs are excluded unless `--include-self` is set. All unique supporting locus
pairs remain recorded on each gene edge. Node radius represents degree within a
small capped range (3–5 diagram units by default); ordinary nodes are charcoal.
Annotation groups do not control node colours or produce group legends.

The representative effect is the most significant **successful adjusted fit**,
with u/v tie-breaking; if no refit succeeds, the edge has a missing effect.
`n_adjusted_pairs` records successful refits. Edge color shows positive, negative,
mixed, neutral, or unknown adjusted directions. Failed fits cannot contribute
to effect direction. ORs are not pooled because supporting pairs may be dependent.
The edge's minimum primary p-value includes all supporting selected pairs and
is descriptive, **not a gene-level p-value**. A gene edge remains a covariation
candidate, not demonstrated functional epistasis.

## Outputs and example

The command writes `PREFIX.distal.tsv` and `PREFIX.selection.json`, which records
settings, input paths, selected counts, and effect-status counts. With `--network`
it also writes `.nodes.tsv`, `.edges.tsv`, `.png`, `.svg`, `.layout.json`, and offline `.html`.
PNG defaults to 300 DPI. PNG, editable SVG and offline D3 HTML start from the
same saved coordinates (`--seed 42`), recorded in `.layout.json` and the node TSV.
The bundled D3 v7.9.0 runs without an internet connection. No browser force
simulation starts automatically; **Compact / reflow** enables it, **Freeze** stops
it and **Reset layout** restores the original coordinates. Search, hover, dragging,
zoom and direct-neighbour highlighting support exploration. SVG/PNG exports use
the current coordinates, including manually moved nodes. Layout JSON saves them.

### Compact display semantics

Connected components are packed close together without community detection. Edge
width uses the absolute beta of the same representative successful locus refit
described above, not the supporting-pair count. Width is capped at the within-map
95th percentile of that magnitude. Positive edges are red (`#C62828`), negative
blue (`#2166AC`), mixed purple (`#7B3294`) and neutral/unknown grey. Mixed and unknown
edges are dashed. These colours describe successful fitted association directions.
A representative beta is a locus-pair coefficient, not a pooled gene-level effect.

Weak/medium/strong are descriptive magnitude categories. Default boundaries are
the within-map tertiles of finite representative |beta|: weak <= lower boundary;
medium > lower and <= upper; strong > upper. Ties stay together, so category
counts need not be equal; identical coefficients all enter weak. Exact boundaries
are saved in the layout JSON and displayed in the legend. Use common fixed
`--strength-cutoffs LOW HIGH` for comparable magnitude categories across species.
These categories do not select edges and do not replace the original P-value rule.
Edges without successful coefficients remain present as unestimated, with thin
lines and the medium target length.

Strong, medium and weak target lengths default to 20, 35 and 55 **diagram units**,
respectively; they are layout preferences, not exact lengths or genomic distances.
Graph constraints can prevent any edge from attaining its target length. Static
labels default to none; `--labels hubs` or `all` enables named-gene labels. Generic
`group_###` labels remain in tables and tooltips rather than cluttering the canvas.
Display options: `--node-radius MIN MAX`, `--link-distances STRONG MEDIUM WEAK`,
`--component-gap GAP`, `--strength-cutoffs LOW HIGH`, and `--labels none|hubs|all`.

### Redraw completed results without annotation or refitting

```bash
python -m ko_variation.network_cli --effects PREFIX.distal.tsv --out PREFIX
```

This command rebuilds the gene summary and display from the completed annotated
effect table. It does not reapply significance/distance filters or rerun numerical
fitting. It writes the network exports and `.network.json` with display settings
and the source effect-table path. Preserve `.selection.json` for the original
selection provenance. To change biological selection, rerun the analysis workflow.

The S. aureus wrapper exposes the same options as environment variables:

```bash
REDRAW_ONLY=1 sbatch run_kovar_plot_saureus.sh
NODE_RADIUS="3 4" LINK_DISTANCES="15 25 40" REDRAW_ONLY=1 sbatch run_kovar_plot_saureus.sh
```

`EFFECTS` defaults to `$PREFIX.distal.tsv`. `NETWORK_LABELS` defaults to `none`,
`COMPONENT_GAP` to 16, and `STRENGTH_CUTOFFS` is optional. Default normal execution
retains the existing resume/progress behaviour. Display changes do not invalidate
refit checkpoints.

```python
from ko_variation.postprocess import SelectionConfig, select_distal_signals, fit_selected_effects

signals = select_distal_signals(results, SelectionConfig(significance_threshold=0.05, ld_distance=0), annotation)
effects = fit_selected_effects(signals, X, prepared_K)
```

`X` is the original sample-by-locus binary matrix and `prepared_K` is the aligned
covariance prepared by `ko_variation.glmm.prepare_kinship`.
Run `python examples/downstream/reproduce.py` for a synthetic star-tree example
with scanner output, alternative fits, and network exports. Minimal focused
checks: `python -m unittest discover -s tests -p test_postprocess.py`.
