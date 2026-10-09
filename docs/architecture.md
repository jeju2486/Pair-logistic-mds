# Package architecture

```text
Pair-logistic-mds/
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- docs/
|   |-- architecture.md
|   |-- interpretation.md
|   |-- model.md
|   |-- research_basis.md
|   `-- validation.md
|-- ko_variation/
|   |-- cli.py          # small public interface and pipeline orchestration
|   |-- io_utils.py     # binary FASTA and pair-table contracts
|   |-- kinship.py      # rooted-tree covariance
|   |-- glmm.py         # PQL null fit and score calculations
|   |-- spa.py          # optional score-tail calibration
|   |-- scan.py         # canonical pairs, filters, task reuse, correction
|   |-- checkpoint.py   # atomic score-task shards and strict resume
|   |-- diagnostics.py  # runtime, thread, memory, and timing metadata
|   |-- selection_cli.py # streamed full-scan selection; preserves n_tests
|   |-- annotation_inputs.py # portable Pyseer/Bakta input preparation
|   |-- locus_annotation.py  # coordinate-based assignment and ambiguity audit
|   |-- annotation_cli.py    # selected effects, resume/progress and exports
|   |-- postprocess.py       # selection rule and selected PQL coefficients
|   |-- effect_checkpoint.py # resumable per-pair coefficient fits
|   |-- network.py           # gene aggregation; no gene-level significance test
|   |-- network_display.py   # shared PNG/SVG/offline D3 layout and display
|   |-- network_cli.py       # redraw saved effects without refitting
|   |-- workflow_cache.py    # shared completed-stage cache
|   `-- trace_tet.py         # read-only determinant tracing
|-- tests/
`-- run_kovar_v083_*.sh
```

## Execution flow

```text
read and validate A/C FASTA, pair table, rooted tree
  -> canonicalize each pair to u < v
  -> apply graph-count, MAF, and joint-cell filters
  -> group eligible response loci by exact/complement pattern
  -> build K eigensystem once
  -> fit one PQL null per canonical response pattern
  -> score all associated predictors in blocks; apply SPA policy
  -> reconstruct checkpointed results
  -> calculate BH and Bonferroni values
  -> atomically write scientific and metadata outputs
```

Checkpointing covers the only multi-day task stage: response-null fitting and
pair scoring. Numerical tuning is internal so the public CLI exposes scientific
choices rather than implementation detail.

## Deliberately separate work

Physical-distance classification, gene annotation, selected alternative fits, and
network exports are implemented in the separate `ko-variation-annotation` CLI
and downstream modules (see `downstream.md`). Keeping them outside the scanner
prevents screening significance, biological annotation, and post-selection
effect estimation from being confused.

## Downstream boundaries

The public commands accept explicit paths and have no ARC scheduler assumptions.
`ko-variation-select` streams a full result table into a small selected table.
`ko-variation-map-loci` prepares and resolves DNA annotations.
`ko-variation-annotation` estimates selected effects and exports networks.
`ko-variation-network` redraws saved effects, while `ko-variation-trace-tet`
diagnoses target coverage using the original full scan. Completed-stage caching
is shared; numerical effect checkpoints remain separate. The R-backed distance
plot is still available through `ko-variation-plot` and is not a network export.

ARC-specific submission scripts and their wrapper-only cache CLI were removed.
The portable v0.8.3 example runners remain because they contain no account paths
or scheduler settings. AMR catalogue construction is future work and is not
introduced by this cleanup.
