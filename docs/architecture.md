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
|   `-- diagnostics.py  # runtime, thread, memory, and timing metadata
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
