# pair-logistic-mds

Pairwise SEER/pyseer-style locus association using tree-derived MDS covariates.

The default workflow is now optimized for large pair sets:

1. **Fast score scan** for all candidate pairs using a logistic score test adjusted for tree-MDS covariates.
2. **Exact logistic/Firth refit** only for the best fraction of pairs.

This is intended for pair lists where exact logistic fitting for every pair is too slow.

## Install

```bash
pip install -e .
```

## Default optimized workflow

```bash
pair-logistic-mds \
  --tree summary_tree.nwk \
  --fasta simulation.fasta \
  --pairs candidate_pairs.tsv \
  --out pair_logistic_mds.top1pct.exact.tsv \
  --mode score-then-exact \
  --top-frac 0.01 \
  --mds-k 10 \
  --min-maf 0.01 \
  --reduce-mds-by-minor-count \
  --threads 8
```

`--threads` is used for the exact refit step. The score scan is vectorized by response locus and is serial.

## Score-only mode

```bash
pair-logistic-mds \
  --tree summary_tree.nwk \
  --fasta simulation.fasta \
  --pairs candidate_pairs.tsv \
  --out pair_logistic_mds.score.tsv \
  --mode score \
  --mds-k 10 \
  --min-maf 0.01
```

## Exact-only mode

```bash
pair-logistic-mds \
  --tree summary_tree.nwk \
  --fasta simulation.fasta \
  --pairs candidate_pairs.tsv \
  --out pair_logistic_mds.exact.tsv \
  --mode exact \
  --threads 8
```

## Inputs

1. Newick tree with branch lengths.
2. Fake FASTA presence/absence matrix. Default: `C=presence`, `A=absence`.
3. Whitespace pair file with first three columns: `u v distance`, where `u` and `v` are 0-based locus indices.

## Main output in score-then-exact mode

Exact refit columns:

- `u`, `v`, `distance`
- `n11`, `n10`, `n01`, `n00`
- `freq_u`, `freq_v`
- `beta_u_to_v`, `p_u_to_v`
- `beta_v_to_u`, `p_v_to_u`
- `p_pair_max`
- `status`

Score-scan columns retained for the selected pairs:

- `score_beta_u_to_v`, `score_p_u_to_v`
- `score_beta_v_to_u`, `score_p_v_to_u`
- `score_p_pair_max`
- `score_status`

## Notes

The score test is an approximation to the same fixed-effect logistic model:

```text
Y_v ~ Y_u + MDS_1 + ... + MDS_K
Y_u ~ Y_v + MDS_1 + ... + MDS_K
```

It fits the null model once per response locus and tests many predictors by score statistics. Exact logistic/Firth fitting is then applied only to top-scoring pairs.
