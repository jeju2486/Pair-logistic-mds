# Pair Logistic MDS

`pair-logistic-mds` tests pairwise locus–locus association while correcting for tree-derived population structure using MDS covariates.

For each pair `(u, v)`, the tool fits both directions:

$$Y_v \sim Y_u + MDS$$

$$Y_u \sim Y_v + MDS$$

The result asks whether one locus predicts the other after correcting for population structure.

This is **not direct proof of epistasis**. It is a structure-corrected locus-pair association test.

## Installation

```bash
git clone https://github.com/jeju2486/Pair-logistic-mds.git
cd Pair-logistic-mds
pip install -e .
````

Check:

```bash
pair-logistic-mds --help
pair-logistic-mds-plot --help
```

For plotting, R must be available with:

```r
install.packages(c("data.table", "ggplot2"))
```

---

## Inputs

### 1. Tree

Newick tree with branch lengths:

```bash
--tree summary_tree.nwk
```

Tip names must match FASTA headers.

### 2. Fake FASTA

Presence/absence matrix:

```bash
--fasta simulation.fasta
```

Default encoding:

| Character | Meaning  |
| --------- | -------- |
| `A`       | absence  |
| `C`       | presence |

### 3. Pair file

Whitespace-delimited file:

```text
u v distance
```

Example:

```text
18176 38950 -1
20899 44115 100001
```

Indices must be **0-based** and match the FASTA locus order.

---

## Recommended run

```bash
pair-logistic-mds \
  --tree summary_tree.nwk \
  --fasta simulation.fasta \
  --pairs simulation.mi_filtered.ud_sgg_0_based \
  --mode score-then-exact \
  --out results/pair_logistic_mds.exact_top.tsv \
  --score-out results/pair_logistic_mds.score_all.tsv \
  --top-score-out results/pair_logistic_mds.score_top.tsv \
  --mds-k 10 \
  --min-maf 0.01 \
  --top-frac 0.01 \
  --distance-bin-size 10000 \
  --min-per-bin 100 \
  --reduce-mds-by-minor-count \
  --threads 8
```

This performs:

1. fast score-test scan for all pairs;
2. distance-stratified top-pair selection;
3. exact logistic/Firth refit for selected pairs.

---

## Output files

### `pair_logistic_mds.score_all.tsv`

Fast approximate score-test result for all pairs.

Important columns:

| Column                     | Meaning                      |
| -------------------------- | ---------------------------- |
| `u`, `v`                   | locus indices                |
| `distance`                 | pair distance                |
| `n11`, `n10`, `n01`, `n00` | raw contingency counts       |
| `freq_u`, `freq_v`         | locus frequencies            |
| `score_p_u_to_v`           | score-test p-value, `u -> v` |
| `score_p_v_to_u`           | score-test p-value, `v -> u` |
| `score_p_pair_max`         | conservative pair p-value    |
| `score_status`             | score-test status            |

### `pair_logistic_mds.score_top.tsv`

Top pairs selected from the score scan.

By default, selection is distance-stratified:

[
\max(100,\lceil 0.01 n_{\text{bin}}\rceil)
]

per 10 kb distance bin.

### `pair_logistic_mds.exact_top.tsv`

Exact logistic/Firth refit for selected top pairs.

Important columns:

| Column        | Meaning                         |
| ------------- | ------------------------------- |
| `beta_u_to_v` | effect estimate, `u -> v`       |
| `p_u_to_v`    | exact p-value, `u -> v`         |
| `beta_v_to_u` | effect estimate, `v -> u`       |
| `p_v_to_u`    | exact p-value, `v -> u`         |
| `p_pair_max`  | conservative exact pair p-value |
| `status`      | exact model status              |

---

## Plotting

Plot score scan with top-score overlay:

```bash
pair-logistic-mds-plot \
  --score results/pair_logistic_mds.score_all.tsv \
  --top results/pair_logistic_mds.score_top.tsv \
  --out plots/pair_logistic_mds.score_top.png \
  --y-col score_p_pair_max \
  --top-y-col score_p_pair_max \
  --bins 250
```

Plot score scan with exact-refit overlay:

```bash
pair-logistic-mds-plot \
  --score results/pair_logistic_mds.score_all.tsv \
  --top results/pair_logistic_mds.exact_top.tsv \
  --out plots/pair_logistic_mds.exact_top.png \
  --y-col score_p_pair_max \
  --top-y-col p_pair_max \
  --bins 250
```

The plot shows:

$$x = \text{distance}$$

$$y = -\log_{10}(p)$$

The background is binned for speed, and top pairs are overlaid in red.

---

## Useful options

| Option                        |            Default | Meaning                              |
| ----------------------------- | -----------------: | ------------------------------------ |
| `--mode`                      | `score-then-exact` | run mode                             |
| `--mds-k`                     |               `10` | number of MDS axes                   |
| `--min-maf`                   |             `0.01` | minimum minor-state frequency        |
| `--top-frac`                  |             `0.01` | top fraction per distance bin        |
| `--distance-bin-size`         |            `10000` | distance bin size                    |
| `--min-per-bin`               |              `100` | minimum selected per bin             |
| `--threads`                   |                `1` | exact-refit worker processes         |
| `--reduce-mds-by-minor-count` |                off | reduce MDS axes for sparse responses |
| `--no-firth`                  |                off | disable Firth fallback               |

---

## Run modes

### Fast score scan only

```bash
pair-logistic-mds \
  --tree summary_tree.nwk \
  --fasta simulation.fasta \
  --pairs pairs.tsv \
  --mode score \
  --out score_all.tsv
```

### Exact logistic/Firth for all pairs

Use only for small pair sets:

```bash
pair-logistic-mds \
  --tree summary_tree.nwk \
  --fasta simulation.fasta \
  --pairs pairs.tsv \
  --mode exact \
  --out exact_all.tsv \
  --threads 8
```

---

## Interpretation

A significant corrected association means:

> one locus predicts the other after tree-MDS population-structure correction.

It does not necessarily mean direct biological epistasis. Hidden ecology is corrected only if it is captured by the tree/MDS structure.

---

## License

MIT

```
```
