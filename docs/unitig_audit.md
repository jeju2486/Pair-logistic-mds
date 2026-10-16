# Whole-catalogue unitig annotation and AMR phylogenetic-distribution audit

`ko-variation-unitig-audit` annotates **every sequence in the supplied unitig
catalogue**, independently of MI selection, KOVAR significance, or network
membership. It reports which unitigs overlap/are near AMR target genes and their
presence/absence frequency, entropy, and Fritz–Purvis D. It never removes loci,
changes the main scanner, reruns MI, or calls a resistance mutation.

## Install

From the development checkout, in the Python environment used to run the helper:

```bash
python -m pip install -e '.[audit]'
```

This installs Biopython and pyahocorasick in addition to KOVAR's base dependencies.
Annotation uses an Aho–Corasick exact sequence matcher; BWA, Pyseer, eggNOG, and
fastBAPS are **not required for this helper**. Both DNA strands, repeated matches,
all supplied analysis-cohort assemblies, and all overlapping/nearby CDS are
searched. There is no first-assembly selection or repeat-hit cap. It performs no
mismatch-tolerant, partial, or contig-boundary-spanning alignment. Assembly gaps,
sequence differences, missing annotation names, and missing references can still
prevent annotation; a missing match is not evidence that a gene is absent.

For D, install R, ape and caper separately. For example:

```bash
conda create -n kovar_d_env -c conda-forge r-base r-ape r-caper
```

Alternatively install the R packages from an existing R environment:

```bash
Rscript --vanilla -e 'install.packages(c("ape", "caper"), repos="https://cloud.r-project.org")'
```

Pass the full Rscript path with `--rscript` when Python and R use different
environments. Runtime versions, inputs, analysis cohort and settings are recorded.

## Required inputs and identity contracts

- `--unitigs`: original **DNA** table, one `zero_based_locus_ID DNA_sequence` per
  line. Never use the artificial A/C presence FASTA as DNA annotation queries.
- `--fasta`: sample-ID binary FASTA. C/c means present; A/a means absent. Every
  sample ID must be unique; do not use repeated `contig00001` headers.
- `--locus-map`: TSV with `fasta_column` and `locus`, each zero-based, covering
  every binary FASTA column. Alternatively explicitly assert
  `--fasta-columns-are-locus-ids` **only when column i really is original locus i**.
  Subset/reordered matrices require the explicit map; row numbers are not guessed.
- `--metadata`: collection TSV with `id`, `isolate`, `species`. The existing cohort
  convention is `id + "_" + isolate.replace(".", "_")`. Override with a one-to-one
  `--sample-map` TSV (`metadata_id`, `sample`) if those are not your FASTA names.
  Non-target metadata species and explicit `--exclude-sample` IDs are excluded.
- `--tree`: core-genome Newick with finite nonnegative branch lengths. Missing
  retained FASTA samples cause an error by default. `--tree-missing-samples drop`
  explicitly changes the cohort and records the exclusions. Tree-only tips are
  pruned from a copy. Midpoint rooting is the default; `--rooting as-is` requires
  an explicitly rooted Newick (`[&R]`). Original inputs are never rewritten.
- `--panaroo`: Roary-style `gene_presence_absence.csv` or
  `gene_presence_absence_roary.csv`, including `Gene`, annotation fields and sample
  columns. Internal cluster IDs are joins, not biological gene names.
- `--bakta`: directory containing `SAMPLE/SAMPLE.fna` and `SAMPLE/SAMPLE.gff3` for
  each retained sample. Missing references stop by default; deliberately incomplete
  annotation requires `--allow-missing-references` and is recorded.

The supplied DNA catalogue may contain more loci than the binary FASTA. These
are annotated, but receive `binary_unavailable` rather than an invented D value.
Every binary locus must exist in the DNA catalogue. If upstream preprocessing
already removed a unitig, this helper cannot recover its distribution.

## Run

```bash
ko-variation-unitig-audit \
  --unitigs /path/to/saureus_k61.unitigs \
  --fasta /path/to/saureus_k61.sample_ids.fasta \
  --fasta-columns-are-locus-ids \
  --tree /path/to/saureus_core.treefile \
  --metadata /path/to/saureus_collection_metadatatxt.txt \
  --panaroo /path/to/gene_presence_absence_roary.csv \
  --bakta /path/to/bakta \
  --species 'Staphylococcus aureus' \
  --out /path/to/results/unitig_amr_audit \
  --rscript /path/to/kovar_d_env/bin/Rscript \
  --permutations 1000 --seed 42 --resume
```

The equivalent module invocation is `python -m ko_variation.unitig_audit`.
Use a new output directory, separate from original inputs and network outputs.
Run one writer per output directory; simultaneous runs must use separate outputs.

To inspect annotation before installing/running R, add `--stage annotate`.
After annotation, start directly at D with:

```bash
ko-variation-unitig-audit --stage d --out /path/to/results/unitig_amr_audit \
  --rscript /path/to/environment/bin/Rscript --resume
```

`--stage d` skips binary FASTA loading, matrix export, reference annotation,
annotation collection and **all gzip exports**. It uses only the saved working
tree, target binary patterns, and AMR annotation state, then updates the final
small TSV reports. Raw-input/cohort/annotation settings are not reapplied; rerun
annotation in a separate output directory to change those. R settings and MAF
report flags can be changed. Add `--zero-tip-policy epsilon` when deliberately
accepting the zero-tip numerical handling described below.

New runs save `annotation_state.json` and a checksum-verified
`annotation_ready.json` before starting D, including when D later fails. Older
runs can import existing `amr_unitig_audit.tsv`, `amr_gene_summary.tsv`,
`pattern_statistics.tsv`, `analysis.treefile` and `target_patterns.tsv` once.
That import checks target/sample/count consistency and creates the new checkpoint.
It does not read or compress assembly evidence. Because old failed runs did not
record raw-source provenance, legacy import explicitly records that source
identity is unverified; it adopts the saved outputs as the prepared analysis.
Missing or inconsistent files stop the restart rather than rerunning annotation.

Annotation-only reports have blank D values and `not_run` statuses. For
`--stage all --resume`, matrix export and collection still repeat; use `--stage d`
to bypass them. Completed D patterns remain independently checkpointed.
Changing MAF updates report flags without changing annotation or D checkpoints.
Changing cohort, tree, D settings, or target-pattern content gives a new D cache.

### Zero-length tree branches

caper cannot use zero terminal lengths. The default is to stop. If deliberately
accepting a tiny numerical replacement in the D working tree, add:

```bash
--zero-tip-policy epsilon --tip-epsilon 1e-12
```

Zero internal edges are collapsed exactly; positive lengths are retained. Support
labels are removed from the working tree because they are not unique node IDs.
Any zero-tip replacement and internal collapse are recorded in
`d_checkpoints/<identity>/tree_adjustments.tsv`; the exact D tree and R session are
saved there. Values close to a proposed D cutoff merit caution, especially after
branch-length adjustments. This is numerical handling, not a biological correction.

## AMR targets and annotations

The bundled species-specific AMR catalogue supplies mecA/mecC, tetK/tetM and other
listed determinants/targets. **graR is added for S. aureus** as the requested
regulatory gene. Add other labels with repeated `--target GENE`. A custom
`--targets FILE` replaces the bundled catalogue; required columns are `determinant`
and `aliases`, with optional `species`, `determinant_type`, `product_require`, and
`product_exclude`. Aliases use semicolons, not substring matching; tet(M) and tetM
normalize together, while tetR is not tetM. The catalogue is a curated starter,
not a complete resistance database or a downloaded CARD release.

Labels combine Bakta gene names and Panaroo annotation names. Product rules
distinguish mecA/PBP2a from the unrelated MecA adaptor. Unverified products remain
visible for review; rejected homonyms do not independently trigger D computation.
The helper preserves multiple genes and all evidence, rather than assigning one
preferred gene. A unitig's annotation can differ across isolates; no single-label
resolution is imposed. `--nearby-bp 500` reports strand-aware upstream/downstream
CDS within 500 bp of the unitig endpoint, separately from coding overlap. A hit
can overlap one CDS and be nearby another; both records remain visible. Nearby
annotation is **not** the network's >10 kb distal-distance definition.

## Outputs and interpretation

Start with these two tables:

| File | Meaning |
| --- | --- |
| `amr_gene_summary.tsv` | Target gene, annotated CDS count, coding/nearby unitigs, unique coding patterns, MAF/D retention counts and D range |
| `amr_unitig_audit.tsv` | Every target–unitig–gene-context match, including product status, coding/nearby status, frequency, entropy, D, and proposed filter flags |

Supporting outputs:

- `unitig_annotation.tsv.gz`: every input DNA locus, mapping status, all names and
  internal IDs; distinguishes unmapped, intergenic, multi-context and binary-unavailable.
- `unitig_gene_hits.tsv.gz`: full coordinate evidence from all supplied assemblies.
  Independently checkpointed assembly evidence is in `reference_annotations/`.
- `pattern_statistics.tsv`: all exact patterns, counts, MAF and total entropy.
- `column_mapping.tsv`: original locus, binary column and exact-pattern ID.
- `pattern_presence_absence.tsv.gz`: one row per unique pattern, one column per
  retained sample. Use `--no-matrix-tsv` to skip this large text export.
- `target_patterns.tsv`: binary patterns actually submitted for D.
- `target_pattern_statistics.tsv`: AMR pattern statistics including D status.
- `sample_inclusion.tsv`, `tree_only_samples.tsv`, `missing_references.tsv`,
  `metadata_filtered.tsv`, `analysis.treefile`, `run_manifest.json`: provenance.

`passes_maf` uses strict MAF > 0.05 by default. `passes_D_nonnegative` uses D >= 0.
`passes_both` combines these flags, **without discarding anything**. Undefined D
does not become zero or a pass. All AMR-matching patterns, including low-MAF ones,
are evaluated before these flags are applied. D is evaluated once per exact
pattern; genomic sequences with that pattern retain their separate annotations.

`D_status` is `ok`, `not_estimable_monomorphic`, `degenerate_reference`, `error`,
or an explicit unrun/unavailable status. Per-pattern numerical errors are reported
and retried on resume; inspect these statuses before interpreting summaries.
`p_random_lower` and `p_brownian_upper` are caper's **directional simulation-tail
probabilities**, not generic two-sided P-values, KOVAR P-values, or FDR-adjusted
tests. No threshold is applied to them. D=0 and D=1 are Brownian-threshold and
random-distribution references; they are not validated association-power cutoffs.

Full-genome exact matching is a potentially long, disk-intensive stage: it searches
all assemblies and retains all coordinates. It currently uses one matching process,
with one progress/checkpoint boundary per assembly, followed by one D process with
per-pattern progress. Do not run the full audit on a login node. No runtime or
memory requirement has been measured on the ARC collection. Check available disk
space for both per-assembly evidence and the merged evidence file.

Gene overlap does not establish a functional mutation, resistance phenotype,
epistasis or a significant covariation edge. A gene without an annotated unitig is
not automatically absent from the genomes. Compare named target retention without
choosing thresholds merely to rescue a preferred gene or pair.

Method references: [Fritz & Purvis (2010)](https://pubmed.ncbi.nlm.nih.gov/20184650/),
[caper phylo.d](https://cran.r-project.org/web/packages/caper/refman/caper.html#phylo.d),
[pyahocorasick exact matcher](https://pyahocorasick.readthedocs.io/en/latest/).
