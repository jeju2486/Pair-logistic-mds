# Trace tet-associated unitigs before network selection

This diagnostic searches the original DNA unitigs around raw Bakta tet CDS
annotations, then joins the results to the **full original** KOVAR pair table.
It does not refit models, modify source files, relax filters, or insert tet nodes
into the scientific network. The main scanner is unchanged.

## Portable command

```bash
ko-variation-trace-tet --unitigs unitigs.tsv --fasta binary.fa \
  --bakta bakta --results results/ko_variation.tsv \
  --panaroo panaroo/gene_presence_absence.csv --pairs candidates.tsv \
  --annotation-status downstream/annotation/annotation_status.tsv \
  --sample-inclusion results/sample_inclusion.tsv \
  --run-summary results/run_summary.txt \
  --out diagnostics/tet --genes tetK tetM \
  --distance-column min_distance --ld-distance 10000 --resume
```

The same command runs as `python -m ko_variation.trace_tet`. Use an environment
with KOVAR's dependencies and BWA on PATH, or pass `--bwa /path/to/bwa`. No Pyseer,
scheduler, ARC account path or server-specific environment is required.
Native BWA requires a compatible Unix environment such as Linux/macOS or WSL.

`--pairs` must identify the exact zero-based candidate file supplied to the
original scan. Omit it to report candidate membership as not checked. The Panaroo
CSV and annotation audit are optional. Supply scanner `sample_inclusion.tsv` and
`run_summary.txt` when available, particularly when the tree excluded isolates.
This preserves the analyzed cohort and checks full-result counts/settings.

Defaults are tetK/tetM, nearby distance 500 bp, distal cutoff 10,000 bp,
`min_distance`, cross-contig exclusion and Bonferroni alpha 0.05. Match selection
settings to your map. Use `--genes tetK tetM tetL tetO tetS tetW tet38` for further
named tet determinants. This diagnostic does not yet support gyr mutation calling.
Raw Bakta inputs use `bakta/SAMPLE/SAMPLE.gff3` and `SAMPLE.fna`. Missing assemblies
stop preparation unless `--allow-missing-assemblies` is explicitly supplied.

## What it checks

1. Find the requested tet genes in **raw** Bakta gene, name or product fields,
   independently of Panaroo cluster labels. Names such as `tet(M)` and `tetM`
   resolve to the same requested determinant. TetR-family regulators are excluded
   by the default gene list. This name-based search is not a new homology-based
   resistance-gene caller: missing or incorrect Bakta annotations remain a limitation.
2. Extract target-containing sequence windows from every matching assembly in
   the scanner cohort. The window extends by the annotation distance plus the
   longest supplied unitig length minus one, so a full unitig overlapping the
   target region can still match. Overlapping windows on a contig are merged.
3. Map **every supplied DNA unitig** using BWA full-length exact matches on both
   strands. All reported matches across all extracted references are retained;
   the first assembly does not determine a unique gene assignment. Unitig IDs
   remain the explicit zero-based PAN-GWES IDs, checked against FASTA dimensions.
4. Record every qualifying tet CDS overlap or same-contig nearby hit, even when
   the unitig also overlaps other genes. The hit table retains the other CDS
   annotations. Coding distance is zero; nearby distance uses the closest interval
   endpoints (adjacent bases are 1 bp apart). Upstream/downstream follow CDS strand.
5. Count unitig presence and minor-state frequency in the scanner cohort and flag
   exact hits whose corresponding binary genotype says absent. These discrepancies
   warrant checking locus indexing and whether assemblies match the PAN-GWES input.
   This comparison requires the scanner encoding A=absence and C=presence.
6. Trace candidate membership, KOVAR statuses, original primary P-values and the
   distance rule. The Bonferroni denominator describes the full scan, never the
   tet subset. Pair statuses such as `LOW_CELL_COUNT` remain visible. The trace
   uses existing statistics and does not attempt to estimate biological epistasis.
7. Join the current annotation audit where supplied. An absent audit row is
   labelled `NOT_IN_SUPPLIED_AUDIT`; this is expected for loci removed before the
   post-selection annotation step and does not by itself establish annotation failure.

Linear contig coordinates do not wrap across circular origins or assembly breaks.
Exact matches can miss sequence differences between the supplied assemblies and
the original PAN-GWES input. BWA's full-hit limit is raised to 10,000; queries above
the limit are listed separately and must not be interpreted as negative mappings.
Repeated identical sequences can map near tet in one isolate and elsewhere in
another; proximity alone does not establish a tet-specific variant or regulation.
Panaroo membership is checked for the raw tet CDS IDs, including its QC suffixes.
An unmapped CDS remains visible in the target table rather than losing its raw name.

## Reports

Reports are written to the directory supplied with `--out`.

| File | Content |
| --- | --- |
| `tet_targets.tsv` | Raw tet CDS copies, references, coordinates, products and Panaroo memberships |
| `tet_unitig_hits.tsv` | Every tet-overlapping/nearby exact unitig hit and all overlapping CDS features |
| `tet_locus_trace.tsv` | Frequency, binary/assembly discrepancies, multi-CDS hits, candidate/result counts and existing annotation per locus |
| `tet_pair_trace.tsv` | Original tet-associated KOVAR rows, statuses, statistics and removal stage |
| `trace_summary.json` | Counts for each stage, actual scanner settings and mapping limitations |
| `repeat_limited_unitigs.tsv` | Queries whose full exact-match count exceeds the BWA limit |

Pair stages are `ineligible_or_failed`, `missing_primary_p`, `not_significant`,
`excluded_distance` and `selected_distal`. Counts are **rows in the supplied files**;
they should not be interpreted as independent biological gene pairs. The script
does not independently annotate the other endpoint of every KOVAR pair; its locus
ID remains available for subsequent inspection.

Start by examining `trace_summary.json` and `tet_locus_trace.tsv`. If tet loci have
no candidate rows, inspect candidate generation. If their rows predominantly have
`LOW_CELL_COUNT`, inspect discordant counts in `tet_pair_trace.tsv`: the current
default minimum count of one in every cell excludes perfectly co-occurring binary
features. If significant rows fail distance selection, the absence concerns the
defined distal network. If selected distal tet-associated rows survive but the
current audit assigns unrelated/ambiguous genes, inspect reference-dependent
annotation and the raw CDS evidence.

Add `--resume` to reuse completed stages. Completed preparation, mapping and tracing reports
are reused only when inputs/settings/source match. Interrupted stages restart;
existing KOVAR models are never rerun. Cache identity uses source paths, sizes and
nanosecond mtimes, plus the tracer's source hash and settings. Outputs use SHA-256
hashes through the shared downstream cache. Older tracer manifests regenerate
once; numerical refit checkpoints are unaffected. Do not alter inputs while
preserving their timestamps. BWA stages are single-process. Preparation and pair scans report
progress. BWA diagnostics are stored in `bwa_index.log` and `bwa_fastmap.log`.

The Python entry point is `python -m ko_variation.trace_tet --help`. Dependencies
are the existing KOVAR NumPy/pandas environment and BWA; pyseer itself is not called.
