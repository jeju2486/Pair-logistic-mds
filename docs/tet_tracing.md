# Trace tet-associated unitigs before network selection

This diagnostic searches the original DNA unitigs around raw Bakta tet CDS
annotations, then joins the results to the **full original** KOVAR pair table.
It does not refit models, modify source files, relax filters, or insert tet nodes
into the scientific network. The main scanner and plotting wrapper are unchanged.

## Run on ARC

```bash
cd /data/biol-micro-genomics/kell7366/kovar/Pair-logistic-mds
git pull --ff-only origin feature/downstream-helpers
sbatch run_trace_tet_saureus.sh
```

The wrapper uses the same S. aureus paths as the plotting workflow and the existing
KOVAR Python and pyseer annotation/BWA environments. `PAIRS` must name the exact
zero-based candidate file supplied to the original KOVAR scan. Its default is
`pangwes_saureus_output/saureus/saureus.mi_filtered.ud_sgg_0_based`. If your file
has a different name, set it explicitly:

```bash
PAIRS=/absolute/path/to/the/original/zero_based_pairs \
  sbatch --export=ALL run_trace_tet_saureus.sh
```

To trace only the full KOVAR results without checking the separate candidate
file, explicitly use `CHECK_CANDIDATES=0`. Candidate membership is then reported
as not checked, rather than inferred from missing KOVAR rows.

Defaults are `GENES="tetK tetM"`, `NEARBY_BP=500`, `LD_DISTANCE_BP=10000`,
`DISTANCE_COLUMN=min_distance`, `CROSS_CONTIG=exclude`, `BONFERRONI_ALPHA=0.05`,
`MAX_HITS=10000` and `CHUNK_ROWS=50000`. Set selection settings to the values used
for your map. To include further named determinants:

```bash
GENES="tetK tetM tetL tetO tetS tetW tet38" \
  sbatch --export=ALL run_trace_tet_saureus.sh
```

The wrapper requires raw Bakta files for each scanner isolate. If you intentionally
need a partial assembly trace, run the Python command with
`--allow-missing-assemblies`; missing isolates are listed explicitly in the summary.
When available, the wrapper passes the scanner's `sample_inclusion.tsv` and
`run_summary.txt`. The former applies the same retained cohort, and the latter
checks sample, result-row and original test counts and reports actual scan settings.
This is important if tree matching excluded isolates. Supply these files manually
when using the Python command outside the wrapper.

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

Default output: `kovar_saureus_output/saureus/results/tet_trace/`.

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

The wrapper enables `--resume`. Completed preparation, mapping and tracing reports
are reused only when inputs/settings/source match. Interrupted stages restart;
existing KOVAR models are never rerun. Cache identity uses source paths, sizes and
nanosecond mtimes, plus the tracer's source hash and settings. Do not alter cached
inputs or outputs while preserving their timestamps. BWA stages are single-process;
the ARC wrapper requests one CPU and 24 GB RAM. Preparation and pair scans report
progress. BWA diagnostics are stored in `bwa_index.log` and `bwa_fastmap.log`.

The Python entry point is `python -m ko_variation.trace_tet --help`. Dependencies
are the existing KOVAR NumPy/pandas environment and BWA; pyseer itself is not called.
