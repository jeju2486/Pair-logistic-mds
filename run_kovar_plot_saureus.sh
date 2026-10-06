#!/bin/bash
#SBATCH --job-name=saureus_kovar_map
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --clusters=arc
#SBATCH --partition=medium
#SBATCH --time=2-00:00:00
#SBATCH --mem=100G
#SBATCH --output=saureus_kovar_map_%j.out
#SBATCH --error=saureus_kovar_map_%j.err

set -euo pipefail

ROOT="${ROOT:-/data/biol-micro-genomics/kell7366/kovar}"
RESULTS="${RESULTS:-$ROOT/kovar_saureus_output/saureus/results}"
SCORE="${SCORE:-$RESULTS/ko_variation.tsv}"
FASTA="${FASTA:-$ROOT/kovar_saureus_output/saureus/prepared_inputs/saureus_k61.sample_ids.fasta}"
TREE="${TREE:-$ROOT/kovar_saureus_output/saureus/prepared_inputs/saureus_core.rooted.treefile}"
KOVAR_REPO="${KOVAR_REPO:-$ROOT/Pair-logistic-mds}"
CONDA_ENV="${CONDA_ENV:-/data/biol-micro-genomics/kell7366/kmer_gwes_env}"
KOVAR_PYTHON="${KOVAR_PYTHON:-python3}"
MAP_DIR="${MAP_DIR:-$RESULTS/covariation_map}"
PREFIX="${PREFIX:-$MAP_DIR/saureus_gene_covariation}"
ANNOTATION_DIR="${ANNOTATION_DIR:-$MAP_DIR/annotation}"
UNITIGS="${UNITIGS:-$ROOT/pangwes_saureus_output/saureus/saureus_k61.unitigs}"
PANAROO_OUT="${PANAROO_OUT:-$ROOT/panaroo_saureus_output/saureus}"
BAKTA_DIR="${BAKTA_DIR:-$ROOT/panaroo_saureus_output/bakta}"
GENE_PRESENCE_ABSENCE="${GENE_PRESENCE_ABSENCE:-$PANAROO_OUT/gene_presence_absence_roary.csv}"
if [[ ! -s "$GENE_PRESENCE_ABSENCE" ]]; then
    GENE_PRESENCE_ABSENCE="$PANAROO_OUT/gene_presence_absence.csv"
fi
ANNOTATION_ENV="${ANNOTATION_ENV:-$ROOT/pyseer_annotation_env}"
ANNOTATION_PYTHON="${ANNOTATION_PYTHON:-$ANNOTATION_ENV/bin/python}"
if [[ -n "${ANNOTATION:-}" ]]; then
    AUTO_ANNOTATE="${AUTO_ANNOTATE:-0}"
else
    ANNOTATION="$ANNOTATION_DIR/locus_to_gene.tsv"
    AUTO_ANNOTATE="${AUTO_ANNOTATE:-1}"
fi

# Native helper: Bonferroni family-wise alpha, NOT a raw P-value or BH q cutoff.
BONFERRONI_ALPHA="${BONFERRONI_ALPHA:-0.05}"
LD_DISTANCE_BP="${LD_DISTANCE_BP:-10000}"
# PAN-GWES minimum separation: require >10 kb across represented genomes.
# Set DISTANCE_COLUMN=distance explicitly if that is your intended distance rule.
DISTANCE_COLUMN="${DISTANCE_COLUMN:-min_distance}"
CROSS_CONTIG="${CROSS_CONTIG:-exclude}"
# Auto-annotation records ambiguous, intergenic and unmapped loci separately.
# Only resolved CDS mappings enter the gene map; retain all selected pair effects.
MISSING_GENES="${MISSING_GENES:-drop}"
NETWORK_SEED="${NETWORK_SEED:-42}"
PNG_DPI="${PNG_DPI:-300}"
KEEP_DISTANCE_PLOT="${KEEP_DISTANCE_PLOT:-0}"
PAIR_CHUNK_ROWS="${PAIR_CHUNK_ROWS:-50000}"
SCREENED_SCORE="$PREFIX.screened.tsv"
SCREENING_MANIFEST="$PREFIX.screening.json"
export PAIR_CHUNK_ROWS
RESUME="${RESUME:-1}"
REFIT_CHECKPOINT="${REFIT_CHECKPOINT:-$PREFIX.refit.sqlite}"
PROGRESS_EVERY="${PROGRESS_EVERY:-10}"
PROGRESS_SECONDS="${PROGRESS_SECONDS:-60}"
ANNOTATION_CACHE="$ANNOTATION.cache.json"
export RESUME
if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
    echo "RESUME must be 0 or 1" >&2
    exit 1
fi

for input in "$SCORE" "$FASTA" "$TREE"; do
    if [[ ! -s "$input" ]]; then
        echo "Missing or empty input: $input" >&2
        exit 1
    fi
done
if [[ ! -s "$KOVAR_REPO/ko_variation/annotation_cli.py" ]]; then
    echo "Missing development helper in: $KOVAR_REPO" >&2
    echo 'Clone feature/downstream-helpers and install its [network] extra; see the accompanying instructions.' >&2
    exit 1
fi
if [[ ! -s "$KOVAR_REPO/ko_variation/effect_checkpoint.py" || ! -s "$KOVAR_REPO/ko_variation/workflow_cache.py" ]]; then
    echo "This plotting script needs the updated downstream resume/progress helper in: $KOVAR_REPO" >&2
    exit 1
fi

module purge
module load Anaconda3
source activate "$CONDA_ENV"

# Select the development source explicitly, not an older installed entry point.
export PYTHONPATH="$KOVAR_REPO${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p "$(dirname "$PREFIX")"

# Read the full scan in bounded chunks. Count tests BEFORE distance/status filters,
# then apply the native selector with that original denominator in every chunk.
"$KOVAR_PYTHON" - "$SCORE" "$SCREENED_SCORE" "$SCREENING_MANIFEST" \
    "$DISTANCE_COLUMN" "$BONFERRONI_ALPHA" "$LD_DISTANCE_BP" "$CROSS_CONTIG" <<'PY'
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from ko_variation.postprocess import SelectionConfig, select_distal_signals
from ko_variation.workflow_cache import stage_identity, cache_matches, cache_fields

score, output, manifest = map(Path, sys.argv[1:4])
distance_column, alpha, ld_distance, cross_contig = sys.argv[4:]
identity = stage_identity("screening", [score], dict(distance_column=distance_column,
                          alpha=float(alpha), ld_distance=float(ld_distance), cross_contig=cross_contig))
if os.environ["RESUME"] == "1" and cache_matches(manifest, identity, [output]):
    print(f"[pipeline] reusing completed screening: {output}", flush=True)
    raise SystemExit(0)
chunk_rows = int(os.environ["PAIR_CHUNK_ROWS"])
if chunk_rows <= 0:
    raise SystemExit("PAIR_CHUNK_ROWS must be positive")
if score.resolve() in {output.resolve(), manifest.resolve()}:
    raise SystemExit("Screening output must not overwrite the original score")
with score.open(encoding="utf-8-sig") as handle:
    header = next(csv.reader(handle, delimiter="\t"))
required = {"u", "v", "status", "p_primary", "n11", "n10", "n01", "n00", distance_column}
if missing := required - set(header):
    raise SystemExit(f"Missing score columns: {sorted(missing)}")
test_columns = ["p_primary"] + (["n_tests"] if "n_tests" in header else [])
input_rows = nonmissing_p = 0
declared_tests = None
print(f"Counting original tests in chunks of {chunk_rows:,} rows", flush=True)
for chunk in pd.read_csv(score, sep="\t", usecols=test_columns, chunksize=chunk_rows):
    p = pd.to_numeric(chunk.p_primary, errors="raise")
    if ((p.dropna() < 0) | (p.dropna() > 1)).any():
        raise ValueError("Significance values must lie in [0, 1] or be missing")
    input_rows += len(chunk)
    nonmissing_p += int(p.notna().sum())
    if "n_tests" in chunk and len(chunk):
        tests = pd.to_numeric(chunk.n_tests, errors="raise")
        if not np.isfinite(tests).all() or (tests < 0).any() or (tests != np.floor(tests)).any() or tests.nunique() != 1:
            raise ValueError("n_tests must consistently describe the original full scan")
        value = int(tests.iloc[0])
        if declared_tests is not None and value != declared_tests:
            raise ValueError("Inconsistent n_tests across score chunks")
        declared_tests = value
n_tests = declared_tests if declared_tests is not None else nonmissing_p
if n_tests < nonmissing_p:
    raise ValueError("n_tests is smaller than the original nonmissing P-value count")
config = SelectionConfig(significance_threshold=float(alpha), distance_column=distance_column,
                         ld_distance=float(ld_distance), cross_contig=cross_contig)
print(f"Original rows: {input_rows:,}; original tests: {n_tests:,}; selecting distal pairs", flush=True)
selected_rows = undefined_rows = scanned_rows = 0
temporary = Path(str(output) + ".partial")
try:
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        first = True
        for index, chunk in enumerate(pd.read_csv(score, sep="\t", chunksize=chunk_rows), 1):
            distance = pd.to_numeric(chunk[distance_column], errors="raise")
            undefined = distance.eq(-1)
            undefined_rows += int(undefined.sum())
            chunk[distance_column + "_pangwes_raw"] = chunk[distance_column]
            chunk[distance_column] = distance.mask(undefined)
            chunk["n_tests"] = n_tests
            selected = select_distal_signals(chunk, config)
            selected.to_csv(handle, sep="\t", index=False, header=first)
            first = False
            selected_rows += len(selected)
            scanned_rows += len(chunk)
            if index == 1 or index % 20 == 0:
                print(f"Screened {scanned_rows:,}/{input_rows:,} rows; retained {selected_rows:,}", flush=True)
    temporary.replace(output)
except BaseException:
    temporary.unlink(missing_ok=True)
    raise
details = dict(original_results=str(score.resolve()), selected_input=str(output.resolve()),
               input_rows=input_rows, nonmissing_primary_p=nonmissing_p, n_tests=n_tests,
               selected_rows=selected_rows, undefined_distance_rows=undefined_rows,
               chunk_rows=chunk_rows, distance_column=distance_column,
               bonferroni_alpha=float(alpha), ld_distance=float(ld_distance), cross_contig=cross_contig)
details.update(cache_fields(identity, [output]))
manifest.write_text(json.dumps(details, indent=2), encoding="utf-8")
print(f"Screening complete: {selected_rows:,} pairs; {undefined_rows:,} undefined distances excluded", flush=True)
PY

# Auto-annotation uses original PAN-GWES DNA sequences, not the binary A/C FASTA.
if [[ "$AUTO_ANNOTATE" == "1" ]]; then
    for input in "$UNITIGS" "$GENE_PRESENCE_ABSENCE"; do
        if [[ ! -s "$input" ]]; then
            echo "Missing annotation input: $input" >&2
            exit 1
        fi
    done
    ANNOTATION_CACHE_ARGS=(
        --manifest "$ANNOTATION_CACHE" --stage annotation
        --inputs "$SCREENED_SCORE" "$FASTA" "$UNITIGS" "$GENE_PRESENCE_ABSENCE"
        --outputs "$ANNOTATION" --directories "$BAKTA_DIR"
        --settings "workflow=panaroo_pyseer_exact_v1" "annotation_python=$ANNOTATION_PYTHON"
    )
    if [[ "$RESUME" == "1" ]] && "$KOVAR_PYTHON" -m ko_variation.workflow_cache check "${ANNOTATION_CACHE_ARGS[@]}"; then
        AUTO_ANNOTATE=0
    fi
fi
if [[ "$AUTO_ANNOTATE" == "1" ]]; then
    echo "[pipeline] preparing unitig annotation"
    mkdir -p "$ANNOTATION_DIR" "$(dirname "$ANNOTATION")"
    ANNOTATION_WORK="$(mktemp -d "$ANNOTATION_DIR/run.XXXXXXXX")"
    "$KOVAR_PYTHON" - "$SCREENED_SCORE" "$FASTA" "$UNITIGS" "$GENE_PRESENCE_ABSENCE" \
        "$BAKTA_DIR" "$ANNOTATION_WORK" "$DISTANCE_COLUMN" \
        "$BONFERRONI_ALPHA" "$LD_DISTANCE_BP" "$CROSS_CONTIG" "$SCREENING_MANIFEST" <<'PY'
import csv
import json
from pathlib import Path
import re
import sqlite3
import sys
from urllib.parse import unquote
import pandas as pd
from ko_variation.postprocess import SelectionConfig, select_distal_signals

score, fasta, unitigs, gpa, bakta, work = map(Path, sys.argv[1:7])
distance_column, alpha, ld_distance, cross_contig = sys.argv[7:11]
screening = json.loads(Path(sys.argv[11]).read_text())
work.mkdir(parents=True, exist_ok=True)
with score.open(encoding="utf-8-sig") as handle:
    header = next(csv.reader(handle, delimiter="\t"))
required = {"u", "v", "status", "p_primary", "n11", "n10", "n01", "n00", distance_column}
if missing := required - set(header):
    raise SystemExit(f"Missing score columns: {sorted(missing)}")
columns = required | ({"n_tests"} if "n_tests" in header else set())
results = pd.read_csv(score, sep="\t", usecols=list(columns))
# PAN-GWES uses -1 for an undefined graph distance. Keep every original row
# and P-value so the Bonferroni denominator still describes the full scan.
undefined_distance = pd.to_numeric(results[distance_column], errors="raise").eq(-1)
undefined_distance_rows = int(undefined_distance.sum())
results.loc[undefined_distance, distance_column] = float("nan")
print(f"Undefined {distance_column} (-1): {undefined_distance_rows} rows treated as missing", flush=True)
selected = select_distal_signals(results, SelectionConfig(
    significance_threshold=float(alpha), distance_column=distance_column,
    ld_distance=float(ld_distance), cross_contig=cross_contig))
loci = sorted(set(selected.u.astype(int)) | set(selected.v.astype(int)))
details = {"original_results": screening["original_results"], "selected_input": str(score), "selected_pairs": len(selected),
           "selected_loci": len(loci), "distance_column": distance_column,
           "bonferroni_alpha": float(alpha), "ld_distance": float(ld_distance),
           "undefined_distance_rows": screening["undefined_distance_rows"], "n_tests": screening["n_tests"],
           "unitigs": str(unitigs), "panaroo_clusters": str(gpa),
           "mapping": "pyseer draft/full-length exact match; first matching assembly",
           "gene_assignment": "all reported hits must overlap one Panaroo cluster"}
del results, selected

# Unitig IDs are explicit zero-based PAN-GWES column IDs; never infer from GFA names.
wanted = set(loci)
found = {}
with unitigs.open() as handle:
    for line in handle:
        fields = line.split()
        if len(fields) < 2 or fields[0].startswith("#"):
            continue
        locus = int(fields[0])
        if locus in wanted:
            if locus in found:
                raise SystemExit(f"Duplicate unitig ID: {locus}")
            sequence = fields[1].upper()
            if set(sequence) - set("ACGT"):
                raise SystemExit(f"Invalid DNA sequence for unitig {locus}")
            found[locus] = sequence
if missing := wanted - set(found):
    raise SystemExit(f"Unitig sequences missing for loci: {sorted(missing)[:10]}")
with (work / "queries.tsv").open("w") as handle:
    handle.write("sequence\tlocus\n")
    for locus in loci:
        handle.write(f"{found[locus]}\t{locus}\n")
(work / "selected_loci.json").write_text(json.dumps(loci))
with (work / "references.txt").open("w") as references:
    if loci:
        sample_names = []
        with fasta.open() as handle:
            sample_names = [line[1:].split()[0] for line in handle if line.startswith(">")]
        sample_set = set(sample_names)
        # Store memberships on disk instead of keeping every isolate's genes in RAM.
        membership_db = sqlite3.connect(work / "panaroo_membership.sqlite")
        membership_db.execute("CREATE TABLE membership (sample TEXT, feature TEXT, cluster TEXT)")
        pending = []
        gene_metadata = {}
        with gpa.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if "Gene" not in (reader.fieldnames or []):
                raise SystemExit("Panaroo table lacks the Gene column")
            sample_columns = [name for name in reader.fieldnames if name in sample_set]
            for row in reader:
                cluster = row["Gene"]
                gene_metadata[cluster] = {
                    "label": row.get("Non-unique Gene name", "") or cluster,
                    "product": row.get("Annotation", "")}
                for sample in sample_columns:
                    for feature in re.split(r"[;\t]", row.get(sample, "") or ""):
                        if feature:
                            pending.append((sample, feature, cluster))
                            if len(pending) >= 50000:
                                membership_db.executemany("INSERT INTO membership VALUES (?, ?, ?)", pending)
                                pending.clear()
        if pending:
            membership_db.executemany("INSERT INTO membership VALUES (?, ?, ?)", pending)
        del pending
        membership_db.commit()
        membership_db.execute("CREATE INDEX membership_sample ON membership (sample)")
        membership_db.commit()
        reference_count = 0
        for index, sample in enumerate(sample_names):
            raw_gff = bakta / sample / f"{sample}.gff3"
            raw_fasta = bakta / sample / f"{sample}.fna"
            if not raw_gff.is_file() or not raw_fasta.is_file():
                continue
            members = {}
            for feature, cluster in membership_db.execute("SELECT feature, cluster FROM membership WHERE sample = ?", (sample,)):
                members.setdefault(feature, set()).add(cluster)
            reference_dir = work / "references" / str(index)
            reference_dir.mkdir(parents=True, exist_ok=True)
            mapped_gff = reference_dir / "genes.gff"
            cds_count = 0
            mapped_cds_count = 0
            with raw_gff.open() as source, mapped_gff.open("w") as target:
                target.write("##gff-version 3\n")
                for line in source:
                    if line.startswith("##FASTA"):
                        break
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) != 9 or fields[2] != "CDS":
                        continue
                    attributes = dict(item.split("=", 1) for item in fields[8].split(";") if "=" in item)
                    feature_id = unquote(attributes.get("ID", ""))
                    clusters = members.get(feature_id, set())
                    if not clusters:
                        # Main Panaroo CSV can append QC flags to original feature IDs.
                        clusters = members.get(feature_id + "_len", set()) | members.get(feature_id + "_pseudo", set()) | members.get(feature_id + "_len_pseudo", set())
                    if len(clusters) == 1:
                        cluster = next(iter(clusters))
                        mapped_cds_count += 1
                    else:
                        # Keep unresolved CDS features visible to overlap tests;
                        # removing them could create a false unambiguous mapping.
                        cluster = f"__UNRESOLVED_CDS_{index}_{cds_count}"
                    if any(char in cluster for char in ";|,\t\n"):
                        raise SystemExit(f"Unsupported delimiter in Panaroo cluster ID: {cluster}")
                    attributes["gene"] = cluster
                    fields[8] = ";".join(f"{key}={value}" for key, value in attributes.items())
                    target.write("\t".join(fields) + "\n")
                    cds_count += 1
            if not mapped_cds_count:
                continue
            # Isolated index files do not modify the Bakta output directories.
            linked_fasta = reference_dir / "assembly.fna"
            linked_fasta.symlink_to(raw_fasta.resolve())
            if any(char.isspace() for char in str(linked_fasta) + str(mapped_gff)):
                raise SystemExit("Pyseer reference paths must not contain whitespace")
            references.write(f"{linked_fasta.absolute()}\t{mapped_gff.absolute()}\tdraft\n")
            reference_count += 1
        membership_db.close()
        if reference_count == 0:
            raise SystemExit("No Bakta CDS IDs matched Panaroo clusters for the scanner samples")
        details["reference_count"] = reference_count
        (work / "gene_metadata.json").write_text(json.dumps(gene_metadata))
(work / "annotation_preparation.json").write_text(json.dumps(details, indent=2))
print(f"Annotation preparation: {len(loci)} loci from {details['selected_pairs']} distal pairs")
PY

    if [[ -s "$ANNOTATION_WORK/references.txt" ]]; then
        if [[ ! -x "$ANNOTATION_PYTHON" ]]; then
            echo "Missing pyseer Python: $ANNOTATION_PYTHON; install the annotation environment described in the instructions." >&2
            exit 1
        fi
        for executable in bwa bedtools gff2bed; do
            if ! PATH="$ANNOTATION_ENV/bin:$PATH" command -v "$executable" >/dev/null; then
                echo "Missing annotation dependency: $executable" >&2
                exit 1
            fi
        done
        # Pyseer uses a fixed tmp_bed filename, so give each job a unique cwd.
        (
            cd "$ANNOTATION_WORK"
            PATH="$ANNOTATION_ENV/bin:$PATH" "$ANNOTATION_PYTHON" \
                -m pyseer.kmer_mapping.annotate_hits \
                "$ANNOTATION_WORK/queries.tsv" "$ANNOTATION_WORK/references.txt" \
                "$ANNOTATION_WORK/pyseer_hits.tsv" --tmp-prefix "$ANNOTATION_WORK"
        )
        PATH="$ANNOTATION_ENV/bin:$PATH" "$ANNOTATION_PYTHON" -c \
            'import pyseer; print("pyseer version:", pyseer.__version__)' \
            > "$ANNOTATION_WORK/pyseer_version.txt"
    else
        : > "$ANNOTATION_WORK/pyseer_hits.tsv"
    fi

    "$KOVAR_PYTHON" - "$ANNOTATION_WORK" "$ANNOTATION" <<'PY'
import csv
from collections import Counter
import json
from pathlib import Path
import sys

work, output = map(Path, sys.argv[1:])
loci = json.loads((work / "selected_loci.json").read_text())
metadata_path = work / "gene_metadata.json"
metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
assigned = {}
raw_hits = {}
with (work / "pyseer_hits.tsv").open() as handle:
    for line in handle:
        fields = line.rstrip("\n").split("\t")
        if len(fields) != 3:
            raise SystemExit("Unexpected pyseer annotation output; expected sequence, locus, annotations")
        locus = int(fields[1])
        if locus in raw_hits:
            raise SystemExit(f"Duplicate pyseer output for locus {locus}")
        raw_hits[locus] = fields[2]
        genes = set()
        has_intergenic_hit = False
        for hit in fields[2].split(","):
            annotation = hit.split(";")
            if len(annotation) != 4:
                raise SystemExit(f"Unexpected hit annotation for locus {locus}")
            inside = set(filter(None, annotation[2].split("|")))
            has_intergenic_hit |= not inside
            genes.update(inside)
        if len(genes) == 1 and not has_intergenic_hit and next(iter(genes)) in metadata:
            assigned[locus] = next(iter(genes))
        elif not genes:
            assigned[locus] = None
        else:
            assigned[locus] = False

counts = Counter()
with output.open("w", newline="") as handle, (work / "annotation_status.tsv").open("w", newline="") as audit:
    writer = csv.writer(handle, delimiter="\t")
    writer.writerow(["locus", "gene", "label", "product", "group"])
    audit_writer = csv.writer(audit, delimiter="\t")
    audit_writer.writerow(["locus", "status", "gene", "pyseer_hits"])
    for locus in loci:
        gene = assigned.get(locus)
        status = "UNMAPPED" if locus not in raw_hits else "INTERGENIC" if gene is None else "AMBIGUOUS" if gene is False else "RESOLVED"
        counts[status] += 1
        audit_writer.writerow([locus, status, gene or "", raw_hits.get(locus, "")])
        if status == "RESOLVED":
            values = metadata.get(gene, {})
            writer.writerow([locus, gene, values.get("label", gene), values.get("product", ""), "Panaroo cluster"])
summary = json.loads((work / "annotation_preparation.json").read_text())
summary["annotation_status_counts"] = dict(counts)
summary["output_annotation"] = str(output)
summary["coordinates"] = "omitted: different draft assemblies are not a shared coordinate system"
(work / "annotation_summary.json").write_text(json.dumps(summary, indent=2))
print("Annotation status counts:", dict(counts))
print("Generated locus-to-gene mapping:", output)
PY
    echo "Annotation details: $ANNOTATION_WORK/annotation_summary.json"
    echo "Unresolved annotation table: $ANNOTATION_WORK/annotation_status.tsv"
    "$KOVAR_PYTHON" -m ko_variation.workflow_cache store "${ANNOTATION_CACHE_ARGS[@]}"
fi

if [[ ! -s "$ANNOTATION" ]]; then
    echo "Missing annotation: $ANNOTATION" >&2
    exit 1
fi

# Check only input headers here. The native helper handles indices and mappings.
"$KOVAR_PYTHON" - "$SCORE" "$ANNOTATION" "$DISTANCE_COLUMN" <<'PY'
import csv
import sys
from ko_variation import __version__
for path, required in [(sys.argv[1], {sys.argv[3]}), (sys.argv[2], {"locus", "gene"})]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        columns = set(next(csv.reader(handle, delimiter="\t"), []))
    if missing := required - columns:
        raise SystemExit(f"Missing columns in {path}: {sorted(missing)}")
print(f"KOVAR version: {__version__}")
PY

REVISION="$(git -C "$KOVAR_REPO" rev-parse HEAD 2>/dev/null || printf 'unknown')"
printf 'Development source: %s\nCommit: %s\nInput: %s\nAnnotation: %s\nOutput prefix: %s\n' \
    "$KOVAR_REPO" "$REVISION" "$SCORE" "$ANNOTATION" "$PREFIX"
printf 'Bonferroni alpha: %s; distance: %s > %s bp; cross-contig: %s\n' \
    "$BONFERRONI_ALPHA" "$DISTANCE_COLUMN" "$LD_DISTANCE_BP" "$CROSS_CONTIG"
printf 'source\t%s\ncommit\t%s\nannotation\t%s\n' \
    "$KOVAR_REPO" "$REVISION" "$ANNOTATION" > "$PREFIX.source.tsv"
printf 'auto_annotation\t%s\nannotation_work\t%s\n' \
    "$AUTO_ANNOTATE" "${ANNOTATION_WORK:-not_applicable}" >> "$PREFIX.source.tsv"

# Selection uses original screening P-values. Only selected pairs are refitted.
# Keep failed-effect pairs in the exports; beta does not select network edges.
ARGS=(
    --results "$SCREENED_SCORE" --fasta "$FASTA" --tree "$TREE"
    --annotation "$ANNOTATION" --out "$PREFIX" --network
    --significance-threshold "$BONFERRONI_ALPHA"
    --ld-distance "$LD_DISTANCE_BP" --distance-column "$DISTANCE_COLUMN"
    --cross-contig "$CROSS_CONTIG" --missing-genes "$MISSING_GENES"
    --seed "$NETWORK_SEED" --dpi "$PNG_DPI" --title ""
    --checkpoint-file "$REFIT_CHECKPOINT"
    --progress-every "$PROGRESS_EVERY" --progress-seconds "$PROGRESS_SECONDS"
)
if [[ "$RESUME" == "1" && -f "$REFIT_CHECKPOINT" ]]; then
    ARGS+=(--resume)
    echo "[pipeline] resuming completed pair fits from $REFIT_CHECKPOINT"
fi

# Run the native annotation helper. Save its identical static figure as editable
# SVG as well as PNG. Original n_tests in the selected input preserves selection.
SCREENING_MANIFEST="$SCREENING_MANIFEST" "$KOVAR_PYTHON" - "${ARGS[@]}" <<'PY'
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import matplotlib
import pandas as pd
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"
from matplotlib.figure import Figure
from ko_variation import annotation_cli

original_select = annotation_cli.select_distal_signals
def select_pangwes_distal(results, config=None, annotation=None):
    column = config.distance_column if config is not None else "distance"
    if column in results:
        values = pd.to_numeric(results[column], errors="raise")
        undefined = values.eq(-1)
        if undefined.any():
            # Preserve the original metadata in the selected-pair export.
            raw_column = column + "_pangwes_raw"
            if raw_column not in results:
                results[raw_column] = results[column]
            results[column] = values.mask(undefined)
            print(f"Undefined {column} (-1): {int(undefined.sum())} rows treated as missing", flush=True)
    return original_select(results, config, annotation)

original_savefig = Figure.savefig
def save_png_and_svg(figure, filename, *args, **kwargs):
    result = original_savefig(figure, filename, *args, **kwargs)
    if isinstance(filename, (str, os.PathLike)) and Path(filename).suffix.lower() == ".png":
        svg_kwargs = dict(kwargs)
        svg_kwargs["format"] = "svg"
        original_savefig(figure, Path(filename).with_suffix(".svg"), *args, **svg_kwargs)
    return result

with patch.object(Figure, "savefig", save_png_and_svg), \
        patch.object(annotation_cli, "select_distal_signals", select_pangwes_distal):
    exit_code = annotation_cli.main(sys.argv[1:])
if exit_code == 0:
    screening = json.loads(Path(os.environ["SCREENING_MANIFEST"]).read_text())
    prefix = sys.argv[sys.argv.index("--out") + 1]
    provenance_path = Path(prefix + ".selection.json")
    provenance = json.loads(provenance_path.read_text())
    provenance["input_rows"] = screening["input_rows"]
    provenance["inputs"][0] = screening["original_results"]
    provenance["screening"] = screening
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
raise SystemExit(exit_code)
PY

# Optional legacy distance profile. This additionally requires R/data.table.
if [[ "$KEEP_DISTANCE_PLOT" == "1" ]]; then
    "$KOVAR_PYTHON" -m ko_variation.plot_cli \
        --score "$SCORE" --out "$RESULTS/ko_variation_distance_pvalue.png" \
        --x-mode distance --distance-col distance --y-col neglog10_p --title ""
fi

echo "Covariation map: $PREFIX.png"
echo "Editable map: $PREFIX.svg"
echo "Interactive map: $PREFIX.html"
echo "Pair effects: $PREFIX.distal.tsv"
echo "Network tables: $PREFIX.nodes.tsv and $PREFIX.edges.tsv"
echo "Selection settings and counts: $PREFIX.selection.json"
echo "Chunked screening: $SCREENING_MANIFEST"
echo "Retained refit checkpoint: $REFIT_CHECKPOINT"
