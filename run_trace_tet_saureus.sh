#!/bin/bash
#SBATCH --job-name=saureus_tet_trace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --clusters=arc
#SBATCH --partition=medium
#SBATCH --time=1-00:00:00
#SBATCH --mem=24G
#SBATCH --output=saureus_tet_trace_%j.out
#SBATCH --error=saureus_tet_trace_%j.err

set -euo pipefail
ROOT="${ROOT:-/data/biol-micro-genomics/kell7366/kovar}"
KOVAR_REPO="${KOVAR_REPO:-$ROOT/Pair-logistic-mds}"
RESULTS="${RESULTS:-$ROOT/kovar_saureus_output/saureus/results}"
SCORE="${SCORE:-$RESULTS/ko_variation.tsv}"
FASTA="${FASTA:-$ROOT/kovar_saureus_output/saureus/prepared_inputs/saureus_k61.sample_ids.fasta}"
UNITIGS="${UNITIGS:-$ROOT/pangwes_saureus_output/saureus/saureus_k61.unitigs}"
BAKTA_DIR="${BAKTA_DIR:-$ROOT/panaroo_saureus_output/bakta}"
PANAROO_OUT="${PANAROO_OUT:-$ROOT/panaroo_saureus_output/saureus}"
PANAROO="${PANAROO:-$PANAROO_OUT/gene_presence_absence_roary.csv}"
if [[ ! -s "$PANAROO" ]]; then
    PANAROO="$PANAROO_OUT/gene_presence_absence.csv"
fi
# Set PAIRS to the EXACT zero-based candidate file used by run_kovar_saureus.sh.
PAIRS="${PAIRS:-$ROOT/pangwes_saureus_output/saureus/saureus.mi_filtered.ud_sgg_0_based}"
CHECK_CANDIDATES="${CHECK_CANDIDATES:-1}"
ANNOTATION_STATUS="${ANNOTATION_STATUS:-$RESULTS/covariation_map/annotation/annotation_status.tsv}"
TRACE_DIR="${TRACE_DIR:-$RESULTS/tet_trace}"
GENES="${GENES:-tetK tetM}"
NEARBY_BP="${NEARBY_BP:-500}"
LD_DISTANCE_BP="${LD_DISTANCE_BP:-10000}"
DISTANCE_COLUMN="${DISTANCE_COLUMN:-min_distance}"
CROSS_CONTIG="${CROSS_CONTIG:-exclude}"
BONFERRONI_ALPHA="${BONFERRONI_ALPHA:-0.05}"
MAX_HITS="${MAX_HITS:-10000}"
CHUNK_ROWS="${CHUNK_ROWS:-50000}"
CONDA_ENV="${CONDA_ENV:-/data/biol-micro-genomics/kell7366/kmer_gwes_env}"
ANNOTATION_ENV="${ANNOTATION_ENV:-$ROOT/pyseer_annotation_env}"
KOVAR_PYTHON="${KOVAR_PYTHON:-python3}"
BWA="${BWA:-bwa}"

for input in "$SCORE" "$FASTA" "$UNITIGS" "$PANAROO" "$KOVAR_REPO/ko_variation/trace_tet.py"; do
    [[ -s "$input" ]] || { echo "Missing input: $input" >&2; exit 1; }
done
if [[ "$CHECK_CANDIDATES" == "1" && ! -s "$PAIRS" ]]; then
    echo "Set PAIRS to the zero-based PAN-GWES candidate file used in the original KOVAR scan." >&2
    echo "Current path: $PAIRS" >&2
    echo "For a trace without candidate-file membership, explicitly set CHECK_CANDIDATES=0." >&2
    exit 1
fi
if [[ "$CHECK_CANDIDATES" != "0" && "$CHECK_CANDIDATES" != "1" ]]; then
    echo "CHECK_CANDIDATES must be 0 or 1" >&2; exit 1
fi
module purge
module load Anaconda3
source activate "$CONDA_ENV"
export PYTHONPATH="$KOVAR_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$ANNOTATION_ENV/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
read -r -a GENE_LIST <<< "$GENES"
ARGS=(--unitigs "$UNITIGS" --fasta "$FASTA" --bakta "$BAKTA_DIR"
      --results "$SCORE" --panaroo "$PANAROO" --out "$TRACE_DIR"
      --genes "${GENE_LIST[@]}" --nearby-bp "$NEARBY_BP"
      --ld-distance "$LD_DISTANCE_BP" --distance-column "$DISTANCE_COLUMN"
      --cross-contig "$CROSS_CONTIG"
      --alpha "$BONFERRONI_ALPHA" --chunk-rows "$CHUNK_ROWS"
      --bwa "$BWA" --max-hits "$MAX_HITS" --resume)
if [[ "$CHECK_CANDIDATES" == "1" ]]; then ARGS+=(--pairs "$PAIRS"); fi
if [[ -s "$ANNOTATION_STATUS" ]]; then ARGS+=(--annotation-status "$ANNOTATION_STATUS"); fi
if [[ -s "$RESULTS/sample_inclusion.tsv" ]]; then ARGS+=(--sample-inclusion "$RESULTS/sample_inclusion.tsv"); fi
if [[ -s "$RESULTS/run_summary.txt" ]]; then ARGS+=(--run-summary "$RESULTS/run_summary.txt"); fi
echo "Read-only tet trace: $TRACE_DIR; source scan: $SCORE"
"$KOVAR_PYTHON" -m ko_variation.trace_tet "${ARGS[@]}"
echo "Send trace_summary.json and tet_locus_trace.tsv first; tet_pair_trace.tsv gives pair-level details."
