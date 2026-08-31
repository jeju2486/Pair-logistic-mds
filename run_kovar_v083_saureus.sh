#!/usr/bin/env bash
set -euo pipefail

# Minimal KOVAR 0.8.3 runner for the S. aureus PAN-GWES dataset.
FASTA_FILE="${FASTA_FILE:-saureus_renamed.fasta}"
PAIR_FILE="${PAIR_FILE:-saureus.mi_filtered.ud_sgg_0_based}"
TREE_FILE="${TREE_FILE:-./panaroo_out/core_gene_alignment_filtered.aln.treefile}"
OUT_DIR="${OUT_DIR:-kovar_v083_saureus_results}"
THREADS="${THREADS:-32}"

for file in "$FASTA_FILE" "$PAIR_FILE" "$TREE_FILE"; do
  [[ -s "$file" ]] || { echo "Missing input: $file" >&2; exit 1; }
done
command -v ko-variation >/dev/null 2>&1 || { echo "ko-variation is not installed" >&2; exit 1; }
[[ "$(ko-variation --version)" == "KO-Variation 0.8.3" ]] || {
  echo "This runner requires KO-Variation 0.8.3" >&2; exit 1;
}

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

cmd=(ko-variation --fasta "$FASTA_FILE" --pairs "$PAIR_FILE" --tree "$TREE_FILE" \
  --out "$OUT_DIR" --threads "$THREADS")
if [[ -f "$OUT_DIR/.kovar_checkpoint/manifest.json" ]]; then
  cmd+=(--resume)
fi

printf ' %q' "${cmd[@]}"; echo
"${cmd[@]}"
echo "[done] $OUT_DIR/ko_variation.tsv"
