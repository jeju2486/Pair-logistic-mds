#!/usr/bin/env bash
set -euo pipefail

# Prepare one existing SLiM/SpydrPick case and run KOVAR 0.8.3.
SCRIPT_DIR="$(cd -- "${BASH_SOURCE[0]%/*}" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
OUTROOT="runs"; REF_ID=""; MU="2e-7"; RHO=""; MODE=""; THREADS=8; FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outroot) OUTROOT="$2"; shift 2 ;;
    --ref-id) REF_ID="$2"; shift 2 ;;
    --mu) MU="$2"; shift 2 ;;
    --rho|--rho-cross) RHO="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$REF_ID" && -n "$RHO" && -n "$MODE" ]] || { echo "Need --ref-id, --rho, and --mode" >&2; exit 2; }
[[ "$(ko-variation --version)" == "KO-Variation 0.8.3" ]] || { echo "KOVAR 0.8.3 is required" >&2; exit 1; }

CASE_ID="ref${REF_ID}__mu_${MU}__rho_${RHO}__mode_${MODE}"
RUN_DIR="$REPO_ROOT/$OUTROOT/ref${REF_ID}/$CASE_ID"
RAW_FASTA="$RUN_DIR/eco_simul_fasta/eco.fa"
REF_FASTA="$RUN_DIR/slim/reference.fa"
TREE="$RUN_DIR/slim/summary_tree.nwk"
SPYDR_DIR="$RUN_DIR/spydrpick_snp"
WORK_DIR="$RUN_DIR/kovar_spydrpick"
FASTA="$WORK_DIR/eco.fake_ac.fa"
PAIRS="$WORK_DIR/spydrpick_pairs.0based.tsv"
RESULTS="$WORK_DIR/results"
mkdir -p "$WORK_DIR"

for item in "$RAW_FASTA" "$REF_FASTA" "$TREE"; do [[ -s "$item" ]] || { echo "Missing input: $item" >&2; exit 1; }; done
shopt -s nullglob; edge_files=("$SPYDR_DIR"/eco.*spydrpick_couplings*edges); shopt -u nullglob
[[ "${#edge_files[@]}" -eq 1 ]] || { echo "Expected one SpydrPick edge file; found ${#edge_files[@]}" >&2; exit 1; }

if [[ "$FORCE" -eq 1 || ! -s "$FASTA" || ! -s "$PAIRS" ]]; then
  python3 - "$REF_FASTA" "$RAW_FASTA" "$FASTA" "${edge_files[0]}" "$PAIRS" <<'PY'
import sys
from pathlib import Path

def fasta(path):
    records=[]; name=None; seq=[]
    for raw in Path(path).read_text().splitlines():
        line=raw.strip()
        if not line: continue
        if line.startswith('>'):
            if name is not None: records.append((name,''.join(seq).upper()))
            name=line[1:].split()[0]; seq=[]
        else: seq.append(line)
    if name is not None: records.append((name,''.join(seq).upper()))
    return records

reference=fasta(sys.argv[1]); samples=fasta(sys.argv[2])
if len(reference) != 1: raise SystemExit('Reference FASTA must contain one record')
ref=reference[0][1]
with Path(sys.argv[3]).open('w') as out:
    for name,seq in samples:
        if len(seq) != len(ref): raise SystemExit(f'Length mismatch: {name}')
        binary=''.join('A' if a == b else 'C' for a,b in zip(seq,ref))
        out.write(f'>{name}\n{binary}\n')

best={}
for raw in Path(sys.argv[4]).read_text().splitlines():
    fields=raw.split()
    if len(fields) < 5: continue
    try: u,v=int(fields[0])-1,int(fields[1])-1; mi=float(fields[4])
    except ValueError: continue
    if u < 0 or v < 0 or u == v: continue
    u,v=sorted((u,v)); row=(u,v,*fields[2:5])
    if (u,v) not in best or mi > best[(u,v)][0]: best[(u,v)]=(mi,row)
with Path(sys.argv[5]).open('w') as out:
    out.write('u\tv\tdistance\tARACNE\tMI\n')
    for _,row in sorted(best.values(), key=lambda item: (-item[0], item[1][0], item[1][1])):
        out.write('\t'.join(map(str,row))+'\n')
PY
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}" NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
cmd=(ko-variation --fasta "$FASTA" --pairs "$PAIRS" --tree "$TREE" --out "$RESULTS" --threads "$THREADS")
if [[ -f "$RESULTS/.kovar_checkpoint/manifest.json" && "$FORCE" -eq 0 ]]; then cmd+=(--resume); else cmd+=(--overwrite); fi
printf ' %q' "${cmd[@]}"; echo
"${cmd[@]}"
echo "[done] $RESULTS/ko_variation.tsv"
