#!/usr/bin/env bash
set -euo pipefail

# Run only simulation mode=0 across the requested rho values.
SCRIPT_DIR="$(cd -- "${BASH_SOURCE[0]%/*}" && pwd)"
RUNNER="${RUNNER:-$SCRIPT_DIR/run_kovar_v083_simulation.sh}"
OUTROOT="runs"; REF_ID=""; MU="2e-7"; THREADS=16; RHOS="4e-8"; FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --outroot) OUTROOT="$2"; shift 2 ;;
    --ref-id) REF_ID="$2"; shift 2 ;;
    --mu) MU="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --rhos) RHOS="$2"; shift 2 ;;
    --runner) RUNNER="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$REF_ID" ]] || { echo "Need --ref-id" >&2; exit 2; }
[[ -f "$RUNNER" ]] || { echo "Missing runner: $RUNNER" >&2; exit 1; }

for rho in $RHOS; do
  cmd=(bash "$RUNNER" --outroot "$OUTROOT" --ref-id "$REF_ID" --mu "$MU" \
    --rho "$rho" --mode 0 --threads "$THREADS")
  [[ "$FORCE" -eq 1 ]] && cmd+=(--force)
  printf ' %q' "${cmd[@]}"; echo
  "${cmd[@]}"
done
