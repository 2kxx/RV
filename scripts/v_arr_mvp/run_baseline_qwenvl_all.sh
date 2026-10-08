#!/usr/bin/env bash
# Run all four Qwen2.5-VL untrained baselines (A1 / A2 / B1 / D6) on the
# 16-dim pointwise hold-out (≈1k rows) and dump per-dim PLCC/SRCC JSONs to
#   runs/baselines/qwenvl_<mode>/<dim>.json
#   runs/baselines/qwenvl_<mode>/_runall_summary.tsv
#
# Each mode takes ~1.5–2.5 h on a single H800 (model loaded once per mode).
# Modes are run sequentially so a single GPU is enough; if you have 4 GPUs
# free, kick them off in parallel by setting CUDA_VISIBLE_DEVICES per mode.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODES="${1:-A1 A2 B1 D6}"
LOG_DIR="$ROOT/runs/baselines/_logs"
mkdir -p "$LOG_DIR"

for mode in $MODES; do
  log="$LOG_DIR/qwenvl_${mode}.log"
  echo ">>> [qwenvl][$mode] running 16-dim pointwise — log $log"
  python scripts/v_arr_mvp/baseline_qwenvl_pointwise.py \
      --mode "$mode" \
      "${@:2}" \
      2>&1 | tee "$log"
done

echo ">>> all done."
