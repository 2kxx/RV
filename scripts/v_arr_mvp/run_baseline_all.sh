#!/usr/bin/env bash
# scripts/v_arr_mvp/run_baseline_all.sh
# ----------------------------------------------------------------------------
# Unified baseline launcher — runs two phases serially:
#
#   Phase 1: VideoScore-v1.1 — full re-run all 12 dims (SKIP_EXISTING=0).
#            Dims with v2 test sets (chromaticity/consistency/logic/pacing)
#            now use the wider-label v2 files; others use v1 (unchanged).
#
#   Phase 2: VideoScore2 — full 12-dim sweep (no prior results exist).
#
# UnifiedReward-7B is NOT run here because the model checkpoint is not
# present on disk (set RM_UNIFIEDREWARD_PATH to a populated dir if you
# obtain the weights; legacy hard-coded path was empty on the original
# /group/40063 host).
#
# Required env:
#   ARR_VIDEOSCORE_DEVICE    default: cuda:0
#   ARR_VIDEOSCORE2_DEVICE   default: cuda:0
#   ARR_VIDEOSCORE_DTYPE     default: bf16  (matches official VideoScore-v1.1 README torch_dtype=bfloat16)
#   ARR_VIDEOSCORE2_DTYPE    default: bf16  (matches official weights torch_dtype=bfloat16)
#   ARR_NUM_FRAMES           default: 48
#
# Usage (background):
#   nohup bash scripts/v_arr_mvp/run_baseline_all.sh \
#     > runs/baselines/_runall_baseline.out 2>&1 & disown
#
#   tail -f runs/baselines/_runall_baseline.out
# ----------------------------------------------------------------------------
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

DEVICE="${ARR_VIDEOSCORE_DEVICE:-cuda:0}"
DEVICE2="${ARR_VIDEOSCORE2_DEVICE:-cuda:0}"
DTYPE="${ARR_VIDEOSCORE_DTYPE:-bf16}"
DTYPE2="${ARR_VIDEOSCORE2_DTYPE:-bf16}"
NUM_FRAMES="${ARR_NUM_FRAMES:-48}"

mkdir -p runs/baselines

T_TOTAL_START="$(date +%s)"

echo "============================================================"
echo "[baseline-all] start=$(date '+%Y-%m-%d %H:%M:%S')"
echo "[baseline-all] device=${DEVICE}  dtype=${DTYPE}  frames=${NUM_FRAMES}"
echo "============================================================"

# ============================================================
# Phase 1: VideoScore-v1.1 — re-run logic/pacing/consistency
#          with v2 test set (SKIP_EXISTING=0 to force overwrite)
# ============================================================
echo ""
echo "============================================================"
echo "[Phase 1/2] VideoScore-v1.1: full re-run all 12 dims (v2 test sets applied)"
echo "============================================================"

SKIP_EXISTING=0 \
ARR_VIDEOSCORE_DEVICE="${DEVICE}" \
ARR_VIDEOSCORE_DTYPE="${DTYPE}" \
ARR_NUM_FRAMES="${NUM_FRAMES}" \
ARR_OUT_DIR="runs/baselines/videoscore" \
  bash scripts/v_arr_mvp/run_videoscore_all_dims.sh

echo ""
echo "[Phase 1/2] VideoScore-v1.1 re-run done."

# ============================================================
# Phase 2: VideoScore2 — full 12-dim sweep
# ============================================================
echo ""
echo "============================================================"
echo "[Phase 2/2] VideoScore2: full 12-dim sweep"
echo "============================================================"

SKIP_EXISTING=1 \
ARR_VIDEOSCORE2_DEVICE="${DEVICE2}" \
ARR_VIDEOSCORE2_DTYPE="${DTYPE2}" \
ARR_NUM_FRAMES="${NUM_FRAMES}" \
ARR_OUT_DIR="runs/baselines/videoscore2" \
  bash scripts/v_arr_mvp/run_videoscore2_all_dims.sh

echo ""
echo "[Phase 2/2] VideoScore2 done."

# ============================================================
# Summary
# ============================================================
T_TOTAL_END="$(date +%s)"
ELAPSED=$((T_TOTAL_END - T_TOTAL_START))
HH=$((ELAPSED / 3600)); MM=$(((ELAPSED % 3600) / 60)); SS=$((ELAPSED % 60))

echo ""
echo "============================================================"
printf "[baseline-all] DONE  elapsed: %02d:%02d:%02d (%ds)\n" "${HH}" "${MM}" "${SS}" "${ELAPSED}"
echo "============================================================"
echo ""
echo "=== VideoScore-v1.1 summary ==="
column -t -s $'\t' runs/baselines/videoscore/_runall_summary.tsv 2>/dev/null || cat runs/baselines/videoscore/_runall_summary.tsv 2>/dev/null || echo "(not found)"
echo ""
echo "=== VideoScore2 summary ==="
column -t -s $'\t' runs/baselines/videoscore2/_runall_summary.tsv 2>/dev/null || cat runs/baselines/videoscore2/_runall_summary.tsv 2>/dev/null || echo "(not found)"
