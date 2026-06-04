#!/usr/bin/env bash
# run_external_vgrb.sh — run RewardVerse's BEST method (listwise + per-rubric
# rubric judge → aggregation) on the VideoGen-RewardBench FULL eval split
# (~25.2k pairs across 4 axes).
#
# This script intentionally drops:
#   - VideoScore-v1.1 / VideoScore-2 baselines on VGRB (no pairwise wrapper
#     yet, see STATUS_SYNC.md §2.3 U2);
#   - the legacy pointwise→tau-sweep path (`pointwise_rubric_judge.py` +
#     `vgrb_headtohead.py`); the head-to-head ablation under
#     `runs/external_benchmarks/videogen_rewardbench/_vq_headtohead/` is
#     considered done.
#
# Pipeline:
#   1. fetch_external_benchmarks.py videogen_rewardbench (full eval split)
#   2. prep_vgrb_pairwise_view.py        (split into 4 axis JSONs)
#   3. pairwise_rubric_judge.py × 4 axes (listwise+per_rubric LLM scoring)
#   4. aggregation_listwise_eval.py × 4 axes (mean / weighted_mean / lp4
#                                              + listwise_direct baseline)
#
# ⚠️ Cost warning (~25.2k pairs × 4 axes ≈ 100k LLM calls).
#    Use VGRB_LIMIT=N for smoke tests before unleashing the full run.
#
# Pre-conditions:
#   * OPENAI_API_KEY / VENUS_SECRET_ID / VENUS_SECRET_KEY exported.
#   * The HF snapshot under
#     /group/40063/jacuzzitang/datasets/VideoGen-RewardBench/ is present
#     (videogen-rewardbench.csv + videos.zip), OR the fetcher is allowed
#     to download it.
#
# Usage:
#   # Smoke (200 pairs)
#   VGRB_LIMIT=200 bash scripts/v_arr_mvp/run_external_vgrb.sh
#
#   # Full eval split (~25.2k pairs, expensive)
#   bash scripts/v_arr_mvp/run_external_vgrb.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ---- Inputs ------------------------------------------------------------
EXT_ROOT="${EXT_ROOT:-/group/40063/jacuzzitang/datasets/_v_arr_external_subsets/videogen_rewardbench}"
VGRB_SUBSET="${VGRB_SUBSET:-}"   # if empty: auto-resolve below
PAIRWISE_VIEW_DIR="${PAIRWISE_VIEW_DIR:-${EXT_ROOT}/pairwise_view_full}"
OUT_ROOT="${OUT_ROOT:-runs/external_benchmarks/videogen_rewardbench}"

# ---- Knobs -------------------------------------------------------------
VGRB_LIMIT="${VGRB_LIMIT:-}"             # smoke-test cap on rows per axis
RUBRIC_CONCURRENCY="${RUBRIC_CONCURRENCY:-4}"
TAU_GRID="${TAU_GRID:-0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5}"

# VGRB axis -> RewardVerse rubric axis used for the listwise prompt.
# Same mapping as the 200-pair experiment; Overall reuses VQ rubric as a
# cheap proxy (can be extended to a 3-rubric vote later).
declare -A VGRB_RUBRIC_AXIS
VGRB_RUBRIC_AXIS[VQ]="visual_quality"
VGRB_RUBRIC_AXIS[MQ]="action"
VGRB_RUBRIC_AXIS[TA]="character"
VGRB_RUBRIC_AXIS[Overall]="visual_quality"

LIMIT_FLAG=()
if [ -n "$VGRB_LIMIT" ]; then
    LIMIT_FLAG=(--limit "$VGRB_LIMIT")
fi

should_skip() { [ -s "$1" ]; }

# =====================================================================
# Step 1. Build / locate the FULL VGRB pairwise JSON.
# =====================================================================
if [ -z "$VGRB_SUBSET" ]; then
    # Prefer the largest existing vgrb_pairwise_*.json in EXT_ROOT.
    cand="$(ls -1 "$EXT_ROOT"/vgrb_pairwise_*.json 2>/dev/null \
            | awk '{print length($0), $0}' | sort -n | tail -1 | cut -d' ' -f2-)"
    if [ -n "$cand" ] && [ -s "$cand" ]; then
        VGRB_SUBSET="$cand"
    fi
fi
if [ -z "$VGRB_SUBSET" ] || [ ! -s "$VGRB_SUBSET" ]; then
    echo "[run_vgrb] no VGRB JSON cached; fetching FULL eval split ..."
    python scripts/v_arr_mvp/fetch_external_benchmarks.py videogen_rewardbench \
        --out_dir "$EXT_ROOT"
    cand="$(ls -1 "$EXT_ROOT"/vgrb_pairwise_*.json 2>/dev/null \
            | awk '{print length($0), $0}' | sort -n | tail -1 | cut -d' ' -f2-)"
    VGRB_SUBSET="$cand"
fi
echo "[run_vgrb] using VGRB JSON: $VGRB_SUBSET"
n_pairs="$(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$VGRB_SUBSET")"
echo "[run_vgrb] n_pairs = $n_pairs"

# =====================================================================
# Step 2. Build the 4 per-axis listwise views (idempotent).
# =====================================================================
if [ ! -f "$PAIRWISE_VIEW_DIR/_axis_index.json" ]; then
    echo "[run_vgrb] building pairwise view: $PAIRWISE_VIEW_DIR"
    python scripts/v_arr_mvp/prep_vgrb_pairwise_view.py \
        --input "$VGRB_SUBSET" \
        --out_dir "$PAIRWISE_VIEW_DIR"
else
    echo "[run_vgrb] reusing pairwise view at $PAIRWISE_VIEW_DIR"
fi

mkdir -p "$OUT_ROOT/rewardverse_pw/pairwise_listwise_perrubric"

# =====================================================================
# Step 3. listwise + per_rubric LLM scoring (4 axes).
# =====================================================================
echo ""
echo "================ listwise + per_rubric LLM scoring ================"
for vgrb_ax in VQ MQ TA Overall; do
    rv_ax="${VGRB_RUBRIC_AXIS[$vgrb_ax]}"
    in_json="$PAIRWISE_VIEW_DIR/${vgrb_ax}.json"
    out_json="$OUT_ROOT/rewardverse_pw/pairwise_listwise_perrubric/${vgrb_ax}.json"
    if should_skip "$out_json"; then
        echo "[skip] listwise/${vgrb_ax} (already exists)"
        continue
    fi
    echo "[run] listwise+per_rubric ${vgrb_ax} (rubric=${rv_ax}) -> $out_json"
    python scripts/v_arr_mvp/pairwise_rubric_judge.py \
        --dataset "$in_json" \
        --output  "$out_json" \
        --axis "$vgrb_ax" \
        --dimension "$rv_ax" \
        --concurrency "$RUBRIC_CONCURRENCY" \
        "${LIMIT_FLAG[@]}"
done

# =====================================================================
# Step 4. Aggregation eval (mean / weighted_mean / lp4 + listwise_direct).
# =====================================================================
echo ""
echo "================ aggregation eval ================"
for vgrb_ax in VQ MQ TA Overall; do
    listwise_pred="$OUT_ROOT/rewardverse_pw/pairwise_listwise_perrubric/${vgrb_ax}.json"
    out_json="$OUT_ROOT/rewardverse_pw/pairwise_listwise_perrubric/aggregation_${vgrb_ax}.json"
    if [ ! -s "$listwise_pred" ]; then
        echo "[warn] missing $listwise_pred; skipping aggregation for ${vgrb_ax}"
        continue
    fi
    echo "[agg] ${vgrb_ax}"
    python scripts/v_arr_mvp/aggregation_listwise_eval.py \
        --listwise_pred "$listwise_pred" \
        --axis "$vgrb_ax" \
        --tau_grid "$TAU_GRID" \
        --output "$out_json"
done

echo ""
echo "[done] VGRB full-split eval complete."
echo "       listwise outputs: $OUT_ROOT/rewardverse_pw/pairwise_listwise_perrubric/{VQ,MQ,TA,Overall}.json"
echo "       aggregation:      $OUT_ROOT/rewardverse_pw/pairwise_listwise_perrubric/aggregation_{VQ,MQ,TA,Overall}.json"
