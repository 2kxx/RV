#!/usr/bin/env bash
# run_vgrb_overall_dp.sh — 8-GPU data-parallel eval of a trained student RM
# on the VideoGen-RewardBench FULL eval split.
#
# Historically Overall-only; now takes AXIS={VQ,MQ,TA,Overall}. Default
# AXIS=Overall preserves the previous invocation. The dataset path,
# rank output dir, and merged JSON name all follow AXIS.
#
# Pipeline:
#   0. (idempotent) ensure FULL VGRB pairwise JSON + <AXIS>.json view exist.
#   1. fan out N processes (one per GPU) — each runs eval_on_vgrb_dp.py with
#      --rank R --world_size N. Each shard writes runs/.../<AXIS>_dp/rank_RR.json.
#   2. merge_vgrb_dp.py glues the shards into a single <AXIS>.json with the
#      same shape eval_on_vgrb.py would have produced single-process.
#
# Required:
#   STUDENT_CKPT=/abs/path/to/checkpoint
#
# Optional knobs (env vars):
#   AXIS=Overall                   # one of VQ | MQ | TA | Overall
#   DIMENSION=                     # override RewardVerse dim (else auto by axis)
#   WORLD_SIZE=8                   # default: 8
#   GPU_IDS="0,1,2,3,4,5,6,7"      # default: 0..WORLD_SIZE-1
#   VGRB_DATASET=/abs/.../<AXIS>.json    # default: pairwise_view_full/<AXIS>.json under EXT_ROOT
#   EXT_ROOT=...                   # auto-resolves /efs vs /group like run_external_vgrb.sh
#   OUT_ROOT=runs/external_benchmarks/videogen_rewardbench/student_rm_dp
#   TAU=0.0
#   VGRB_LIMIT=                    # row cap (smoke test); empty = full
#
# Usage:
#   # legacy Overall run
#   STUDENT_CKPT=/group/.../checkpoint-final \
#     bash scripts/eval/run_vgrb_overall_dp.sh
#
#   # TA-axis run on the same ckpt
#   AXIS=TA STUDENT_CKPT=ckpts/student_grpo_8gpu_zero3_joint/checkpoint-45 \
#     bash scripts/eval/run_vgrb_overall_dp.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# ---- Config -----------------------------------------------------------
: "${STUDENT_CKPT:?STUDENT_CKPT must be set (path to student checkpoint)}"
AXIS="${AXIS:-Overall}"
case "$AXIS" in
  VQ|MQ|TA|Overall) : ;;
  *) echo "[run-dp][fatal] AXIS must be one of VQ|MQ|TA|Overall (got '$AXIS')" >&2; exit 2 ;;
esac
DIMENSION="${DIMENSION:-}"
WORLD_SIZE="${WORLD_SIZE:-8}"
TAU="${TAU:-0.0}"
VGRB_LIMIT="${VGRB_LIMIT:-}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"   # sdpa | flash_attention_2 | eager
OUT_ROOT="${OUT_ROOT:-runs/external_benchmarks/videogen_rewardbench/student_rm_dp}"

# Default GPU IDs = 0..WORLD_SIZE-1
if [[ -z "${GPU_IDS:-}" ]]; then
  GPU_IDS="$(seq -s, 0 $((WORLD_SIZE - 1)))"
fi
IFS=',' read -ra GPU_LIST <<< "$GPU_IDS"
if [[ "${#GPU_LIST[@]}" -ne "$WORLD_SIZE" ]]; then
  echo "[run-dp] GPU_IDS count (${#GPU_LIST[@]}) != WORLD_SIZE ($WORLD_SIZE)" >&2
  exit 2
fi

# Same EXT_ROOT auto-resolution as run_external_vgrb.sh, kept independent
# so this script can be used in isolation.
if [[ -z "${EXT_ROOT:-}" ]]; then
  for cand in \
      /efs/jacuzzitang/datasets/_v_arr_external_subsets/videogen_rewardbench \
      /group/40063/jacuzzitang/datasets/_v_arr_external_subsets/videogen_rewardbench \
      ; do
    if [[ -d "$cand" ]]; then
      EXT_ROOT="$cand"
      break
    fi
  done
fi
EXT_ROOT="${EXT_ROOT:-/group/40063/jacuzzitang/datasets/_v_arr_external_subsets/videogen_rewardbench}"
PAIRWISE_VIEW_DIR="${PAIRWISE_VIEW_DIR:-${EXT_ROOT}/pairwise_view_full}"

# ---- Step 0. Ensure FULL VGRB JSON + <AXIS>.json exist ----------------
if [[ -z "${VGRB_DATASET:-}" ]]; then
  if [[ ! -s "$PAIRWISE_VIEW_DIR/$AXIS.json" ]]; then
    # Find or build the full pairwise JSON under EXT_ROOT.
    cand="$(ls -1 "$EXT_ROOT"/vgrb_pairwise_*.json 2>/dev/null \
            | awk '{print length($0), $0}' | sort -n | tail -1 | cut -d' ' -f2-)"
    if [[ -z "$cand" || ! -s "$cand" ]]; then
      echo "[run-dp] no VGRB pairwise JSON cached; fetching FULL eval split ..."
      python scripts/v_arr_mvp/fetch_external_benchmarks.py videogen_rewardbench \
          --out_dir "$EXT_ROOT"
      cand="$(ls -1 "$EXT_ROOT"/vgrb_pairwise_*.json 2>/dev/null \
              | awk '{print length($0), $0}' | sort -n | tail -1 | cut -d' ' -f2-)"
    fi
    echo "[run-dp] building 4-axis pairwise view from: $cand"
    python scripts/v_arr_mvp/prep_vgrb_pairwise_view.py \
        --input "$cand" \
        --out_dir "$PAIRWISE_VIEW_DIR"
  fi
  VGRB_DATASET="$PAIRWISE_VIEW_DIR/$AXIS.json"
fi

if [[ ! -s "$VGRB_DATASET" ]]; then
  echo "[run-dp][fatal] VGRB_DATASET not found: $VGRB_DATASET" >&2
  exit 2
fi
n_pairs="$(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" "$VGRB_DATASET")"
echo "[run-dp] axis:       $AXIS"
echo "[run-dp] dataset:    $VGRB_DATASET  (n=$n_pairs)"
echo "[run-dp] ckpt:       $STUDENT_CKPT"
echo "[run-dp] world_size: $WORLD_SIZE  gpus=$GPU_IDS  tau=$TAU"

RANK_DIR="$OUT_ROOT/${AXIS}_dp"
mkdir -p "$RANK_DIR"
LOG_DIR="$OUT_ROOT/_logs"
mkdir -p "$LOG_DIR"

LIMIT_FLAG=()
if [[ -n "$VGRB_LIMIT" ]]; then
  LIMIT_FLAG=(--limit "$VGRB_LIMIT")
fi
DIM_FLAG=()
if [[ -n "$DIMENSION" ]]; then
  DIM_FLAG=(--dimension "$DIMENSION")
fi

# ---- Step 1. Fan out per-rank processes -------------------------------
echo ""
echo "================ launching $WORLD_SIZE shards ================"
PIDS=()
for r in $(seq 0 $((WORLD_SIZE - 1))); do
  gpu="${GPU_LIST[$r]}"
  log="$LOG_DIR/${AXIS}_rank_$(printf '%02d' "$r").log"
  echo "[run-dp] rank=$r gpu=$gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" \
    python rm_training/stage3_deploy/eval_on_vgrb_dp.py \
      --rank "$r" \
      --world_size "$WORLD_SIZE" \
      --student_ckpt "$STUDENT_CKPT" \
      --vgrb_dataset "$VGRB_DATASET" \
      --output_dir "$RANK_DIR" \
      --tau "$TAU" \
      --attn_implementation "$ATTN_IMPL" \
      "${DIM_FLAG[@]}" \
      "${LIMIT_FLAG[@]}" \
      > "$log" 2>&1 &
  PIDS+=("$!")
done

# Wait for all ranks; if any fails, kill the rest and exit non-zero.
fail=0
for i in "${!PIDS[@]}"; do
  pid="${PIDS[$i]}"
  if ! wait "$pid"; then
    echo "[run-dp][fatal] rank=$i (pid=$pid) failed; see $LOG_DIR/${AXIS}_rank_$(printf '%02d' "$i").log" >&2
    fail=1
  fi
done
if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

# ---- Step 2. Merge ----------------------------------------------------
echo ""
echo "================ merging $WORLD_SIZE shards ================"
MERGED_OUT="$OUT_ROOT/${AXIS}.json"
python rm_training/stage3_deploy/merge_vgrb_dp.py \
    --rank_dir "$RANK_DIR" \
    --output   "$MERGED_OUT"

echo ""
echo "[done] merged summary: $MERGED_OUT"
echo "       per-rank shards: $RANK_DIR/rank_*.json"
echo "       per-rank logs:   $LOG_DIR/${AXIS}_rank_*.log"
