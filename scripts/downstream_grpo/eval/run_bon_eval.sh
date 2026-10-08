#!/usr/bin/env bash
# End-to-end evaluation pipeline for the reward-hacking figure.
#
# Runs generation (one arm at a time) → cross-RM scoring → plots.
#
# Common invocations:
#   # Base model only (fastest smoke test)
#   bash scripts/downstream_grpo/eval/run_bon_eval.sh base ""
#
#   # A trained LoRA arm (adapter dir must exist)
#   bash scripts/downstream_grpo/eval/run_bon_eval.sh arm_D_rewardverse \
#       ckpts/downstream/arm_D_rewardverse/checkpoint-final
#
# Env overrides:
#   N_PER_PROMPT    (default 1)
#   RMS             (default rewardverse,v3_zeroshot,videoscore_v2,videoalign)
#   SPLITS          (default scripts/downstream_grpo/configs/prompts/evalverse_eval.jsonl)
#   EVAL_ROOT       (default runs/downstream/eval)
#   SAMPLER_GPU     (default 0) — GPU for video generation
#   RM_GPU          (default 0) — GPU for reward server
set -euo pipefail

ARM_NAME="${1:?usage: run_bon_eval.sh <arm_name> <ckpt_dir_or_empty>}"
CKPT="${2:-}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

N_PER_PROMPT="${N_PER_PROMPT:-1}"
RMS="${RMS:-rewardverse,v3_zeroshot,videoscore_v2,videoalign}"
SPLITS="${SPLITS:-scripts/downstream_grpo/configs/prompts/evalverse_eval.jsonl}"
EVAL_ROOT="${EVAL_ROOT:-runs/downstream/eval}"
SAMPLER_GPU="${SAMPLER_GPU:-0}"
RM_GPU="${RM_GPU:-0}"

VIDEOS_DIR="${EVAL_ROOT}/videos"
SCORES_DIR="${EVAL_ROOT}/scores"
FIG_DIR="${EVAL_ROOT}/figures"

mkdir -p "${VIDEOS_DIR}" "${SCORES_DIR}" "${FIG_DIR}"

# --- 0) prompt splits (idempotent) ---
if [ ! -f "${SPLITS}" ]; then
    echo "[bon_eval] prompt splits missing, building..."
    python -c "from scripts.downstream_grpo.src.prompt_pool import build_and_dump_splits; \
from scripts.downstream_grpo.src.train_grpo_wan import load_yaml; \
build_and_dump_splits(load_yaml('scripts/downstream_grpo/configs/wan_grpo_default.yaml'), \
                      'scripts/downstream_grpo/configs/prompts')"
fi

# --- 1) generate videos for this arm ---
echo "[bon_eval] === generating videos: arm=${ARM_NAME} ckpt=${CKPT:-'(base)'} ==="
CUDA_VISIBLE_DEVICES="${SAMPLER_GPU}" python -m scripts.downstream_grpo.eval.generate_eval_videos \
    --arm_name "${ARM_NAME}" \
    ${CKPT:+--ckpt "${CKPT}"} \
    --splits "${SPLITS}" \
    --out "${VIDEOS_DIR}" \
    --n_per_prompt "${N_PER_PROMPT}"

# --- 2) score under all RMs (sequential) ---
echo "[bon_eval] === scoring under RMs: ${RMS} ==="
python -m scripts.downstream_grpo.eval.score_matrix \
    --manifest "${VIDEOS_DIR}/manifest.jsonl" \
    --out "${SCORES_DIR}" \
    --rms "${RMS}" \
    --gpu_ordinal "${RM_GPU}"

# --- 3) plots ---
echo "[bon_eval] === plotting ==="
python -m scripts.downstream_grpo.eval.plot_cross_rm_matrix \
    --scores "${SCORES_DIR}/scores.jsonl" \
    --out_dir "${FIG_DIR}"
python -m scripts.downstream_grpo.eval.plot_per_dim_radar \
    --scores "${SCORES_DIR}/scores.jsonl" \
    --out_dir "${FIG_DIR}"

echo "[bon_eval] done. Figures at ${FIG_DIR}"
