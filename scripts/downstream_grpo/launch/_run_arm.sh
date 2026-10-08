#!/usr/bin/env bash
# Launch a single downstream GRPO arm on 8xH100:
#   * Rank 7 GPU hosts the reward server for the selected RM type
#   * Ranks 0-6 run the DeepSpeed ZeRO-2 diffusion trainer
#
# Usage:
#   ARM=arm_D_rewardverse RM_TYPE=rewardverse bash launch/_run_arm.sh
#
# Env overrides:
#   MAX_STEPS, EVAL_EVERY, SAVE_EVERY, LOG_EVERY, RM_CKPT (for rewardverse),
#   WANDB_PROJECT, CUDA_VISIBLE_DEVICES

set -euo pipefail

: "${ARM:?ARM must be set, e.g. arm_D_rewardverse}"
: "${RM_TYPE:?RM_TYPE must be set (rewardverse|v3_zeroshot|videoscore_v2|videoalign)}"
: "${SOCKET:=/tmp/downstream_${RM_TYPE}.sock}"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
# Reduces allocator fragmentation for the alternating rollout/VAE-decode/PPO
# footprint — recommended by the OOM error message itself.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[launch] arm=${ARM} rm_type=${RM_TYPE} socket=${SOCKET}"
mkdir -p logs/downstream

# --- 1) Reward server on GPU 7 ---
REWARD_LOG="logs/downstream/${ARM}_rewardserver.log"
echo "[launch] starting reward server (GPU 7) -> ${REWARD_LOG}"
CUDA_VISIBLE_DEVICES=7 nohup python -m scripts.downstream_grpo.src.reward_server \
    --rm_type "$RM_TYPE" \
    --socket  "$SOCKET" \
    > "$REWARD_LOG" 2>&1 &
RWSRV_PID=$!
echo "[launch] reward_server PID=${RWSRV_PID}; waiting for socket..."

# wait for socket up to 5 minutes
for i in $(seq 1 300); do
    if [ -S "$SOCKET" ]; then
        echo "[launch] reward server ready"
        break
    fi
    sleep 1
    if [ "$i" = "300" ]; then
        echo "[launch] reward server didn't come up; log:"
        tail -60 "$REWARD_LOG" || true
        kill "$RWSRV_PID" 2>/dev/null || true
        exit 1
    fi
done

# --- 2) DeepSpeed trainer on GPUs 0-6 ---
NPROC=${NPROC:-7}
MASTER_PORT=${MASTER_PORT:-29517}
MAX_STEPS=${MAX_STEPS:-500}
EVAL_EVERY=${EVAL_EVERY:-100}
SAVE_EVERY=${SAVE_EVERY:-100}
LOG_EVERY=${LOG_EVERY:-10}
WANDB_PROJECT=${WANDB_PROJECT:-}

# Bounce trainer's CUDA_VISIBLE_DEVICES to just the 7 diffusion GPUs
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_TRAIN:-0,1,2,3,4,5,6}

# hand off KV
if [ "$RM_TYPE" = "rewardverse" ] && [ -n "${RM_CKPT:-}" ]; then
    export RM_CKPT
fi

set +e
torchrun \
    --nnodes=1 \
    --nproc_per_node=${NPROC} \
    --master_port=${MASTER_PORT} \
    -m scripts.downstream_grpo.src.train_grpo_wan \
    --arm "$ARM" \
    --config scripts/downstream_grpo/configs/wan_grpo_default.yaml \
    --reward_socket "$SOCKET" \
    --max_steps "$MAX_STEPS" \
    --eval_every "$EVAL_EVERY" \
    --save_every "$SAVE_EVERY" \
    --log_every "$LOG_EVERY" \
    --wandb_project "$WANDB_PROJECT"
TRAINER_RC=$?
set -e

echo "[launch] trainer exit code = ${TRAINER_RC}"

# --- 3) shut down reward server ---
echo "[launch] shutting down reward server"
python -c "from scripts.downstream_grpo.src.reward_client import RewardClient; RewardClient('${SOCKET}').shutdown()" || true
wait "$RWSRV_PID" 2>/dev/null || true
rm -f "$SOCKET"

exit "$TRAINER_RC"
