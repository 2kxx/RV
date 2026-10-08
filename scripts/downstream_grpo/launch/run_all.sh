#!/usr/bin/env bash
# Runs the 3 core arms sequentially on one 8xH100 node:
#   D (RewardVerse, main) -> B (VideoScore-v2, hacking demo) -> C (V3-zeroshot).
# Arm A (Base, no RL) contributes no training run — its videos are
# generated at eval time from the base Wan-2.1-1.3B checkpoint.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../../.."

echo "[run_all] === Arm D: RewardVerse ==="
bash "$HERE/run_arm_D_rewardverse.sh"

echo "[run_all] === Arm B: VideoScore-v2 ==="
bash "$HERE/run_arm_B_videoscore.sh"

echo "[run_all] === Arm C: V3-zeroshot ==="
bash "$HERE/run_arm_C_v3zeroshot.sh"

echo "[run_all] all arms complete."
echo "[run_all] next: bash scripts/downstream_grpo/eval/run_all_eval.sh"
