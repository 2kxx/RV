#!/usr/bin/env bash
# Arm D (main) — GRPO with the fully RGPO-trained RewardVerse scorer.
#
# BEFORE RUNNING: export RM_CKPT to point at your trained RGPO LoRA
# adapter directory, e.g.
#     export RM_CKPT=/path/to/ckpts/student_grpo_.../checkpoint-final
#
# If RM_CKPT is unset, the RewardVerse adapter transparently falls
# back to V3-zeroshot so smoke-tests can still run.
set -euo pipefail
export ARM=arm_D_rewardverse
export RM_TYPE=rewardverse
bash "$(dirname "$0")/_run_arm.sh"
