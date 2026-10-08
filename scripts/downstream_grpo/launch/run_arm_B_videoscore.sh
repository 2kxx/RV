#!/usr/bin/env bash
# Arm B — GRPO with the unconstrained pointwise scorer VideoScore-v2.
# This is the paper's designated reward-hacking demonstrator.
set -euo pipefail
export ARM=arm_B_videoscore
export RM_TYPE=videoscore_v2
bash "$(dirname "$0")/_run_arm.sh"
