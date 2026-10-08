#!/usr/bin/env bash
# Arm C — GRPO with V3-zeroshot (rubric + soft-logits, no RGPO training).
# Isolates the "learned calibration" axis from the "rubric anchoring" axis.
set -euo pipefail
export ARM=arm_C_v3zeroshot
export RM_TYPE=v3_zeroshot
bash "$(dirname "$0")/_run_arm.sh"
