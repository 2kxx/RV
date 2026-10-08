#!/usr/bin/env bash
# Arm E (optional, appendix) — GRPO with VideoAlign/VideoReward as an
# external oracle-ish RM. Requires KwaiVGI/VideoReward on HF.
set -euo pipefail
export ARM=arm_E_videoalign
export RM_TYPE=videoalign
bash "$(dirname "$0")/_run_arm.sh"
