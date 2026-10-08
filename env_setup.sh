#!/usr/bin/env bash
# env_setup.sh — thin convenience wrapper around install_rm_env.sh.
#
# Kept for backwards compatibility with the old README, which pointed
# users at this file. The actual heavy lifting lives in
# ``install_rm_env.sh``; this script just delegates so both entry points
# stay in sync.
#
# Usage (identical to install_rm_env.sh):
#   bash env_setup.sh                        # install into current python
#   bash env_setup.sh --conda rewardverse    # create+use a fresh conda env
#   bash env_setup.sh --venv .venv           # create+use a fresh venv
#   bash env_setup.sh --no-flash             # skip flash-attn (default: install)
#   bash env_setup.sh --extras vllm          # also install vllm stack
#
# flash-attn is installed by default because the main training entrypoint
# ``scripts/train_8gpu_zero3_joint.sh`` requires it
# (``--attn_implementation flash_attention_2``). Pass ``--no-flash`` only
# if you'll only run single-card SDPA paths.
#
# All flags are forwarded verbatim to install_rm_env.sh.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec bash "$ROOT/install_rm_env.sh" "$@"
