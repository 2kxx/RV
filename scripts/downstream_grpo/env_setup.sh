#!/usr/bin/env bash
# Env setup for the downstream GRPO experiment. Idempotent.
# On the 8xH100 machine, run this in the conda env used for the paper.
#
# Optional: --dry-run only prints the pip commands.

set -euo pipefail

DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; fi

pkgs=(
    "diffusers>=0.33.0"       # Wan-2.1 classes (AutoencoderKLWan, WanPipeline, WanTransformer3DModel) landed in 0.33
    "transformers>=4.49.0"    # UMT5EncoderModel + Wan tokenizer paths
    "accelerate>=0.34.0"
    "deepspeed>=0.15.0"
    "peft>=0.17.0"            # required by diffusers>=0.33
    "torch>=2.4.0"
    "torchvision"
    "einops"
    "opencv-python-headless"
    "openai>=1.40.0"
    "tqdm"
    "pyyaml"
    "wandb"
    "matplotlib"
    "numpy"
)
optional_vbench=(
    "vbench"  # official VBench; falls back to CLIP proxy if not present
)
optional_videoscore=(
    "mantis-vl"  # for VideoScore-v1.1 head; only needed for arm B
)

echo "[env_setup] core deps"
if [ "$DRY" = "0" ]; then
    pip install -U --no-cache-dir "${pkgs[@]}"
else
    echo pip install -U "${pkgs[@]}"
fi

echo "[env_setup] optional: VBench"
if [ "$DRY" = "0" ]; then
    pip install --no-cache-dir "${optional_vbench[@]}" || echo "[env_setup] vbench install failed — plot_vbench will use CLIP fallback"
else
    echo pip install "${optional_vbench[@]}"
fi

echo "[env_setup] optional: mantis-vl for VideoScore adapter"
if [ "$DRY" = "0" ]; then
    pip install --no-cache-dir "${optional_videoscore[@]}" || echo "[env_setup] mantis-vl install failed — arm B will refuse to launch until installed"
else
    echo pip install "${optional_videoscore[@]}"
fi

echo "[env_setup] validating imports"
if [ "$DRY" = "0" ]; then
    python - <<'PY'
import diffusers, transformers, peft, deepspeed, torch, openai
print(f"diffusers  = {diffusers.__version__}")
print(f"transformers = {transformers.__version__}")
print(f"peft       = {peft.__version__}")
print(f"deepspeed  = {deepspeed.__version__}")
print(f"torch      = {torch.__version__}, CUDA={torch.cuda.is_available()}")
print(f"openai     = {openai.__version__}")
PY
fi

echo "[env_setup] done"
