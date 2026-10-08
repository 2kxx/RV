#!/usr/bin/env bash
# install_rm_env.sh — one-shot installer for the RewardVerse Stage ② RM
# training environment. Reproduces the verified-working env captured
# from ``miniconda3/envs/rewardverse`` on 2026-07-09.
#
# Usage
# -----
#   bash install_rm_env.sh                     # install into current python
#   bash install_rm_env.sh --conda rewardverse # create+use a fresh conda env
#   bash install_rm_env.sh --venv .venv        # create+use a fresh venv
#   bash install_rm_env.sh --no-flash          # skip flash-attn (default: install)
#   bash install_rm_env.sh --extras vllm       # also install vllm+bitsandbytes
#   bash install_rm_env.sh --no-smoke          # skip the import/dummy smoke
#
# Combine as needed, e.g.
#   bash install_rm_env.sh --conda rewardverse --extras vllm
#
# flash-attn is REQUIRED by the main training entrypoint
# ``scripts/train_8gpu_zero3_joint.sh`` (and the other zero3 scripts):
# they pass ``--attn_implementation flash_attention_2``. Loading Qwen2.5-VL
# with that flag hard-fails if the ``flash_attn`` module isn't importable.
# The installer therefore builds it by default. Only pass ``--no-flash``
# if you know you'll only run single-card SDPA paths.
#
# What this installs (verified versions)
# --------------------------------------
#   * Python 3.10 (via ``--conda`` if requested)
#   * PyTorch 2.6.0 + torchvision 0.21.0 + torchaudio 2.6.0, cu124 wheels
#     (matches the cluster's CUDA 12.4 driver)
#   * HuggingFace stack:
#       transformers 4.51.3, tokenizers 0.21.4, accelerate 1.13.0,
#       peft 0.15.2, huggingface_hub 0.36.2, sentencepiece 0.2.1,
#       safetensors 0.8.0, hf-xet 1.5.1
#   * Qwen2.5-VL multimodal helpers:
#       qwen-vl-utils 0.0.14, av 17.0.1, decord 0.6.0, pillow 10.2.0,
#       opencv-python-headless 4.10.0.84, einops 0.8.0, ftfy 6.3.0
#   * Diffusion / kernels: diffusers 0.35.0, torchdiffeq 0.2.4,
#       liger_kernel 0.4.1
#   * Data / numeric: numpy 2.2.6, pandas 2.3.3, scipy 1.15.3,
#       datasets 2.19.1, pyarrow 24.0.0, PyYAML 6.0.1, matplotlib 3.9.2
#   * Sentence embeddings: sentence-transformers 5.5.0, scikit-learn 1.7.2
#   * Trackers / clients: wandb 0.27.2, openai 2.42.0, tiktoken 0.7.0,
#       tenacity 9.1.4, loguru 0.7.3
#   * DeepSpeed 0.16.5 — REQUIRED (scripts/train_8gpu_zero3*.sh preflight-
#       checks ``import deepspeed`` and exits early if missing).
#   * flash-attn 2.7.4.post1 (default; disable with ``--no-flash``) —
#       REQUIRED for the zero3 training scripts. Built with
#       ``--no-build-isolation`` because its setup.py imports torch at
#       build time, which pip's PEP 517 sandbox hides.
#   * (--extras vllm) vllm + bitsandbytes 0.44.1 — legacy auto-rubric
#       image path (rubric_pipeline/vllm_serve.sh, eval_pointwise).
#
# What this does NOT install
# --------------------------
#   * By default: vllm / bitsandbytes  → opt in with ``--extras vllm``.
#
# Memory / time
# -------------
#   * ~3 GB downloaded (mostly torch wheels)
#   * ~10-15 min on a fresh env (flash-attn build is ~5-10 min with nvcc
#     + ninja), +5 min with ``--extras vllm``.

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────
# Args.
# ─────────────────────────────────────────────────────────────────────────
TARGET_PY=""           # path to a python binary; empty = use current
USE_CONDA=""           # name of a conda env to create+activate
USE_VENV=""            # path to a venv to create+activate
INSTALL_FLASH=1        # default ON — required by scripts/train_8gpu_zero3_joint.sh
INSTALL_EXTRAS=""      # comma-separated: e.g. "vllm"
RUN_SMOKE=1            # 0 → skip the smoke test at the end
PYTORCH_INDEX="https://download.pytorch.org/whl/cu124"
PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-}"  # honour user override

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda)     USE_CONDA="$2";  shift 2 ;;
    --venv)      USE_VENV="$2";   shift 2 ;;
    --python)    TARGET_PY="$2";  shift 2 ;;
    --flash)     INSTALL_FLASH=1; shift ;;   # kept for backwards compat; now default
    --no-flash)  INSTALL_FLASH=0; shift ;;
    --extras)    INSTALL_EXTRAS="$2"; shift 2 ;;
    --no-smoke)  RUN_SMOKE=0;     shift ;;
    --pytorch-index) PYTORCH_INDEX="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,60p' "$0"
      exit 0
      ;;
    *)
      echo "[install] unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

ROOT="$(cd "$(dirname "$0")" && pwd)"
echo "[install] repo root = $ROOT"

# ─────────────────────────────────────────────────────────────────────────
# Pick / create the target python.
# ─────────────────────────────────────────────────────────────────────────
if [[ -n "$USE_CONDA" ]]; then
  if ! command -v conda >/dev/null; then
    echo "[install] conda not on PATH — install miniconda first or omit --conda" >&2
    exit 3
  fi
  echo "[install] creating conda env: $USE_CONDA (python=3.10)"
  # ``conda create`` is idempotent enough; reuse if already exists.
  conda create -y -n "$USE_CONDA" python=3.10 2>/dev/null || true
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$USE_CONDA"
  TARGET_PY="$(which python)"
elif [[ -n "$USE_VENV" ]]; then
  if [[ ! -d "$USE_VENV" ]]; then
    echo "[install] creating venv at $USE_VENV"
    python3 -m venv "$USE_VENV"
  fi
  # shellcheck disable=SC1091
  source "$USE_VENV/bin/activate"
  TARGET_PY="$(which python)"
elif [[ -z "$TARGET_PY" ]]; then
  TARGET_PY="$(command -v python3 || command -v python)"
  if [[ -z "$TARGET_PY" ]]; then
    echo "[install] no python found on PATH; pass --python /path or --conda NAME" >&2
    exit 3
  fi
fi

PIP="$TARGET_PY -m pip"
echo "[install] target python         = $TARGET_PY"
echo "[install] target python version = $($TARGET_PY -V)"

# ─────────────────────────────────────────────────────────────────────────
# Pre-flight: detect CUDA, warn on mismatch.
# ─────────────────────────────────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
  DRIVER_CUDA="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
  echo "[install] nvidia driver: $DRIVER_CUDA"
fi
if command -v nvcc >/dev/null 2>&1; then
  NVCC_VER="$(nvcc --version 2>/dev/null | grep release || true)"
  echo "[install] nvcc: $NVCC_VER"
fi

# Compute total steps for pretty progress log.
TOTAL_STEPS=5
[[ $INSTALL_FLASH  -eq 1 ]] && TOTAL_STEPS=$((TOTAL_STEPS + 1))
[[ -n "$INSTALL_EXTRAS" ]]  && TOTAL_STEPS=$((TOTAL_STEPS + 1))
STEP=0
next_step() { STEP=$((STEP + 1)); echo; echo "[install] [${STEP}/${TOTAL_STEPS}] $1"; }

# ─────────────────────────────────────────────────────────────────────────
# Step 1 — pip itself + build prerequisites.
# ─────────────────────────────────────────────────────────────────────────
next_step "upgrading pip + build deps"
$PIP install --upgrade 'pip>=24.0' 'setuptools>=68' 'wheel>=0.42' 'packaging>=24'

# ─────────────────────────────────────────────────────────────────────────
# Step 2 — PyTorch from the CUDA 12.4 wheel index.
# We pin to 2.6.0 because student_wrapper.py exercises PyTorch 2.6's
# SDPA flash-kernel branch and the FSDP fast path.
# ─────────────────────────────────────────────────────────────────────────
next_step "installing torch 2.6.0 + cu124 (largest download, ~2.5GB)"
$PIP install \
    --index-url "$PYTORCH_INDEX" \
    'torch==2.6.0' 'torchvision==0.21.0' 'torchaudio==2.6.0'

# ─────────────────────────────────────────────────────────────────────────
# Step 3 — RM-training dependencies from requirements.txt (standard PyPI).
# ─────────────────────────────────────────────────────────────────────────
next_step "installing pinned deps from requirements.txt"
EXTRA_FLAG=""
if [[ -n "$PIP_EXTRA_INDEX_URL" ]]; then
  EXTRA_FLAG="--extra-index-url $PIP_EXTRA_INDEX_URL"
fi
# shellcheck disable=SC2086
$PIP install $EXTRA_FLAG -r "$ROOT/requirements.txt"

# ─────────────────────────────────────────────────────────────────────────
# Step 4 — flash-attn 2 build (default ON).
# Required by scripts/train_8gpu_zero3_joint.sh + siblings, which pass
# ``--attn_implementation flash_attention_2``. Built with
# ``--no-build-isolation`` because setup.py imports torch at build time.
# ─────────────────────────────────────────────────────────────────────────
if [[ $INSTALL_FLASH -eq 1 ]]; then
  next_step "building flash-attn 2.7.4.post1 (5-10 min with ninja, needs nvcc on PATH)"
  $PIP install ninja
  $PIP install 'flash-attn==2.7.4.post1' --no-build-isolation
else
  echo
  echo "[install] WARNING: --no-flash was passed. The zero3 training scripts"
  echo "[install]          (scripts/train_8gpu_zero3_joint.sh etc.) REQUIRE"
  echo "[install]          flash_attn and will fail to load Qwen2.5-VL with"
  echo "[install]          --attn_implementation flash_attention_2."
fi

# ─────────────────────────────────────────────────────────────────────────
# Step 5 — optional extras (vllm / bitsandbytes for legacy image path).
# ─────────────────────────────────────────────────────────────────────────
if [[ -n "$INSTALL_EXTRAS" ]]; then
  next_step "installing optional extras: [$INSTALL_EXTRAS]"
  # shellcheck disable=SC2086
  $PIP install $EXTRA_FLAG -e "$ROOT[$INSTALL_EXTRAS]"
fi

# ─────────────────────────────────────────────────────────────────────────
# Step 6 — install the project itself in editable mode so
# ``import rm_training`` / ``import rubric_pipeline`` work without
# PYTHONPATH gymnastics. Falls through silently if pyproject.toml has
# unrelated build issues (the RM trainer is import-from-cwd friendly).
# ─────────────────────────────────────────────────────────────────────────
next_step "installing project in editable mode (best-effort)"
if [[ -z "$INSTALL_EXTRAS" ]]; then
  # If we already ran ``pip install -e .[extras]`` above, this is a no-op.
  $PIP install -e "$ROOT" || \
      echo "[install] (non-fatal) editable install failed; rm_training still importable from cwd."
else
  echo "[install] (already installed via --extras step above)"
fi

# ─────────────────────────────────────────────────────────────────────────
# Smoke test — import every critical module.
# ─────────────────────────────────────────────────────────────────────────
if [[ $RUN_SMOKE -eq 1 ]]; then
  next_step "running import smoke test"
  RM_SMOKE_FLASH_REQUIRED="$INSTALL_FLASH" $TARGET_PY - <<'PY'
import importlib, os, sys
flash_required = os.environ.get("RM_SMOKE_FLASH_REQUIRED", "1") == "1"
required = [
    # (module, must_have)
    ("torch", True), ("torchvision", True), ("torchaudio", True),
    ("transformers", True), ("tokenizers", True),
    ("accelerate", True), ("peft", True),
    ("huggingface_hub", True), ("sentencepiece", True),
    ("safetensors", True),
    ("qwen_vl_utils", True),
    ("av", True), ("decord", True), ("PIL", True),
    ("cv2", True), ("einops", True), ("ftfy", True),
    ("diffusers", True), ("torchdiffeq", True),
    ("liger_kernel", True),
    ("numpy", True), ("pandas", True), ("scipy", True),
    ("yaml", True), ("datasets", True), ("pyarrow", True),
    ("matplotlib", True),
    ("sentence_transformers", True), ("sklearn", True),
    ("wandb", True), ("openai", True), ("tiktoken", True),
    ("tenacity", True), ("loguru", True),
    ("tqdm", True), ("requests", True), ("pydantic", True),
    ("deepspeed", True),          # scripts/train_8gpu_zero3*.sh preflight-imports this
    ("flash_attn", flash_required),
    ("vllm", False),              # optional extras
    ("bitsandbytes", False),      # optional extras
]
print("\n[smoke] checking imports …")
ok, missing = 0, []
for pkg, must_have in required:
    try:
        m = importlib.import_module(pkg)
        v = getattr(m, "__version__", "n/a")
        print(f"  {pkg:<25s} {v}")
        ok += 1
    except ImportError as e:
        print(f"  {pkg:<25s} MISSING ({e})")
        if must_have:
            missing.append(pkg)
print(f"[smoke] {ok}/{len(required)} imports OK")
if missing:
    print(f"[smoke] FATAL: missing required deps: {missing}")
    sys.exit(1)

import torch
print(f"\n[smoke] torch                     = {torch.__version__}")
print(f"[smoke] torch.version.cuda        = {torch.version.cuda}")
print(f"[smoke] torch.cuda.is_available() = {torch.cuda.is_available()}")
if torch.cuda.is_available():
    devs = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(f"[smoke] devices                   = {devs}")
print(f"[smoke] flash_sdp_enabled         = {torch.backends.cuda.flash_sdp_enabled()}")
PY
else
  echo "[install] skipping smoke test (--no-smoke)"
fi

# ─────────────────────────────────────────────────────────────────────────
# Done.
# ─────────────────────────────────────────────────────────────────────────
echo
echo "[install] DONE. Activate with:"
if [[ -n "$USE_CONDA" ]]; then
  echo "    conda activate $USE_CONDA"
elif [[ -n "$USE_VENV" ]]; then
  echo "    source $USE_VENV/bin/activate"
else
  echo "    (already active in $TARGET_PY)"
fi
echo
echo "[install] Next:"
echo "    # single-card LoRA run"
echo "    python rm_training/stage2_grpo/train_grpo.py \\"
echo "        --student_model /path/to/Qwen2.5-VL-7B-Instruct \\"
echo "        --gold_rubrics  data/rm_train/gold_rubrics_18d.jsonl \\"
echo "        --lora_config   rm_training/stage2_grpo/configs/lora_qwen25vl.yaml \\"
echo "        --output_dir    ckpts/run_lora"
echo
echo "    # 8-card FSDP full-param run"
echo "    bash scripts/train_8gpu_fsdp.sh ckpts/run_fsdp"
echo
echo "    # Rubric pipeline (Gemini judge)"
echo "    export OPENAI_API_KEY=...    # Gemini or OpenAI-compatible key"
echo "    bash scripts/v_arr_mvp/run_mvp_pipeline.sh"
