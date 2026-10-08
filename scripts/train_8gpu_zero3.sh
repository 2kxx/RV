#!/usr/bin/env bash
# Stage ② GRPO training on 8 × H100 with DeepSpeed ZeRO-3 + full-parameter
# fine-tuning + KL ref policy + wandb logging.
#
# Usage
# -----
#   bash scripts/train_8gpu_zero3.sh [output_dir] [extra_train_args...]
#
# Defaults
# --------
#   output_dir: ckpts/student_grpo_8gpu_zero3
#   wandb:      enabled (--wandb_project rewardverse-rm)
#   ref policy: same path as student model (KL against base instruct)
#   epochs:     5  (use --max_steps N to override with a step budget instead)
#   G:          8 rollouts/instance (8 cards × G=8 = effective 64 rollouts/step)
#
# Memory budget (per H100 80GB, verified analytically):
#   Model:    8.3B / 8 = 1.04 GB     (peak burst on layer-gather: ~14 GB)
#   Optim:    8.3B × 12B / 8 = 12.5 GB  (m + v fp32, master fp32)
#   Grads:    8.3B × 2B / 8 = 2.1 GB
#   Activ:    ~2 GB / rollout (grad-ckpt + flash-attn-2)
#   KL ref:   ~14 GB / card (full-replicated, NOT sharded — see _load_ref_model)
#   Total steady ~30 GB / card, peak ~50 GB / card → fits comfortably.

set -e
cd "$(dirname "$0")/.."

OUTPUT_DIR="${1:-ckpts/student_grpo_8gpu_zero3}"
shift || true

# Model path resolution — env var overrides, then fallback list.
# Set RM_MODEL_PATH=/your/path if your weights live somewhere else.
if [[ -n "${RM_MODEL_PATH:-}" ]]; then
  MODEL_PATH="$RM_MODEL_PATH"
else
  for cand in \
      /efs/jacuzzitang/models/Qwen2.5-VL-7B-Instruct \
      /group/40063/jacuzzitang/models/Qwen2.5-VL-7B-Instruct \
      ; do
    if [[ -d "$cand" ]]; then
      MODEL_PATH="$cand"
      break
    fi
  done
fi
if [[ -z "${MODEL_PATH:-}" || ! -d "$MODEL_PATH" ]]; then
  echo "[train] ERROR: Qwen2.5-VL-7B-Instruct not found locally."
  echo "[train] Set RM_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct and re-run." >&2
  exit 4
fi
ACCELERATE_CFG="rm_training/stage2_grpo/configs/accelerate_zero3_8gpu.yaml"

# Quick env sanity. deepspeed must be importable in this venv.
if ! python -c "import deepspeed" 2>/dev/null; then
  echo "[train] ERROR: deepspeed is not installed in the active Python env." >&2
  echo "[train]   pip install deepspeed" >&2
  exit 5
fi

mkdir -p "$OUTPUT_DIR"

# Use all 8 H100 cards; the accelerate config asks for 8 processes.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false   # quiet a noisy fork warning
export TRANSFORMERS_VERBOSITY=warning
# DeepSpeed-side: use NCCL (default) and avoid bringing up its own
# distributed launcher (accelerate handles process spawn).
export ACCELERATE_USE_DEEPSPEED=true

# ── CUDA allocator tuning to fight backward-time CUBLAS_STATUS_ALLOC_FAILED.
#
# Things we tried and DROPPED:
#   * ``expandable_segments:True`` — DOES NOT work with DeepSpeed ZeRO-3.
#     DS allocates the gather flat-buffer via plain ``torch.empty`` and the
#     CUDACachingAllocator asserts ``!block->expandable_segment_`` on those
#     paths. Crashes immediately on the first param all_gather (see
#     CUDACachingAllocator.cpp:2586). Keep this OFF when running ZeRO-3.
#
# What we keep:
#   * ``max_split_size_mb=512`` — caps how aggressively the caching pool
#     splits a free block to satisfy a small request, which keeps small
#     backward-time allocations (cuBLAS workspaces, attn buffers) from
#     carving up the larger free chunks the next forward needs.
#   * ``garbage_collection_threshold=0.8`` — when ≥80% of reserved memory
#     is allocated, the allocator opportunistically returns the LARGEST
#     free block to the driver. Empirically the gentlest knob to relieve
#     fragmentation under PPO/GRPO-style alloc churn without breaking DS.
#   * (Optional) ``RM_NO_ALLOC_CONF=1`` to wipe this config entirely if
#     you suspect it's interfering with diagnosis.
if [[ "${RM_NO_ALLOC_CONF:-0}" != "1" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8"
fi

# ── NCCL memory tuning to keep P2P channel init within budget.
#
# Symptom (2026-06-13): on the FIRST reduce_scatter of training, NCCL
# tries to allocate ~4.5 KB per channel for control structs + ring
# buffers (init.cc:525 → channel.cc:40). With ZeRO-3 sharded params +
# full-replicated ref model + Trick A's past_kv + video-token
# activations occupying nearly all 80 GB, the cudaCalloc fails and
# NCCL aborts with ``Cuda failure 2 'out of memory'`` → wrapped as
# ``ncclUnhandledCudaError`` in the trainer.
#
# The fix has two prongs:
#
# 1. Cap ``NCCL_MAX_NCHANNELS`` to reduce the number of P2P channels
#    NCCL allocates control structs for. Default is 24 channels (one
#    per ring × 3 rings); we cap at 4. This frees ~80-100 KB of
#    NCCL-side allocations.
#
#    Cost: communication bandwidth drops ~30% (fewer parallel SMs
#    handling NCCL). Acceptable during warm-up; revisit when revisiting
#    the score-PG memory budget.
#
# 2. Cap ``NCCL_BUFFSIZE`` (per-channel staging buffer) at 1 MB
#    (default 4 MB). Each channel reserves this much for collective
#    ring buffers. 4 channels × 1 MB = 4 MB instead of 24 × 4 = 96 MB.
#
#    Cost: very small messages may be split into more chunks. For
#    reduce_scatter on 7B params (~14 GB) this is invisible.
#
# 3. ``TORCH_NCCL_AVOID_RECORD_STREAMS=1`` tells PyTorch to skip the
#    record_stream call on NCCL-allocated tensors, which keeps
#    NCCL's caching alloc more predictable on tight cards.
#
# If you have headroom and want to restore full bandwidth (e.g. after
# Trick A's past_kv issue is solved), unset these or raise the caps.
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-2}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-4}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-1048576}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"

# ── Per-phase micro-batch caps. Default raised to 8 so all G=8 rollouts
# fold into a single forward pass per phase — saves ~30-40s/step on
# rubric generation. Lower to 4 if OOM appears under your specific
# config (e.g. larger G, longer prompts). Raise above 8 only if you
# have headroom on a card.
#
# 2026-06-11 update: ``RM_SCORE_MB`` defaulted to 4 (down from 8).
# Empirically mb=8 on the score forward triggers a NaN-logits failure
# mode under ZeRO-3 + grad-ckpt + bf16 — the 8-row batch causes
# ``step_scores[t]`` to come back with NaN for ~30% of rows even when
# ``sequences`` looks fine. Halving to mb=4 keeps the prefill batched
# (still ~2x speedup vs mb=1) without triggering the bad path. See
# pref_diag + score_diag dump 2026-06-11 (logits_nan_anchors=4 at mb=8).
#
# 2026-06-13 update: ``RM_SCORE_MB`` defaulted to 1 (down from 4).
# probe_v8_bisect variants A vs U conclusively isolate the bug to
# batched left-padding under ZeRO-3 + SDPA + sharded attention:
#   * variant A (mb=4): 33% finite, 17 stub, 14 no_anchor / 320 cells
#   * variant U (mb=1): 100% finite across all 8 ranks, 0 stub, 0 no_anchor
# Other suspects all ruled out: greedy/sampling (Q), budget (Q),
# grad-ckpt (T), rubric form (R), single-card baseline (probe_score_v4
# v8cfg = 100%). The remaining variable that flips the outcome is
# whether more than one rubric shares a generate() call. Cost: each
# score forward now does G=8 prefills per side instead of 2; on H100
# this adds ~30-50 s per train step relative to mb=4. Cheap insurance
# against silent reward poisoning.
#
# If you need to override (e.g. to reproduce the bug, or once a
# kernel-level fix lands), set ``RM_SCORE_MB=4`` in the env before
# launching this script.
export RM_RUBRIC_MB="${RM_RUBRIC_MB:-8}"
export RM_SCORE_MB="${RM_SCORE_MB:-1}"

echo "[train] launching ZeRO-3 run -> $OUTPUT_DIR"
echo "[train] config:           $ACCELERATE_CFG"
echo "[train] student:          $MODEL_PATH (full-param)"
echo "[train] ref:              $MODEL_PATH (frozen)"

accelerate launch \
    --config_file "$ACCELERATE_CFG" \
    rm_training/stage2_grpo/train_grpo.py \
        --student_model        "$MODEL_PATH" \
        --ref_model_path       "$MODEL_PATH" \
        --gold_rubrics         data/rm_train/gold_rubrics_18d.clean.jsonl \
        --output_dir           "$OUTPUT_DIR" \
        --num_generations      8 \
        --epochs               5 \
        --gold_rubric_warmup_epochs 1 \
        --log_every            1 \
        --save_every           30 \
        --val_every            0 \
        --val_full_every       0 \
        --val_max_samples_per_dim 10 \
        --learning_rate        2e-6 \
        --warmup_steps         30 \
        --pref_temperature     1.5 \
        --beta_kl              0.1 \
        --lambda_0             0.5 \
        --lambda_min           0.0 \
        --lambda_warmup_steps  30 \
        --lambda_decay_steps   100 \
        --beta_format          0.5 \
        --beta_format_min      0.1 \
        --beta_format_decay_steps 200 \
        --beta_format_schedule linear \
        --align_floor          0.3 \
        --score_sample \
        --score_pg_alpha_max   1.0 \
        --score_pg_start_step  50 \
        --score_pg_full_step   150 \
        --score_format_gate    0.65 \
        --max_grad_norm        1.0 \
        --attn_implementation  flash_attention_2 \
        --gradient_checkpointing \
        --wandb_project        rewardverse-rm \
        --wandb_name           "$(basename "$OUTPUT_DIR")" \
        --wandb_tags           zero3 8gpu full-param hard-fmt score-pg gold-warmup token-kl \
        --wandb_sample_every   50 \
        "$@"
