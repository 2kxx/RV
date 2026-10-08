#!/usr/bin/env bash
# Stage ② GRPO — PHASE A: score-head warm-up only.
#
# What this does
# --------------
# Runs ONLY the score-head warm-up (1 epoch = 60 steps with the current
# 480-row balanced dataset / 8 ranks). The rubric stays at gold for every
# rollout so:
#   * NO rubric forward (no log-prob, no rubric-side KL forward)
#   * Score forward + score-side KL only
#   * score_pg_alpha forced to alpha_max during warm-up (see
#     train_grpo.py:1730)
#
# Dataset (2026-06-18 rebuild)
# ----------------------------
# 16 vision dimensions × 30 pairs each = 480 rows. The 4 originally
# under-covered dims (logic 17, rhythm 22, action 29, expression 29)
# were extended to 30 each via build_dim_seed_v2.py with
# --max_pairs_per_prompt=2 --min_score_gap=0.05; the other 12 dims are
# unchanged. Two audio dims (soundscape, vocal) stay excluded because
# Qwen2.5-VL is vision-only. See scripts/v_arr_mvp/extend_data.sh for
# the full reproduction recipe.
#
# Memory peak is exactly the same as steps 0-59 of the 2026-06-14 run
# (which proved the per-step footprint fits comfortably). The OOM at
# step 60→61 in that run was the warm-up→joint TRANSITION suddenly
# turning on the rubric-side KL forward (~6000-token ref forward × G
# rollouts in addition to the score forwards). Splitting that transition
# into two separate runs avoids the spike entirely.
#
# Output
# ------
# Writes to ckpts/student_grpo_8gpu_zero3_warmup/checkpoint-{30,60} (or
# whatever output dir you pass as $1). Pass the FINAL ckpt path to
# train_8gpu_zero3_joint.sh as the starting point for phase B.
#
# Usage
# -----
#   bash scripts/train_8gpu_zero3_warmup.sh
#   bash scripts/train_8gpu_zero3_warmup.sh ckpts/my_warmup_run
#
# The script ends naturally after 1 epoch (60 steps); no need to ctrl-C.

set -e
cd "$(dirname "$0")/.."

OUTPUT_DIR="${1:-ckpts/student_grpo_8gpu_zero3_warmup}"
shift || true

# ── Model path resolution (same logic as train_8gpu_zero3.sh).
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
  echo "[train-warmup] ERROR: Qwen2.5-VL-7B-Instruct not found locally." >&2
  echo "[train-warmup] Set RM_MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct and re-run." >&2
  exit 4
fi
ACCELERATE_CFG="rm_training/stage2_grpo/configs/accelerate_zero3_8gpu.yaml"

if ! python -c "import deepspeed" 2>/dev/null; then
  echo "[train-warmup] ERROR: deepspeed not installed in active venv." >&2
  exit 5
fi

mkdir -p "$OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning
export ACCELERATE_USE_DEEPSPEED=true

# Allocator + NCCL tuning — copy from train_8gpu_zero3.sh, same trade-offs.
if [[ "${RM_NO_ALLOC_CONF:-0}" != "1" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8"
fi
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-2}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-4}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-1048576}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"

export RM_RUBRIC_MB="${RM_RUBRIC_MB:-8}"
export RM_SCORE_MB="${RM_SCORE_MB:-1}"

# Score-side KL soft brake (added 2026-06-19).
# Standalone coefficient on score-leg KL added to loss as
# `RM_SCORE_KL_BETA · kl_score`, independent of beta_kl / score_pg_alpha.
# Reason for adding: 2026-06-18 warmup run drifted from kl_s≈0.6 (step 0,
# path-noise floor) to kl_s≈4500 (step 30) — score head wandered far
# enough that S_pos/S_neg collapsed from [5,5,5,5,5] to [1,3,1,1,1] and
# preference reversed in some samples. lambda_pw only constrains
# (S_pos − S_neg), nothing constrains the absolute level. A small KL
# anchor (~10 nats * 0.005 = 0.05 added to loss when kl_s=2000)
# supplies the soft pull-back toward ref without dominating pg/pw the
# way RM_KL_INCLUDE_SCORE=1 would (that path uses beta_kl × alpha = 0.04
# coefficient → ~96 added to loss at the same kl_s).
# Set to 0 to disable; tune up to ~0.02 for harder constraint.
#
# Raised 0.005 → 0.02 on 2026-06-28. v3 joint per-dim eval revealed
# the warmup-trained score head was already drifting absolute scale
# down (avg S_mean − label_mean = −0.45 by the time joint started
# from warmup ckpt-60), and joint training accelerated this drift to
# −0.98 by ckpt-60. Hypothesis: warmup is the cleanest training window
# for the score head (rubric is locked to gold, advantage attribution
# flows entirely through r_pref → score head), so we should be doing
# more score-head training there, not less. Tripling the KL anchor to
# 0.02 lets us train the score head harder in warmup without it
# drifting off the ref absolute scale. 0.02 is the upper end the
# original comment flagged for kl_s ~50 territory; with current kl_s
# 5-10 nats in warmup, 0.02 contributes ~0.1-0.2 to loss, comparable
# to lambda_pw·pw (~0.06) and well below pg_loss magnitude (~1-3) so
# it shouldn't dominate.
#
# Reverted 0.02 → 0.005 on 2026-06-30. Empirical result of joint v6/
# v7/v8 (all using warmup-v2 ckpt-60 as start) collapsed below v3's
# 0.415 (v6 ckpt-30 plcc 0.295 from rubric corruption, v7 valid → 1-2
# from step 30 onward, v8 ckpt-45 plcc 0.272 with rubric leg fully
# blowing up at kl=45 nat in steps 45-60). Mechanism: warmup-v2's
# stronger KL anchor (0.02) trained score head into a TIGHTER local
# minimum that is brittle when joint switches input distribution
# (rubric becomes a moving target instead of fixed gold). "Trained
# more stably" ≠ "better starting point for joint" — the v1 score
# head's noisier training trajectory left it in a flatter region of
# the loss landscape, which transferred better to joint. v1 KL=0.005
# is still the best-known config for warmup → joint v3 → 0.415.
export RM_SCORE_KL_BETA="${RM_SCORE_KL_BETA:-0.005}"

# ── Log noise reduction (added 2026-06-21).
# Silence per-step trace breadcrumbs by default. Per-sample rubric dumps
# are emitted every 10 steps via --sample_log_every so we can watch
# S_pos / S_neg / scores_pos / scores_neg / r_pref evolve across the
# 60-step warm-up (added 2026-06-23 — train_warmup-new.log shows pos/neg
# barely separating; without this dump we can't tell whether the score
# head is putting [5,4,4,3,3] on pos and [4,3,3,2,2] on neg, or just
# returning [3,3,3,3,3] for both). Set RM_SAMPLE_LOG_EVERY=0 to silence.
export RM_TRACE_SAMPLE="${RM_TRACE_SAMPLE:-0}"
SAMPLE_LOG_EVERY="${RM_SAMPLE_LOG_EVERY:-10}"

echo "[train-warmup] phase A: score-head warm-up only (1 epoch ≈ 60 steps)"
echo "[train-warmup] output:  $OUTPUT_DIR"
echo "[train-warmup] student: $MODEL_PATH"

# ── Crash diagnostics (added 2026-06-20).
# Repeated SIGSEGV on rank 2 with no Python traceback. Enable core dumps
# and force unbuffered stderr so the breadcrumb in train_grpo.py is
# flushed before the launcher kills siblings on segfault.
#
# RM_FAULT_DIR: where train_grpo.py's faulthandler writes per-rank
# faulthandler_rankN.log (Python stacks of every thread on every rank,
# captured by the SIGSEGV handler before the process dies).
#
# core_pattern is system-wide; we can't chmod it from a non-root user
# inside the container, so we just bump ulimit -c and hope the system
# default writes to /var/lib/systemd/coredump or the cwd. If you have
# root, also run: echo '/tmp/core.%e.%p.%t' > /proc/sys/kernel/core_pattern
ulimit -c unlimited || true
export RM_FAULT_DIR="${RM_FAULT_DIR:-$OUTPUT_DIR/faulthandler}"
mkdir -p "$RM_FAULT_DIR" || true
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
echo "[train-warmup] core ulimit: $(ulimit -c)"
echo "[train-warmup] faulthandler dir: $RM_FAULT_DIR"

# Phase A specifics:
#   --epochs 3                        → 180 steps total (60 steps/epoch ×
#                                        3 epochs). Bumped from 1 on
#                                        2026-06-24 after the 60-step
#                                        warmup ckpt evaluated at plcc
#                                        ~0.4 with valid_ratio_ema and
#                                        reward_std both still climbing
#                                        (see logs/train_warmup.log /
#                                        wandb run 06-23). 60 steps was
#                                        a "stop early to avoid the
#                                        warmup→joint OOM transition"
#                                        decision, not a "score head
#                                        converged" decision. With ckpt
#                                        every 30 steps we get plcc at
#                                        30/60/90/120/150/180 to draw
#                                        the convergence curve directly.
#                                        If plcc plateaus by ckpt-120,
#                                        drop to --epochs 2 in future
#                                        runs; if still climbing at
#                                        ckpt-180, bump to 4.
#   --gold_rubric_warmup_epochs 3     → match --epochs so the entire run
#                                        stays in warmup mode (rubric
#                                        held to gold for every rollout).
#                                        Keeping it < --epochs would
#                                        trigger the warmup→joint
#                                        transition mid-run, which is
#                                        the OOM scenario phase A is
#                                        explicitly avoiding.
#   --score_pg_alpha_max 0.4          → score-PG ON, MODERATE strength
#                                        (lowered from 1.0 on 2026-06-18
#                                        after a run crashed at step 30/57:
#                                        kl_s ramped 0.36 → 4137 nats over
#                                        30 steps, score outputs drifted
#                                        far enough off-format that all 8
#                                        rollouts simultaneously fell below
#                                        --score_format_gate, leaving the
#                                        loss with no grad-bearing tensor
#                                        and triggering "element 0 ... does
#                                        not have a grad_fn" on backward).
#                                        With α=0.4 + the lower λ_bt below,
#                                        the 30-step kl_s drift should stay
#                                        in the O(10) nat range. Note: in
#                                        warm-up α is held at α_max for
#                                        every step (train_grpo.py:1729-32)
#                                        — start_step / full_step are NOT
#                                        used here, only in phase B.
#   --score_pg_start_step / full_step → ignored during warm-up but kept
#                                        for legibility
#   --lambda_bt 0                     → BT-aux OFF (2026-06-18). Replaced
#                                        by score-gap supervision below.
#                                        Reason: warmup-2 step 30 dumped
#                                        scores_pos = scores_neg =
#                                        [1,1,1,1,1] — BT pushes margin
#                                        unbounded which collapses both
#                                        scores to a flat constant when
#                                        the visual signal is weak. Kept
#                                        as a 0-default arg so we can
#                                        ablate (set to 0.2 to recover
#                                        old behaviour).
#   --bt_temperature 1.5              → unused while lambda_bt=0; kept
#                                        for legibility / ablation.
#   --lambda_pw 0.2                   → score-gap L1 ON (lowered from 0.5
#                                        on 2026-06-21). Supervises
#                                        S⁺ − S⁻ to match 4·(winner_score
#                                        − loser_score) joined from each
#                                        row's source_pair_file. Avoids
#                                        the BT mode-collapse because the
#                                        target gap for a 0.05 evalverse
#                                        pair is 0.2 in 1-5 range — no
#                                        incentive to saturate.
#                                        Why 0.5 → 0.2: with lambda_pw=0.5
#                                        the L1 term sits ~0.5 in loss
#                                        magnitude while the score-PG term
#                                        (0.4 × adv × Δlog_prob) only
#                                        contributes ~0.005-0.04 in
#                                        gradient — pw outweighs PG by
#                                        10-100×, so score head learns
#                                        "give 3.0 to everything and pin
#                                        S⁺−S⁻ at 0.2" (low pw_loss, low
#                                        PLCC). Observed full-set eval
#                                        on 2026-06-21: warmup ckpt-60
#                                        mae=0.637 (very low → constants)
#                                        + plcc=0.272 (collapse). Halving
#                                        pw to 0.2 gives PG room to push
#                                        rank without losing magnitude
#                                        anchor. Bump back to 0.5 if S⁺−S⁻
#                                        explodes (>1.0 in absolute terms);
#                                        drop to 0.1 if collapse persists.
#                                        Pointwise calibration (MSE on S
#                                        itself) is intentionally NOT
#                                        done: student rubric themes are
#                                        dynamically generated and don't
#                                        share an axis with evalverse's
#                                        sub_keys, so absolute-scale
#                                        alignment is unsound. Differences
#                                        ARE comparable under a fixed
#                                        student rubric.
#   --score_format_gate 0.55          → relaxed from 0.65 (2026-06-18) so
#                                        score-PG / BT-aux still fire on
#                                        rollouts whose format is mostly
#                                        ok but not strict-mode clean.
#                                        Combined with the policy_update
#                                        defensive early-return (in case
#                                        every rollout is gated), this
#                                        avoids the "all rolls below gate
#                                        → zero-tensor backward" failure.
#                                        2026-06-28 v2 tried 0.55 → 0.60
#                                        alongside KL=0.02; warmup itself
#                                        was stable but the resulting
#                                        ckpt-60 made joint v6/v7/v8
#                                        collapse below 0.42 (vs v3's
#                                        0.415 from the 0.55 baseline).
#                                        Reverted to 0.55 on 2026-06-30
#                                        as part of full warmup → v1
#                                        baseline rollback.
#
# --ref_zero3: REMOVED 2026-06-20 — warmup now matches joint (no ref_zero3,
# per-rank full ref replica ~14 GB).
#
# Why this works without OOM
# --------------------------
# Earlier attempts to drop --ref_zero3 OOM'd on step-0 backward because
# warmup OVERRIDES score_pg_alpha to alpha_max from step 0 (gold-warmup
# override at train_grpo.py:1885-88), so backward touches the full
# score-PG path immediately and the ref 14 GB + score-PG backward graph +
# DeepSpeed comm buffers don't all fit on a fresh allocator.
#
# Fix: train() now runs ONE preflight step before the real loop —
# rollout + forward + backward identical to step 0, but optimizer.step()
# is skipped and gradients zeroed. This materializes every lazy alloc
# (DS reduce buffers, NCCL comms, flash-attn workspace, Adam state) and
# rearranges the cached pool so step 0 sees a settled allocator. Joint
# already gets a "free" settle window from its step-0-α=0 ramp; preflight
# gives warmup the equivalent without changing training dynamics.
#
# Disable preflight with RM_PREFLIGHT=0 (e.g. for debugging; you'll then
# need --ref_zero3 back).
accelerate launch \
    --config_file "$ACCELERATE_CFG" \
    rm_training/stage2_grpo/train_grpo.py \
        --student_model        "$MODEL_PATH" \
        --ref_model_path       "$MODEL_PATH" \
        --gold_rubrics         data/rm_train/gold_rubrics_18d.clean.jsonl \
        --output_dir           "$OUTPUT_DIR" \
        --num_generations      8 \
        --epochs               1 \
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
        --score_pg_alpha_max   0.4 \
        --score_pg_start_step  0 \
        --score_pg_full_step   1 \
        --score_format_gate    0.55 \
        --lambda_bt            0.0 \
        --bt_temperature       1.5 \
        --lambda_pw            0.2 \
        --max_grad_norm        1.0 \
        --attn_implementation  flash_attention_2 \
        --gradient_checkpointing \
        --wandb_project        rewardverse-rm \
        --wandb_name           "$(basename "$OUTPUT_DIR")" \
        --wandb_tags           zero3 8gpu full-param phase-a warmup gap-pw score-pg kl-fix \
        --wandb_sample_every   50 \
        --sample_log_every     "$SAMPLE_LOG_EVERY" \
        "$@"

echo "[train-warmup] phase A complete."
echo "[train-warmup] next: bash scripts/train_8gpu_zero3_joint.sh $OUTPUT_DIR/checkpoint-latest"
