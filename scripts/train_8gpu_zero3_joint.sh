#!/usr/bin/env bash
# Stage ② GRPO — PHASE B: joint training from a warmed-up checkpoint.
#
# What this does
# --------------
# Resumes training FROM a phase-A warm-up checkpoint (where the score
# head is already trained against gold rubrics) and now trains BOTH
# legs jointly:
 
#   * Student emits its OWN rubric (rubric forward + rubric-side KL on)
#   * Score forward operates on student rubric (not gold)
#   * score_pg_alpha ramps up linearly from --score_pg_start_step to
#     --score_pg_full_step (set both small so it kicks in immediately)
# By skipping warm-up entirely (--gold_rubric_warmup_epochs 0) the
# warmup→joint memory-spike transition is gone — every step from the
# very first one runs the full rubric+score forward stack, so the
# allocator settles into the joint-mode steady state from step 0.
#
# Why this avoids the OOM
# -----------------------
# In a single-script run the warm-up path NEVER touches the rubric
# forward (rubric is held to gold), so the cached allocator's blocks
# only contain score-forward sized buffers. At step 60→61, suddenly
# G=8 rollouts × 2 sides (pos+neg score) × 2 forwards (policy + ref)
# = 32 fresh ~6000-token forwards land in the pool. The allocator
# has no free block big enough → tries to grab 30 GiB → OOM
# (observed 2026-06-14 on rank 2). Splitting the run lets phase B
# start with a clean allocator that grows organically into joint-mode
# peaks.
#
# Usage
# -----
#   bash scripts/train_8gpu_zero3_joint.sh <warmup_ckpt> [output_dir]
#
# Examples
#   bash scripts/train_8gpu_zero3_joint.sh \
#       ckpts/student_grpo_8gpu_zero3_warmup/checkpoint-60
#   bash scripts/train_8gpu_zero3_joint.sh \
#       ckpts/student_grpo_8gpu_zero3_warmup/checkpoint-latest \
#       ckpts/student_grpo_8gpu_zero3_joint
#
# Output
# ------
# Writes to ckpts/student_grpo_8gpu_zero3_joint by default.
# (Different dir from phase A so the warm-up ckpts don't get clobbered.)

set -e
cd "$(dirname "$0")/.."

WARMUP_CKPT="${1:?Pass the warm-up checkpoint dir as the first argument.}"
shift

if [[ ! -d "$WARMUP_CKPT" ]]; then
  echo "[train-joint] ERROR: warm-up ckpt not found: $WARMUP_CKPT" >&2
  exit 4
fi

OUTPUT_DIR="${1:-ckpts/student_grpo_8gpu_zero3_joint}"
shift || true

# Resolve symlink (checkpoint-latest → checkpoint-N) so logs show the
# real dir rather than the alias.
WARMUP_CKPT_RESOLVED="$(readlink -f "$WARMUP_CKPT")"

# ── Base model resolution.
#
# KL anchor decision (revised 2026-06-16)
# ---------------------------------------
# Both legs of the KL term (rubric leg + score leg) anchor at the WARM-UP
# CKPT, not the original untrained base. Reasoning:
#
#   * Standard PPO/GRPO convention is ref = init_policy_of_this_stage.
#     KL collapses to 0 at step 0 by design; that's correct PG behaviour,
#     not a bug. KL is a *future-drift* penalty, not a "prove you're far
#     from base" requirement.
#
#   * The score head was trained against gold rubrics in phase A. Anchoring
#     the score-side KL at the original base would penalise the student for
#     deviating from a position it has never inhabited (phase A already
#     moved the score head). Empirically (run 20260615_082421) this drove
#     kl_s from 0 → ~5 over 200 steps and pulled the reward μ from 1.0 → 0.5.
#
#   * On the rubric leg, anchoring at warmup is benign: rubric was held to
#     gold during phase A so the warmup ckpt's rubric distribution is
#     essentially the base model's rubric distribution (the gen forward
#     was never gradient-updated in phase A). KL ref = warmup ≈ KL ref =
#     base for the rubric leg, plus a small correction from any incidental
#     drift caused by score-head co-training.
#
# Net: warmup ckpt for both refs is the conservative, principled choice.
BASE_MODEL_PATH="$WARMUP_CKPT_RESOLVED"

ACCELERATE_CFG="rm_training/stage2_grpo/configs/accelerate_zero3_8gpu_joint.yaml"

if ! python -c "import deepspeed" 2>/dev/null; then
  echo "[train-joint] ERROR: deepspeed not installed in active venv." >&2
  exit 5
fi

mkdir -p "$OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=warning
export ACCELERATE_USE_DEEPSPEED=true

if [[ "${RM_NO_ALLOC_CONF:-0}" != "1" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8"
fi
export NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-2}"
export NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-4}"
export NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-1048576}"
export TORCH_NCCL_AVOID_RECORD_STREAMS="${TORCH_NCCL_AVOID_RECORD_STREAMS:-1}"

export RM_RUBRIC_MB="${RM_RUBRIC_MB:-8}"
export RM_SCORE_MB="${RM_SCORE_MB:-1}"

# Score-side KL soft brake (added 2026-06-19, mirrors warm-up).
# Standalone coefficient on score-leg KL added to loss as
# `RM_SCORE_KL_BETA · kl_score`, independent of beta_kl / score_pg_alpha.
# Reason: lambda_pw only constrains (S_pos − S_neg); nothing constrains the
# absolute level of the score head. A small KL anchor (~10 nats * 0.005 =
# 0.05 added to loss when kl_s=2000) pulls the score head softly back toward
# the warm-up ref without dominating pg/pw. Tune up to ~0.02 for a
# harder constraint if kl_s blows out past ~50 in the first epoch.
#
# Default raised from 0.0 → 0.005 on 2026-06-24. The 06-23 joint-1 run
# was the canonical "good early window" run (steps 21-37 had kl_s in
# 0.5-1.6, pw in 0.1-1.5, valid 7-8/8) but kl_s drifted past 4 around
# step 37 and combined with the λ_align decay to produce the same
# step-38 valid=0/8 cliff seen in joint-freeze. With λ_align now held
# constant (see lambda_min below) the soft brake is the second leg
# preventing the same cliff: anchors kl_s before it can drag rubrics
# off the format manifold.
#
# Raised 0.005 → 0.015 on 2026-06-28 (v4). v3 per-dim eval revealed
# systematic absolute-scale drift: avg(S_mean − label_mean) went −0.45
# (ckpt-15) → −0.58 (ckpt-30) → −0.79 (ckpt-45) → −0.98 (ckpt-60),
# monotone across all 16 dims. Gap supervision (lambda_pw=0.3) keeps
# rank correct (PLCC stayed ~0.41 from ckpt-30 onward) but lets the
# score-head emit lower and lower absolute scores — pred S_mean ≈ 3.0
# vs label_mean ≈ 4.1 at ckpt-60, and 4 dims (expression, pacing,
# consistency, progression) saw score_failed jump 0 → 23/16/7/8 in
# the last 15 steps as outputs drifted off the digit-token manifold.
# Tripling the KL anchor to 0.015 increases pull toward warm-up ref
# proportional to drift, without crossing the 0.02 ceiling that the
# warmup script flagged as "harder constraint" territory. kl_s in v3
# stayed at 1.3-1.4 — at that level 0.015 contributes ~0.02 to loss,
# small relative to pg loss (~1-3) but enough to bias the absolute
# level back toward ref over 60 steps.
#
# Lowered 0.015 → 0.01 on 2026-06-29 (v6). Pairs with switching to a
# warmup-v2 starting ckpt (which itself was trained at KL=0.02 + gate
# 0.60). The warmup-v2 score head is already pulled hard toward ref
# absolute scale by warmup's stronger anchor, so joint doesn't need
# to keep the same pull strength — that would over-constrain and
# leave score-PG without room to specialise per-dim. 0.01 ≈ half-way
# back toward the 0.005 default; with kl_s in joint expected to stay
# ~1-2 (warmup-v2 ended at kl_s ≈ 3.4), 0.01 contributes ~0.01-0.02
# to loss — same order of magnitude as v4's contribution under v3's
# higher kl_s, but better matched to v2's already-anchored start.
#
# Reverted 0.01 → 0.015 on 2026-06-29 (v7) — v6 eval revealed silent
# rubric corruption: ckpt-15 plcc 0.387 (vs v3 0.399, OK) but ckpt-30
# 0.295 (vs v3 0.415, catastrophic), with sample dumps at step 30/40
# showing rubric_text ending in malformed JSON like "]\n  }\n}" or
# "]\n  ]\n)" — extra closing brackets that fail parse_rubric.
# Mechanism: the v6 combo (gate 0.55→0.60 + KL 0.015→0.01) made
# score-PG train on a narrower high-fmt rollout subset under weaker
# KL pressure, so score-leg gradients reaching the SHARED LM backbone
# pushed the rubric distribution off-format too. Rubric β_kl=0.1
# only penalises rubric-forward KL, not the backbone shift induced
# by score-PG. v6 ran with two changes pulling the same direction
# (more "pure / aggressive" score training); reverting just the KL
# anchor restores v3-level damping while keeping gate=0.60 (which
# matches warmup-v2's gate, the only change v7 keeps from v6).
export RM_SCORE_KL_BETA="${RM_SCORE_KL_BETA:-0.015}"

# ── Log noise reduction (added 2026-06-21, mirrors warm-up).
# Silence per-step trace breadcrumbs by default. Per-sample rubric dumps
# are emitted every 10 steps via --sample_log_every (changed from 0 on
# 2026-06-23, mirrors warm-up) so we can watch S_pos / S_neg / scores_pos
# / scores_neg / r_pref evolve across the 120-step joint run. The 06-22
# joint run collapsed at step 78 (reward μ=0, valid=0/8, kl=2.2e+04) with
# no per-sample signal in the log, making the failure mode unrecoverable.
# wandb_sample_every is unchanged so the wandb table still gets samples —
# only stderr is muted. Set RM_SAMPLE_LOG_EVERY=0 to silence.
export RM_TRACE_SAMPLE="${RM_TRACE_SAMPLE:-0}"
SAMPLE_LOG_EVERY="${RM_SAMPLE_LOG_EVERY:-10}"

echo "[train-joint] phase B: joint training (rubric + score, no warmup)"
echo "[train-joint] student: $WARMUP_CKPT_RESOLVED  (loaded from warm-up)"
echo "[train-joint] ref:     $BASE_MODEL_PATH       (untrained base, for KL)"
echo "[train-joint] output:  $OUTPUT_DIR"

# Phase B specifics:
#   --student_model       <warmup_ckpt>  → load warm-up weights
#   --ref_model_path      <warmup_ckpt>  → KL anchored at warm-up (both legs)
#   --gold_rubric_warmup_epochs 0        → NO warm-up; joint from step 0
#   --lambda_0/min/decay  0.5/0.5/0      → λ_align HELD CONSTANT at 0.5
#                                          (no warmup, no decay) on
#                                          2026-06-24. Previously 0.5→0.0
#                                          over warmup=30+decay=100.
#                                          Both joint runs of 06-23 (joint-1
#                                          with full score-PG, joint-freeze
#                                          with score frozen) collapsed at
#                                          step 38: r_align cracked first,
#                                          then valid → 0/8, then kl → 1e3+.
#                                          Common factor: λ decay had been
#                                          eating the only theme-side anchor.
#                                          _r_align is forward-Chamfer cosine
#                                          over THEME STRINGS ONLY (rubric
#                                          tips and weights are not in the
#                                          loss), so a constant λ doesn't
#                                          fight tips/query-conditioning —
#                                          it only keeps the student's
#                                          theme set close to the gold theme
#                                          set. With score-PG and lambda_pw
#                                          on, the score head still gets the
#                                          BT/L1 signal it needs; r_align
#                                          just provides the format-manifold
#                                          floor so the rubric-side rollout
#                                          doesn't drift off the parser.
#   --score_pg_alpha_max  0.4            → score-PG MODERATE. Tuned down
#                                          from 1.0 on 2026-06-18 in lock-
#                                          step with the warm-up fix:
#                                          phase A's 30-step crash showed
#                                          α=1.0 + lr=2e-6 ramps kl_s into
#                                          the thousands and pushes the
#                                          score head off the format
#                                          manifold. Joint mode adds
#                                          rubric-leg gradient that
#                                          counter-balances, but for the
#                                          first joint epoch we keep α at
#                                          the warm-up value (0.4) to
#                                          inherit the warm-up's stability
#                                          regime; bump to 0.6 once kl_s
#                                          stays tame for a full epoch
#                                          and PLCC stalls.
#   --score_pg_full_step  30             → α ramp end-step. v2 (2026-06-27)
#                                          tried pushing this to 45 to defer
#                                          α=α_max into a "safer" window,
#                                          assuming fixed-run's ckpt-60
#                                          drop (0.415 → 0.389 plcc) was
#                                          caused by 30 full-strength PG
#                                          steps. Eval contradicted that:
#                                          v2 ckpt-45 plcc=0.361 (vs fixed
#                                          0.415, −0.054), ckpt-60 0.319
#                                          (vs 0.389, −0.070), and v2 block
#                                          45-60 kl_s = 7.46 vs fixed 1.27.
#                                          The aprоблеm wasn't 30 steps at
#                                          α_max — it was crossing into the
#                                          2nd epoch (rerun of same data).
#                                          Pushing full_step to 45 just
#                                          aligned the kl_s explosion with
#                                          the 2nd-epoch boundary. Reverted
#                                          to 30 on 2026-06-27 (post-v2).
#                                          A real fix needs either α decay
#                                          tail (--score_pg_decay_*) or
#                                          stopping at --epochs 1 = 60
#                                          steps with no overlap.
#   --score_format_gate   0.55           → joint stays at 0.55. v6/v7
#                                          (2026-06-29) tried 0.60 to
#                                          match warmup-v2's gate, both
#                                          collapsed: v6 ckpt-30 plcc
#                                          0.295 (rubric corruption from
#                                          shared backbone), v7 step 30+
#                                          valid dropped to 1-2/8 with
#                                          score_pos emitting "?" tokens
#                                          and zero rollouts passing
#                                          score-PG (kls=0 for blocks
#                                          30-60). Mechanism: in warmup
#                                          rubric is gold (fixed input),
#                                          gate=0.60 acts as a quality
#                                          filter; in joint rubric is
#                                          a moving target, so
#                                          marginal-fmt rollouts (0.55-
#                                          0.60) ARE the score-PG signal
#                                          that corrects emerging digit-
#                                          token slips. Tightening gate
#                                          starves score-PG of corrective
#                                          signal → score head drifts
#                                          unchecked → cascade. v3 found
#                                          0.55 was correct; v6/v7
#                                          re-confirmed it from the
#                                          failure side. Keeping at 0.55
#                                          on v8 (2026-06-29) so the only
#                                          variable vs v3 is the warmup
#                                          starting ckpt (warmup-v2 vs
#                                          warmup-v1 ckpt-60).
#   --lambda_bt           0.0            → BT-aux OFF (2026-06-19, mirrors
#                                          warm-up). Reason: phase A run
#                                          confirmed score-gap L1 (lambda_pw)
#                                          is the right pointwise supervision
#                                          — BT pushes margin unbounded which
#                                          can collapse both scores to a flat
#                                          constant when the visual signal is
#                                          weak. lambda_pw stays bounded
#                                          because the target gap for a 0.05
#                                          evalverse pair is only 0.2 in the
#                                          1-5 range. Set to 0.05 to re-enable
#                                          for ablation.
#   --bt_temperature      1.5            → unused while lambda_bt=0; kept
#                                          for legibility / ablation. Same
#                                          τ as warm-up.
#   --lambda_pw           0.3            → score-gap L1. 0.2 → 0.3 on
#                                          2026-06-27 in v2; this is the
#                                          one v2 change that DIDN'T look
#                                          harmful — block 30-44 r_pref
#                                          actually edged up vs fixed
#                                          (0.520 vs 0.498) while pw
#                                          dropped less aggressively
#                                          (0.404 vs 0.505), suggesting
#                                          the magnitude anchor was indeed
#                                          a bit weak in fixed. Kept at
#                                          0.3 in v3 as the only ablation
#                                          variable vs fixed, so v3 vs
#                                          fixed isolates the effect of
#                                          this single bump. If v3 ckpt-45
#                                          beats fixed's 0.415, attribute
#                                          to lambda_pw; if not, revert.
#                                          Still well below 0.5 which had
#                                          previously caused mode-collapse
#                                          to a constant predictor.
#   --beta_format_min     0.1            → format-floor. v2 (2026-06-27)
#                                          lifted to 0.2 hoping to hold
#                                          valid_count above 7 in the
#                                          back half. Did not help: v2
#                                          block 45-60 valid=5.38 vs
#                                          fixed 6.69, AND v2 ckpt-60
#                                          eval n dropped 994→980 same
#                                          as fixed. β_format wasn't the
#                                          binding constraint; valid drop
#                                          was caused by score-head drift
#                                          (kl_s=7.46) not weak format
#                                          pressure. Reverted to 0.1.
#   --epochs 2                           → 2 more epochs joint training
#                                           (phase A did 1 epoch already);
#                                           480 rows / 8 ranks = 60
#                                           steps/epoch, so total ≈ 120
#                                           steps + 4 ckpts at every 30.
#                                           Lowered from 4 on 2026-06-21:
#                                           current warmup config is
#                                           still collapse-prone, no
#                                           point pre-committing to 240
#                                           steps before seeing the
#                                           30/60 trajectory. Re-launch
#                                           with a higher --epochs (or
#                                           chain another joint run from
#                                           the latest ckpt) if val PLCC
#                                           still climbing at step 120.
#
# --ref_zero3: REMOVED 2026-06-19. Patch 7's `summon_full_params` ref path
# was OK in warmup (single score-side ref forward per step) but in joint
# (rubric ref forward + score-pos + score-neg) it triggered SIGSEGV on
# rank 7 at step 5 — almost certainly a NCCL all-gather edge case under
# the heavier summon traffic. Fall back to per-rank full ref replica, the
# same path the 06-15 phaseB run used for 195+ steps. Trade-off: each rank
# carries an extra ~14 GB constant for the bf16 ref, but no all-gather on
# every ref forward. kl_s gets a ~0.5 nat path-noise floor at step 0 (not
# bit-identical with policy forward); no effect on training quality.
#
# Memory note (2026-06-22)
# ------------------------
# This phase uses accelerate_zero3_8gpu_joint.yaml (separate from the
# warmup yaml) which offloads BOTH optimizer state AND params to CPU:
#   * offload_optimizer_device: cpu  → DeepSpeedCPUAdam (frees ~12 GB/card)
#   * offload_param_device:     cpu  → params live on CPU, layer-gathered
#                                       on demand (frees ~14 GB/card)
# Why param offload is needed here but not in warmup
# --------------------------------------------------
# Joint adds 3 ~6500-token forwards on top of warmup's stack (rubric
# policy + rubric ref + score ref × 2). The 06-22 attempt with
# param=none SIGSEGV'd at step 43 inside partition_parameters.py:1171
# _all_gather_dtype with 89 allocator cache flushes already accumulated
# — same NCCL-cudaMalloc-fails-silently signature as the 06-21 saturation
# crash. param=cpu gives the ~14 GB headroom NCCL/flash-attn need to
# allocate without colliding. Cost: ~1.5s/step extra (host↔dev layer
# fetches), so a 120-step joint takes ~7-8h vs ~7h with param=none.
accelerate launch \
    --config_file "$ACCELERATE_CFG" \
    rm_training/stage2_grpo/train_grpo.py \
        --student_model        "$WARMUP_CKPT_RESOLVED" \
        --ref_model_path       "$BASE_MODEL_PATH" \
        --gold_rubrics         data/rm_train/gold_rubrics_18d.clean.jsonl \
        --output_dir           "$OUTPUT_DIR" \
        --num_generations      8 \
        --epochs               1 \
        --gold_rubric_warmup_epochs 0 \
        --log_every            1 \
        --save_every           15 \
        --val_every            0 \
        --val_full_every       0 \
        --val_max_samples_per_dim 10 \
        --learning_rate        2e-6 \
        --warmup_steps         30 \
        --pref_temperature     1.5 \
        --beta_kl              0.1 \
        --lambda_0             0.5 \
        --lambda_min           0.5 \
        --lambda_warmup_steps  0 \
        --lambda_decay_steps   0 \
        --beta_format          0.5 \
        --beta_format_min      0.1 \
        --beta_format_decay_steps 200 \
        --beta_format_schedule linear \
        --align_floor          0.3 \
        --score_sample \
        --score_pg_alpha_max   0.4 \
        --score_pg_start_step  20 \
        --score_pg_full_step   30 \
        --score_format_gate    0.55 \
        --lambda_bt            0.0 \
        --bt_temperature       1.5 \
        --lambda_pw            0.3 \
        --max_grad_norm        1.0 \
        --attn_implementation  flash_attention_2 \
        --gradient_checkpointing \
        --wandb_project        rewardverse-rm \
        --wandb_name           "$(basename "$OUTPUT_DIR")" \
        --wandb_tags           zero3 8gpu full-param phase-b joint gap-pw score-pg warmup-ref kl-fix \
        --wandb_sample_every   50 \
        --sample_log_every     "$SAMPLE_LOG_EVERY" \
        "$@"
