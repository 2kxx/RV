
"""rm_training.stage2_grpo.train_grpo
==================================
Main entry-point for Stage ② training.

Why we hand-roll the GRPO loop instead of using ``trl.GRPOTrainer``
-------------------------------------------------------------------
``trl.GRPOTrainer`` (v0.14) assumes:
    1. The reward function consumes one (prompt, completion) at a time.
    2. The student is a plain causal-LM with text-only forward.

Our setup violates BOTH:
    A. Each rollout needs THREE student forwards
       (rubric-gen + score y⁺ + score y⁻), with the rubric output of
       the first feeding into the next two prompts.
    B. The score forward is **video-multimodal** (Qwen2.5-VL).

Rather than wrestle a video-multimodal monkey-patch into TRL we keep
the GRPO math explicit here. The mathematical core is::

    advantage_g = (r_g - r.mean()) / (r.std() + ε)            # GRPO normalisation
    log_p       = teacher_force(prompt, sampled, ref=False)   # grad ON, LoRA ON
    log_p_ref   = teacher_force(prompt, sampled, ref=True)    # no grad, LoRA OFF
    loss_g      = - advantage_g · log_p.mean()  +  β_kl · (log_p - log_p_ref).mean()
    loss_total  = mean over G rollouts

Sampling is done under no_grad (``model.generate``) — we recover
grad-on log-probs by re-running a teacher-forcing forward with the
saved (prompt_ids, sampled_ids) pair. This is the standard PPO/GRPO
trick: sampling and learning use different forwards.

Only the **rubric-gen** step participates in the policy gradient. The
two score forwards are part of the reward environment (they read the
rubric, look at one video, emit numbers). Letting score tokens carry
gradient would teach the model to "fake the score" rather than
generate better rubrics, so we run those in greedy + no_grad.

For the MVP scaffold we ship two execution paths:
    • DUMMY  — no GPU, skips the LM update; verifies the data flow.
    • REAL   — Qwen2.5-VL-7B-Instruct + LoRA + accelerate.

Usage
-----
::

    # CPU dummy:
    RM_DUMMY=1 python rm_training/stage2_grpo/train_grpo.py \\
        --gold_rubrics data/rm_train/gold_rubrics_18d.jsonl \\
        --output_dir   ckpts/student_grpo_dummy \\
        --num_generations 8 --max_steps 4

    # Real run (single-node multi-GPU; accelerate handles dispatch):
    accelerate launch rm_training/stage2_grpo/train_grpo.py \\
        --student_model /group/.../Qwen2.5-VL-7B-Instruct \\
        --gold_rubrics  data/rm_train/gold_rubrics_18d.jsonl \\
        --embed_model   /group/.../bge-m3 \\
        --lora_config   rm_training/stage2_grpo/configs/lora_qwen25vl.yaml \\
        --output_dir    ckpts/student_grpo \\
        --num_generations 8 --max_steps 1000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rm_training.stage1_teacher.task_descriptions import (
    TaskSpec,
    load_task_specs,
)
from rm_training.stage2_grpo.data_collator   import GoldRubricDataset
from rm_training.stage2_grpo.eval_pointwise  import (
    DIMENSIONS_18,
    prefetch_pointwise_videos,
    run_pointwise_eval,
)
from rm_training.stage2_grpo.gold_align      import DummyEmbedder
from rm_training.stage2_grpo.reward_fn       import (
    BatchRewardConfig,
    BetaFormatScheduler,
    RolloutPair,
    compute_batch_rewards,
)
from rm_training.stage2_grpo.rubric_parser   import (
    build_full_from_scores,
    parse_rubric_only,
)
from rm_training.stage2_grpo.student_wrapper import (
    DimensionContext,
    StudentConfig,
    StudentWrapper,
    _summon_full_params_ctx,
    _param_full_numel as _param_full_numel_safe,
)
from rm_training.stage2_grpo.valid_filter    import ValidRatioTracker


# ---------------------------------------------------------------------------
# Accelerate helpers (single-process and DeepSpeed ZeRO-3 both supported).
#
# When run as plain ``python train_grpo.py`` (no accelerate), Accelerator()
# initialises a 1-process world and acts as a thin wrapper that does not
# change behaviour. When run as ``accelerate launch --config_file ...``,
# Accelerator() picks up DeepSpeed ZeRO-3 from the env vars set by the
# launcher and produces a real distributed runtime.
# ---------------------------------------------------------------------------
def _make_accelerator(*, num_generations: int = 1):
    """Lazy-construct an Accelerator only when actually needed.

    Why lazy: importing accelerate triggers a torch.distributed init that
    we don't want during dummy / CPU smoke tests. Returns ``None`` if
    accelerate isn't installed (the trainer then falls back to a single-
    process path that's identical to the pre-DDP code).

    ``num_generations`` is forwarded into the DeepSpeed plugin's
    ``gradient_accumulation_steps``. Each GRPO step issues G per-rollout
    backwards (one per rollout); we want DeepSpeed to treat all G as a
    single accumulation window so reduce-scatter runs ONCE per step
    rather than G times. See the ``policy_update`` boundary control for
    how the per-rollout flag is flipped.
    """
    try:
        from accelerate import Accelerator
    except ImportError:                                          # pragma: no cover
        print("[accel] accelerate not installed — running single-process.",
              file=sys.stderr)
        return None
    try:
        accelerator = Accelerator()
    except Exception as e:                                       # pragma: no cover
        print(f"[accel] Accelerator() failed: {e} — single-process fallback.",
              file=sys.stderr)
        return None

    # ── DeepSpeed-only patch: ``accelerate.prepare()`` requires
    # ``train_micro_batch_size_per_gpu`` to be set BEFORE prepare runs.
    # The accelerate yaml's ``deepspeed_config:`` block only accepts a
    # short whitelist of keys (zero_stage, offload_*, gradient_*, ...) —
    # any unknown key is silently dropped, so writing the field in yaml
    # has no effect. The supported escape hatch is to set it on the
    # plugin's runtime config, which we do here.
    #
    # Our trainer hand-rolls the data loop and never passes a dataloader
    # to ``prepare()``, so 1 instance / step / GPU is the right value
    # (G rollouts are an inner loop, not a batch axis).
    try:
        plugin = getattr(accelerator.state, "deepspeed_plugin", None)
        if plugin is not None:
            cfg = plugin.deepspeed_config
            if cfg.get("train_micro_batch_size_per_gpu", "auto") in ("auto", None):
                cfg["train_micro_batch_size_per_gpu"] = 1
            # Force gradient_accumulation_steps = G so DeepSpeed treats
            # the G per-rollout backwards as one accumulation window. The
            # yaml's value (1) was correct for batch-=1-per-step training
            # but wrong for our hand-rolled GRPO loop. Override unless
            # the user explicitly disables this by setting the env var
            # ``RM_KEEP_DS_ACCUM_STEPS=1``.
            #
            # NOTE: don't be tempted to set this to G/2 to "save grad
            # memory" without ALSO changing the boundary cadence in
            # ``policy_update`` (which currently flags boundary=True only
            # on the LAST usable rollout). DS's internal accumulation
            # counter and our explicit set_gradient_accumulation_boundary
            # calls have to agree, otherwise reduce-scatter fires at the
            # wrong time and grad scaling drifts. If you really want to
            # split the accumulation window, change BOTH places together.
            if (os.environ.get("RM_KEEP_DS_ACCUM_STEPS", "0") != "1"
                    and num_generations > 1):
                prev = cfg.get("gradient_accumulation_steps", 1)
                cfg["gradient_accumulation_steps"] = int(num_generations)
                if accelerator.is_main_process and prev != num_generations:
                    print(f"[accel] gradient_accumulation_steps: "
                          f"{prev} → {num_generations} (= G; one "
                          f"reduce-scatter per training step).",
                          file=sys.stderr)
            if cfg.get("train_batch_size", "auto") in ("auto", None):
                cfg["train_batch_size"] = (
                    cfg["train_micro_batch_size_per_gpu"]
                    * cfg.get("gradient_accumulation_steps", 1)
                    * accelerator.num_processes
                )
    except Exception as e:                                       # pragma: no cover
        print(f"[accel] DeepSpeed plugin patch skipped: {e}",
              file=sys.stderr)

    if accelerator.is_main_process:
        print(f"[accel] num_processes={accelerator.num_processes}  "
              f"distributed_type={accelerator.distributed_type}  "
              f"mixed_precision={accelerator.mixed_precision}  "
              f"local_rank={accelerator.local_process_index}",
              file=sys.stderr)
    return accelerator


def _is_main(accelerator) -> bool:
    """rank-0 / main process check that works when accelerator is None."""
    return accelerator is None or accelerator.is_main_process


def _mm_kwargs_from_meta(meta) -> dict | None:
    """Extract multi-modal tensors from a RubricRollout score-meta object.

    Returns a dict suitable to pass into student.compute_log_probs /
    compute_digit_logits as ``mm_kwargs``, or None when the meta carries
    no mm tensors (which happens for the rubric leg, where the prompt is
    text-only). See Patch 6 (2026-06-16): without these tensors the
    grad-on re-forward computes the LANGUAGE-ONLY marginal of π_θ rather
    than the multimodal π_θ that produced the sample, biasing the GRPO
    score-PG ratio and making BT-aux supervision vision-blind.
    """
    if meta is None:
        return None
    pvv = getattr(meta, "pixel_values_videos", None)
    vgt = getattr(meta, "video_grid_thw", None)
    if pvv is None or vgt is None:
        # Image-only rollouts could still have pixel_values + image_grid_thw,
        # but our pipeline produces video-only score forwards.
        pvi = getattr(meta, "pixel_values", None)
        igt = getattr(meta, "image_grid_thw", None)
        if pvi is None and igt is None:
            return None
        out: dict = {}
        if pvi is not None: out["pixel_values"] = pvi
        if igt is not None: out["image_grid_thw"] = igt
        return out
    out = {"pixel_values_videos": pvv, "video_grid_thw": vgt}
    spgt = getattr(meta, "second_per_grid_ts", None)
    if spgt is not None:
        out["second_per_grid_ts"] = spgt
    return out


# ---------------------------------------------------------------------------
# 1. Group-relative advantage.
# ---------------------------------------------------------------------------
def grpo_advantage(
    rewards: list[float],
    *,
    std_floor: float = 1e-2,
    zero_if_low_signal: bool = True,
    breakdowns: list[dict] | None = None,
) -> list[float]:
    """Standard GRPO normalisation::

        a_g = (r_g − mean(r)) / (max(std(r), std_floor) + ε)

    Notes
    -----
    1. ``std_floor`` clamps the denominator from below so that low-variance
       reward groups (all G rollouts on the same r_format ≈ same r_align)
       don't get advantages amplified by 1/std → ±20-something noise that
       shreds the policy.
    2. ``zero_if_low_signal``: when r_pref is dead (every rollout has
       r_pref == 0, i.e. valid=0/G) AND r_align is also degenerate (all
       equal across G), the only reward variance left is format-noise.
       Returning all-zero advantages in that case stops the policy from
       chasing format jitter and lets the rollout step accumulate KL
       safely (KL term stays well-defined). This is the documented "skip
       the PG update when the group is signal-dead" path that the v1
       version of grpo_advantage was missing — symptom: r_align collapses
       to 0 around step ~13 and never recovers.
    """
    if not rewards:
        return []
    if len(rewards) == 1:
        return [0.0]
    m = statistics.mean(rewards)
    try:
        s = statistics.stdev(rewards)
    except statistics.StatisticsError:
        s = 0.0

    # Signal-dead guard: when the only reward variance comes from format-
    # reward noise (r_pref dead, r_align identical), zero out advantages.
    if zero_if_low_signal and breakdowns is not None and len(breakdowns) == len(rewards):
        all_pref_zero  = all(float(b.get("r_pref", 0.0))  == 0.0 for b in breakdowns)
        align_vals     = [float(b.get("r_align", 0.0))           for b in breakdowns]
        align_spread   = max(align_vals) - min(align_vals)
        if all_pref_zero and align_spread < 1e-4:
            return [0.0 for _ in rewards]

    eps = 1e-6
    denom = max(float(s), float(std_floor)) + eps
    return [(r - m) / denom for r in rewards]


# ---------------------------------------------------------------------------
# 2. One training-step worker (one (q, y⁺, y⁻) instance).
# ---------------------------------------------------------------------------
def _ctx_from_specs(task_specs: dict[str, TaskSpec], dim: str) -> DimensionContext:
    """Lookup a DimensionContext for ``dim`` in the loaded TaskSpec map.

    Falls back to ``DimensionContext.generic(dim)`` when missing — keeps
    smoke runs (no yaml available) and ablation runs (custom dim slugs)
    working without crashing.
    """
    spec = task_specs.get(dim) if task_specs else None
    if spec is None:
        return DimensionContext.generic(dim)
    return DimensionContext(
        dimension        = spec.dimension,
        evalverse_path   = (
            f"{spec.evalverse_aspect} > {spec.evalverse_main}"
            if spec.evalverse_aspect and spec.evalverse_main
            else spec.dimension
        ),
        task_description = spec.task_description_short,
        # SCORE-side prompt uses the one-liner. ``oneline()`` falls back
        # to ``task_description_short`` if the yaml didn't yield a clean
        # one-line sentence — that fallback is still better than nothing,
        # but newer yamls all match the regex so we get the clean form.
        task_description_score = spec.oneline(),
        sub_aspects      = list(spec.sub_aspects),
    )


def step_one_instance(
    student:    StudentWrapper,
    instance:   dict,
    *,
    G:          int,
    cfg:        BatchRewardConfig,
    embedder,
    tracker:    ValidRatioTracker,
    beta_scheduler: "BetaFormatScheduler | None" = None,
    step:       int = 0,
    task_specs: dict[str, TaskSpec] | None = None,
    score_pg_active: bool = False,
    use_gold_warmup: bool = False,
    gold_by_dim: dict[str, list[dict]] | None = None,
) -> dict:
    """Roll out G times → reward → advantage → optional LM backward.

    Returns a metrics dict including ``rollout_meta`` (the per-rollout
    RubricRollout objects, real mode only) so the driver loop can run
    the policy-gradient update without re-sampling.

    ``score_pg_active``
    -------------------
    True when the score-side curriculum is past its start step: tells
    ``score_videos_batch`` to capture per-row (prompt_ids, sampled_ids)
    metadata so ``policy_update`` can compute log-probs on the score
    forward and back-prop the score-side advantage. Requires sampling
    on the score forward (``cfg.score_greedy=False``) — greedy mode
    silently downgrades to no meta capture inside the wrapper.

    ``use_gold_warmup``
    -------------------
    When True, skip ``student.generate_rubrics_with_meta_batch`` entirely
    and use the per-dim Stage ① gold rubric (from ``gold_by_dim``) for all
    G rollouts. This yields:
      * ``rubric_metas = [None]*G`` so policy_update's rubric-PG path
        skips for every rollout (no rubric-side gradient during warm-up).
      * ``rubric_texts = [<gold>]*G`` — all G rollouts share the same
        rubric, so the only variance across rollouts comes from the
        score-forward sampling. r_align ≈ 1.0 (saturated) for all G,
        which means r_align contributes no advantage variance — exactly
        what we want during warm-up: r_pref drives the gradient.
      * Combined with the caller forcing ``score_pg_active=True``, this
        gives the score head a full-strength gradient against a clean
        rubric to bootstrap on.
    """
    # NOTE: the v_arr_mvp pair-file field literally called "query" is the
    # *video generation prompt* — i.e. our caption — not the per-dim task
    # description. The latter is loaded via ``task_specs`` and injected
    # into the prompt as DimensionContext.
    caption = instance["query"]
    dim     = instance["dimension"]
    y_pos   = instance["y_pos"]
    y_neg   = instance["y_neg"]
    gold    = instance["gold_rubric"]
    dim_ctx = _ctx_from_specs(task_specs or {}, dim)

    rollouts: list[RolloutPair] = []
    rollout_meta: list = []        # type: list[RubricRollout | None]

    # ── (i) sample G rubrics in ONE generate call.
    # Why batched: the previous G× loop re-paid the full prefill cost
    # (~700 input tokens through 7B params) plus the ZeRO-3 gather +
    # grad-ckpt-pause ceremony G times for an identical prompt. With
    # ``num_return_sequences=G`` prefill runs once and only the suffix
    # decode is replicated. On 7B + H100 this is 5-7× faster than the
    # naive loop. Dummy mode also implements the batched path so smoke
    # tests follow the same code shape.
    #
    # ``use_gold_warmup`` short-circuits both the dummy and real branches:
    # we use the per-dim gold rubric verbatim for all G rollouts (no
    # student forward), and stamp ``rubric_metas`` to None so the
    # downstream rubric-PG path is skipped.
    if use_gold_warmup:
        gold_entries = (gold_by_dim or {}).get(dim) or []
        if not gold_entries:
            # Defensive: shouldn't happen since gold_by_dim is built from
            # the same dataset, but if it does we fall through to the
            # student path so the step doesn't degenerate to "score an
            # empty rubric".
            print(f"[gold-warmup] WARNING: dim={dim!r} has no gold rubric "
                  f"in gold_by_dim; falling back to student rubric for "
                  f"this step.", file=sys.stderr)
            use_gold_warmup = False  # local fall-through, doesn't leak out

    if use_gold_warmup:
        gold_text = json.dumps({"rubrics": gold_entries}, ensure_ascii=False)
        rubric_metas = [None] * G
        rubric_texts = [gold_text] * G
    elif student.is_dummy():
        rubric_metas = [None] * G
        rubric_texts = [
            student.generate_rubric(caption, dim_ctx=dim_ctx) for _ in range(G)
        ]
    else:
        rubric_metas = student.generate_rubrics_with_meta_batch(
            caption, dim_ctx=dim_ctx, num_return_sequences=G,
        )
        rubric_texts = [m.text for m in rubric_metas]

    # Parse all G rubrics up-front so we can pass rubric_themes_per to the
    # batched score. Each rubric may have a different number of themes
    # (parser truncates malformed JSON), so n_themes is per-row.
    parsed_rubrics = [parse_rubric_only(rt) for rt in rubric_texts]
    rubric_themes_per: list[list[str] | None] = [
        [r.theme for r in pr.rubrics] or None for pr in parsed_rubrics
    ]

    # ── (ii) cross-score all G rubrics on y⁺ in one batched generate,
    # then on y⁻ in another. The video is decoded once per side; the G
    # different prompts share the visual prefix and differ only in the
    # appended rubric body.
    if student.is_dummy():
        pos_results = [
            student.score_video(caption, y_pos, rubric_texts[g],
                                dim_ctx=dim_ctx,
                                rubric_themes=rubric_themes_per[g])
            for g in range(G)
        ]
        neg_results = [
            student.score_video(caption, y_neg, rubric_texts[g],
                                dim_ctx=dim_ctx,
                                rubric_themes=rubric_themes_per[g])
            for g in range(G)
        ]
        # Dummy mode never does score-side PG (no real sampling). Pad
        # results to the (text, scores, None) shape so the unpack below
        # is uniform.
        if score_pg_active:
            pos_results = [(t, s, None) for (t, s) in pos_results]
            neg_results = [(t, s, None) for (t, s) in neg_results]
    else:
        # ── Force greedy score forward when score-PG is OFF (2026-06-12).
        #
        # ``--score_sample`` (cfg.score_greedy=False) is required for the
        # score-side policy gradient because PG needs stochastic rollouts
        # to estimate advantage. But it's only USED when score_pg_active
        # is True (step >= --score_pg_start_step, default 50).
        #
        # In the warm-up window (steps 0..score_pg_start_step-1) the
        # trainer never reads the sampled score logits for PG, but cfg
        # still has score_greedy=False — so the score forward samples for
        # nothing, and pays the full cost:
        #   * sampling at score_max_new_tokens=32 budget early-EOSes
        #     ("theme", "theme_1:") and stub-fails ~25-30% of rows under
        #     student-generated rubrics (probe v8 variant=A: 27-35%
        #     finite, mirrors the [step] valid=2-3/8 you see live)
        #   * gold rubric in K stays 100% under the same sampling because
        #     the prompt-boundary autoregressive prior is tighter — but
        #     student rubrics' multi-line JSON + instruction-shaped tips
        #     drift the model into rubric continuation
        #
        # During warm-up there's no reason to pay this cost; flip to
        # greedy so valid_count and r_pref get a clean signal to
        # bootstrap on. Once score-PG kicks in at step_pg_start, we
        # restore the original cfg so the sampling rollouts are real.
        cfg_was_greedy = None
        if not score_pg_active and hasattr(student, "cfg"):
            cfg_was_greedy = getattr(student.cfg, "score_greedy", None)
            student.cfg.score_greedy = True
        try:
            pos_results = student.score_videos_batch(
                caption, y_pos, rubric_texts,
                dim_ctx=dim_ctx, rubric_themes_per=rubric_themes_per,
                return_meta=score_pg_active,
            )
            neg_results = student.score_videos_batch(
                caption, y_neg, rubric_texts,
                dim_ctx=dim_ctx, rubric_themes_per=rubric_themes_per,
                return_meta=score_pg_active,
            )
        finally:
            if cfg_was_greedy is not None and hasattr(student, "cfg"):
                student.cfg.score_greedy = cfg_was_greedy

    for g in range(G):
        rmeta         = rubric_metas[g]
        rubric_text   = rubric_texts[g]
        parsed_rubric = parsed_rubrics[g]
        if score_pg_active:
            score_pos_text, scores_pos, score_pos_meta = pos_results[g]
            score_neg_text, scores_neg, score_neg_meta = neg_results[g]
        else:
            score_pos_text, scores_pos = pos_results[g]
            score_neg_text, scores_neg = neg_results[g]
            score_pos_meta = None
            score_neg_meta = None

        # Build ParsedFull objects from the (rubric, scores) pair. The
        # rubric is locked to the generation step's output (P0-#2
        # fairness invariant); soft_logits already produced floats so we
        # don't re-parse text.
        parsed_pos = build_full_from_scores(
            parsed_rubric.rubrics, scores_pos, raw_text=score_pos_text,
        )
        parsed_neg = build_full_from_scores(
            parsed_rubric.rubrics, scores_neg, raw_text=score_neg_text,
        )

        rollouts.append(RolloutPair(
            student_rubric=parsed_rubric,
            student_full_pos=parsed_pos,
            student_full_neg=parsed_neg,
            score_pos_meta=score_pos_meta,
            score_neg_meta=score_neg_meta,
            meta={"rollout_index": g},
        ))
        rollout_meta.append(rmeta)

    # --- (iii) reward + group-relative advantage.
    # β(t) = scheduler.beta(step) when a scheduler is present, else cfg's
    # static β (back-compat).
    beta_t = (
        beta_scheduler.beta(step)
        if beta_scheduler is not None else cfg.beta_format
    )
    rewards, breakdowns = compute_batch_rewards(
        rollouts, gold,
        embedder=embedder, cfg=cfg, tracker=tracker,
        beta_format_t=beta_t,
        step=step,
    )
    # ── r_pref diagnostic: dump parsed scores when valid_count == 0 in the
    # cold-start window so we can see WHY S(y⁺) and S(y⁻) collapse to the
    # same value. Symptom in the prior run: first 8 steps had valid=0/8
    # despite healthy r_align. Likely culprits (one of):
    #   * score forward decodes the same digit token for every theme
    #   * format_score returns weighted-mean over identical scores
    #   * fairness lock pulled in a malformed rubric → both sides 0
    # The dump writes ONE jsonl row per call (rank-0 only) under
    # ``$RM_PREF_DIAG_DIR`` (default: $output_dir/pref_diag/), so we can
    # diff S(y⁺)/S(y⁻) and see which case applies. Gated by env so it
    # doesn't run by default.
    diag_dir = os.environ.get("RM_PREF_DIAG_DIR", "")
    diag_max_steps = int(os.environ.get("RM_PREF_DIAG_STEPS", "0"))
    if diag_dir and step < diag_max_steps:
        try:
            from pathlib import Path as _P
            _P(diag_dir).mkdir(parents=True, exist_ok=True)
            valid_n = sum(1 for b in breakdowns if b["valid"])
            row = {
                "step": step,
                "dim": instance.get("dimension"),
                "pair_id": instance.get("pair_id"),
                "valid_count": valid_n,
                "G": G,
                "rollouts": [
                    {
                        "g": g,
                        "s_pos": float(ro.s_pos) if ro.s_pos is not None and math.isfinite(ro.s_pos) else None,
                        "s_neg": float(ro.s_neg) if ro.s_neg is not None and math.isfinite(ro.s_neg) else None,
                        "delta": (
                            float(ro.s_pos - ro.s_neg)
                            if (ro.s_pos is not None and ro.s_neg is not None
                                and math.isfinite(ro.s_pos) and math.isfinite(ro.s_neg))
                            else None
                        ),
                        "n_rubrics_locked": len(ro.student_rubric.rubrics),
                        "scores_pos": [float(x) if x is not None and math.isfinite(x) else None
                                       for x in (ro.student_full_pos.scores or [])],
                        "scores_neg": [float(x) if x is not None and math.isfinite(x) else None
                                       for x in (ro.student_full_neg.scores or [])],
                        "rubric_themes": [r.theme for r in ro.student_rubric.rubrics],
                        # ── Score raw text dump (2026-06-13 added).
                        # When a row's s_pos / s_neg is None, the cause is
                        # one of: stub ("theme"), no-anchor ("themespeech ..."),
                        # or NaN logits in some position. Without the raw
                        # text we can't tell which. Trim to 200 chars to
                        # bound jsonl size; replace newlines so each row
                        # stays a single line.
                        "score_pos_raw": (
                            (getattr(ro.student_full_pos, "raw_text", "") or "")[:200]
                            .replace("\n", " ⏎ ")
                        ),
                        "score_neg_raw": (
                            (getattr(ro.student_full_neg, "raw_text", "") or "")[:200]
                            .replace("\n", " ⏎ ")
                        ),
                    }
                    for g, ro in enumerate(rollouts)
                ],
            }
            fpath = _P(diag_dir) / f"pref_diag_step{step:05d}.jsonl"
            with fpath.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False, default=str))
                fh.write("\n")
        except Exception as _e:
            print(f"[pref_diag] dump failed at step {step}: {_e}", file=sys.stderr)
    advantages = grpo_advantage(rewards, breakdowns=breakdowns)

    # --- (iv) telemetry.
    pref_mean   = statistics.mean(b["r_pref"]    for b in breakdowns)
    align_mean  = statistics.mean(b["r_align"]   for b in breakdowns)
    fmt_mean    = statistics.mean(b["r_format"]  for b in breakdowns)
    valid_count = sum(1 for b in breakdowns if b["valid"])
    return {
        "pair_id":      instance["pair_id"],
        "dimension":    instance["dimension"],
        "G":            G,
        "rewards":      rewards,
        "advantages":   advantages,
        "rollout_meta": rollout_meta,
        "rollouts":     rollouts,        # for wandb sample dump
        "breakdowns":   breakdowns,      # per-rollout reward components
        "caption":      caption,         # raw video-gen prompt for the sample
        # Score-gap supervision targets (in raw 0-1 evalverse units; the
        # consumer applies the ×4 scale to match the model's 1-5 digit
        # range). May be None when the source pair file lacked them.
        "s_pos_target": instance.get("s_pos_target"),
        "s_neg_target": instance.get("s_neg_target"),
        "pref_mean":    pref_mean,
        "align_mean":   align_mean,
        "fmt_mean":     fmt_mean,
        "valid_count":  valid_count,
        "lambda_t":     breakdowns[0]["lambda_t"],
        "beta_t":       breakdowns[0]["beta_format"],
        "valid_ratio_ema": tracker.current(),
    }


# ---------------------------------------------------------------------------
# 3. Policy-gradient update over G rollouts of one instance.
# ---------------------------------------------------------------------------
def policy_update(
    student:     "StudentWrapper",
    rollout_meta: list,
    advantages:  list[float],
    *,
    optimizer,
    beta_kl:     float,
    max_grad_norm: float = 1.0,
    accelerator = None,
    rollouts:           list | None = None,
    score_pg_alpha:     float = 0.0,
    score_format_gate:  float = 0.65,
    breakdowns:         list[dict] | None = None,
    # ── BT-aux (added 2026-06-16): supervised BT loss directly on the
    # 5-digit logits at colon-anchored positions of the score scaffold.
    # ``lambda_bt > 0`` enables it; ``bt_temperature`` is τ in
    # ``L_BT = -log σ(τ·(S⁺ − S⁻))``. Pass ``rubric_weights_per_rollout``
    # so the trainer can produce a weighted aggregate score (same lock as
    # r_pref). Pass-through gating: BT-aux only fires when score-side
    # sampling meta exists AND format gate passed (same as score-PG).
    lambda_bt:          float = 0.0,
    bt_temperature:     float = 3.0,
    # ── Score-gap supervision (added 2026-06-18): replaces BT-aux with a
    # rubric-invariant L1 between the model's S⁺ − S⁻ and the annotated
    # gap. Reasoning:
    #   * Pointwise alignment (MSE on S itself) is unsound — the student's
    #     5 themes are dynamically generated and don't share an axis with
    #     evalverse's sub_keys, so the absolute scale is meaningless.
    #   * The differences are scale-free: with a fixed student rubric,
    #     S⁺ − S⁻ measures "how much better is y⁺ under YOUR rubric",
    #     which is directly comparable to evalverse's winner − loser.
    #   * BT pushes margin to ∞ regardless of annotated magnitude;
    #     gap-L1 says "match the magnitude exactly", which kills the
    #     mode-collapse-to-extremes failure observed in warmup-2.
    # ``s_pos_target`` / ``s_neg_target`` are in evalverse's 0-1 range;
    # we scale by 4 to match S∈[1,5]. Gating mirrors BT-aux.
    lambda_pw:          float = 0.0,
    s_pos_target:       float | None = None,
    s_neg_target:       float | None = None,
) -> dict:
    """Run one optimiser step over G rollouts of a single training instance.

    Single-summon, per-rollout backward
    -----------------------------------
    Two constraints simultaneously:

    1. **Correctness**: under sharded backends (FSDP / ZeRO-3),
       repeatedly entering ``summon_full_params(writeback=True)`` within
       one optimiser step corrupts the gathered parameters — only the
       first rollout's log-probs are valid; rollouts 2..G return garbage.
       Empirically: at step 0 (policy == ref) we should see KL ≈ 0 for
       every rollout, but the buggy run shows KL ≈ 0 for rollout 0 and
       KL ≈ −15.5 for rollouts 1..7 — exactly this corruption.

       Fix: **open ONE summon scope around the whole G-rollout loop**
       and tell ``compute_log_probs`` to skip its own internal summon.

    2. **Memory**: a fully-batched forward of G=8 rollouts simultaneously
       (one big (8, P+T) tensor) holds 8× the activation graph alive,
       which OOMs 80GB H100s once the 7B params + optimiser state are
       there. The original "per-rollout backward releases the graph
       immediately" pattern is correct on memory.

       Fix: **keep the per-rollout backward loop** so peak activation
       memory is bounded by ONE rollout. Grads accumulate into the
       optimiser's param tensors; backward is linear in the loss so the
       optimiser sees ``∇ Σ_g L_g / G`` regardless of whether we batched
       the loss or summed per-rollout backwards.

    Score-side PG curriculum (optional)
    -----------------------------------
    When ``score_pg_alpha > 0`` AND ``rollouts`` carries score metadata
    (``score_pos_meta`` / ``score_neg_meta``), we ADD the score-forward
    log-probs to the per-rollout PG signal:

        logp_total = logp_rubric + α · (logp_score_pos + logp_score_neg)
        loss       = (-adv · logp_total + β_kl · kl_rubric) / n_usable

    Score-side KL (added 2026-06-14)
    --------------------------------
    The score forward used to skip its KL term entirely. The original
    rationale ("ref distribution is identical when LoRA is off") only
    holds at step 0; once any backward has run, policy weights diverge
    from the frozen ref and the score forward's distribution diverges
    too. Without a KL anchor on the score leg, the score head is free
    to drift into a degenerate mode (e.g. always emit ``5/5/5/5/5`` or
    its inverse) that maximises ``pref_advantage`` while ignoring the
    video — observed empirically as ``loss`` magnitude exploding from
    ±0.5 (warm-up early) to ±27 (warm-up step 59) with no KL counter-
    pressure to slow it.

    The fix runs two extra ref forwards per score-active rollout (pos
    and neg score scaffolds), computes the same Schulman k3 token-level
    estimator used for the rubric leg, and folds the average into the
    composite loss:

        kl_score = 0.5 · (mean_t[k3(Δ_pos_t)] + mean_t[k3(Δ_neg_t)])
        loss     = (pg_loss + β_kl · (kl_rubric + α · kl_score)) / n_usable

    The score-side KL is scaled by the same ``score_pg_alpha`` curriculum
    coefficient as the score-PG term, so warm-up's α=alpha_max gets full
    KL pressure on the score leg while the post-warmup linear ramp
    (start_step → full_step) ramps both PG and KL together. Cost: each
    score-active rollout pays one extra 6000-token ref forward × 2 sides
    (~1-2 s on H100). Cheap insurance against silent score-head drift.

    With score_pg_alpha == 0 (default) or rollouts == None, this is the
    legacy rubric-only PG.

    Combined:

        with summon_full_params(writeback=True):       # gather ONCE
            for g in 1..G:
                logp_g     = forward(prompt_g, sampled_g)         # (T,)
                logp_ref_g = forward_ref(prompt_g, sampled_g)     # (T,)
                kl_g       = mean_t[exp(Δ_g,t) − 1 − Δ_g,t]        # token-level k3
                L_g        = (-adv_g · mean(logp_g) + β_kl · kl_g) / G
                L_g.backward()                          # graph released
        # exit scope → ONE reduce-scatter of accumulated grads
        optimizer.step()

    Notes
    -----
    * The ref forward goes through ``compute_log_probs(ref=True)``. The
      ref model is unsharded (one full copy per rank, set up by the
      trainer), so its inner summon is a NullCtx anyway — but for safety
      we also pass ``_skip_summon=True`` to make the contract explicit.
    * KL is the **token-level** Schulman k3 estimator (DeepSeek-Math GRPO):
      ``KL ≈ mean_t[ exp(Δ_t) − 1 − Δ_t ]`` where
      ``Δ_t = log π_θ(s_t|·) − log π_ref(s_t|·)``. Non-negative, equals
      0 iff policy matches ref. Computing per-token then averaging (rather
      than ``exp(mean Δ) − 1 − mean Δ``) avoids the Jensen under-estimate
      of the previous implementation and matches the GRPO paper exactly,
      so β_kl=0.1 acts as the paper's 0.1 with no surprises.
    """
    out = {"loss": 0.0, "pg_loss": 0.0, "kl": 0.0, "skipped": 0, "applied": 0,
           "score_pg_applied": 0, "bt_aux_applied": 0, "bt_aux": 0.0,
           "pw_aux_applied": 0, "pw_aux": 0.0}
    if optimizer is None or student.is_dummy() or not rollout_meta:
        out["skipped"] = len(rollout_meta or [])
        return out

    import torch

    # ── Decide which rollouts will contribute gradient.
    # Two PG legs feed into ``loss``:
    #   * rubric-PG (always-on): needs ``rollout_meta[i]`` with non-empty
    #     ``sampled_ids``.
    #   * score-PG (curriculum): needs ``score_pg_alpha > 0`` AND
    #     ``rollouts[i]`` carrying score_{pos,neg}_meta with non-empty
    #     ``sampled_ids``.
    # During the gold-rubric warm-up, rubric_metas is [None]*G so rubric-PG
    # contributes nothing — but score-PG can still drive the optimiser.
    # Skip the optimiser step ONLY when neither leg has anything to do.
    def _has_score_meta(idx: int) -> bool:
        if rollouts is None or idx >= len(rollouts):
            return False
        ro = rollouts[idx]
        m_pos = getattr(ro, "score_pos_meta", None)
        m_neg = getattr(ro, "score_neg_meta", None)
        if m_pos is None or m_neg is None:
            return False
        sp = getattr(m_pos, "sampled_ids", None)
        sn = getattr(m_neg, "sampled_ids", None)
        return (sp is not None and sp.numel() > 0
                and sn is not None and sn.numel() > 0)

    score_pg_eligible = (
        score_pg_alpha > 0.0
        and rollouts is not None
        and any(_has_score_meta(i) for i in range(len(rollout_meta)))
    )
    rubric_pg_eligible = any(
        m is not None
        and getattr(m, "sampled_ids", None) is not None
        and m.sampled_ids.numel() > 0
        for m in rollout_meta
    )
    if not rubric_pg_eligible and not score_pg_eligible:
        out["skipped"] = len(rollout_meta)
        return out

    if student.model is not None:
        student.model.train()

    # ── Per-rollout eligibility: a rollout contributes to the loss if EITHER
    # leg is eligible (rubric-PG or score-PG). During the gold-rubric warm-up
    # only score-PG is eligible; in normal training rubric-PG is always
    # eligible (and score-PG when α>0 + format-gate passes).
    def _rubric_ok(idx: int) -> bool:
        if idx >= len(rollout_meta):
            return False
        m = rollout_meta[idx]
        return (m is not None
                and getattr(m, "sampled_ids", None) is not None
                and m.sampled_ids.numel() > 0)

    eligible = [
        (_rubric_ok(i) or _has_score_meta(i))
        for i in range(len(rollout_meta))
    ]

    # ── Score-side per-rollout format gate.
    # Pre-compute the gate decision here (instead of inside the loop) so
    # we can ALL-REDUCE both ``eligible`` and ``score_gate_ok`` across
    # ranks before the forward loop starts. The reason this matters:
    #
    # ZeRO-3 collective alignment
    # ---------------------------
    # Each rollout's policy forward triggers N_layers per-layer all_gather
    # collectives, plus the eventual reduce_scatter on backward. NCCL
    # matches collectives by call-order, not by tag, so every rank MUST
    # issue the same sequence of collectives in the same order — otherwise
    # rank A's forward k+1 reduce_scatter waits for rank B's forward k
    # reduce_scatter and the watchdog kills the job at 600 s.
    #
    # In phase B each rank samples its own rubric, so per-rank
    # ``fmt_rubric`` / ``fmt_pos`` / ``fmt_neg`` differ → ``score_gate_ok``
    # differs → some ranks run the 4 extra score-side log-prob forwards
    # while others skip them, giving each rank a different collective
    # count. Symptom (observed 2026-06-15): rank 5 ``last_enqueued=132,
    # last_completed=129`` after only one optimiser step.
    #
    # Fix: take the per-rollout MIN across all ranks, so a rollout is
    # "eligible" / "score-PG firing" only when EVERY rank agrees. The
    # cost is a slight pessimism early in training (one bad rubric on
    # any rank drops that rollout for everyone); the benefit is that
    # per-rollout collective counts are guaranteed identical across
    # ranks. DP-averaging semantics are also restored — previously some
    # ranks contributed a "with score-PG" gradient and others a "rubric-
    # only" gradient to the same reduce_scatter, which was silently
    # producing wrong DP averages even when NCCL didn't time out.
    score_gate_ok = [False] * len(rollout_meta)
    if (score_pg_alpha > 0.0 and rollouts is not None
            and breakdowns is not None):
        for i in range(min(len(rollout_meta), len(rollouts), len(breakdowns))):
            ro = rollouts[i]
            bd = breakdowns[i]
            fmt_rub  = float(bd.get("fmt_rubric", 0.0))
            fmt_pos_ = float(bd.get("fmt_pos",    0.0))
            fmt_neg_ = float(bd.get("fmt_neg",    0.0))
            local_gate = (
                fmt_rub  >= score_format_gate
                and fmt_pos_ >= score_format_gate
                and fmt_neg_ >= score_format_gate
                and ro.score_pos_meta is not None
                and ro.score_neg_meta is not None
                and getattr(ro.score_pos_meta, "sampled_ids", None) is not None
                and getattr(ro.score_neg_meta, "sampled_ids", None) is not None
                and ro.score_pos_meta.sampled_ids.numel() > 0
                and ro.score_neg_meta.sampled_ids.numel() > 0
            )
            score_gate_ok[i] = bool(local_gate)

    # ── Cross-rank intersection of (eligible, score_gate_ok).
    # MIN over an int32 tensor of {0,1} gives logical AND. We pack both
    # masks into one tensor of shape (G, 2) so it's one collective, not
    # two.
    if (accelerator is not None
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()):
        try:
            device = (student.model.device
                      if student.model is not None
                      else torch.device(f"cuda:{accelerator.local_process_index}"))
            G_n = len(rollout_meta)
            mask_t = torch.zeros((G_n, 2), dtype=torch.int32, device=device)
            for i in range(G_n):
                mask_t[i, 0] = 1 if eligible[i] else 0
                mask_t[i, 1] = 1 if score_gate_ok[i] else 0
            torch.distributed.all_reduce(
                mask_t, op=torch.distributed.ReduceOp.MIN,
            )
            eligible      = [bool(int(mask_t[i, 0].item())) for i in range(G_n)]
            score_gate_ok = [bool(int(mask_t[i, 1].item())) for i in range(G_n)]
        except Exception as _e:
            # Graceful fallback: if the all_reduce fails for any reason
            # (single-process, weird backend), proceed with the local masks
            # — single-rank runs don't have the divergence problem anyway.
            print(f"[policy_update] mask all_reduce skipped: {_e}",
                  file=sys.stderr)

    # ── Warm-up eligibility alignment (added 2026-06-19).
    # Under gold-rubric warm-up rubric_meta[i] is None for every rollout, so
    # the only grad-bearing branch is score-PG / BT / pw — all three gated on
    # ``score_gate_ok[i]``. The original ``eligible[i] = _rubric_ok or
    # _has_score_meta`` collapses to True for every rollout in warm-up and
    # waves through gate-failing rollouts; their per-rollout loss then
    # degenerates to a 0-tensor with no grad_fn and ``accelerator.backward``
    # raises ``element 0 of tensors does not require grad and does not have
    # a grad_fn`` (observed at warm-up step 37, 2026-06-18 run).
    #
    # Tighten the mask: when this rollout has no rubric leg, require
    # ``score_gate_ok``. Because both masks went through the same cross-rank
    # MIN above, every rank takes the same decision and the collective
    # sequence stays aligned. Mixed-state batches (some pass / some fail) are
    # now handled correctly — passing rollouts backward as before, failing
    # rollouts counted into ``skipped`` and never entering the loop.
    def _rubric_empty(idx: int) -> bool:
        if idx >= len(rollout_meta):
            return True
        m = rollout_meta[idx]
        return (m is None
                or getattr(m, "sampled_ids", None) is None
                or m.sampled_ids.numel() == 0)
    eligible = [
        (score_gate_ok[i] if _rubric_empty(i) else eligible[i])
        for i in range(len(rollout_meta))
    ]

    n_usable = sum(eligible)
    if n_usable == 0:
        out["skipped"] = len(rollout_meta)
        if student.model is not None:
            student.model.eval()
        return out

    # ── Defensive early-return: warm-up + score-gate fully closed.
    #
    # In gold-rubric warm-up rubric_meta[i] is None for every rollout, so the
    # only branch that can produce a grad-bearing tensor is the score-PG /
    # BT-aux block, both gated on ``score_gate_ok[i]``. If every rank
    # collectively fails the format gate (cross-rank MIN over an int32 mask
    # earlier in this function), the per-rollout loss collapses to a bare
    # zero-tensor with no grad_fn — ``accelerator.backward(loss)`` then
    # raises ``RuntimeError: element 0 of tensors does not require grad and
    # does not have a grad_fn``. Detect that state up front, skip the step
    # cleanly, and let the next instance try again.
    warmup_no_rubric = all(
        (m is None or getattr(m, "sampled_ids", None) is None
         or m.sampled_ids.numel() == 0)
        for m in rollout_meta
    )
    if warmup_no_rubric and not any(score_gate_ok):
        optimizer.zero_grad(set_to_none=True)
        out["skipped"] = len(rollout_meta)
        if student.model is not None:
            student.model.eval()
        try:
            rank_dbg = int(os.environ.get("LOCAL_RANK", "0"))
        except Exception:
            rank_dbg = 0
        if rank_dbg == 0:
            print(
                "[policy_update] all rollouts gated out (warm-up + every rank "
                "below score_format_gate). Skipping backward this step; lower "
                "--score_format_gate or reduce --score_pg_alpha_max if this "
                "persists.",
                file=sys.stderr, flush=True,
            )
        return out

    optimizer.zero_grad(set_to_none=True)
    pg_values:   list[float] = []
    kl_values:   list[float] = []
    kl_score_values: list[float] = []
    bt_loss_values: list[float] = []
    pw_loss_values: list[float] = []
    loss_values: list[float] = []

    # Under ZeRO-3 the engine's per-layer pre/post-forward hooks gather
    # and scatter automatically during forward + backward, so the
    # ``writeback=True`` branch of ``_summon_full_params_ctx`` resolves
    # to a no-op. The historical FSDP path used a single explicit
    # ``summon_full_params(writeback=True)`` here; we keep the call site
    # so the trainer interface stays uniform across backends and so the
    # ``_skip_summon=True`` baton in compute_log_probs continues to make
    # sense (it tells the inner read-only ref forward to skip its own
    # gather, which is still relevant on ZeRO-3 when the read scope was
    # opened higher up).
    policy_summon = _summon_full_params_ctx(student.model, writeback=True)

    # ── DeepSpeed gradient-accumulation boundary control.
    # ``accelerate.backward`` calls ``DeepSpeedEngine.backward`` which by
    # default runs a full reduce-scatter at the end of EVERY backward
    # call. With G rollouts that's G reduce-scatters per training step —
    # the dominant communication cost on this run.
    #
    # DeepSpeed exposes ``set_gradient_accumulation_boundary(False)`` to
    # mark "this backward is NOT a sync boundary; just locally accumulate
    # grads into the param shards and skip the reduce-scatter". Combined
    # with ``gradient_accumulation_steps: G`` in the yaml (so the engine
    # knows when the boundary IS), we get exactly ONE reduce-scatter per
    # step — the same wall-clock as a single G-batched backward but with
    # the per-rollout activation memory of the original loop.
    #
    # The flag must be reset to True on the FINAL rollout so the engine
    # actually syncs and ``optimizer.step()`` sees the reduced grad. We
    # also force True on the very last iteration of the loop regardless
    # of where it landed, as a defensive measure against early ``continue``
    # from a stray skipped rollout.
    ds_engine = None
    cand = student.model
    if accelerator is not None and cand is not None and \
            hasattr(cand, "set_gradient_accumulation_boundary"):
        ds_engine = cand
    # Pre-compute which rollout is the LAST usable one so we can flag it
    # as the boundary regardless of skipped rows ahead of it. "Usable" is
    # the unified eligibility (rubric-PG OR score-PG) computed above.
    usable_indices = [i for i, e in enumerate(eligible) if e]
    last_usable = usable_indices[-1] if usable_indices else -1

    with policy_summon:
        for i, (adv, meta) in enumerate(zip(advantages, rollout_meta)):
            if not eligible[i]:
                out["skipped"] += 1
                continue

            rubric_ok_i = _rubric_ok(i)
            score_ok_i  = _has_score_meta(i)

            is_last = (i == last_usable)
            if ds_engine is not None:
                # False on inner backwards (no reduce-scatter), True on
                # the last so the engine flushes accumulated grads.
                ds_engine.set_gradient_accumulation_boundary(bool(is_last))

            # Per-rollout memory diag — gated, off by default. Set
            # RM_MEM_DIAG=1 to print alloc/peak right before forward and
            # after backward of each rollout. Helps localise WHICH rollout
            # blows the budget when the OOM is intermittent.
            _mem_diag = os.environ.get("RM_MEM_DIAG", "0") == "1"
            if _mem_diag and torch.cuda.is_available():
                a0 = torch.cuda.memory_allocated() / 2**30
                p0 = torch.cuda.max_memory_allocated() / 2**30
                print(f"[mem rollout={i}/{len(usable_indices)} "
                      f"is_last={is_last}] pre-fwd alloc={a0:.1f}G "
                      f"peak={p0:.1f}G", file=sys.stderr, flush=True)

            # ── Rubric-side log-prob + KL.
            # Skip the rubric forward entirely when this rollout has no
            # rubric meta (the gold-rubric warm-up case). KL is then 0
            # for the rubric leg; the score leg below carries all the
            # gradient. This is correct because rubric_metas[i] is None
            # exactly when the rubric came from the dataset (gold), not
            # from the student's policy — there is no policy emission to
            # KL-regularise against ref.
            if rubric_ok_i:
                with torch.enable_grad():
                    logp = student.compute_log_probs(
                        meta.prompt_ids, meta.sampled_ids,
                        ref=False, _skip_summon=True,
                    )
                logp_ref = student.compute_log_probs(
                    meta.prompt_ids, meta.sampled_ids,
                    ref=True, _skip_summon=True,
                ).detach()

                logp_mean   = logp.mean()
                logp_ref_mn = logp_ref.mean()
                # ── Token-level Schulman k3 KL estimator (DeepSeek-Math GRPO §3.2).
                # ``logp`` and ``logp_ref`` are both shape (T,) per-token tensors.
                # Compute the k3 estimator per token, then mean over T:
                #     KL ≈ (1/T) Σ_t [exp(Δ_t) - 1 - Δ_t]   where Δ_t = logp_t − logp_ref_t
                # This matches the GRPO-paper formulation. The previous
                # implementation collapsed to ``exp(mean Δ) − 1 − mean Δ`` which is
                # ≤ token-level KL by Jensen, systematically under-estimating the
                # penalty (β_kl had to be tuned higher to compensate). Using the
                # correct estimator here makes β_kl=0.1 act exactly as the paper's
                # 0.1, with no surprises in cross-impl ports.
                log_ratio_t = logp - logp_ref          # (T,) per-token
                # ── log_ratio clip for numerical safety (PPO/GRPO standard).
                # Same RM_KL_LOG_RATIO_CLIP env var as the score leg below.
                _kl_clip_r = float(os.environ.get("RM_KL_LOG_RATIO_CLIP", "10.0"))
                log_ratio_t = log_ratio_t.clamp(-_kl_clip_r, _kl_clip_r)
                kl_per_t    = log_ratio_t.exp() - 1.0 - log_ratio_t   # (T,)
                kl_value    = kl_per_t.mean()          # scalar
                # Sequence-level Δ retained for telemetry / sanity prints elsewhere.
                log_ratio   = logp_mean - logp_ref_mn
            else:
                # Warm-up rollout: rubric is held to gold, so no rubric-side
                # log-prob and no rubric-side KL. logp_mean is initialised
                # to a 0-tensor on the policy's device so the score-PG
                # branch below can `+=` into it; KL is a 0-tensor so the
                # composite loss expression stays well-typed.
                _dev = (student.model.device
                        if student.model is not None
                        else torch.device("cpu"))
                logp_mean = torch.zeros((), device=_dev)
                kl_value  = torch.zeros((), device=_dev)

            # ── Optional score-side PG (curriculum gated on caller).
            #
            # When ``score_pg_alpha > 0`` AND this rollout has score-side
            # sampling metadata AND the rollout's format passes the hard
            # gate, fold the score forward's mean log-prob into the
            # advantage signal:
            #
            #     logp_total = logp_rubric + α · (logp_score_pos + logp_score_neg)
            #
            # Why hard format gate
            # --------------------
            # If fmt_rubric < gate, the rubric contained malformed paths /
            # weights / etc., and the fairness lock pulled that into the
            # score forward's prompt scaffold. Training the score forward
            # against a corrupt rubric teaches it to operate in noise —
            # opposite of what we want. The gate ensures score-side PG
            # only fires when the upstream rubric is schema-clean.
            score_logp_added = False
            kl_score_value = None    # set below when score-PG fires
            # ``score_gate_ok`` was computed and ALL-REDUCEd across ranks
            # before the loop began (see the "Cross-rank intersection"
            # block earlier in this function), so all ranks see the same
            # decision here — collective sequence stays aligned.
            if (score_pg_alpha > 0.0 and rollouts is not None
                    and i < len(rollouts) and score_gate_ok[i]):
                ro = rollouts[i]
                with torch.enable_grad():
                    # detach_prompt=True (Trick A): score forwards have
                    # ~6000-token video prefixes that dominate activation
                    # memory. Detach the prompt prefill (no_grad +
                    # use_cache) and only run the ~32-token score
                    # scaffold tail under autograd. Mathematically
                    # equivalent to the legacy single-forward path —
                    # see _compute_log_probs_detach_prompt docstring
                    # for the slice-vs-cache equivalence proof. This
                    # is the same simplification VeRL / TRL / OpenRLHF
                    # / EasyR1 / DeepSpeed-Chat all make: prompt is a
                    # condition for log π_θ(generated | prompt), not
                    # a target, so ∇_θ through the prompt forward is
                    # not required for valid PG.
                    # ── Patch 6d (2026-06-16): build per-side mm_kwargs
                    # from the score meta. The score forward's prompt
                    # contains video placeholder tokens that MUST be
                    # accompanied by pixel_values_videos / video_grid_thw
                    # in any grad-on re-forward; otherwise the vision
                    # tower is skipped and the resulting logp is the
                    # language-only marginal of π_θ rather than the
                    # multimodal π_θ that produced the sample. Both
                    # logp_pos / logp_neg (policy) AND logp_ref_pos /
                    # logp_ref_neg below take the same mm_kwargs so
                    # ref vs policy stay numerically aligned and
                    # KL_s reflects real divergence rather than a
                    # vision-on vs vision-off mismatch.
                    mm_pos = _mm_kwargs_from_meta(ro.score_pos_meta)
                    mm_neg = _mm_kwargs_from_meta(ro.score_neg_meta)
                    logp_pos = student.compute_log_probs(
                        ro.score_pos_meta.prompt_ids,
                        ro.score_pos_meta.sampled_ids,
                        ref=False, _skip_summon=True,
                        detach_prompt=True,
                        mm_kwargs=mm_pos,
                    )
                    logp_neg = student.compute_log_probs(
                        ro.score_neg_meta.prompt_ids,
                        ro.score_neg_meta.sampled_ids,
                        ref=False, _skip_summon=True,
                        detach_prompt=True,
                        mm_kwargs=mm_neg,
                    )
                # ── Score-side KL (A1, added 2026-06-14; refined 2026-06-16).
                # Two extra ref forwards over the same score scaffolds.
                # Token-level Schulman k3 matching the rubric KL.
                # Weighted by score_pg_alpha so curriculum off → 0.
                #
                # IMPORTANT (Patch 5c, 2026-06-16): the ref forwards must
                # take the SAME path as the policy forward above —
                # ``detach_prompt=True`` — otherwise the policy uses
                # Trick A's two-stage forward (no_grad prompt prefill +
                # KV-cached tail in eval mode) while the ref uses the
                # legacy single-forward, and the bf16 numerical asymmetry
                # makes step-0 KL_s = 4–22 nats of pure path noise even
                # when student == ref. The Schulman k3 estimator cannot
                # distinguish that noise from a real KL, so the ref-side
                # gradient through ``β_kl · α · KL_s`` pushes the score
                # head in a meaningless direction. With both forwards on
                # the same path, KL_s ≈ 0 at step 0 by construction.
                #
                # Memory: detach_prompt for ref adds one KV-cache
                # allocation (~tens of MB for our P≈6000) but skips the
                # full-sequence activation memory the legacy path was
                # holding, so net memory is comparable or slightly lower.
                logp_ref_pos = student.compute_log_probs(
                    ro.score_pos_meta.prompt_ids,
                    ro.score_pos_meta.sampled_ids,
                    ref=True, _skip_summon=True,
                    detach_prompt=True,
                    mm_kwargs=mm_pos,
                ).detach()
                logp_ref_neg = student.compute_log_probs(
                    ro.score_neg_meta.prompt_ids,
                    ro.score_neg_meta.sampled_ids,
                    ref=True, _skip_summon=True,
                    detach_prompt=True,
                    mm_kwargs=mm_neg,
                ).detach()
                # ── Token-level Schulman k3 KL with log_ratio CLIP.
                #
                # Standard PPO/GRPO numerical safety: clamp(log_ratio,
                # -RM_KL_LOG_RATIO_CLIP, +RM_KL_LOG_RATIO_CLIP) before
                # the exp() so a single token with a path-noise-induced
                # large |log_ratio| can't dominate the mean. Used by
                # TRL, OpenRLHF, DeepSpeed-Chat verbatim. The clamp is
                # mathematically conservative (it only ever decreases
                # the kl estimate; sign of gradient is preserved); it
                # bounds Schulman k3's worst-case per-token contribution
                # to (e^C - 1 - C) regardless of how badly bf16 / vision
                # tower / KV-cache asymmetry leaked into a token.
                #
                # Default C=10 (ratio cap ≈ e^10 ≈ 22000) matches OpenRLHF.
                # Set RM_KL_LOG_RATIO_CLIP=5 if you want tighter (e^5 ≈ 150).
                _kl_clip = float(os.environ.get("RM_KL_LOG_RATIO_CLIP", "10.0"))
                log_ratio_pos = (logp_pos - logp_ref_pos).clamp(-_kl_clip, _kl_clip)
                log_ratio_neg = (logp_neg - logp_ref_neg).clamp(-_kl_clip, _kl_clip)
                kl_pos = (log_ratio_pos.exp() - 1.0 - log_ratio_pos).mean()
                kl_neg = (log_ratio_neg.exp() - 1.0 - log_ratio_neg).mean()
                kl_score_value = 0.5 * (kl_pos + kl_neg)

                # Defensive: NaN/Inf would still corrupt optimiser even
                # after the clamp. Drop kl term for this rollout if so.
                if not torch.isfinite(kl_score_value).all():
                    try:
                        rank = int(os.environ.get("LOCAL_RANK", "0"))
                    except Exception:
                        rank = 0
                    if rank == 0:
                        print(
                            f"[kl_clamp] non-finite kl_score on rollout {i}; "
                            f"setting KL term to 0 for this rollout.",
                            file=sys.stderr, flush=True,
                        )
                    kl_score_value = None

                logp_mean = (
                    logp_mean
                    + float(score_pg_alpha)
                        * (logp_pos.mean() + logp_neg.mean())
                )
                score_logp_added = True

            # ── BT-aux / score-gap supervision (revised 2026-06-18).
            # Both legs share the SAME S⁺ / S⁻ derived from the 5-digit
            # logits at colon-anchored positions of the score scaffold.
            # We compute S once and route it to either:
            #   * BT-aux (lambda_bt > 0): -log σ(τ·(S⁺ − S⁻)) — scale-free
            #     direction-only signal. Kept for ablation; pushes margin
            #     to ∞ regardless of annotated magnitude.
            #   * gap-L1 (lambda_pw > 0): |(S⁺ − S⁻) − Δ̂|  with Δ̂ =
            #     4·(s_pos_target − s_neg_target). Matches the annotated
            #     magnitude in the model's 1-5 digit range. Default for
            #     warmup; replaces BT to fix the mode-collapse-to-1-or-5
            #     failure observed when BT was sole score-head supervisor.
            #
            # Gating mirrors score-PG (alpha + format gate + score meta),
            # plus a hard requirement that BOTH pos and neg sides have at
            # least one matched digit — otherwise we'd be computing the
            # signal on partial coverage.
            bt_loss = None
            pw_loss = None
            need_S = (lambda_bt > 0.0 or lambda_pw > 0.0)
            if (need_S and rollouts is not None
                    and i < len(rollouts) and score_gate_ok[i]):
                ro_bt = rollouts[i]
                spm = getattr(ro_bt, "score_pos_meta", None)
                snm = getattr(ro_bt, "score_neg_meta", None)
                if (spm is not None and snm is not None
                        and getattr(spm, "digit_token_offsets", None)
                        and getattr(snm, "digit_token_offsets", None)
                        and getattr(spm, "digit_mask", None) is not None
                        and getattr(snm, "digit_mask", None) is not None):
                    with torch.enable_grad():
                        # Patch 6d: pass mm tensors so vision tower runs.
                        digit_logits_pos = student.compute_digit_logits(
                            spm.prompt_ids, spm.sampled_ids,
                            spm.digit_token_offsets,
                            _skip_summon=True, detach_prompt=True,
                            mm_kwargs=_mm_kwargs_from_meta(spm),
                        )                                       # (Kp, 5)
                        digit_logits_neg = student.compute_digit_logits(
                            snm.prompt_ids, snm.sampled_ids,
                            snm.digit_token_offsets,
                            _skip_summon=True, detach_prompt=True,
                            mm_kwargs=_mm_kwargs_from_meta(snm),
                        )                                       # (Kn, 5)
                    if (digit_logits_pos.numel() > 0
                            and digit_logits_neg.numel() > 0):
                        device_bt = digit_logits_pos.device
                        digits_v = torch.tensor(
                            [1., 2., 3., 4., 5.],
                            device=device_bt, dtype=digit_logits_pos.dtype,
                        )
                        # Expected score per theme position.
                        e_pos = (digit_logits_pos.softmax(-1)
                                 * digits_v).sum(-1)            # (Kp,)
                        e_neg = (digit_logits_neg.softmax(-1)
                                 * digits_v).sum(-1)            # (Kn,)
                        # Align with rubric weights via the per-side mask.
                        # ``digit_mask[k]`` is True at theme positions whose
                        # offset was recorded; e_pos / e_neg are packed in
                        # mask order (one row per True entry).
                        rubric_entries = ro_bt.student_rubric.rubrics
                        weights_full = torch.tensor(
                            [float(r.weight) for r in rubric_entries],
                            device=device_bt, dtype=digit_logits_pos.dtype,
                        )                                       # (n_themes,)
                        mask_pos = spm.digit_mask.to(device_bt)
                        mask_neg = snm.digit_mask.to(device_bt)
                        # Both sides must have a digit at the same theme.
                        common = mask_pos & mask_neg            # (n_themes,)
                        if bool(common.any().item()):
                            # Pack e_pos / e_neg back to theme-indexed dense
                            # vectors so we can apply ``common`` uniformly.
                            n_themes_loc = mask_pos.numel()
                            e_pos_dense = torch.zeros(
                                n_themes_loc, device=device_bt,
                                dtype=e_pos.dtype,
                            )
                            e_neg_dense = torch.zeros(
                                n_themes_loc, device=device_bt,
                                dtype=e_neg.dtype,
                            )
                            e_pos_dense[mask_pos] = e_pos
                            e_neg_dense[mask_neg] = e_neg
                            w_common = weights_full[common]
                            w_sum = w_common.sum()
                            if float(w_sum.item()) > 1e-6:
                                S_pos = (w_common * e_pos_dense[common]).sum() / w_sum
                                S_neg = (w_common * e_neg_dense[common]).sum() / w_sum
                                if lambda_bt > 0.0:
                                    # -log σ(τ·(S⁺ − S⁻)) — minimised when S⁺ ≫ S⁻.
                                    bt_loss = -torch.nn.functional.logsigmoid(
                                        bt_temperature * (S_pos - S_neg)
                                    )
                                if (lambda_pw > 0.0
                                        and s_pos_target is not None
                                        and s_neg_target is not None):
                                    # Δ̂ in the 1-5 digit range:
                                    # 4·(target_pos − target_neg).
                                    target_gap = 4.0 * (
                                        float(s_pos_target)
                                        - float(s_neg_target)
                                    )
                                    pred_gap = S_pos - S_neg
                                    target_gap_t = torch.tensor(
                                        target_gap, device=device_bt,
                                        dtype=pred_gap.dtype,
                                    )
                                    pw_loss = (pred_gap - target_gap_t).abs()

            pg_loss = -float(adv) * logp_mean
            # ── KL composition: rubric leg only.
            #
            # Score-leg KL is computed for telemetry (logged as kl_s) but
            # NOT added to loss. Reason: the policy and ref forward paths
            # for the score leg are not bit-identical (policy is wrapped
            # in DeepSpeedEngine + per-layer hooks, ref under ref_zero3=True
            # uses manual GatheredParameters + disable_hooks; bf16 layout
            # / kernel selection differ). Cascaded over 28 layers + a
            # 6000-token video prefix this can produce kl_s of O(1)–O(100)
            # nats from path noise alone, which when added to loss produces
            # spurious gradients that swamp the BT-aux supervised signal.
            # Standard supervised-RM training (OpenAI/Anthropic) uses BT
            # loss WITHOUT a KL term; with score-PG also providing logp
            # magnitude signal, the KL term is not needed for stability
            # in either phase. Set RM_KL_INCLUDE_SCORE=1 to re-enable.
            #
            # RM_SCORE_KL_BETA (added 2026-06-19): independent coefficient
            # on score-side KL — added to loss as `beta_score_kl · kl_score`
            # OUTSIDE the (beta_kl × score_pg_alpha) cascade. Motivation:
            # the original ``RM_KL_INCLUDE_SCORE=1`` path multiplies
            # score-KL by ``beta_kl × score_pg_alpha`` (= 0.04 with current
            # defaults), which on the 2026-06-18 warmup run would have
            # contributed ~96 to loss by step 9 (kl_s ~2400) and dominated
            # pg/pw entirely. A separate small coefficient (e.g. 0.005)
            # supplies the soft brake without crushing supervised signal.
            # Takes precedence over the legacy include-score path so the
            # two switches don't compound.
            kl_total = kl_value
            score_kl_beta_extra = 0.0
            try:
                score_kl_beta_extra = float(
                    os.environ.get("RM_SCORE_KL_BETA", "0.0")
                )
            except ValueError:
                score_kl_beta_extra = 0.0
            if (kl_score_value is not None
                    and score_kl_beta_extra <= 0.0
                    and os.environ.get("RM_KL_INCLUDE_SCORE", "0") == "1"):
                kl_total = kl_total + float(score_pg_alpha) * kl_score_value
            loss    = (pg_loss + beta_kl * kl_total) / n_usable
            if (kl_score_value is not None
                    and score_kl_beta_extra > 0.0):
                loss = loss + score_kl_beta_extra * kl_score_value
            # BT-aux: add OUTSIDE the n_usable normalisation. The PG loss is
            # already a per-rollout average via /n_usable; BT-aux is itself
            # a per-rollout scalar (not summed over G), so we apply lambda_bt
            # directly. This keeps the BT magnitude predictable across
            # different effective n_usable values within a step.
            if bt_loss is not None:
                loss = loss + float(lambda_bt) * bt_loss
            if pw_loss is not None:
                loss = loss + float(lambda_pw) * pw_loss
            if pw_loss is not None:
                loss = loss + float(lambda_pw) * pw_loss

            # ── Pre-backward cache flush to give NCCL room (FIRST rollout only).
            #
            # The first reduce_scatter of an optimiser step needs ~80-100 KB
            # of GPU memory to set up its P2P channel control structs +
            # ring buffers (NCCL init.cc:525). With ZeRO-3 + ref model +
            # video-token activations + Trick A's past_kv, the allocator
            # reserves nearly the full card and NCCL's tiny calloc fails
            # with "Cuda failure 2 'out of memory'" → wrapped as
            # ``ncclUnhandledCudaError`` on the first backward of the run.
            # An empty_cache() here returns reserved-but-unallocated blocks
            # to the driver so NCCL's communicator init can grab its
            # control buffers.
            #
            # Why only on the first usable rollout (changed 2026-06-15)
            # ---------------------------------------------------------
            # The original implementation flushed before EVERY rollout's
            # backward. Phase B then logged ~35-45 "pytorch allocator
            # cache flushes since last step" warnings per training step
            # (8 rollouts × ~5 forward each → 40 sync points), each one
            # costing a cudaDeviceSynchronize stall and forcing the
            # caching pool to re-grow on the next forward. Wall-clock
            # impact on phase B was ~1-2 min/step. After the FIRST
            # backward of the step succeeds, NCCL's communicator is
            # already initialised on every rank and won't re-init within
            # the step — the protective flush is only needed once. The
            # allocator can then keep its blocks across rollouts 2..G,
            # which is exactly what we want.
            if torch.cuda.is_available() and out["applied"] == 0:
                torch.cuda.empty_cache()

            if accelerator is not None:
                accelerator.backward(loss)
            else:
                loss.backward()

            pg_values.append(float(pg_loss.detach()))
            kl_values.append(float(kl_value.detach()))
            kl_score_scalar = (
                float(kl_score_value.detach())
                if kl_score_value is not None else 0.0
            )
            kl_score_values.append(kl_score_scalar)
            # BT-aux: track the per-rollout BT loss (0.0 when not fired) so
            # we can log a step-level mean to wandb / stdout.
            bt_loss_scalar = (
                float(bt_loss.detach()) if bt_loss is not None else 0.0
            )
            bt_loss_values.append(bt_loss_scalar)
            if bt_loss is not None:
                out["bt_aux_applied"] += 1
            pw_loss_scalar = (
                float(pw_loss.detach()) if pw_loss is not None else 0.0
            )
            pw_loss_values.append(pw_loss_scalar)
            if pw_loss is not None:
                out["pw_aux_applied"] += 1
            pw_loss_scalar = (
                float(pw_loss.detach()) if pw_loss is not None else 0.0
            )
            pw_loss_values.append(pw_loss_scalar)
            if pw_loss is not None:
                out["pw_aux_applied"] += 1
            loss_values.append(float(loss.detach()) * n_usable)
            out["applied"] += 1
            if score_logp_added:
                out["score_pg_applied"] += 1

            if _mem_diag and torch.cuda.is_available():
                a1 = torch.cuda.memory_allocated() / 2**30
                p1 = torch.cuda.max_memory_allocated() / 2**30
                print(f"[mem rollout={i}/{len(usable_indices)} "
                      f"is_last={is_last}] post-bwd alloc={a1:.1f}G "
                      f"peak={p1:.1f}G", file=sys.stderr, flush=True)
                torch.cuda.reset_peak_memory_stats()

    # Defensive: if every usable rollout got skipped after the boundary
    # flag was last set, leave it True so the engine is in a clean state
    # for the NEXT step (no stale "still inside accumulation window").
    if ds_engine is not None:
        ds_engine.set_gradient_accumulation_boundary(True)

    grad_norm: float | None = None
    if max_grad_norm and max_grad_norm > 0:
        if accelerator is not None:
            gn = accelerator.clip_grad_norm_(
                [p for p in student.trainable_parameters() if p.grad is not None],
                max_grad_norm,
            )
        else:
            gn = torch.nn.utils.clip_grad_norm_(
                (p for p in student.trainable_parameters() if p.grad is not None),
                max_grad_norm,
            )
        try:
            grad_norm = float(gn) if gn is not None else None
        except Exception:
            grad_norm = None
    optimizer.step()

    if student.model is not None:
        student.model.eval()

    out["loss"]    = sum(loss_values) / max(1, len(loss_values))
    out["pg_loss"] = sum(pg_values)   / max(1, len(pg_values))
    out["kl"]      = sum(kl_values)   / max(1, len(kl_values))
    out["kl_score"] = sum(kl_score_values) / max(1, len(kl_score_values))
    out["bt_aux"]  = sum(bt_loss_values) / max(1, len(bt_loss_values))
    out["pw_aux"]  = sum(pw_loss_values) / max(1, len(pw_loss_values))
    out["grad_norm"]   = grad_norm
    out["kl_per_rollout"]      = list(kl_values)
    out["kl_score_per_rollout"] = list(kl_score_values)
    out["bt_aux_per_rollout"]  = list(bt_loss_values)
    out["pw_aux_per_rollout"]  = list(pw_loss_values)
    out["pg_loss_per_rollout"] = list(pg_values)
    return out


# ---------------------------------------------------------------------------
# 4. Driver loop.
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset.
    ds = GoldRubricDataset.from_jsonl(args.gold_rubrics, rebase_root=ROOT)
    print(f"[train] loaded {len(ds)} training rows from {args.gold_rubrics}",
          file=sys.stderr)
    if len(ds) == 0:
        raise SystemExit("[train] dataset is empty — did Stage ① run?")

    # ── Accelerator (DeepSpeed ZeRO-3 / single-process). Created BEFORE the
    # student so we can pass the right ``device_map`` choice via env
    # detection inside StudentWrapper.
    dummy_env = os.environ.get("RM_DUMMY", "0") == "1"
    accelerator = None
    if not (dummy_env or args.student_model is None) and not args.no_accelerate:
        accelerator = _make_accelerator(num_generations=args.num_generations)

    # ── Student.
    student = StudentWrapper(StudentConfig(
        model_path=args.student_model,
        dummy=dummy_env or args.student_model is None,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        lora_config_path=args.lora_config,
        ref_model_path=(args.ref_model_path or
                        (args.student_model if args.lora_config is None and args.beta_kl > 0 else None)),
        ref_zero3=args.ref_zero3,
        attn_implementation=args.attn_implementation,
        score_greedy=not args.score_sample,
        score_method=args.score_method,
        score_max_new_tokens=args.score_max_new_tokens,
        use_gradient_checkpointing=args.gradient_checkpointing,
        grad_checkpoint_use_reentrant=args.grad_ckpt_use_reentrant,
    ))

    # ── Embedder.
    embedder = _load_embedder(args.embed_model)

    # ── Reward config + tracker.
    cfg = BatchRewardConfig(
        lambda_0     = args.lambda_0,
        beta_format  = args.beta_format,
        eps_valid    = args.eps_valid,
        hard_pref    = args.hard_pref,
        pref_temperature = args.pref_temperature,
        lock_weights_to_generation = not args.no_lock_weights,
        beta_format_min        = args.beta_format_min,
        beta_format_decay_steps= args.beta_format_decay_steps,
        beta_format_schedule   = args.beta_format_schedule,
        lambda_min            = args.lambda_min,
        lambda_warmup_steps   = args.lambda_warmup_steps,
        lambda_decay_steps    = args.lambda_decay_steps,
        align_floor    = args.align_floor,
        align_saturate = args.align_saturate,
    )
    tracker = ValidRatioTracker(window=args.valid_window, alpha=args.valid_alpha)
    beta_scheduler = BetaFormatScheduler.from_cfg(cfg)
    print(f"[reward] β_format schedule={cfg.beta_format_schedule}  "
          f"β_0={cfg.beta_format}  β_min={cfg.beta_format_min}  "
          f"decay_steps={cfg.beta_format_decay_steps}", file=sys.stderr)
    if cfg.lambda_warmup_steps > 0 or cfg.lambda_decay_steps > 0:
        print(f"[reward] λ_align anneal: λ_0={cfg.lambda_0}  λ_min={cfg.lambda_min}  "
              f"warmup={cfg.lambda_warmup_steps}  decay={cfg.lambda_decay_steps} "
              f"(applied on top of ValidRatioTracker scaling)",
              file=sys.stderr)

    # ── Per-dim prompt scaffolding (loaded from rubric_pipeline yamls).
    # The trainer passes a DimensionContext into student.generate_rubric /
    # score_video so the model sees the same task description Stage ①
    # teacher saw. Without this, student rubrics drift away from the
    # teacher's sub-aspect taxonomy and r_align cannot anchor.
    task_specs = load_task_specs(ROOT) if args.use_task_specs else {}
    if task_specs:
        print(f"[task_specs] loaded {len(task_specs)} dim specs from "
              f"{ROOT}/rubric_pipeline/config", file=sys.stderr)
    else:
        print("[task_specs] disabled or not found — student will see a "
              "generic dimension scaffold (fine for dummy smoke runs).",
              file=sys.stderr)

    # ── Per-dim gold rubric lookup (one canonical entry per dim, for
    # the val_use_gold path).
    gold_by_dim = _build_gold_by_dim(ds)
    if gold_by_dim:
        print(f"[gold_by_dim] {len(gold_by_dim)} dims have a gold rubric "
              f"(used when --val_use_gold).", file=sys.stderr)

    # ── Optimiser (real mode only — dummy returns no trainable params).
    optimizer = _build_optimizer(student, args)

    # ── LR warmup scheduler (linear from 0 → lr over args.warmup_steps,
    # then constant). Rationale: at step 0 the policy is identical to the
    # ref, advantages are computed off near-zero reward variance, and a
    # fresh AdamW with lr=2e-6 + bf16 grads will already nudge the LM head
    # enough to break the rubric format on the first ~5 steps. A short
    # linear warmup (~30 steps) lets the policy ride out the cold-start
    # phase where r_pref is still 0/G and only format noise drives the
    # gradient — the dominant failure mode in the prior run.
    lr_scheduler = None
    if optimizer is not None and args.warmup_steps and args.warmup_steps > 0:
        def _warmup_lambda(current_step: int, *, w: int = int(args.warmup_steps)):
            if current_step < w:
                return float(current_step + 1) / float(max(1, w))
            return 1.0
        import torch
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup_lambda)
        if _is_main(accelerator):
            print(f"[optim] LR warmup: linear over first {args.warmup_steps} steps "
                  f"(0 → {args.learning_rate})", file=sys.stderr)

    # ── Wrap with accelerator (DeepSpeed ZeRO-3). Single-process
    # mode skips this entirely. Note: when LoRA is on, only the (small) adapter
    # weights need preparing; ZeRO-3 would still wrap them but provide
    # almost no memory benefit. The wrap is only critical for full-param
    # training. We always call .prepare() when accelerator is non-None
    # so the rest of the train loop has a single code path.
    if accelerator is not None and student.model is not None and optimizer is not None:
        student._model, optimizer = accelerator.prepare(student._model, optimizer)
        if student._ref_model is not None:
            # ── Patch 8 reverted (2026-06-18): do NOT call
            # ``accelerator.prepare_model(ref, evaluation_mode=True)``.
            #
            # Empirically, wrapping the ref in a second DeepSpeedEngine
            # alongside the policy engine produced rank-0 SIGSEGV at
            # step 7 of warm-up (logs/train_warmup.log, 2026-06-18 run);
            # the previous warm-up using the legacy Patch 7 path
            # (manual ``_summon_full_params_ctx(writeback=False)`` for
            # ref forwards) ran 60 steps cleanly with the same data and
            # seed (logs/grpo_phaseA_v3_20260616_153754.log). Two
            # DSEngines sharing the NCCL group + per-layer hooks on a
            # frozen ref appears to be the trigger.
            #
            # Now: ref stays as a plain HF module with ZeRO-3-partitioned
            # params (ds_id stamps from the partition init in
            # StudentWrapper); ``compute_log_probs(ref=True)`` falls into
            # the ``ref_is_partitioned`` branch in student_wrapper.py and
            # uses manual GatheredParameters around the forward — the
            # same path that worked on 06-16. We still .eval() and freeze
            # requires_grad here.
            student._ref_model = student._ref_model.to(accelerator.device)
            student._ref_model.eval()
            for p in student._ref_model.parameters():
                p.requires_grad = False
        if accelerator.is_main_process:
            print(f"[accel] model + optimizer prepared "
                  f"(ref_model={'yes' if student._ref_model is not None else 'no'})",
                  file=sys.stderr)

    # ── Loop.
    rng = random.Random(args.seed)
    is_main = _is_main(accelerator)
    log_path = out_dir / "train_log.jsonl"
    # Only rank-0 writes logs / ckpts / val files. Other ranks open
    # /dev/null to keep the same code path (their writes are silently
    # dropped). For val we additionally guard the run_pointwise_eval
    # call itself because it would re-do the same forward 8x otherwise.
    log_fh = log_path.open("w", encoding="utf-8") if is_main else open(os.devnull, "w")
    if is_main:
        print(f"[train] log -> {log_path}", file=sys.stderr)

    # ── Optional wandb (only main process — _wandb_init already
    # handles the RANK env, but we re-check via is_main for the
    # accelerate-Accelerator path which doesn't set RANK explicitly).
    wandb_run = _wandb_init(args, cfg) if is_main else None

    indices = list(range(len(ds)))
    t0 = time.time()
    step = 0

    # ── Data-parallel sharding across ranks.
    # When running under ZeRO-3 + DP, each rank should consume a DIFFERENT
    # instance per step so the 8 cards aren't all redundantly chewing on
    # the same row. The model itself is ZeRO-3-sharded across ranks, but
    # the training data wasn't — that's a 1-line oversight that wasted
    # 7/8 of the throughput. With this fix:
    #
    #   step k:  rank r picks indices_shuffled[k * world_size + r]
    #
    # Effective throughput per logged ``step`` becomes ``world_size``
    # instances (8× on 8 ranks). Gradients still reduce-scatter via
    # the engine's ``backward`` so the optimizer sees the same DP-averaged
    # gradient as if we'd run a (world_size × G) batched update.
    #
    # The shuffle uses the SAME seeded ``rng`` on every rank so the
    # ordering is deterministic and identical — slicing by rank then
    # gives non-overlapping coverage of the dataset within each epoch.
    world_size = (
        accelerator.num_processes
        if accelerator is not None and not args.no_accelerate else 1
    )
    rank = (
        accelerator.process_index
        if accelerator is not None and not args.no_accelerate else 0
    )

    # ── Resolve --epochs into --max_steps.
    # An "epoch" here means: every rank consumes len(ds)//world_size
    # instances, so collectively the cluster sees the dataset once
    # (each row processed by exactly one rank under the seeded shuffle).
    # We translate epochs into a max_steps override so the rest of the
    # loop stays single-currency in steps. --max_steps still wins if
    # --epochs is unset.
    if args.epochs is not None:
        if args.epochs <= 0:
            raise SystemExit(f"[train] --epochs must be > 0, got {args.epochs}")
        steps_per_epoch = max(1, len(ds) // world_size)
        new_max = args.epochs * steps_per_epoch
        if is_main:
            print(f"[train] --epochs={args.epochs} resolved to max_steps={new_max} "
                  f"(len(ds)={len(ds)}, world_size={world_size}, "
                  f"steps_per_epoch={steps_per_epoch}); "
                  f"overrides --max_steps={args.max_steps}",
                  file=sys.stderr)
        args.max_steps = new_max
    else:
        steps_per_epoch = max(1, len(ds) // world_size)

    # ── Resolve --gold_rubric_warmup_epochs into a step count.
    # During warm-up, step_one_instance is fed the per-dim Stage ① gold
    # rubric instead of student-generated rubrics. This decouples the
    # bootstrap of the score head (learn to discriminate y⁺/y⁻ given a
    # clean rubric) from the bootstrap of the rubric head (learn to
    # generate good rubrics) — the two were locked into a chicken-and-
    # egg deadlock at cold start (poor rubrics → r_pref=0 → no advantage
    # → student can't learn to score → student can't learn rubrics).
    #
    # See RGR-GRPO / VQ-Insight / DAPO "warm-up curriculum" pattern.
    # Recommended: 1 epoch (≈ steps_per_epoch). After warm-up, both heads
    # train jointly with the standard reward composition (r_pref + λ·r_align
    # + β·r_format). Set to 0 to disable.
    warmup_n = max(0, int(getattr(args, "gold_rubric_warmup_epochs", 0) or 0))
    gold_warmup_steps = warmup_n * steps_per_epoch
    if is_main and gold_warmup_steps > 0:
        print(f"[gold-warmup] using gold rubric for first "
              f"{gold_warmup_steps} steps "
              f"({warmup_n} epoch{'s' if warmup_n != 1 else ''} × "
              f"steps_per_epoch={steps_per_epoch}); "
              f"score_pg_alpha forced to {args.score_pg_alpha_max} during warm-up.",
              file=sys.stderr)
        if (args.score_pg_alpha_max <= 0.0
                and args.lambda_bt <= 0.0
                and args.lambda_pw <= 0.0):
            print("[gold-warmup] WARNING: --score_pg_alpha_max is 0 AND "
                  "--lambda_bt is 0 AND --lambda_pw is 0. During warm-up "
                  "rubric-PG is OFF (rubric is gold) AND score-PG is OFF "
                  "(alpha_max=0) AND BT-aux / gap-L1 are both OFF → no "
                  "leg gets gradient. Set --score_pg_alpha_max 1.0 OR "
                  "--lambda_pw > 0 (recommended) OR --lambda_bt > 0 to "
                  "actually train the score head during warm-up.",
                  file=sys.stderr)
        elif args.score_pg_alpha_max <= 0.0 and (args.lambda_bt > 0.0 or args.lambda_pw > 0.0):
            print(f"[gold-warmup] score-PG OFF (alpha_max=0); supervised "
                  f"score-head signal ON (lambda_bt={args.lambda_bt}, "
                  f"lambda_pw={args.lambda_pw}). Score head trained by "
                  f"the supervised loss on the 5-digit logits.",
                  file=sys.stderr)

    # ── Preflight (added 2026-06-20). ─────────────────────────────────
    # Run ONE light-weight rubric forward + backward to materialise lazy
    # GPU allocations (DeepSpeed reduce-scatter buffers, NCCL channels,
    # flash-attn workspace, Adam state lazy init, cuBLAS algo selection)
    # so the real step 0's allocator is already settled.
    #
    # Why "light" — earlier preflight that called step_one_instance +
    # policy_update deadlocked: per-rank inst variation (different videos,
    # different decode times) made first-call NCCL collectives wait
    # asymmetrically until 6/8 ranks held 80GB GPU mem and 2/8 still
    # decoding. This version runs IDENTICAL token IDs on every rank with
    # a small fixed prompt+sampled pair so the collective sequence is
    # bit-identical across ranks; no video, no scoring, no reward path.
    #
    # What this DOES warm up:
    #   * DeepSpeed Stage 3 reduce-scatter / all-gather buffers
    #   * Adam state (m, v) — created lazily after first non-zero grad
    #   * NCCL ring channels and per-comm buffers
    #   * flash-attn forward+backward workspace
    #   * cuBLAS algo cache
    # What this does NOT warm up (acceptable):
    #   * Vision tower (no video here) — re-runs first time at step 0,
    #     but its activations are bounded and don't need a single contiguous
    #     allocation (allocator pool is already shaped correctly).
    #   * generate() KV-cache buffers — same: bounded, allocator settled.
    #
    # Optimizer NEVER takes a step here, so step-0 KL≈0 invariant
    # (Patch 5c) is preserved bit-for-bit.
    #
    # Disable with RM_PREFLIGHT=0 (e.g. for memory diagnosis).
    if (os.environ.get("RM_PREFLIGHT", "1") == "1"
            and student.model is not None
            and not student.is_dummy()):
        if is_main:
            print("[preflight] running 1 light forward+backward "
                  "(no optimizer.step) to warm DS/NCCL/flash-attn buffers; "
                  "all ranks run identical tokens to avoid inter-rank skew.",
                  file=sys.stderr, flush=True)
        try:
            import torch as _torch
            # Build a deterministic, identical-across-ranks token sequence.
            # We don't need anything semantic — DS reduce-scatter and NCCL
            # only care about model-shape, not content. Pick small numbers
            # so the forward is cheap (a few MB of activations).
            tok = student._processor.tokenizer
            pf_text   = "Preflight warm-up. " * 4   # short, fixed
            pf_chat   = student._processor.apply_chat_template(
                [{"role": "user", "content": [{"type": "text", "text": pf_text}]}],
                tokenize=False, add_generation_prompt=True,
            )
            pf_inputs = student._processor(text=[pf_chat], return_tensors="pt").to(
                student._model.device
            )
            pf_prompt_ids = pf_inputs["input_ids"][0].detach().clone()
            # Synthetic 4-token "sampled" tail — deterministic per rank.
            # Any valid token IDs work; use the first 4 of the prompt's vocab
            # range for stability.
            pf_sampled_ids = _torch.tensor(
                [pf_prompt_ids[0].item()] * 4,
                device=pf_prompt_ids.device, dtype=pf_prompt_ids.dtype,
            )
            student._model.train()
            # grad-on rubric forward through the same path policy_update uses.
            with _torch.enable_grad():
                pf_logp = student.compute_log_probs(
                    pf_prompt_ids, pf_sampled_ids,
                    ref=False, _skip_summon=False,
                    detach_prompt=False, mm_kwargs=None,
                )
            pf_loss = pf_logp.mean() * 0.0   # zero coefficient: NO real grad,
            # but the autograd graph still gets walked — DS reduce-scatter and
            # NCCL still allocate their buffers. Multiplying by 0 means the
            # gradient values themselves are zero, so even if zero_grad(set_to_none=True)
            # somehow fails to drop them, optimizer.step() being skipped means
            # weights are bit-identical.
            if accelerator is not None:
                accelerator.backward(pf_loss)
            else:
                pf_loss.backward()
            try:
                optimizer.zero_grad(set_to_none=True)
            except TypeError:
                optimizer.zero_grad()
            if student.model is not None:
                student.model.eval()
            del pf_logp, pf_loss, pf_inputs, pf_prompt_ids, pf_sampled_ids
        except Exception as e:
            # If preflight itself fails, log and continue. Step 0 will
            # exhibit whatever the underlying issue is; preflight's job
            # is best-effort.
            if is_main:
                print(f"[preflight] WARN: preflight forward failed with "
                      f"{type(e).__name__}: {e}; continuing without warm-up. "
                      f"Set RM_PREFLIGHT=0 to silence this path.",
                      file=sys.stderr, flush=True)
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
                _torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        if is_main:
            print("[preflight] done; entering main loop.",
                  file=sys.stderr, flush=True)

    while step < args.max_steps:
        rng.shuffle(indices)
        # Walk the shuffled list in stride-world_size groups; each group
        # of ``world_size`` rows becomes ONE training step (each rank
        # claims one row of the group).
        group_start = 0
        while group_start + world_size <= len(indices) and step < args.max_steps:
            group = indices[group_start : group_start + world_size]
            group_start += world_size
            idx = group[rank]
            inst = ds[idx]

            # ── Crash-diag breadcrumb (every rank, every step).
            # Prints BEFORE any forward/generate so the LAST line in
            # ``[trace rank=...]`` after a SIGSEGV identifies which (rank,
            # sample) was about to run when the C extension crashed. Cost:
            # one fprintf per rank per step (~negligible at 4-min/step).
            # Disable with RM_TRACE_SAMPLE=0 if it ever clutters logs.
            if os.environ.get("RM_TRACE_SAMPLE", "1") != "0":
                try:
                    _ypos = inst.get("y_pos", "")
                    _yneg = inst.get("y_neg", "")
                    # Trim long video paths to the basename + parent for
                    # readability; full path is already in the dataset jsonl.
                    def _short(p):
                        if not p:
                            return ""
                        parts = str(p).rsplit("/", 2)
                        return "/".join(parts[-2:]) if len(parts) >= 2 else str(p)
                    print(
                        f"[trace rank={rank} step={step} idx={idx} "
                        f"dim={inst.get('dimension','?')} "
                        f"pair_id={inst.get('pair_id','?')} "
                        f"y_pos={_short(_ypos)} y_neg={_short(_yneg)}",
                        file=sys.stderr, flush=True,
                    )
                except Exception:
                    pass

            # Rollout phase. Each rank processes its own instance; the
            # inner generate paths each open their own ``summon_full_params
            # (writeback=False)`` around the autoregressive forward. Those
            # summons ARE collectives, but NCCL matches by call order, not
            # wall clock — fast ranks just block at the next collective
            # waiting for slow ranks, no deadlock. (The DP-deadlock
            # concern that motivated an outer summon was wrong; per-call
            # summons under writeback=False are safe.)
            # ── Score-side PG curriculum.
            #
            # Phase schedule (per the RGR-GRPO / VQ-Insight curriculum
            # discussion): hold rubric-only PG until the rubric format
            # has stabilised, then linearly ramp the score-side α
            # coefficient up to its max.
            #
            #   step < score_pg_start_step       → α = 0  (rubric-only)
            #   score_pg_start_step ≤ step < full → α linearly 0 → α_max
            #   step ≥ score_pg_full_step         → α = α_max
            #
            # ``score_pg_active`` is True iff α > 0 — only then do we
            # ask the wrapper to sample (vs greedy) the score forward
            # and capture meta. The tradeoff: sampling injects extra
            # variance into the reward (scores no longer deterministic
            # from rubric+video), but it's the only path that gives
            # the score forward a gradient.
            if args.score_pg_alpha_max <= 0.0 or args.score_pg_start_step < 0:
                score_pg_alpha = 0.0
            elif step < args.score_pg_start_step:
                score_pg_alpha = 0.0
            elif step >= args.score_pg_full_step:
                score_pg_alpha = args.score_pg_alpha_max
            else:
                span = max(1, args.score_pg_full_step - args.score_pg_start_step)
                p    = (step - args.score_pg_start_step) / span
                score_pg_alpha = args.score_pg_alpha_max * float(p)
            score_pg_active = score_pg_alpha > 0.0

            # ── Gold-rubric warm-up override.
            # During the warm-up window the rubric is held constant (per-dim
            # gold), so r_pref carries a clean signal and the score head can
            # be trained at full strength immediately. We OVERRIDE the
            # score_pg ramp to α_max so warm-up doesn't end up training
            # nothing (rubric-PG off because metas are None, score-PG off
            # because step < score_pg_start_step). After warm-up the normal
            # ramp resumes from whatever step we landed on.
            use_gold_warmup = (gold_warmup_steps > 0 and step < gold_warmup_steps)
            if use_gold_warmup and args.score_pg_alpha_max > 0.0:
                score_pg_alpha  = float(args.score_pg_alpha_max)
                score_pg_active = True

            metrics = step_one_instance(
                student, inst,
                G=args.num_generations,
                cfg=cfg,
                embedder=embedder,
                tracker=tracker,
                beta_scheduler=beta_scheduler,
                step=step,
                task_specs=task_specs,
                score_pg_active=score_pg_active,
                use_gold_warmup=use_gold_warmup,
                gold_by_dim=gold_by_dim,
            )# ── Free the rollout phase's cached allocator blocks before
            # backward starts. The G generates and 2 score batches each
            # allocate big KV-cache tensors that are now dead Python
            # references but still live in PyTorch's caching pool. cuBLAS
            # in backward needs a contiguous workspace allocation; if the
            # pool is fragmented from generate-side churn, that allocation
            # can fail with CUBLAS_STATUS_ALLOC_FAILED even when nvidia-smi
            # reports plenty of free memory. ``empty_cache`` returns the
            # cached blocks to the driver so cuBLAS sees a fresh pool.
            #
            # Costs: a single ``cudaDeviceSynchronize`` (~ms) plus the
            # next forward has to re-grow the pool. Not a hot-loop concern
            # at our 2-3min/step cadence.
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Optional per-step memory snapshot, gated by env so the
                # logs aren't spammed. Set RM_MEM_DIAG=1 to enable.
                if os.environ.get("RM_MEM_DIAG", "0") == "1" and is_main:
                    alloc = torch.cuda.memory_allocated() / 2**30
                    resv  = torch.cuda.memory_reserved()  / 2**30
                    peak  = torch.cuda.max_memory_allocated() / 2**30
                    print(f"[mem step={step}] post-rollout "
                          f"alloc={alloc:.1f}G  reserved={resv:.1f}G  "
                          f"peak={peak:.1f}G",
                          file=sys.stderr, flush=True)
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

            # ── Policy gradient step.
            #
            # ── Patch 8 (2026-06-16): step-0 KL kill-switch.
            # Even with the bit-identical ref/policy fix in
            # student_wrapper.py:_compute_log_probs (Patch 8), the very
            # first call into the model on each rank pays a one-shot
            # lazy-init cost (flash-attn kernel selection, ZeRO-3
            # partition mapping, KV cache dtype resolution) that can
            # leave the bf16 numerical residue elevated for the FIRST
            # forward only. Combined with score-PG's α=1.0 and the
            # 6000-token video prefix, the resulting kl_s spike at step 0
            # was producing loss ≈ 6.6e5 — which would be absorbed into
            # Adam's v accumulator on the first optimiser step and
            # poison the effective learning rate for the next ~50 steps.
            #
            # Setting beta_kl=0 on step 0 lets the warm-up forward pay
            # its lazy-init cost without contaminating the optimiser
            # state. From step 1 onwards the KL term is back online; if
            # Patch 8's bit-identity fix is correct, kl_s should drop
            # to ~1e-5 at step 1 and stay near zero throughout warm-up.
            beta_kl_eff = 0.0 if step == 0 else args.beta_kl
            if step == 0 and is_main:
                print(
                    "[step-0] KL kill-switch active "
                    f"(beta_kl: {args.beta_kl} → 0 for this step only). "
                    "From step 1 onward kl_s should be near zero; if it "
                    "is not, the bit-identity assumption between policy "
                    "and ref forwards is violated.",
                    file=sys.stderr,
                )
            update = policy_update(
                student,
                metrics.pop("rollout_meta"),
                metrics["advantages"],
                optimizer=optimizer,
                beta_kl=beta_kl_eff,
                max_grad_norm=args.max_grad_norm,
                accelerator=accelerator,
                rollouts=metrics.get("rollouts"),
                breakdowns=metrics.get("breakdowns"),
                score_pg_alpha=score_pg_alpha,
                score_format_gate=args.score_format_gate,
                lambda_bt=args.lambda_bt,
                bt_temperature=args.bt_temperature,
                lambda_pw=args.lambda_pw,
                s_pos_target=metrics.get("s_pos_target"),
                s_neg_target=metrics.get("s_neg_target"),
            )
            # Step the LR warmup scheduler after each optimizer.step().
            # policy_update internally calls optimizer.step() exactly once
            # per training step (or short-circuits when there are no usable
            # rollouts), so stepping the scheduler here keeps lr in sync.
            if lr_scheduler is not None and update.get("applied", 0) > 0:
                try:
                    lr_scheduler.step()
                except Exception:
                    pass
            metrics["loss"]    = update["loss"]
            metrics["pg_loss"] = update["pg_loss"]
            metrics["kl"]      = update["kl"]
            metrics["kl_score"] = update.get("kl_score", 0.0)
            metrics["bt_aux"]   = update.get("bt_aux", 0.0)
            metrics["bt_aux_applied"] = update.get("bt_aux_applied", 0)
            metrics["pw_aux"]   = update.get("pw_aux", 0.0)
            metrics["pw_aux_applied"] = update.get("pw_aux_applied", 0)
            metrics["pw_aux"]   = update.get("pw_aux", 0.0)
            metrics["pw_aux_applied"] = update.get("pw_aux_applied", 0)
            metrics["applied"] = update["applied"]
            metrics["skipped"] = update["skipped"]
            metrics["score_pg_alpha"]   = score_pg_alpha
            metrics["gold_warmup"]      = bool(use_gold_warmup)
            metrics["score_pg_applied"] = update.get("score_pg_applied", 0)
            metrics["grad_norm"]      = update.get("grad_norm")
            metrics["kl_per_rollout"] = update.get("kl_per_rollout") or []
            metrics["kl_score_per_rollout"] = update.get("kl_score_per_rollout") or []
            metrics["bt_aux_per_rollout"]   = update.get("bt_aux_per_rollout") or []
            metrics["pw_aux_per_rollout"]   = update.get("pw_aux_per_rollout") or []
            metrics["pg_loss_per_rollout"] = update.get("pg_loss_per_rollout") or []

            metrics["step"]   = step
            metrics["wall_s"] = round(time.time() - t0, 2)
            metrics["rank"]   = rank
            metrics["pair_id_local"] = inst.get("pair_id")
            # train_log.jsonl is plain JSON; the wandb-only fields hold
            # parsed dataclasses that don't serialise. Strip before write,
            # but keep them in the live ``metrics`` so wandb_log_sample
            # below can still read them.
            #
            # Under DP, every rank produced its own metrics for THIS step.
            # We log only rank-0's row to jsonl + wandb to keep the dashboard
            # readable; every rank still ran a full forward+backward and
            # contributed to the engine's reduce-scattered gradient.
            if is_main:
                jsonl_safe = {
                    k: v for k, v in metrics.items()
                    if k not in ("rollouts", "breakdowns", "rollout_meta")
                }
                log_fh.write(json.dumps(jsonl_safe, ensure_ascii=False))
                log_fh.write("\n")
                log_fh.flush()
                _wandb_log_step(wandb_run, metrics)
                # Dump a rollout sample (rubric text + scores + reward parts)
                # every N steps so the wandb UI carries a human-readable
                # trace of how generations evolve.
                if args.wandb_sample_every > 0 and step % args.wandb_sample_every == 0:
                    _wandb_log_sample(wandb_run, step, metrics)
                if args.sample_log_every > 0 and step % args.sample_log_every == 0:
                    _stderr_log_sample(step, metrics)
                if step % max(1, args.log_every) == 0:
                    print(
                        f"[step {step:5d}] dim={inst['dimension']:18s} "
                        f"reward μ={statistics.mean(metrics['rewards']):.3f}  "
                        f"r_pref={metrics['pref_mean']:.3f}  "
                        f"r_align={metrics['align_mean']:.3f}  "
                        f"r_fmt={metrics['fmt_mean']:.3f}  "
                        f"valid={metrics['valid_count']}/{args.num_generations}  "
                        f"λ={metrics['lambda_t']:.3f} β={metrics['beta_t']:.3f}  "
                        f"loss={metrics['loss']:.4f} "
                        f"kl={metrics['kl']:.3e} kl_s={metrics['kl_score']:.3e} "
                        f"bt={metrics.get('bt_aux', 0.0):.3e} "
                        f"pw={metrics.get('pw_aux', 0.0):.3e} "
                        f"(of {world_size} parallel instances/step)",
                        file=sys.stderr,
                    )

            step += 1

            # ── Periodic checkpoint.
            # Under DeepSpeed ZeRO-3, save_pretrained on rank-0 must
            # be flanked by an all-rank state_dict gather. EVERY rank
            # has to enter ``_save_ckpt`` so ``accelerator.get_state_dict``
            # (the gather collective) can run in lockstep — gating this
            # on ``is_main`` was the bug that produced the
            # ``copying a param with shape torch.Size([0])`` corrupted
            # ckpts in earlier runs. ``_save_ckpt`` itself branches on
            # ``is_main`` for the disk-side I/O after the gather returns.
            if optimizer is not None and args.save_every > 0 and \
                    (step % args.save_every == 0 or step >= args.max_steps):
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                _save_ckpt(student, out_dir, step,
                           accelerator=accelerator, is_main=is_main)
                if accelerator is not None:
                    accelerator.wait_for_everyone()

            # ── Periodic PLCC/SRCC val on the EvalVerse hold-out.
            #
            # ZeRO-3 deadlock fix
            # -------------------
            # Previously this branch ran ``_run_val_eval`` ONLY on rank-0,
            # while the other ranks waited at the trailing
            # ``wait_for_everyone()``. The val path internally calls
            # ``student.score_video`` / ``generate_rubric``, both of which
            # enter ``_summon_full_params_ctx`` — a ZeRO-3 ``all_gather``
            # collective that REQUIRES every rank to participate in
            # lockstep. With only rank-0 inside, NCCL hung for the
            # 1800-second watchdog timeout and the whole job aborted at
            # SeqNum=157035 on (Rank 0 = _ALLGATHER_BASE 188160 elements,
            # Ranks 1-7 = ALLREDUCE 1 element). Hard fail at step 30 with
            # ``--val_every 30``.
            #
            # Fix: every rank runs the val forwards (the gather is then
            # served by the local layer-shard collective like a normal
            # training forward), but only rank-0 writes logs / wandb / the
            # per-step val_step{step}.json. The duplicate forwards on
            # ranks 1-7 are wasted compute, but they keep the collective
            # in sync — the alternative (sharding val rows across ranks +
            # all-reduce-mean) needs run_pointwise_eval support that
            # doesn't exist yet. ``--val_max_samples_per_dim 10`` keeps
            # the val sweep ~170 forwards * ~4s = ~10 min, acceptable.
            if args.val_every > 0 and (step % args.val_every == 0 or step >= args.max_steps):
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                # Rank-0 prefetches every URL the val sweep will need, then
                # barriers; all ranks then run val with prefetch_missing=False
                # so they all see the same set of locally-resolvable videos.
                # Letting each rank prefetch independently used to deadlock
                # (see ``prefetch_pointwise_videos`` docstring) — DO NOT undo.
                if is_main:
                    try:
                        prefetch_pointwise_videos(
                            ROOT,
                            max_samples_per_dim=args.val_max_samples_per_dim,
                        )
                    except Exception as e:
                        print(f"[val   {step:5d}] prefetch failed: {e} — "
                              "continuing; rows whose URL didn't materialise "
                              "will be skip_no_video.", file=sys.stderr)
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                # Every rank runs the eval forwards (collectives must match);
                # only rank-0 keeps the result for logging.
                ovr, heatmap = _run_val_eval(
                    student, task_specs, gold_by_dim,
                    use_gold_rubric=args.val_use_gold,
                    max_samples_per_dim=args.val_max_samples_per_dim,
                    repo_root=ROOT,
                    step=step,
                    out_dir=out_dir,
                    write_artifacts=is_main,
                    prefetch_missing=False,
                )
                if is_main:
                    metrics["val_plcc"] = ovr.get("plcc", 0.0)
                    metrics["val_srcc"] = ovr.get("srcc", 0.0)
                    metrics["val_mae"]  = ovr.get("mae",  0.0)
                    metrics["val_n"]    = ovr.get("n",    0)
                    # Re-write this step's log line with the val numbers
                    # appended (we already wrote it above without val).
                    log_fh.write("# val update for step={}: plcc={:.4f} srcc={:.4f} mae={:.4f} n={}\n".format(
                        step, ovr.get("plcc", 0.0), ovr.get("srcc", 0.0),
                        ovr.get("mae", 0.0), ovr.get("n", 0)
                    ))
                    log_fh.flush()
                    _wandb_log_val(wandb_run, step, ovr, heatmap, tag="val")
                    print(f"[val   {step:5d}] plcc={ovr.get('plcc',0):+.3f} "
                          f"srcc={ovr.get('srcc',0):+.3f} mae={ovr.get('mae',0):.3f} "
                          f"n={ovr.get('n',0)}  use_gold={args.val_use_gold}",
                          file=sys.stderr)
                if accelerator is not None:
                    accelerator.wait_for_everyone()

            if args.val_full_every > 0 and (step % args.val_full_every == 0 or step >= args.max_steps):
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                # Rank-0 prefetch + barrier (full sweep variant).
                if is_main:
                    try:
                        prefetch_pointwise_videos(ROOT, max_samples_per_dim=None)
                    except Exception as e:
                        print(f"[val-FULL {step:5d}] prefetch failed: {e} — "
                              "continuing.", file=sys.stderr)
                if accelerator is not None:
                    accelerator.wait_for_everyone()
                # Same ZeRO-3 deadlock fix: every rank runs forwards.
                ovr_full, heatmap_full = _run_val_eval(
                    student, task_specs, gold_by_dim,
                    use_gold_rubric=args.val_use_gold,
                    max_samples_per_dim=None,    # full sweep
                    repo_root=ROOT,
                    step=step,
                    out_dir=out_dir,
                    progress_every=20,
                    write_artifacts=is_main,
                    prefetch_missing=False,
                )
                if is_main:
                    _wandb_log_val(wandb_run, step, ovr_full, heatmap_full,
                                   tag="val_full")
                    print(f"[val-FULL {step:5d}] plcc={ovr_full.get('plcc',0):+.3f} "
                          f"srcc={ovr_full.get('srcc',0):+.3f} "
                          f"mae={ovr_full.get('mae',0):.3f} "
                          f"n={ovr_full.get('n',0)}", file=sys.stderr)
                if accelerator is not None:
                    accelerator.wait_for_everyone()

    log_fh.close()
    summary_path = out_dir / "summary.json"
    summary = {
        "max_steps":      args.max_steps,
        "G":              args.num_generations,
        "valid_ratio_ema": tracker.current(),
        "lambda_at_end":  tracker.lambda_t(args.lambda_0),
        "beta_at_end":    beta_scheduler.beta(max(0, step - 1)),
        "mode":           "dummy" if (dummy_env or args.student_model is None) else "real",
        "wall_s":         round(time.time() - t0, 2),
        "config":         asdict(cfg),
    }
    if is_main:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                encoding="utf-8")
        # Stash a final summary blob into wandb (graceful no-op when disabled).
        if wandb_run is not None:
            try:
                wandb_run.summary.update(summary)
            except Exception:
                pass
        _wandb_finish(wandb_run)
        print(f"[train] finished. summary -> {summary_path}", file=sys.stderr)
    if accelerator is not None:
        accelerator.wait_for_everyone()


# ---------------------------------------------------------------------------
# 5. Optimiser + checkpoint helpers.
# ---------------------------------------------------------------------------
def _build_optimizer(student: StudentWrapper, args: argparse.Namespace):
    """Create AdamW over the student's trainable params (LoRA only).

    Returns None when the student is in dummy mode or has no trainable
    params; the policy_update path detects this and short-circuits.

    NOTE on lazy loading
    --------------------
    StudentWrapper defers ``_ensure_model`` until the first forward, but
    LoRA is injected INSIDE ``_ensure_model``. That means before any
    forward, ``trainable_parameters()`` is empty even with --lora_config.
    Force the load here so the optimiser sees the LoRA params.
    """
    if student.is_dummy():
        print("[optim] dummy mode — no optimiser created.", file=sys.stderr)
        return None
    # Trigger lazy LoRA injection before counting trainable params.
    student._ensure_model()
    params = list(student.trainable_parameters())
    if not params:
        print("[optim] no trainable params (did you forget --lora_config?). "
              "Continuing in evaluation-only mode.", file=sys.stderr)
        return None
    import torch
    optim = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    n = sum(_param_full_numel_safe(p) for p in params)
    print(f"[optim] AdamW over {len(params)} param tensors ({n/1e6:.1f}M elems) "
          f"lr={args.learning_rate} wd={args.weight_decay}", file=sys.stderr)
    return optim


def _save_ckpt(student: StudentWrapper, out_dir: Path, step: int,
               *, accelerator=None, is_main: bool = True) -> None:
    """Save the model under ``out_dir/checkpoint-<step>``.

    Two modes:
      * LoRA: ``save_pretrained`` writes only the small adapter (~40MB).
        The base model lives at ``cfg.model_path`` and is unchanged.
      * Full-param: writes the entire 7B (~15 GB) as a consolidated
        bf16 state_dict. Under DeepSpeed ZeRO-3 each rank holds a 1/N
        shard of every param, so a naive ``save_pretrained`` from rank-0
        writes a state_dict full of empty placeholders — a successfully
        loaded such ckpt produces the
        ``Error(s) in loading state_dict: size mismatch ...``
        ``copying a param with shape torch.Size([0])`` failure mode
        observed 2026-06-14 on checkpoint-60.

        The fix uses ``accelerator.get_state_dict(model)`` which is the
        accelerate API that triggers DeepSpeed's
        ``zero3_save_16bit_model`` consolidation: every rank participates
        in the gather (collective — DO NOT gate on rank-0), the result
        materialises only on rank-0 as a complete bf16 state_dict, and
        rank-0 then passes it explicitly to ``save_pretrained(state_dict=...)``.
        Non-main ranks return early after the gather without writing
        anything to disk.

    Threading
    ---------
    This function MUST be invoked on EVERY rank (not just rank-0). The
    state_dict gather is a collective and rank-0-only invocation hangs
    on the watchdog timeout under ZeRO-3. The caller passes ``is_main``
    so we can branch the disk-side I/O after the collective completes.
    """
    if student.model is None:
        return
    ck = out_dir / f"checkpoint-{step}"

    # Decide save_model + state_dict based on backend:
    #
    # ZeRO-3 / DeepSpeed: every rank must participate in
    # ``accelerator.get_state_dict`` (it's a collective). Result is a
    # full bf16 state_dict on rank-0, empty dict on others.
    #
    # Non-DeepSpeed (single-process or DDP): unwrap and save_pretrained
    # directly — every param is fully replicated, no gather needed.
    state_dict = None
    using_accel_consolidate = False
    if accelerator is not None:
        # ``get_state_dict`` is the accelerate API that handles
        # ZeRO-3 → 16bit / FSDP / plain DDP consolidations uniformly.
        # Under DeepSpeed ZeRO-3 + zero3_save_16bit_model=True, this
        # gathers on rank-0; on other ranks it returns {} (or None on
        # some versions). It's a no-op for the unwrapped CPU model.
        try:
            state_dict = accelerator.get_state_dict(student._model)
            using_accel_consolidate = True
        except Exception as e:
            # Older accelerate versions may not implement get_state_dict
            # for every backend; fall through to direct save_pretrained
            # (correct for non-ZeRO-3 paths but BROKEN under ZeRO-3 —
            # warn loudly so this isn't silently corrupting ckpts).
            if is_main:
                print(f"[ckpt] accelerator.get_state_dict failed at step "
                      f"{step}: {e}; falling back to unwrap+save_pretrained "
                      f"(THIS IS UNSAFE under ZeRO-3 — verify ckpt loads).",
                      file=sys.stderr)

    if not is_main:
        # Non-main ranks: participated in the collective above, now bail
        # out without touching the filesystem.
        return

    ck.mkdir(parents=True, exist_ok=True)
    try:
        save_model = (accelerator.unwrap_model(student._model)
                      if accelerator is not None else student._model)
        if using_accel_consolidate and state_dict is not None:
            # Pass the consolidated state_dict so save_pretrained writes
            # the gathered weights instead of the per-rank shards.
            save_model.save_pretrained(str(ck), state_dict=state_dict)
        else:
            save_model.save_pretrained(str(ck))
    except Exception as e:
        print(f"[ckpt] save_pretrained failed at step {step}: {e}", file=sys.stderr)
        return
    try:
        if student._tokenizer is not None:
            student._tokenizer.save_pretrained(str(ck))
    except Exception:
        pass
    # ── Save the AutoProcessor (preprocessor_config.json + image-processor
    # cfg + chat-template). Without this, downstream loaders that call
    # ``AutoProcessor.from_pretrained(ckpt_dir)`` (e.g. eval_ckpt.py,
    # deploy paths) fail with "Can't load image processor for ..." — the
    # ckpt has only tokenizer + model weights, no preprocessor metadata.
    # Saving here costs <100 KB and zero wall-time; the omission was a
    # historical oversight.
    try:
        proc = getattr(student, "_processor", None)
        if proc is not None:
            proc.save_pretrained(str(ck))
    except Exception as e:
        print(f"[ckpt] processor.save_pretrained failed at step {step}: {e}",
              file=sys.stderr)
    # Refresh "latest" symlink for convenience.
    latest = out_dir / "checkpoint-latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(ck.name)
    except Exception:
        pass
    print(f"[ckpt] step {step} -> {ck}", file=sys.stderr)


def _make_val_score_fn(
    student:    StudentWrapper,
    task_specs: dict[str, TaskSpec],
    *,
    use_gold_rubric: bool,
    gold_by_dim:     dict[str, list[dict]] | None = None,
) -> "Callable[[str, str, str], float]":
    """Build a ``score_fn(dim, caption, video_path) -> float`` for eval.

    Two flavours:

    ``use_gold_rubric=True``
        Use the per-dim gold rubric (from Stage ① v2.txt, looked up via
        ``gold_by_dim``) as the rubric input to ``student.score_video``.
        This is the train/test-aligned validation: it measures how well
        the student SCORES given a fixed rubric, isolating score quality
        from rubric quality. Equivalent to stage0 D6 baseline path.

    ``use_gold_rubric=False`` (default during training)
        Sample the rubric from the student first (no_grad), then pass it
        to ``score_video``. This is the deployment-equivalent path —
        what the trained RM will actually do at inference time. Strictly
        the "real" deployment metric.

    The returned callable is closure-captured; the trainer calls it once
    per (dim, caption, video) inside ``run_pointwise_eval``.
    """
    from rm_training.stage2_grpo.rubric_parser import (
        build_full_from_scores, format_score, parse_rubric_only,
    )
    from rm_training.stage2_grpo.student_wrapper import DimensionContext

    def _ctx_for(dim: str) -> DimensionContext:
        spec = task_specs.get(dim) if task_specs else None
        if spec is None:
            return DimensionContext.generic(dim)
        return DimensionContext(
            dimension        = spec.dimension,
            evalverse_path   = (
                f"{spec.evalverse_aspect} > {spec.evalverse_main}"
                if spec.evalverse_aspect and spec.evalverse_main else spec.dimension
            ),
            task_description = spec.task_description_short,
            # SCORE-side prompt uses the one-liner (see _ctx_from_specs).
            task_description_score = spec.oneline(),
            sub_aspects      = list(spec.sub_aspects),
        )

    def _gold_rubric_text(dim: str) -> str:
        """Build a JSON rubric blob from the per-dim gold list."""
        entries = (gold_by_dim or {}).get(dim) or []
        if not entries:
            # Fallback: an empty rubric — student.score_video will still
            # produce a 5-theme output via the default scaffold prompt.
            return '{"rubrics": []}'
        # Match the JSON schema RUBRIC_PROMPT_TMPL emits so the parser
        # path is the same.
        return json.dumps({"rubrics": entries}, ensure_ascii=False)

    def score_fn(dim: str, caption: str, video_path: str) -> float:
        ctx = _ctx_for(dim)
        if use_gold_rubric:
            rubric_text   = _gold_rubric_text(dim)
            rubric_parsed = parse_rubric_only(rubric_text)
        else:
            rubric_text   = student.generate_rubric(caption, dim_ctx=ctx)
            rubric_parsed = parse_rubric_only(rubric_text)

        score_text, scores = student.score_video(
            caption, video_path, rubric_text,
            dim_ctx=ctx,
            rubric_themes=[r.theme for r in rubric_parsed.rubrics] or None,
        )
        if not rubric_parsed.rubrics:
            # Degenerate: empty rubric → fall back to uniform-mean of scores.
            valid = [s for s in scores if s is not None]
            return float(sum(valid) / len(valid)) if valid else float("nan")
        full = build_full_from_scores(rubric_parsed.rubrics, scores)
        S = format_score(full)
        return float(S) if S is not None else float("nan")

    return score_fn


def _build_gold_by_dim(ds: GoldRubricDataset) -> dict[str, list[dict]]:
    """Pull one canonical gold rubric per dim out of the training jsonl.

    Stage ① emits one ``gold_rubric`` per pair, but every pair within a
    dim shares the same per-dim v2.txt (Stage ① is per-dim, not per-pair).
    We just take the first row per dim.
    """
    out: dict[str, list[dict]] = {}
    for row in ds.rows:
        dim = row.get("dimension")
        gold = row.get("gold_rubric")
        if dim and gold and dim not in out:
            out[dim] = gold
    return out


def _run_val_eval(
    student:        StudentWrapper,
    task_specs:     dict[str, TaskSpec],
    gold_by_dim:    dict[str, list[dict]],
    *,
    use_gold_rubric: bool,
    max_samples_per_dim: int | None,
    repo_root:       Path,
    step:            int,
    out_dir:         Path,
    progress_every:  int = 0,
    write_artifacts: bool = True,
    prefetch_missing: bool = False,
) -> dict:
    """Run a PLCC/SRCC val pass, write a per-step JSON, return the summary.

    Side-effect file: ``<out_dir>/val_step{step}.json`` carries the full
    heatmap; the caller inlines the ``overall`` block into ``train_log``.

    ``write_artifacts``
    -------------------
    True on rank-0 (caller writes the JSON / wandb payloads); False on
    non-main ranks under ZeRO-3 (those ranks must still execute the
    ``score_fn`` forwards because the underlying ``_summon_full_params``
    + ``_pause_grad_ckpt`` path issues ZeRO-3 ``all_gather`` collectives
    that REQUIRE every rank to participate in lockstep — otherwise NCCL
    deadlocks for the watchdog timeout). Non-main ranks compute, then
    drop their result on the floor.

    ``prefetch_missing``
    --------------------
    Default False. Prefetching MUST run rank-0-only BEFORE this call (see
    ``prefetch_pointwise_videos``). Leaving True here would let every rank
    independently race on the same ``<sha1>_<basename>.mp4.part`` file,
    which causes per-rank "found locally vs not" divergence and a
    downstream ZeRO-3 all_gather deadlock at the next ``score_fn`` call.
    """
    if student.model is None and not student.is_dummy():
        return {"plcc": 0.0, "srcc": 0.0, "mae": 0.0, "n": 0}, {}

    score_fn = _make_val_score_fn(
        student, task_specs,
        use_gold_rubric=use_gold_rubric,
        gold_by_dim=gold_by_dim,
    )
    # Switch to eval mode. The trainer flips back to train() inside
    # policy_update — this won't conflict because val runs OUTSIDE the
    # backward path.
    if student.model is not None:
        was_training = student.model.training
        student.model.eval()

    # ── Force GREEDY + larger budget on the val score forward (2026-06-12).
    #
    # Why
    # ---
    # Training enables ``--score_sample`` (cfg.score_greedy=False, ~32-token
    # budget) so the score-side policy gradient gets stochastic rollouts.
    # Val inherits the same cfg by accident, which collapses scoring under
    # student-generated rubrics:
    #   * sampling at low budget early-EOSes ("theme", "theme_1:") and
    #     stub-fails ~25-30% of rows (probe v8 variant=A: 27-35% finite,
    #     stub=24-30 / 320, no_anchor=8-20 / 320),
    #   * student rubrics' instruction-shaped tips ("Ensure ...",
    #     "Avoid ...") + multi-line JSON tail bias the model into
    #     CONTINUING the rubric instead of switching to the scaffold,
    #   * gold rubrics avoid both because they're tighter and more
    #     reference-shaped (probe v8 variant=K: 100% finite at the same
    #     budget + sampling).
    #
    # Greedy + 64 tokens removes ALL three failure modes for val:
    #   * argmax after "theme_k:" is always a digit, never EOS → no early stop
    #   * no sampling noise to drift into rubric continuation
    #   * 2× budget absorbs any prefix drift if it does happen
    #
    # We restore the cfg flags after run_pointwise_eval so policy_update
    # next step still sees the trainer's own settings (sampling on, original
    # budget). This is the same pattern as the train()/eval() toggle above.
    cfg_was_greedy = None
    cfg_was_budget = None
    if hasattr(student, "cfg"):
        cfg_was_greedy = getattr(student.cfg, "score_greedy", None)
        cfg_was_budget = getattr(student.cfg, "score_max_new_tokens", None)
        student.cfg.score_greedy = True
        # Use 64 (matches probe_score_v4_e2e known-good setup); only bump
        # if the trainer was already using a larger value.
        if cfg_was_budget is not None and cfg_was_budget >= 64:
            pass  # already big enough
        else:
            student.cfg.score_max_new_tokens = 64

    val_t0 = time.time()
    report = run_pointwise_eval(
        score_fn=score_fn,
        repo_root=repo_root,
        max_samples_per_dim=max_samples_per_dim,
        progress_every=progress_every,
        prefetch_missing=prefetch_missing,
    )
    wall = round(time.time() - val_t0, 2)

    # Restore train/eval mode and cfg flags before returning.
    if student.model is not None and was_training:
        student.model.train()
    if hasattr(student, "cfg"):
        if cfg_was_greedy is not None:
            student.cfg.score_greedy = cfg_was_greedy
        if cfg_was_budget is not None:
            student.cfg.score_max_new_tokens = cfg_was_budget

    if write_artifacts:
        val_path = out_dir / f"val_step{step}.json"
        payload = {
            "step":           step,
            "use_gold_rubric": use_gold_rubric,
            "max_samples_per_dim": max_samples_per_dim,
            "wall_s":         wall,
            "result":         report,
        }
        val_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    return report["overall"], report.get("heatmap", {})


# ---------------------------------------------------------------------------
# 5b. Optional wandb logger.
#
# Defaults to OFF — pass ``--wandb_project <name>`` to enable. Designed so
# every call site has a no-op shape when ``run`` is None (no module-wide
# globals; trivially DDP-friendly: the trainer only constructs ``run`` on
# rank 0, so a plain rank-1+ ``train()`` instance never touches wandb).
# ---------------------------------------------------------------------------
def _wandb_init(args: argparse.Namespace, cfg: BatchRewardConfig):
    """Best-effort wandb.init; returns the run object or None if disabled.

    Skips silently when:
      * wandb is not installed (``ImportError``);
      * ``--wandb_project`` is empty/None;
      * a non-zero ``RANK`` env var is set (DDP non-main rank).
    """
    if not getattr(args, "wandb_project", None):
        return None
    # In a future DDP launch, accelerate / torchrun set RANK on every
    # worker; only rank-0 should talk to wandb to avoid duplicate runs.
    if int(os.environ.get("RANK", "0")) != 0:
        return None
    try:
        import wandb                                            # type: ignore
    except ImportError:
        print("[wandb] wandb not installed — skipping. "
              "(`pip install wandb` to enable)", file=sys.stderr)
        return None

    init_kwargs = dict(
        project=args.wandb_project,
        name=args.wandb_name or Path(args.output_dir).name,
        config={
            **vars(args),
            **{f"reward_cfg.{k}": v for k, v in asdict(cfg).items()},
        },
        dir=str(Path(args.output_dir).resolve()),
        reinit=True,
    )
    if getattr(args, "wandb_entity", None):
        init_kwargs["entity"] = args.wandb_entity
    if getattr(args, "wandb_offline", False):
        init_kwargs["mode"] = "offline"
    if getattr(args, "wandb_tags", None):
        init_kwargs["tags"] = list(args.wandb_tags)

    try:
        run = wandb.init(**init_kwargs)
    except Exception as e:                                      # pragma: no cover
        print(f"[wandb] init failed: {e} — continuing without wandb",
              file=sys.stderr)
        return None
    print(f"[wandb] run initialised: {run.url if hasattr(run, 'url') else run.name}",
          file=sys.stderr)
    return run


def _wandb_log_step(run, metrics: dict) -> None:
    """Log per-step training metrics to wandb (no-op when run is None).

    GRPO-specific stats
    -------------------
    Slim payload (12 keys) — every key is something I'd actually look at
    to decide if training is healthy. The previous version emitted ~50
    keys (mean/std/min/max for every reward component) which buried the
    signal under noise.

    Health-check guide
    ------------------
      * ``train/reward_std → 0``  ⇒ all G rollouts got the same reward
                                     ⇒ advantages ≈ 0 ⇒ policy gradient dead.
                                     This is the single most diagnostic field.
      * ``train/kl`` jumps up     ⇒ policy drifting from ref. Schulman k3
                                     KL is non-negative; healthy values are
                                     0.001 – 0.1. Persistent ``> 1`` means
                                     KL coefficient too small or LR too big.
      * ``train/grad_norm`` huge  ⇒ unstable. Tighten ``max_grad_norm``.
      * ``train/valid_ratio_ema`` ⇒ fraction of rollouts where
                                     ``|s_pos − s_neg| > eps_valid``. Cold
                                     start sits low; should rise as the model
                                     learns to discriminate the preference pair.
    """
    if run is None:
        return
    import statistics as _stats

    rewards    = metrics.get("rewards") or []
    advantages = metrics.get("advantages") or []
    breakdowns = metrics.get("breakdowns") or []

    r_pref_pr  = [b.get("r_pref")   for b in breakdowns if b.get("r_pref")   is not None]
    r_align_pr = [b.get("r_align")  for b in breakdowns if b.get("r_align")  is not None]
    r_fmt_pr   = [b.get("r_format") for b in breakdowns if b.get("r_format") is not None]

    def _safe_mean(xs):
        return (sum(xs) / len(xs)) if xs else 0.0

    def _safe_std(xs):
        if len(xs) < 2:
            return 0.0
        try:
            return _stats.stdev(xs)
        except _stats.StatisticsError:
            return 0.0

    # Slim payload — one key per *independent* diagnostic axis. Anything
    # that's a transform of another key (pg_loss ≈ loss − β·kl,
    # reward_mean is bias-only, advantage_abs ∝ reward_std, valid_count
    # vs valid_ratio_ema) gets dropped. Add fields back only if you
    # catch a bug that needed them.
    payload = {
        # Optimizer health
        "train/loss":            metrics.get("loss"),
        "train/kl":              metrics.get("kl"),
        "train/grad_norm":       metrics.get("grad_norm"),
        # Reward signal (std is THE diagnostic — →0 means GRPO is dead)
        "train/reward_std":      _safe_std(rewards),
        # Sub-rewards — every component the loss sees
        "train/r_pref_mean":     _safe_mean(r_pref_pr),
        "train/r_align_mean":    _safe_mean(r_align_pr),
        "train/r_format_mean":   _safe_mean(r_fmt_pr),
        # Validity gate (smoothed; raw count visible in stderr)
        "train/valid_ratio_ema": metrics.get("valid_ratio_ema"),
        # Schedules
        "train/lambda_t":        metrics.get("lambda_t"),
        "train/beta_format_t":   metrics.get("beta_t"),
    }

    # Strip None / NaN payload (wandb prefers absent over null).
    cleaned = {}
    for k, v in payload.items():
        if v is None: continue
        try:
            if isinstance(v, float) and v != v:    # NaN
                continue
        except Exception:
            pass
        cleaned[k] = v

    try:
        run.log(cleaned, step=metrics.get("step"))
    except Exception as e:                                      # pragma: no cover
        print(f"[wandb] log_step failed: {e}", file=sys.stderr)


def _wandb_log_val(run, step: int, overall: dict, heatmap: dict | None = None,
                   tag: str = "val") -> None:
    """Log a validation pass's overall + per-dim heatmap to wandb."""
    if run is None:
        return
    try:
        import wandb                                            # type: ignore
    except ImportError:
        return
    payload = {
        f"{tag}/plcc":       overall.get("plcc"),
        f"{tag}/srcc":       overall.get("srcc"),
        f"{tag}/mae":        overall.get("mae"),
        f"{tag}/n":          overall.get("n"),
        f"{tag}/S_mean":     overall.get("S_mean"),
        f"{tag}/label_mean": overall.get("label_mean"),
    }
    # macro-PLCC = mean over per-dim PLCCs. More informative than the
    # flat overall when dim distributions vary widely (see B-task notes).
    if heatmap:
        per_dim_plcc = [h["plcc"] for h in heatmap.values()
                        if isinstance(h, dict) and "plcc" in h]
        if per_dim_plcc:
            payload[f"{tag}/macro_plcc"] = sum(per_dim_plcc) / len(per_dim_plcc)
        # Per-dim scalars — easy to plot a single dim over time.
        for dim, h in heatmap.items():
            if not isinstance(h, dict):
                continue
            for k in ("plcc", "srcc", "mae"):
                v = h.get(k)
                if v is not None:
                    payload[f"{tag}_dim/{dim}/{k}"] = v
    try:
        run.log(payload, step=step)
        # Also drop a sortable per-dim heatmap table for the wandb UI.
        if heatmap:
            tbl = wandb.Table(columns=["dim", "plcc", "srcc", "mae", "n", "S_mean", "label_mean"])
            for dim, h in heatmap.items():
                if isinstance(h, dict):
                    tbl.add_data(
                        dim,
                        h.get("plcc"),  h.get("srcc"),  h.get("mae"),
                        h.get("n"),     h.get("S_mean"), h.get("label_mean"),
                    )
            run.log({f"{tag}_table/step_{step}": tbl}, step=step)
    except Exception as e:                                      # pragma: no cover
        print(f"[wandb] log_val failed: {e}", file=sys.stderr)


def _wandb_log_sample(run, step: int, metrics: dict,
                      *, max_rubric_chars: int = 1500,
                      max_caption_chars: int = 200) -> None:
    """Dump ONE rollout's full text payload to wandb.

    For every ``--wandb_sample_every`` train step we pick the first
    rollout (g=0) of the current instance and emit a wandb.Table row
    containing:

      step, dim, caption_head, rubric_text,
      scores_pos (5-vec), scores_neg (5-vec),
      S_pos, S_neg,                          # weighted means
      r_pref, r_align, r_format, reward,     # per-rollout reward parts
      advantage,                             # GRPO-normalised advantage
      valid                                  # |S_pos - S_neg| > eps?

    Logged under the key ``samples/step_<step>``. wandb's UI appends
    each new step's table to a single growing artifact, so you can
    scroll through the rubric evolution from step 0 to step N.

    Caps:
      * caption truncated to ``max_caption_chars`` (the v_arr_mvp
        prompts are very long; full text isn't useful here).
      * rubric truncated to ``max_rubric_chars`` (≈ 5 themes × 3 tips
        × 60 chars).
    """
    if run is None:
        return
    rollouts    = metrics.get("rollouts") or []
    breakdowns  = metrics.get("breakdowns") or []
    advantages  = metrics.get("advantages") or []
    rewards     = metrics.get("rewards") or []
    if not rollouts or not breakdowns:
        return
    g_idx = 0     # always the first rollout — comparable across steps
    if g_idx >= len(rollouts):
        return
    ro = rollouts[g_idx]
    bd = breakdowns[g_idx]

    def _trunc(s: str | None, n: int) -> str:
        if not s: return ""
        return s if len(s) <= n else s[:n] + " …"

    # The student's emitted rubric — try the parsed object's raw_text
    # first, then fall back to a json render of its rubric entries.
    rubric_text = ""
    if hasattr(ro, "student_rubric"):
        rubric_text = getattr(ro.student_rubric, "raw_text", "") or ""
        if not rubric_text and getattr(ro.student_rubric, "rubrics", None):
            try:
                rubric_text = json.dumps(
                    {"rubrics": [
                        {"theme": r.theme, "weight": r.weight,
                         "tips": list(r.tips)}
                        for r in ro.student_rubric.rubrics
                    ]},
                    ensure_ascii=False, indent=2,
                )
            except Exception:
                rubric_text = repr(ro.student_rubric.rubrics)

    def _fmt_scores(parsed_full):
        if not parsed_full or not getattr(parsed_full, "scores", None):
            return "[]"
        return "[" + ", ".join(
            ("?" if s is None or not (s == s) else f"{float(s):.2f}")
            for s in parsed_full.scores
        ) + "]"

    try:
        import wandb                                            # type: ignore
    except ImportError:
        return

    cols = ["step", "dim", "caption_head", "rubric_text",
            "scores_pos", "scores_neg", "S_pos", "S_neg",
            "r_pref", "r_align", "r_format", "reward", "advantage", "valid"]
    table = wandb.Table(columns=cols)
    table.add_data(
        step,
        metrics.get("dimension", ""),
        _trunc(metrics.get("caption"), max_caption_chars),
        _trunc(rubric_text, max_rubric_chars),
        _fmt_scores(getattr(ro, "student_full_pos", None)),
        _fmt_scores(getattr(ro, "student_full_neg", None)),
        bd.get("s_pos") if bd.get("s_pos") is not None else float("nan"),
        bd.get("s_neg") if bd.get("s_neg") is not None else float("nan"),
        bd.get("r_pref"),
        bd.get("r_align"),
        bd.get("r_format"),
        rewards[g_idx] if g_idx < len(rewards) else None,
        advantages[g_idx] if g_idx < len(advantages) else None,
        bool(bd.get("valid")),
    )
    try:
        run.log({f"samples/step_{step}": table}, step=step)
    except Exception as e:                                      # pragma: no cover
        print(f"[wandb] log_sample failed: {e}", file=sys.stderr)


def _stderr_log_sample(step: int, metrics: dict,
                       *, max_rubric_chars: int = 1500,
                       max_caption_chars: int = 200) -> None:
    """Dump ONE rollout's full text payload to stderr (captured by train log).

    Mirrors :func:`_wandb_log_sample` but writes a multi-line plaintext block
    that survives in the redirected training log without any wandb dependency.
    Useful for inspecting how rubric / scores evolve when wandb is offline,
    or when you want the trace inline with stderr metrics.
    """
    rollouts   = metrics.get("rollouts") or []
    breakdowns = metrics.get("breakdowns") or []
    advantages = metrics.get("advantages") or []
    rewards    = metrics.get("rewards") or []
    if not rollouts or not breakdowns:
        return
    g_idx = 0
    if g_idx >= len(rollouts):
        return
    ro = rollouts[g_idx]
    bd = breakdowns[g_idx]

    def _trunc(s, n):
        if not s: return ""
        return s if len(s) <= n else s[:n] + " …"

    rubric_text = ""
    if hasattr(ro, "student_rubric"):
        rubric_text = getattr(ro.student_rubric, "raw_text", "") or ""
        if not rubric_text and getattr(ro.student_rubric, "rubrics", None):
            try:
                rubric_text = json.dumps(
                    {"rubrics": [
                        {"theme": r.theme, "weight": r.weight,
                         "tips": list(r.tips)}
                        for r in ro.student_rubric.rubrics
                    ]},
                    ensure_ascii=False, indent=2,
                )
            except Exception:
                rubric_text = repr(ro.student_rubric.rubrics)

    def _fmt_scores(parsed_full):
        if not parsed_full or not getattr(parsed_full, "scores", None):
            return "[]"
        return "[" + ", ".join(
            ("?" if s is None or not (s == s) else f"{float(s):.2f}")
            for s in parsed_full.scores
        ) + "]"

    dim     = metrics.get("dimension", "")
    caption = _trunc(metrics.get("caption"), max_caption_chars)
    rubric  = _trunc(rubric_text, max_rubric_chars)
    s_pos   = bd.get("s_pos")
    s_neg   = bd.get("s_neg")
    reward  = rewards[g_idx]    if g_idx < len(rewards)    else None
    adv     = advantages[g_idx] if g_idx < len(advantages) else None

    def _f(x, fmt="{:.3f}"):
        try:
            return fmt.format(float(x))
        except Exception:
            return "nan"

    print(
        f"[sample step={step} dim={dim}] caption=\"{caption}\"",
        file=sys.stderr,
    )
    # rubric_text — keep on its own block; multi-line is fine.
    print(f"[sample step={step}] rubric_text=\n{rubric}", file=sys.stderr)
    print(
        f"[sample step={step}] scores_pos={_fmt_scores(getattr(ro, 'student_full_pos', None))} "
        f"S_pos={_f(s_pos)}",
        file=sys.stderr,
    )
    print(
        f"[sample step={step}] scores_neg={_fmt_scores(getattr(ro, 'student_full_neg', None))} "
        f"S_neg={_f(s_neg)}",
        file=sys.stderr,
    )
    print(
        f"[sample step={step}] r_pref={_f(bd.get('r_pref'))} "
        f"r_align={_f(bd.get('r_align'))} r_fmt={_f(bd.get('r_format'))} "
        f"reward={_f(reward)} adv={_f(adv, '{:+.3f}')} valid={bool(bd.get('valid'))}",
        file=sys.stderr, flush=True,
    )


def _wandb_finish(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:                                           # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# 6. Embedder loader (real ↔ dummy).
# ---------------------------------------------------------------------------
def _load_embedder(model_path: str | None):
    if not model_path or os.environ.get("RM_DUMMY", "0") == "1":
        print("[embedder] using DummyEmbedder.", file=sys.stderr)
        return DummyEmbedder()
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        print("[embedder] sentence-transformers not installed; falling back "
              "to DummyEmbedder. (pip install sentence-transformers)",
              file=sys.stderr)
        return DummyEmbedder()
    print(f"[embedder] loading SentenceTransformer({model_path})", file=sys.stderr)
    return SentenceTransformer(model_path)


# ---------------------------------------------------------------------------
# 7. CLI.
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold_rubrics",  required=True,
                   help="Path to Stage ① jsonl, e.g. data/rm_train/gold_rubrics_18d.jsonl")
    p.add_argument("--student_model", default=None,
                   help="Path to Qwen2.5-VL-7B-Instruct (omit to force dummy mode)")
    p.add_argument("--embed_model",   default=None,
                   help="Path to a sentence-transformers model (omit for DummyEmbedder)")
    p.add_argument("--lora_config",   default=None,
                   help="Path to lora_qwen25vl.yaml. When provided, only "
                        "the LoRA adapter is trained (~10M trainable params, "
                        "fits a single 80GB card). When None, full-param "
                        "training is enabled — REQUIRES launching under "
                        "``accelerate launch --config_file <zero3.yaml>`` "
                        "(8×H100 budget) and a separate KL ref via "
                        "--ref_model_path.")
    p.add_argument("--ref_model_path", default=None,
                   help="Path to a frozen reference policy for KL. Only "
                        "used in full-param mode (LoRA mode uses "
                        "model.disable_adapter() instead). Defaults to "
                        "--student_model when full-param + beta_kl > 0.")
    p.add_argument("--ref_zero3", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="When True, partition the reference policy with "
                        "DeepSpeed ZeRO-3 (same as the policy). This "
                        "fixes the 'kl_s ≈ 4.6 nats at step 0 when "
                        "student==ref' bug (Patch 7, 2026-06-16): both "
                        "policy and ref forwards then go through the "
                        "same ZeRO-3 hooks (gather → forward → "
                        "re-shard, fused-kernel selection on sharded "
                        "weights), so KL_s ≈ 0 by construction at step 0. "
                        "Also saves ~12 GB / card (ref shard 1.75 GB vs "
                        "full 14 GB). Default False to preserve legacy "
                        "behaviour for older runs; recommended True for "
                        "any new full-param run.")
    p.add_argument("--attn_implementation",
                   choices=["sdpa", "flash_attention_2", "eager"],
                   default="sdpa",
                   help="Attention kernel. 'sdpa' (default) uses PyTorch "
                        "2.6's built-in flash kernel — no extra deps. "
                        "'flash_attention_2' needs the flash-attn pip "
                        "package (~10-20%% faster on H100). 'eager' "
                        "disables fusion, mostly for debugging.")
    p.add_argument("--no_accelerate", action="store_true",
                   help="Force single-process mode even when accelerate "
                        "is installed. Mostly for unit tests.")
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--num_generations", type=int, default=8,
                   help="G — number of rubric rollouts per training instance")
    p.add_argument("--max_steps",     type=int, default=4)
    p.add_argument("--epochs",        type=int, default=None,
                   help="Train for N full passes over the dataset. When set, "
                        "overrides --max_steps with epochs * (len(ds) // "
                        "world_size). Convenient when you want 'see each "
                        "row N times' rather than counting global steps. "
                        "Leave unset to use --max_steps directly.")
    p.add_argument("--gold_rubric_warmup_epochs", type=int, default=0,
                   help="Use the per-dim Stage ① gold rubric (instead of "
                        "student-generated rubrics) for the FIRST N epochs "
                        "of training. During this warm-up window the rubric "
                        "is held constant per-dim, so r_pref carries a clean "
                        "discrimination signal and the score head can be "
                        "trained at full strength. score_pg_alpha is forced "
                        "to score_pg_alpha_max during warm-up so the score "
                        "leg actually receives gradient (the start_step / "
                        "full_step ramp is suppressed). After warm-up, both "
                        "the rubric and score legs train jointly under the "
                        "standard composition. 0 disables (legacy behaviour).")
    p.add_argument("--log_every",     type=int, default=1)
    p.add_argument("--save_every",    type=int, default=100,
                   help="Save LoRA adapter every N steps; 0 to disable.")

    # ── PLCC/SRCC val on EvalVerse hold-out (test_pointwise_*).
    p.add_argument("--val_every", type=int, default=100,
                   help="Run a fast PLCC/SRCC val pass every N training "
                        "steps. 0 disables. The fast pass caps at "
                        "--val_max_samples_per_dim videos per dim, so "
                        "18 dims × 5 ≈ 90 forwards (a few minutes).")
    p.add_argument("--val_max_samples_per_dim", type=int, default=5,
                   help="Per-dim cap for the fast val pass. Set 0 / None "
                        "(via not passing it) to disable the cap; the "
                        "default 5 keeps the val budget under 5 minutes "
                        "on a single H100/A100 at 18 dims.")
    p.add_argument("--val_full_every", type=int, default=0,
                   help="Run a FULL PLCC/SRCC sweep (no per-dim cap, "
                        "≈540 video × 1 forward) every N steps. Default 0 "
                        "(off). Use sparingly — single full pass is ~3 h.")
    p.add_argument("--val_use_gold",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Val rubric source. Default: student-sampled "
                        "(deployment-equivalent). --val_use_gold uses "
                        "the per-dim Stage ① gold rubric instead, which "
                        "isolates 'how well does the model SCORE' from "
                        "'how well does the model WRITE rubrics'.")

    # ── Wandb logging (optional). All flags default to OFF so existing
    # smoke / dummy / single-card runs are not affected. Pass at least
    # --wandb_project to enable; everything else falls back to sane
    # defaults. DDP-friendly: only rank-0 will actually call wandb.init.
    p.add_argument("--wandb_project", default=None,
                   help="Wandb project name; setting this enables wandb. "
                        "Leave empty to disable. The run name defaults to "
                        "the output_dir basename unless --wandb_name is "
                        "given.")
    p.add_argument("--wandb_name",    default=None,
                   help="Wandb run name (default: output_dir basename).")
    p.add_argument("--wandb_entity",  default=None,
                   help="Wandb entity / team. Defaults to your default "
                        "entity in ~/.netrc.")
    p.add_argument("--wandb_offline", action="store_true",
                   help="Run wandb in offline mode — logs are written "
                        "locally and can be `wandb sync`'d later. Useful "
                        "on GPU nodes without outbound HTTPS.")
    p.add_argument("--wandb_tags",    nargs="*", default=None,
                   help="Optional list of tags for the wandb run.")
    p.add_argument("--wandb_sample_every", type=int, default=50,
                   help="Every N training steps, dump ONE rollout's full "
                        "text payload (rubric_text, scores_pos, scores_neg, "
                        "reward parts, advantage) to wandb as a Table row. "
                        "Lets you scroll through generation evolution in "
                        "the UI. 0 disables.")
    p.add_argument("--sample_log_every", type=int, default=0,
                   help="Every N training steps, dump ONE rollout's full "
                        "text payload (rubric_text / scores / reward parts) "
                        "to stderr — independent of --wandb_sample_every. "
                        "Useful when wandb is offline or you want the trace "
                        "inline with the stderr training log. 0 disables.")
    p.add_argument("--seed",          type=int, default=0)

    # reward knobs
    p.add_argument("--lambda_0",      type=float, default=0.3)
    p.add_argument("--lambda_min",    type=float, default=0.0,
                   help="Floor for λ_align after the step-based anneal "
                        "completes. Default 0.0 (gold rubric is fully retired). "
                        "Set above 0 to keep a small permanent gold anchor.")
    p.add_argument("--lambda_warmup_steps", type=int, default=0,
                   help="Hold λ at λ_0 for the first N optimizer steps "
                        "before retiring it. Default 0 (no warmup → no anneal). "
                        "Recommended 30 — gives the policy a few steps to "
                        "land in the gold rubric's neighbourhood, then hands "
                        "control back to r_pref.")
    p.add_argument("--lambda_decay_steps",  type=int, default=0,
                   help="Linear decay from λ_0 to λ_min over N steps after "
                        "warmup. Default 0 (constant λ_0 forever, legacy). "
                        "Recommended 100 with warmup_steps=30 and λ_min=0 "
                        "for a 5-epoch / 300-step run.")
    p.add_argument("--align_floor",   type=float, default=0.7,
                   help="Saturation threshold for r_align. sim ≥ floor → "
                        "r_align = 1.0 (no extra reward for matching gold "
                        "more closely). Below floor, r_align decays "
                        "linearly. Treat the gold rubric as a warm-start "
                        "anchor, not a ground-truth answer.")
    p.add_argument("--align_saturate", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Toggle the warm-start anchor behaviour. Disable "
                        "with --no-align_saturate to recover the legacy "
                        "linear sim → r_align mapping.")
    p.add_argument("--beta_format",   type=float, default=1.0,
                   help="β_format coefficient. Default 1.0 (same scale as "
                        "r_pref / r_align). With --beta_format_schedule "
                        "constant (default), β stays at this value all "
                        "training. r_format converges to ~1.0 quickly so "
                        "β only matters for the first ~50 steps when "
                        "the schema isn't learned yet — higher β nudges "
                        "the model to nail the format faster.")
    p.add_argument("--beta_format_min", type=float, default=0.05,
                   help="Floor for β_format after annealing. Keep slightly "
                        "above 0 so the format signal stays active and "
                        "prevents schema drift late in training.")
    p.add_argument("--beta_format_decay_steps", type=int, default=500,
                   help="Steps over which β_format decays from β_0 to β_min. "
                        "Beyond this, β stays at β_min. Default 500 ≈ first "
                        "half of the README's max_steps=1000 plan.")
    p.add_argument("--beta_format_schedule",
                   choices=["linear", "cosine", "step", "constant"],
                   default="constant",
                   help="Anneal shape for β_format. Default 'constant': "
                        "schema compliance is a hard contract that should "
                        "stay enforced for the full run. r_format converges "
                        "to ~1.0 once the model learns the schema, so a "
                        "constant β just contributes a small steady offset "
                        "without distorting the pref/align gradient. Use "
                        "linear/cosine only for ablation. (Note: r_align's "
                        "λ schedule is the right place to anneal teacher "
                        "guidance — gold rubric is a warm-start anchor, "
                        "not a permanent contract.)")
    p.add_argument("--eps_valid",     type=float, default=0.05)
    p.add_argument("--hard_pref",     action="store_true",
                   help="Use 0/1 hard preference reward instead of σ(S⁺-S⁻).")
    p.add_argument("--pref_temperature", type=float, default=1.0,
                   help="Sharpen the BT reward: r_pref = σ(τ·(S⁺-S⁻)). "
                        "τ>1 makes the σ steeper around 0, which lifts |S⁺-S⁻| "
                        "above eps_valid more often (raises valid_ratio) and "
                        "gives the policy gradient a stronger signal when "
                        "scores are close. Try 1.5-2.0 if you observe "
                        "valid_ratio < 0.5 after warm-up.")
    p.add_argument("--no_lock_weights", action="store_true",
                   help="Ablation only: disable the fairness lock that "
                        "forces S(y⁺) and S(y⁻) to share the rubric-gen "
                        "weight vector. Default keeps the lock ON.")
    p.add_argument("--valid_window",  type=int,   default=256)
    p.add_argument("--valid_alpha",   type=float, default=0.05)

    # student / sampling knobs
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature",    type=float, default=0.7)
    p.add_argument("--score_sample",   action="store_true",
                   help="Sample (instead of greedy) during the score-video "
                        "forward. Default greedy keeps reward variance from "
                        "the rubric only.")
    p.add_argument("--score_method",
                   choices=["soft_logits", "integer_text"],
                   default="soft_logits",
                   help="How to extract per-theme scores from the score "
                        "forward. soft_logits (default) reads logits over "
                        "{1..5} digit ids and reports E[k]=Σ k·p_k; "
                        "integer_text parses the model's text scaffold.")
    p.add_argument("--score_max_new_tokens", type=int, default=32,
                   help="Max new tokens for the score forward. The plain-"
                        "text scaffold ``theme_1: X theme_2: X ... theme_5: X`` "
                        "tokenises to ~25-30 ids; 32 leaves a small slack and "
                        "halves the score-side step_scores list relative to "
                        "the legacy 64. Bump back to 64 if you observe "
                        "missed digit hits in the parsing path.")
    p.add_argument("--use_task_specs", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Load per-dim task descriptions from "
                        "rubric_pipeline/config/v_arr_mvp_<dim>_t2v.yaml and "
                        "inject them into the rubric/score prompts. Disable "
                        "with --no-use_task_specs for ablation (student "
                        "will see only a generic 'evaluate this dimension' "
                        "stub).")
    p.add_argument("--gradient_checkpointing",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Trade ~30%% wall-time for ~30-50%% activation memory. "
                        "Required when going past G=8 or 1024 max_new_tokens "
                        "on a single 80GB card. Auto-handles the LoRA + "
                        "input_require_grads + use_cache toggling. Disable "
                        "with --no-gradient_checkpointing for max-throughput "
                        "ablation runs.")
    p.add_argument("--grad_ckpt_use_reentrant",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Reentrant=False (default) is the recommended path "
                        "post-PyTorch 1.11 — faster backward, compatible "
                        "with multimodal forwards. Reentrant=True works but "
                        "is 2-3x slower; use only if you hit a graph-rewrite "
                        "edge case.")

    # optimisation knobs
    p.add_argument("--learning_rate",  type=float, default=5.0e-6)
    p.add_argument("--weight_decay",   type=float, default=0.01)
    p.add_argument("--max_grad_norm",  type=float, default=1.0)
    p.add_argument("--warmup_steps",   type=int, default=0,
                   help="Linear LR warmup over the first N optimizer steps. "
                        "Default 0 (disabled). Recommended 20-50 for GRPO "
                        "fine-tuning of a pretrained VLM — protects the LM "
                        "head from cold-start noise when r_pref is still "
                        "zero (valid=0/G) and only format noise drives the "
                        "advantage. After N steps the LR floors at "
                        "--learning_rate.")
    p.add_argument("--beta_kl",        type=float, default=0.04,
                   help="KL coefficient against the reference policy "
                        "(LoRA-disabled forward of the same model).")

    # ── Score-side PG curriculum (off by default).
    #
    # When score_pg_alpha_max > 0, the trainer enables the
    # multi-stage curriculum where the score forward also flows gradient,
    # not just the rubric-gen forward. Implements the policy from the
    # CE-RM (arXiv:2601.20327) and VQ-Insight (arXiv:2506.18564)
    # discussions:
    #
    #   step < score_pg_start_step       → rubric-only PG (legacy)
    #   start ≤ step < score_pg_full_step → α linearly 0 → α_max
    #   step ≥ score_pg_full_step         → α = α_max
    #
    # AND a per-rollout hard format gate: score_pg only fires when
    # fmt_rubric / fmt_pos / fmt_neg ALL ≥ score_format_gate (default
    # 0.65 — paired with the hard 0/1 format reward this means "rubric
    # fully schema-compliant" since each fmt component is now 0 or 1).
    #
    # Requirements: ``--no-score_greedy`` (so score forward samples).
    # Greedy + score_pg_alpha > 0 silently downgrades to no PG capture.
    p.add_argument("--score_pg_alpha_max",  type=float, default=0.0,
                   help="Max α coefficient on score-side log-probs in "
                        "the policy gradient. 0.0 (default) disables the "
                        "score-side PG entirely (rubric-only legacy). "
                        "Recommended 1.0 once you've validated the "
                        "rubric-only baseline and want to unlock score "
                        "training. Lower (0.3-0.5) for a more conservative "
                        "first attempt.")
    p.add_argument("--score_pg_start_step", type=int, default=50,
                   help="Step at which the score-side PG ramp begins. "
                        "Default 50 — gives the rubric forward time to "
                        "settle on schema-clean rubrics before training "
                        "the score forward against them.")
    p.add_argument("--score_pg_full_step",  type=int, default=150,
                   help="Step at which α reaches its max. Linear ramp "
                        "from start_step to full_step. Default 150 → "
                        "100-step ramp window, matching a 5-epoch / "
                        "300-step plan.")
    p.add_argument("--score_format_gate",   type=float, default=0.65,
                   help="Per-rollout hard format gate for score-side PG. "
                        "Score-forward log-probs only enter the loss when "
                        "fmt_rubric / fmt_pos / fmt_neg ALL ≥ this value. "
                        "With the hard 0/1 format reward, 0.65 means "
                        "\"all three components passed\". Lower to 0.5 "
                        "to admit only 2/3 components if the gate is too "
                        "aggressive.")
    # ── BT-aux loss (added 2026-06-16) ──
    p.add_argument("--lambda_bt", type=float, default=0.0,
                   help="Weight of the supervised BT-aux loss applied "
                        "directly on the 5-digit logits at colon-anchored "
                        "positions of the score scaffold:\n"
                        "  L_BT = -log σ(τ·(S⁺ − S⁻))\n"
                        "where S = Σ_k w_k · E[d_k|softmax(digit_logits_k)] "
                        "and the weights w come from the (locked) student "
                        "rubric. Bypasses GRPO's high-variance score-PG "
                        "path and pulls Δ apart deterministically.\n"
                        "  • 0.0 (default) — disabled, recover legacy "
                        "GRPO-only behaviour.\n"
                        "  • 0.05–0.10    — recommended for Stage B "
                        "(joint), as a tie-breaker on top of PG.\n"
                        "  • 0.3–0.5      — recommended for Stage A "
                        "(warmup), where score is the main task.\n"
                        "Gating: only fires when score-side sampling meta "
                        "exists AND the format gate passed (same as "
                        "score-PG) AND both pos / neg sides have ≥1 matched "
                        "digit at the same theme position.")
    p.add_argument("--bt_temperature", type=float, default=3.0,
                   help="τ in -log σ(τ·(S⁺ − S⁻)). Higher = sharper BT "
                        "(stronger gradient near Δ≈0, smaller far away). "
                        "Default 3.0; the existing pref_temperature "
                        "(default 1.5) is unchanged — they live in "
                        "different paths (BT-aux is a supervised loss, "
                        "pref_temperature scales the r_pref reward).")
    # ── Score-gap supervision (added 2026-06-18) ──
    p.add_argument("--lambda_pw", type=float, default=0.0,
                   help="Weight of the score-gap L1 supervision applied "
                        "directly to S⁺ − S⁻ derived from the same "
                        "5-digit logits as BT-aux:\n"
                        "  L_pw = | (S⁺ − S⁻) − Δ̂ |   "
                        "where Δ̂ = 4·(s_pos_target − s_neg_target).\n"
                        "Targets come from data_collator.attach_score_targets "
                        "which joins winner_score / loser_score from each "
                        "row's source_pair_file (range 0-1; ×4 lifts to the "
                        "model's 1-5 digit range). Replaces BT for warm-up:\n"
                        "  • BT pushes margin to ∞ regardless of magnitude\n"
                        "    → mode-collapses to all-1 / all-5 (warmup-2 "
                        "    log step 30: scores_pos = scores_neg = "
                        "    [1,1,1,1,1]).\n"
                        "  • gap-L1 matches the annotated magnitude, so a "
                        "    pair with target gap 0.05 only asks for a 0.2 "
                        "    spread in the 1-5 range — no incentive to "
                        "    saturate.\n"
                        "Recommended: 0.5 (warm-up). Set 0 to disable.\n"
                        "Pointwise calibration is INTENTIONALLY NOT done — "
                        "the student's 5 themes are dynamically generated "
                        "and don't share an axis with evalverse's sub_keys, "
                        "so absolute-scale alignment is unsound. Differences "
                        "ARE comparable: with a fixed student rubric, "
                        "S⁺ − S⁻ is rubric-invariant.")
    return p


def _install_crash_diagnostics() -> None:
    """Install faulthandler so SIGSEGV / SIGABRT / SIGBUS dumps the Python
    stacks of every thread on every rank to stderr (and to a per-rank file)
    *before* the process dies.

    Why: ZeRO-3 + Qwen-VL + flash-attn + decord are all C-extension stacks
    that have segfaulted us repeatedly with no Python traceback. Without
    faulthandler we only see ``exitcode: -11`` from the launcher — useless.
    With it we get the Python frame the C call dispatched from, which
    almost always points at the offending sample / forward path.

    Also enables SIGUSR1 as a manual "dump all stacks now" trigger:
        kill -USR1 <pid>   # works on any rank's process, prints to stderr
    Useful when a rank is wedged in a NCCL collective and we want to know
    which Python line each rank is parked on.
    """
    import faulthandler
    import signal as _sig

    rank_str = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    out_dir  = os.environ.get("RM_FAULT_DIR", "/tmp")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass
    fh_path = os.path.join(out_dir, f"faulthandler_rank{rank_str}.log")
    try:
        # Keep a long-lived file handle so faulthandler can write into it
        # even after the interpreter is in a bad state. Stash on the module
        # so it isn't GC'd.
        global _FAULTHANDLER_FH
        _FAULTHANDLER_FH = open(fh_path, "a", buffering=1)
        faulthandler.enable(file=_FAULTHANDLER_FH, all_threads=True)
        # Also mirror to stderr so the launcher log captures it.
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception as e:
        print(f"[faulthandler] enable failed: {e}", file=sys.stderr)
        return

    # Manual dump trigger: kill -USR1 <pid>
    try:
        faulthandler.register(_sig.SIGUSR1, file=sys.stderr,
                              all_threads=True, chain=False)
    except Exception:
        pass

    print(f"[faulthandler] enabled rank={rank_str} -> {fh_path} "
          f"(SIGUSR1 dumps stacks on demand)",
          file=sys.stderr, flush=True)


_FAULTHANDLER_FH = None  # held alive across the run


def main() -> None:
    _install_crash_diagnostics()
    p = build_argparser()
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
