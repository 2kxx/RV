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
)
from rm_training.stage2_grpo.valid_filter    import ValidRatioTracker


# ---------------------------------------------------------------------------
# 1. Group-relative advantage.
# ---------------------------------------------------------------------------
def grpo_advantage(rewards: list[float]) -> list[float]:
    """Standard GRPO normalisation::

        a_g = (r_g − mean(r)) / (std(r) + ε)
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
    eps = 1e-6
    return [(r - m) / (s + eps) for r in rewards]


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
) -> dict:
    """Roll out G times → reward → advantage → optional LM backward.

    Returns a metrics dict including ``rollout_meta`` (the per-rollout
    RubricRollout objects, real mode only) so the driver loop can run
    the policy-gradient update without re-sampling.
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
    for g in range(G):
        # --- (i) sample rubric.
        if student.is_dummy():
            rubric_text = student.generate_rubric(caption, dim_ctx=dim_ctx)
            rmeta = None
        else:
            rmeta = student.generate_rubric_with_meta(caption, dim_ctx=dim_ctx)
            rubric_text = rmeta.text
        parsed_rubric = parse_rubric_only(rubric_text)

        # --- (ii) cross-score on y⁺ then y⁻ (each forward sees ONE video).
        # Each call returns ``(text, list[float|None])`` of length 5
        # (matching the rubric size). soft_logits gives continuous floats
        # in [1.0, 5.0]; integer_text gives {1..5} ints.
        score_pos_text, scores_pos = student.score_video(
            caption, y_pos, rubric_text,
            dim_ctx=dim_ctx,
            rubric_paths=[r.path for r in parsed_rubric.rubrics],
        )
        score_neg_text, scores_neg = student.score_video(
            caption, y_neg, rubric_text,
            dim_ctx=dim_ctx,
            rubric_paths=[r.path for r in parsed_rubric.rubrics],
        )
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
    )
    advantages = grpo_advantage(rewards)

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
) -> dict:
    """Run one optimiser step over G rollouts of a single training instance.

    The loss for each rollout is::

        L_g = - advantage_g · mean_t log π_θ(a_t | s_<t)
              + β_kl · mean_t (log π_θ - log π_ref)

    The total loss averages over the G rollouts. We do a single
    backward + step per training instance, which matches the GRPO
    paper's "one update per group" recipe and keeps memory bounded
    (we don't have to hold G separate computation graphs).

    In dummy mode (rollout_meta entries are None) this returns a
    zero-diagnostic dict and never touches the optimiser, so the
    same code path also smoke-tests the data flow on CPU.
    """
    out = {"loss": 0.0, "pg_loss": 0.0, "kl": 0.0, "skipped": 0, "applied": 0}
    if optimizer is None or student.is_dummy() or not rollout_meta:
        out["skipped"] = len(rollout_meta or [])
        return out

    import torch
    if all(m is None for m in rollout_meta):
        out["skipped"] = len(rollout_meta)
        return out

    # The model needs to be in train mode for dropout / LoRA grads.
    if student.model is not None:
        student.model.train()

    losses    : list = []
    kl_values : list = []
    pg_values : list = []
    for adv, meta in zip(advantages, rollout_meta):
        if meta is None or meta.sampled_ids is None or meta.sampled_ids.numel() == 0:
            out["skipped"] += 1
            continue

        # Grad-on log-probs from the policy.
        with torch.enable_grad():
            logp = student.compute_log_probs(meta.prompt_ids, meta.sampled_ids, ref=False)
        # Detached log-probs from the reference (LoRA disabled).
        logp_ref = student.compute_log_probs(meta.prompt_ids, meta.sampled_ids, ref=True)

        # Per-token mean keeps short and long generations on equal
        # footing — without it, longer outputs would dominate the loss.
        logp_mean    = logp.mean()
        logp_ref_mn  = logp_ref.mean()
        kl           = logp_mean - logp_ref_mn          # KL(π‖π_ref) on this sequence

        pg_loss = -float(adv) * logp_mean
        loss    = pg_loss + beta_kl * kl
        losses.append(loss)
        pg_values.append(float(pg_loss.detach()))
        kl_values.append(float(kl.detach()))
        out["applied"] += 1

    if not losses:
        return out

    total = torch.stack(losses).mean()
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    if max_grad_norm and max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(
            (p for p in student.trainable_parameters() if p.grad is not None),
            max_grad_norm,
        )
    optimizer.step()

    if student.model is not None:
        student.model.eval()

    out["loss"]    = float(total.detach())
    out["pg_loss"] = sum(pg_values) / len(pg_values)
    out["kl"]      = sum(kl_values) / len(kl_values)
    return out


# ---------------------------------------------------------------------------
# 4. Driver loop.
# ---------------------------------------------------------------------------
def train(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset.
    ds = GoldRubricDataset.from_jsonl(args.gold_rubrics)
    print(f"[train] loaded {len(ds)} training rows from {args.gold_rubrics}",
          file=sys.stderr)
    if len(ds) == 0:
        raise SystemExit("[train] dataset is empty — did Stage ① run?")

    # ── Student.
    dummy_env = os.environ.get("RM_DUMMY", "0") == "1"
    student = StudentWrapper(StudentConfig(
        model_path=args.student_model,
        dummy=dummy_env or args.student_model is None,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        lora_config_path=args.lora_config,
        score_greedy=not args.score_sample,
        score_method=args.score_method,
        score_max_new_tokens=args.score_max_new_tokens,
    ))

    # ── Embedder.
    embedder = _load_embedder(args.embed_model)

    # ── Reward config + tracker.
    cfg = BatchRewardConfig(
        lambda_0     = args.lambda_0,
        beta_format  = args.beta_format,
        eps_valid    = args.eps_valid,
        hard_pref    = args.hard_pref,
        lock_weights_to_generation = not args.no_lock_weights,
        beta_format_min        = args.beta_format_min,
        beta_format_decay_steps= args.beta_format_decay_steps,
        beta_format_schedule   = args.beta_format_schedule,
        align_floor    = args.align_floor,
        align_saturate = args.align_saturate,
    )
    tracker = ValidRatioTracker(window=args.valid_window, alpha=args.valid_alpha)
    beta_scheduler = BetaFormatScheduler.from_cfg(cfg)
    print(f"[reward] β_format schedule={cfg.beta_format_schedule}  "
          f"β_0={cfg.beta_format}  β_min={cfg.beta_format_min}  "
          f"decay_steps={cfg.beta_format_decay_steps}", file=sys.stderr)

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

    # ── Optimiser (real mode only — dummy returns no trainable params).
    optimizer = _build_optimizer(student, args)

    # ── Loop.
    rng = random.Random(args.seed)
    log_path = out_dir / "train_log.jsonl"
    log_fh = log_path.open("w", encoding="utf-8")
    print(f"[train] log -> {log_path}", file=sys.stderr)

    indices = list(range(len(ds)))
    t0 = time.time()
    step = 0
    while step < args.max_steps:
        rng.shuffle(indices)
        for idx in indices:
            if step >= args.max_steps:
                break
            inst = ds[idx]
            metrics = step_one_instance(
                student, inst,
                G=args.num_generations,
                cfg=cfg,
                embedder=embedder,
                tracker=tracker,
                beta_scheduler=beta_scheduler,
                step=step,
                task_specs=task_specs,
            )

            # ── Policy gradient step.
            update = policy_update(
                student,
                metrics.pop("rollout_meta"),
                metrics["advantages"],
                optimizer=optimizer,
                beta_kl=args.beta_kl,
                max_grad_norm=args.max_grad_norm,
            )
            metrics["loss"]    = update["loss"]
            metrics["pg_loss"] = update["pg_loss"]
            metrics["kl"]      = update["kl"]
            metrics["applied"] = update["applied"]
            metrics["skipped"] = update["skipped"]

            metrics["step"]   = step
            metrics["wall_s"] = round(time.time() - t0, 2)
            log_fh.write(json.dumps(metrics, ensure_ascii=False))
            log_fh.write("\n")
            log_fh.flush()
            if step % max(1, args.log_every) == 0:
                print(
                    f"[step {step:5d}] dim={inst['dimension']:18s} "
                    f"reward μ={statistics.mean(metrics['rewards']):.3f}  "
                    f"r_pref={metrics['pref_mean']:.3f}  "
                    f"r_align={metrics['align_mean']:.3f}  "
                    f"r_fmt={metrics['fmt_mean']:.3f}  "
                    f"valid={metrics['valid_count']}/{args.num_generations}  "
                    f"λ={metrics['lambda_t']:.3f} β={metrics['beta_t']:.3f}  "
                    f"loss={metrics['loss']:.4f} kl={metrics['kl']:.4f}",
                    file=sys.stderr,
                )

            step += 1

            # ── Periodic checkpoint.
            if optimizer is not None and args.save_every > 0 and \
                    (step % args.save_every == 0 or step >= args.max_steps):
                _save_ckpt(student, out_dir, step)

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
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"[train] finished. summary -> {summary_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 5. Optimiser + checkpoint helpers.
# ---------------------------------------------------------------------------
def _build_optimizer(student: StudentWrapper, args: argparse.Namespace):
    """Create AdamW over the student's trainable params (LoRA only).

    Returns None when the student is in dummy mode or has no trainable
    params; the policy_update path detects this and short-circuits.
    """
    if student.is_dummy():
        print("[optim] dummy mode — no optimiser created.", file=sys.stderr)
        return None
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
    n = sum(p.numel() for p in params)
    print(f"[optim] AdamW over {len(params)} param tensors ({n/1e6:.1f}M elems) "
          f"lr={args.learning_rate} wd={args.weight_decay}", file=sys.stderr)
    return optim


def _save_ckpt(student: StudentWrapper, out_dir: Path, step: int) -> None:
    """Save the LoRA adapter (and tokenizer) under ``out_dir/checkpoint-<step>``.

    We save adapter weights only — the base model lives at
    ``cfg.model_path`` and is never modified.
    """
    if student.model is None:
        return
    ck = out_dir / f"checkpoint-{step}"
    ck.mkdir(parents=True, exist_ok=True)
    try:
        student.model.save_pretrained(str(ck))
    except Exception as e:
        print(f"[ckpt] save_pretrained failed at step {step}: {e}", file=sys.stderr)
        return
    try:
        if student._tokenizer is not None:
            student._tokenizer.save_pretrained(str(ck))
    except Exception:
        pass
    # Refresh "latest" symlink for convenience.
    latest = out_dir / "checkpoint-latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(ck.name)
    except Exception:
        pass
    print(f"[ckpt] step {step} -> {ck}", file=sys.stderr)


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
                   help="Path to lora_qwen25vl.yaml. Required for real-mode training.")
    p.add_argument("--output_dir",    required=True)
    p.add_argument("--num_generations", type=int, default=8,
                   help="G — number of rubric rollouts per training instance")
    p.add_argument("--max_steps",     type=int, default=4)
    p.add_argument("--log_every",     type=int, default=1)
    p.add_argument("--save_every",    type=int, default=100,
                   help="Save LoRA adapter every N steps; 0 to disable.")
    p.add_argument("--seed",          type=int, default=0)

    # reward knobs
    p.add_argument("--lambda_0",      type=float, default=0.3)
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
    p.add_argument("--beta_format",   type=float, default=0.5,
                   help="β_format starting coefficient (β_0). Decays toward "
                        "--beta_format_min over --beta_format_decay_steps "
                        "according to --beta_format_schedule. Set "
                        "schedule=constant to disable annealing.")
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
                   default="linear",
                   help="Anneal shape for β_format. linear (default) and "
                        "cosine are smooth; step is a hard cliff at "
                        "decay_steps; constant disables annealing.")
    p.add_argument("--eps_valid",     type=float, default=0.05)
    p.add_argument("--hard_pref",     action="store_true",
                   help="Use 0/1 hard preference reward instead of σ(S⁺-S⁻).")
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
    p.add_argument("--score_max_new_tokens", type=int, default=64,
                   help="Max new tokens for the score forward. The plain-"
                        "text scaffold needs ~32; default 64 leaves slack "
                        "for tokenisers that split 'theme_3:' into many ids.")
    p.add_argument("--use_task_specs", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Load per-dim task descriptions from "
                        "rubric_pipeline/config/v_arr_mvp_<dim>_t2v.yaml and "
                        "inject them into the rubric/score prompts. Disable "
                        "with --no-use_task_specs for ablation (student "
                        "will see only a generic 'evaluate this dimension' "
                        "stub).")

    # optimisation knobs
    p.add_argument("--learning_rate",  type=float, default=5.0e-6)
    p.add_argument("--weight_decay",   type=float, default=0.01)
    p.add_argument("--max_grad_norm",  type=float, default=1.0)
    p.add_argument("--beta_kl",        type=float, default=0.04,
                   help="KL coefficient against the reference policy "
                        "(LoRA-disabled forward of the same model).")
    return p


def main() -> None:
    p = build_argparser()
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
