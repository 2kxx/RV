"""rm_training.stage2_grpo.reward_fn
=================================
The composite reward used by GRPO::

    Reward(R̂_g) = r_pref + λ(t)·r_align + β·r_format

Each scalar lives in roughly [0, 1]; the trainer sums them and lets
GRPO's group-relative advantage do the rest.

Design choices (all baked in by spec)
-------------------------------------
* ``r_pref``  is a **soft Bradley-Terry**: σ(S⁺ − S⁻). MVP can switch
  to a 0/1 hard reward (set ``hard_pref=True``).
* ``r_align`` is **bipartite Hungarian** cosine over rubric paths only
  (see ``gold_align.py``). Never enforces score equality.
* ``r_format`` is the granular schema-compliance score from
  ``rubric_parser.format_reward``.
* Invalid rubrics (``|S⁺-S⁻| ≤ eps``) get **r_pref = 0** and feed into
  the ``ValidRatioTracker`` which in turn drives ``λ(t)``.

Fairness invariant ("weight lock")
----------------------------------
The student emits the rubric **once** during the rubric-gen forward and
is then asked to score y⁺ and y⁻ in two separate score forwards. The
score forward is prompted to echo the same rubric block back, but the
LM may quietly perturb the weights / order / paths in either of the
two echoes. If we trusted those echoes, S(y⁺) and S(y⁻) would be
weighted by **different** vectors and the BT comparison ``σ(S⁺-S⁻)``
would no longer measure what the rubric says — it would measure what
two divergent re-emissions say.

To preserve the algorithm's "cross-scoring" fairness contract, the
reward function locks ``student_full_pos.rubrics`` and
``student_full_neg.rubrics`` to the rubric the trainer generated up
front (``student_rubric.rubrics``), then aggregates with that single
weight vector. Scores are taken from the score forwards verbatim;
``format_score`` already returns None when len(scores) ≠ len(rubrics)
so a stingy / over-eager score forward is correctly penalised.

Set ``cfg.lock_weights_to_generation = False`` for ablation only.

The function is **stateless w.r.t. the model**: it consumes already-
parsed structured outputs. The ``train_grpo.py`` script is responsible
for running the student forward 3 times (gen rubric, score y⁺, score
y⁻) and packaging results into ``RolloutPair`` before calling here.

Public API
----------
- ``RolloutPair``                              — single G rollout
- ``BatchRewardConfig``                        — knobs at train start
- ``compute_pair_reward(rollout, gold, cfg)``  — scalar [0, ~2]
- ``compute_batch_rewards(rollouts, gold, cfg, tracker)``  — list[float]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .gold_align import Embedder, bipartite_sim
from .rubric_parser import (
    ParsedFull,
    ParsedRubric,
    format_reward,
    format_score,
)
from .valid_filter import ValidRatioTracker, is_valid_score_pair

__all__ = [
    "RolloutPair",
    "BatchRewardConfig",
    "BetaFormatScheduler",
    "compute_pair_reward",
    "compute_batch_rewards",
]


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------
@dataclass
class BatchRewardConfig:
    lambda_0:       float = 0.3       # gold-align starting weight
    beta_format:    float = 0.5       # format reward STARTING coefficient (β_0)
    eps_valid:      float = 0.05      # |S⁺-S⁻| threshold
    hard_pref:      bool  = False     # MVP can flip to 0/1 reward
    pref_temperature: float = 1.0     # σ(τ·(S⁺-S⁻));  τ>1 sharpens

    # β_format anneal:
    #   β(t) = decay(β_0 → β_min) over decay_steps, then floors at β_min.
    # Rationale: early-high β grounds the model on schema compliance
    # (valid rubric JSON + valid score scaffold) while the policy is
    # near-random and would otherwise hill-climb on r_pref / r_align via
    # malformed outputs. Past decay_steps the format signal stays at a
    # small floor so the LM doesn't drift back into bad formats once the
    # main pref/align signal dominates.
    beta_format_min:        float = 0.05
    beta_format_decay_steps: int  = 500
    beta_format_schedule:   str   = "linear"   # {"linear", "cosine", "step", "constant"}

    # Soft clip of r_align to non-negative (cosine can be slightly negative
    # for unrelated paths; we treat that as 0 contribution rather than
    # negative reward to avoid punishing exploration).
    clip_align_to_nonneg: bool = True

    # r_align warm-start anchor (NOT a "match the gold rubric exactly"
    # reward). Once student rubric similarity to gold reaches ``align_floor``,
    # r_align saturates at 1.0 and stops contributing to advantage variance —
    # the model is then free to specialise toward whichever rubric phrasing
    # discriminates pairs best (driven by r_pref). Below the floor, r_align
    # decays linearly to 0 at sim=0, providing a strong corrective gradient.
    #
    # Rationale (per training-design discussion, 2026-06-04):
    #   The gold rubric is a *warm-start anchor*, not a ground-truth answer.
    #   We want the student to land in the gold's conceptual neighbourhood
    #   (so it covers the right sub-aspects), not to copy it verbatim. A
    #   saturating reward implements exactly that: clip when good enough,
    #   penalise only when off-topic.
    align_floor:    float = 0.7      # sim ≥ floor → r_align = 1.0
    align_saturate: bool  = True     # set False to recover the legacy linear behaviour

    # Fairness lock: replace student_full_{pos,neg}.rubrics with
    # student_rubric.rubrics before aggregation, so S(y⁺) and S(y⁻) are
    # weighted by the SAME vector. Disable only for ablation.
    lock_weights_to_generation: bool = True


@dataclass
class BetaFormatScheduler:
    """Decays β_format from ``beta_0`` to ``beta_min`` over ``decay_steps``.

    Schedules
    ---------
    - ``linear`` — β(t) = β_0 - (β_0 - β_min) · min(t/decay, 1)
    - ``cosine`` — β(t) = β_min + 0.5·(β_0 - β_min)·(1 + cos(π·min(t/decay, 1)))
    - ``step``   — β(t) = β_0 if t < decay else β_min  (hard cliff)
    - ``constant`` — β(t) = β_0  (no anneal; equivalent to old behaviour)

    The scheduler is stateless w.r.t. the trainer — the trainer holds the
    step counter and asks ``beta(step)``. This keeps it trivially testable
    without mocking out a dataloader.

    Built directly from a ``BatchRewardConfig`` via ``from_cfg``.
    """
    beta_0:        float = 0.5
    beta_min:      float = 0.05
    decay_steps:   int   = 500
    schedule:      str   = "linear"

    @classmethod
    def from_cfg(cls, cfg: "BatchRewardConfig") -> "BetaFormatScheduler":
        return cls(
            beta_0       = float(cfg.beta_format),
            beta_min     = float(cfg.beta_format_min),
            decay_steps  = int(cfg.beta_format_decay_steps),
            schedule     = str(cfg.beta_format_schedule),
        )

    def beta(self, step: int) -> float:
        # Clamp to non-negative integer steps; treat ``step < 0`` as 0.
        t = max(0, int(step))
        D = max(1, int(self.decay_steps))   # avoid div-by-zero on misconfig
        # If β_min > β_0, swap so the schedule still moves monotonically;
        # the "anneal" semantic is "interpolate from beta_0 toward beta_min".
        b0, bmin = float(self.beta_0), float(self.beta_min)
        sched = (self.schedule or "linear").lower()

        if sched == "constant":
            return b0
        if sched == "step":
            return b0 if t < D else bmin

        # Normalised progress in [0, 1]; clamps after decay_steps.
        p = min(1.0, t / D)
        if sched == "linear":
            return b0 + (bmin - b0) * p
        if sched == "cosine":
            # Cosine half-cycle: 1 → 0 as p goes 0 → 1.
            half = 0.5 * (1.0 + math.cos(math.pi * p))
            return bmin + (b0 - bmin) * half
        # Unknown schedule → linear fallback (with a one-time warning by
        # caller; we don't print here to keep the function pure).
        return b0 + (bmin - b0) * p


# ---------------------------------------------------------------------------
# Rollout container.
# ---------------------------------------------------------------------------
@dataclass
class RolloutPair:
    """One of the G rollouts for a single (q, y⁺, y⁻) training instance."""
    student_rubric:  ParsedRubric            # parsed rubric-only emission
    student_full_pos: ParsedFull             # parsed scores on y⁺
    student_full_neg: ParsedFull             # parsed scores on y⁻
    # Optional cached: weighted aggregate score (weighted_mean over c_k).
    # We compute it lazily in ``compute_pair_reward`` if missing.
    s_pos: float | None = None
    s_neg: float | None = None
    # Free-form for debugging.
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Fairness lock — see module docstring "Fairness invariant".
# ---------------------------------------------------------------------------
def _apply_weight_lock(rollout: RolloutPair) -> None:
    """Replace pos/neg score-forward rubrics with the rubric-gen output.

    Rationale: ``format_score`` aggregates as ``Σ w_k · c_k / Σ w_k`` zipping
    ``parsed.rubrics`` and ``parsed.scores`` in order. If the score forward
    quietly rewrites weights or reorders entries, the resulting S(y⁺) and
    S(y⁻) are weighted by **different** vectors, which breaks the BT
    fairness contract that ``σ(S⁺-S⁻)`` relies on. We force a single source
    of truth: the rubric the trainer generated up front.

    Mutation, not copy: the caller (``compute_pair_reward``) re-reads the
    same RolloutPair after this call and ``s_pos``/``s_neg`` are recomputed
    against the locked rubrics. We deliberately do NOT touch ``scores`` —
    the score forward's score values still drive S(y), and any length
    mismatch between locked rubrics and scores triggers ``format_score`` to
    return None (correct behaviour: penalise the rollout).
    """
    locked = list(rollout.student_rubric.rubrics)
    rollout.student_full_pos.rubrics = locked
    rollout.student_full_neg.rubrics = locked
    # s_pos / s_neg may have been pre-computed before the lock — invalidate
    # so compute_pair_reward / compute_batch_rewards re-aggregate with the
    # locked weights.
    rollout.s_pos = None
    rollout.s_neg = None


# ---------------------------------------------------------------------------
# r_pref.
# ---------------------------------------------------------------------------
def _r_pref(s_pos: float, s_neg: float, cfg: BatchRewardConfig) -> float:
    if not (math.isfinite(s_pos) and math.isfinite(s_neg)):
        return 0.0
    delta = (s_pos - s_neg) * cfg.pref_temperature
    if cfg.hard_pref:
        if delta > 0: return 1.0
        if delta < 0: return 0.0
        return 0.5
    # Soft Bradley-Terry: σ(τ·δ).
    if delta >= 0:
        return 1.0 / (1.0 + math.exp(-delta))
    # Stable form for negative delta.
    e = math.exp(delta)
    return e / (1.0 + e)


# ---------------------------------------------------------------------------
# r_align.
# ---------------------------------------------------------------------------
def _r_align(
    student_rubric: ParsedRubric,
    gold_rubric:    list[dict] | list[str],
    embedder:       Embedder | None,
    cfg:            BatchRewardConfig,
) -> float:
    if not student_rubric.rubrics or not gold_rubric:
        return 0.0
    student_paths = [r.path for r in student_rubric.rubrics if r.path]
    if isinstance(gold_rubric[0], str):
        gold_paths = list(gold_rubric)        # type: ignore[arg-type]
    else:
        # List of dicts (the Stage ① schema): ``{"path": ..., "weight": ...}``.
        gold_paths = [g.get("path", "") for g in gold_rubric if g.get("path")]   # type: ignore[union-attr]
    if not student_paths or not gold_paths:
        return 0.0
    sim = bipartite_sim(student_paths, gold_paths, embedder)
    if cfg.clip_align_to_nonneg:
        sim = max(0.0, sim)
    if not cfg.align_saturate:
        return float(sim)

    # Warm-start anchor: saturate at the floor so the student isn't
    # rewarded for *exactly* matching the gold once it's already in the
    # right neighbourhood. Below the floor, scale linearly toward 0 so
    # off-topic rubrics still receive a corrective gradient.
    floor = max(1e-6, float(cfg.align_floor))   # avoid div-by-zero on misconfig
    if sim >= floor:
        return 1.0
    return float(sim) / floor


# ---------------------------------------------------------------------------
# Pair-level scalar reward.
# ---------------------------------------------------------------------------
def compute_pair_reward(
    rollout:     RolloutPair,
    gold_rubric: list[dict] | list[str],
    *,
    embedder:    Embedder | None = None,
    lambda_t:    float | None = None,
    beta_format_t: float | None = None,
    cfg:         BatchRewardConfig | None = None,
) -> tuple[float, dict]:
    """Return ``(reward, breakdown_dict)`` for one G rollout.

    Breakdown carries the per-component values so the trainer can log
    them to wandb / tensorboard.

    ``beta_format_t``: format-reward coefficient for THIS step, after
    annealing. When ``None``, we fall back to the static ``cfg.beta_format``
    so callers that haven't migrated to the scheduler keep working.
    """
    cfg = cfg or BatchRewardConfig()
    lam  = cfg.lambda_0    if lambda_t      is None else lambda_t
    beta = cfg.beta_format if beta_format_t is None else beta_format_t

    # --- fairness lock (see module docstring).
    if cfg.lock_weights_to_generation:
        _apply_weight_lock(rollout)

    # --- score aggregation (weighted-mean over per-criterion scores).
    if rollout.s_pos is None:
        rollout.s_pos = format_score(rollout.student_full_pos) or float("nan")
    if rollout.s_neg is None:
        rollout.s_neg = format_score(rollout.student_full_neg) or float("nan")

    s_pos = rollout.s_pos
    s_neg = rollout.s_neg
    valid = is_valid_score_pair(s_pos, s_neg, eps=cfg.eps_valid)

    # --- components.
    r_pref = _r_pref(s_pos, s_neg, cfg) if valid else 0.0
    r_align = _r_align(rollout.student_rubric, gold_rubric, embedder, cfg)
    # Format reward: rubric-gen forward + the two score forwards each
    # contribute. The two share the SAME locked rubric (P0-#2 fairness
    # invariant), so the rubric-gen evaluation runs once on the canonical
    # rubric, while the two score forwards are evaluated against the
    # plain-text scaffold schema (length / range / cross-theme spread).
    fmt_rubric = format_reward(rollout.student_rubric, mode="rubric_json")
    fmt_pos    = format_reward(rollout.student_full_pos, mode="score_scaffold")
    fmt_neg    = format_reward(rollout.student_full_neg, mode="score_scaffold")
    r_format   = (fmt_rubric + fmt_pos + fmt_neg) / 3.0

    total = r_pref + lam * r_align + beta * r_format

    breakdown = {
        "r_pref":       r_pref,
        "r_align":      r_align,
        "r_format":     r_format,
        "lambda_t":     lam,
        "beta_format":  beta,
        "s_pos":        s_pos if math.isfinite(s_pos) else None,
        "s_neg":        s_neg if math.isfinite(s_neg) else None,
        "valid":        valid,
        "total":        total,
    }
    return float(total), breakdown


# ---------------------------------------------------------------------------
# Batch helper: also updates the ValidRatioTracker → returns lambda_t used.
# ---------------------------------------------------------------------------
def compute_batch_rewards(
    rollouts:    Sequence[RolloutPair],
    gold_rubric: list[dict] | list[str],
    *,
    embedder:    Embedder | None = None,
    cfg:         BatchRewardConfig | None = None,
    tracker:     ValidRatioTracker | None = None,
    beta_format_t: float | None = None,
) -> tuple[list[float], list[dict]]:
    """Compute reward for the G rollouts of one (q, y⁺, y⁻) instance.

    Updates ``tracker`` with the validity flags so λ(t) reflects the
    most recent batch's discrimination rate.

    ``beta_format_t``: per-step β coefficient (from BetaFormatScheduler).
    Same semantics as in ``compute_pair_reward`` — None falls back to
    ``cfg.beta_format`` for callers that haven't migrated.
    """
    cfg = cfg or BatchRewardConfig()

    # First pass: compute validity flags so we can update λ(t) BEFORE
    # the reward — ensures the same batch's λ is consistent across the
    # G rollouts. We must lock weights here too, otherwise the validity
    # flag would be based on score-forward-rewritten weights while the
    # reward in compute_pair_reward uses the locked weights — they'd
    # disagree on which rollouts are valid.
    valids: list[bool] = []
    for ro in rollouts:
        if cfg.lock_weights_to_generation:
            _apply_weight_lock(ro)
        if ro.s_pos is None:
            ro.s_pos = format_score(ro.student_full_pos) or float("nan")
        if ro.s_neg is None:
            ro.s_neg = format_score(ro.student_full_neg) or float("nan")
        valids.append(is_valid_score_pair(ro.s_pos, ro.s_neg, eps=cfg.eps_valid))

    if tracker is not None:
        tracker.update(valids)
        lambda_t = tracker.lambda_t(cfg.lambda_0)
    else:
        lambda_t = cfg.lambda_0

    rewards: list[float] = []
    breakdowns: list[dict] = []
    for ro in rollouts:
        r, bd = compute_pair_reward(
            ro, gold_rubric,
            embedder=embedder,
            lambda_t=lambda_t,
            beta_format_t=beta_format_t,
            cfg=cfg,
        )
        rewards.append(r)
        breakdowns.append(bd)
    return rewards, breakdowns
