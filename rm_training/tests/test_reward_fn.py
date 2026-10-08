"""Unit tests for ``rm_training.stage2_grpo.reward_fn``.

Validates:
  • r_pref direction matches sign(S⁺ − S⁻);
  • soft and hard pref modes both behave;
  • valid filter zeroes r_pref when |S⁺ − S⁻| ≤ eps;
  • λ(t) annealing: when valid_ratio increases, λ shrinks.
"""

from __future__ import annotations

import math

try:
    import pytest                                    # noqa: F401  (optional)
except ImportError:                                  # pragma: no cover
    pytest = None                                    # type: ignore

from rm_training.stage2_grpo.gold_align    import DummyEmbedder
from rm_training.stage2_grpo.reward_fn     import (
    BatchRewardConfig,
    BetaFormatScheduler,
    RolloutPair,
    compute_batch_rewards,
    compute_pair_reward,
)
from rm_training.stage2_grpo.rubric_parser import (
    parse_rubric_and_scores,
    parse_rubric_only,
)
from rm_training.stage2_grpo.valid_filter  import ValidRatioTracker


GOOD_RUBRIC = """{"rubrics": [
  {"theme": "Theme: VQ",   "weight": 0.3},
  {"theme": "Theme: Char", "weight": 0.2},
  {"theme": "Theme: Env",  "weight": 0.2},
  {"theme": "Theme: Act",  "weight": 0.15},
  {"theme": "Theme: Cam",  "weight": 0.15}
]}"""


def _rollout_with_scores(s_per_pos: list[float], s_per_neg: list[float]) -> RolloutPair:
    """Build a RolloutPair whose pos / neg scores hit the requested averages."""
    rubric_parsed = parse_rubric_only(GOOD_RUBRIC)
    pos_json = """{
      "rubrics": [
        {"theme": "Theme: VQ",   "weight": 0.3},
        {"theme": "Theme: Char", "weight": 0.2},
        {"theme": "Theme: Env",  "weight": 0.2},
        {"theme": "Theme: Act",  "weight": 0.15},
        {"theme": "Theme: Cam",  "weight": 0.15}
      ],
      "scores": [
    """ + ",".join(f'{{"rubric_index":{i+1},"score":{v}}}' for i, v in enumerate(s_per_pos)) + "]}"
    neg_json = """{
      "rubrics": [
        {"theme": "Theme: VQ",   "weight": 0.3},
        {"theme": "Theme: Char", "weight": 0.2},
        {"theme": "Theme: Env",  "weight": 0.2},
        {"theme": "Theme: Act",  "weight": 0.15},
        {"theme": "Theme: Cam",  "weight": 0.15}
      ],
      "scores": [
    """ + ",".join(f'{{"rubric_index":{i+1},"score":{v}}}' for i, v in enumerate(s_per_neg)) + "]}"
    return RolloutPair(
        student_rubric  = rubric_parsed,
        student_full_pos= parse_rubric_and_scores(pos_json),
        student_full_neg= parse_rubric_and_scores(neg_json),
    )


GOLD_PATHS = [
    {"theme": "Theme: VQ",   "weight": 0.3},
    {"theme": "Theme: Char", "weight": 0.2},
    {"theme": "Theme: Env",  "weight": 0.2},
    {"theme": "Theme: Act",  "weight": 0.15},
    {"theme": "Theme: Cam",  "weight": 0.15},
]


# ---------------------------------------------------------------------------
# r_pref behaviour.
# ---------------------------------------------------------------------------
def test_pref_positive_when_pos_better():
    ro = _rollout_with_scores([3, 3, 3, 3, 3], [2, 2, 2, 2, 2])
    r, bd = compute_pair_reward(ro, GOLD_PATHS, embedder=DummyEmbedder(),
                                 cfg=BatchRewardConfig())
    assert bd["r_pref"] > 0.5      # σ(+1) > 0.5
    assert bd["valid"]


def test_pref_low_when_neg_better():
    ro = _rollout_with_scores([2, 2, 2, 2, 2], [3, 3, 3, 3, 3])
    r, bd = compute_pair_reward(ro, GOLD_PATHS, embedder=DummyEmbedder(),
                                 cfg=BatchRewardConfig())
    assert bd["r_pref"] < 0.5      # σ(-1) < 0.5


def test_invalid_zeroes_pref():
    # |S⁺ - S⁻| = 0 < eps_valid (default 0.05) → invalid → r_pref = 0.
    ro = _rollout_with_scores([2.5]*5, [2.5]*5)
    r, bd = compute_pair_reward(ro, GOLD_PATHS, embedder=DummyEmbedder())
    assert not bd["valid"]
    assert bd["r_pref"] == 0.0


def test_hard_pref_mode():
    ro = _rollout_with_scores([3]*5, [2]*5)
    r, bd = compute_pair_reward(
        ro, GOLD_PATHS,
        embedder=DummyEmbedder(),
        cfg=BatchRewardConfig(hard_pref=True),
    )
    assert bd["r_pref"] == 1.0


# ---------------------------------------------------------------------------
# r_align behaviour (rubric themes identical to gold → ~1.0).
# ---------------------------------------------------------------------------
def test_align_high_when_rubric_matches_gold():
    ro = _rollout_with_scores([3]*5, [2]*5)
    r, bd = compute_pair_reward(ro, GOLD_PATHS, embedder=DummyEmbedder(dim=64))
    assert bd["r_align"] > 0.99


# ---------------------------------------------------------------------------
# Batch + tracker integration.
# ---------------------------------------------------------------------------
def test_batch_with_tracker_drives_lambda_down_when_valid():
    rollouts = [_rollout_with_scores([3]*5, [2]*5) for _ in range(8)]
    tracker = ValidRatioTracker(window=8, alpha=0.5)
    rewards1, _ = compute_batch_rewards(
        rollouts, GOLD_PATHS,
        embedder=DummyEmbedder(),
        cfg=BatchRewardConfig(lambda_0=0.3),
        tracker=tracker,
    )
    lam_after_valid = tracker.lambda_t(0.3)
    # 8/8 valid → ema → 1.0 → lambda → 0.0 (after enough EMA steps).
    # We push 4 more batches to let alpha=0.5 saturate.
    for _ in range(4):
        compute_batch_rewards(rollouts, GOLD_PATHS,
                               embedder=DummyEmbedder(),
                               cfg=BatchRewardConfig(lambda_0=0.3),
                               tracker=tracker)
    lam_saturated = tracker.lambda_t(0.3)
    assert lam_saturated < lam_after_valid + 1e-9
    assert lam_saturated < 0.05            # should be near 0


def test_batch_lambda_stays_high_when_all_invalid():
    rollouts = [_rollout_with_scores([2.5]*5, [2.5]*5) for _ in range(8)]
    tracker = ValidRatioTracker(window=8, alpha=0.5)
    compute_batch_rewards(rollouts, GOLD_PATHS,
                           embedder=DummyEmbedder(),
                           cfg=BatchRewardConfig(lambda_0=0.3),
                           tracker=tracker)
    # 0/8 valid → ema near 0 → lambda near lambda_0.
    assert tracker.lambda_t(0.3) > 0.25


# ---------------------------------------------------------------------------
# Fairness lock — see reward_fn module docstring "Fairness invariant".
#
# We construct a *deliberately adversarial* rollout where the score forward
# returns the SAME numeric scores on y⁺ and y⁻ but has rewritten the weights
# in opposite directions. With the lock disabled, the BT comparison flips
# (because S⁺ and S⁻ are weighted by different vectors); with the lock on,
# both sides use the rubric-gen weight and the weighted means are equal
# → r_pref reverts to 0.5 (or marked invalid by the eps filter).
# ---------------------------------------------------------------------------
def _adversarial_rollout(scores: list[float],
                         weights_gen: list[float],
                         weights_pos: list[float],
                         weights_neg: list[float]) -> RolloutPair:
    """Build a rollout whose pos/neg score forwards rewrote weights differently."""
    def _full(rubric_themes, weights, scores):
        rubrics = ", ".join(
            f'{{"theme": "{p}", "weight": {w}}}'
            for p, w in zip(rubric_themes, weights)
        )
        scores_blob = ", ".join(
            f'{{"rubric_index": {i+1}, "score": {s}}}'
            for i, s in enumerate(scores)
        )
        return f'{{"rubrics": [{rubrics}], "scores": [{scores_blob}]}}'

    themes = ["Theme: VQ", "Theme: Char", "Theme: Env", "Theme: Act", "Theme: Cam"]
    rubric_only = '{"rubrics": [' + ", ".join(
        f'{{"theme": "{t}", "weight": {w}}}' for t, w in zip(themes, weights_gen)
    ) + "]}"
    return RolloutPair(
        student_rubric  = parse_rubric_only(rubric_only),
        student_full_pos= parse_rubric_and_scores(_full(themes, weights_pos, scores)),
        student_full_neg= parse_rubric_and_scores(_full(themes, weights_neg, scores)),
    )


def test_fairness_lock_flips_bt_direction_vs_no_lock():
    """Same scores, weights rewritten differently → lock decides BT direction.

    Concrete construction:
      • generation weights = [0.20]*5 (uniform)
      • scores             = [4, 4, 4, 1, 1] on y⁺
                             [1, 1, 1, 4, 4] on y⁻
      • pos forward rewrites weights → [0.4, 0.4, 0.2, 0.0, 0.0]   (favours high-score dims of y⁺)
      • neg forward rewrites weights → [0.4, 0.4, 0.2, 0.0, 0.0]   (same vector → favours its own LOW-score dims)

    Without the lock:
      S⁺ = (0.4*4 + 0.4*4 + 0.2*4) = 4.0
      S⁻ = (0.4*1 + 0.4*1 + 0.2*1) = 1.0    →  r_pref ≈ σ(+3) → ~0.95   (positive)

    Wait — that still says y⁺ wins. The *real* fairness break is when the
    score forwards rewrite the weights in OPPOSITE directions. Use that:
    """
    themes = ["Theme: VQ", "Theme: Char", "Theme: Env", "Theme: Act", "Theme: Cam"]
    s_pos = [4.0, 4.0, 4.0, 1.0, 1.0]
    s_neg = [1.0, 1.0, 1.0, 4.0, 4.0]    # same total under uniform weights

    # Score forwards rewrite weights in opposite directions: pos echo says
    # "the first 3 dims matter", neg echo says "the last 2 dims matter".
    w_gen = [0.2, 0.2, 0.2, 0.2, 0.2]
    w_pos_echo = [0.0, 0.0, 0.0, 0.5, 0.5]    # cherry-picks y⁺'s WORST dims
    w_neg_echo = [0.5, 0.5, 0.0, 0.0, 0.0]    # cherry-picks y⁻'s WORST dims

    # Without lock:
    #   S⁺ uses w_pos_echo on s_pos = (0+0+0+0.5*1+0.5*1) = 1.0
    #   S⁻ uses w_neg_echo on s_neg = (0.5*1+0.5*1+0+0+0) = 1.0
    #   r_pref ≈ 0.5  (tied, but for adversarial reasons — and since
    #                  |S⁺-S⁻| ≤ eps, the validity filter zeros r_pref).
    # With lock (uniform 0.2 weights on both sides):
    #   S⁺ = mean(s_pos) = 2.8
    #   S⁻ = mean(s_neg) = 2.2
    #   r_pref ≈ σ(+0.6) ≈ 0.65    → POSITIVE, valid.
    ro_locked    = _adversarial_rollout(s_pos, w_gen, w_pos_echo, w_neg_echo)
    ro_no_lock   = _adversarial_rollout(s_pos, w_gen, w_pos_echo, w_neg_echo)
    # NB: _adversarial_rollout puts the same s_pos in both sides; we patched
    # the pos/neg score blobs to differ via the wrapper below.

    # Re-patch with distinct pos vs neg scores.
    def _full_blob(weights, scs):
        rubrics = ", ".join(
            f'{{"theme": "{t}", "weight": {w}}}' for t, w in zip(themes, weights)
        )
        scores_blob = ", ".join(
            f'{{"rubric_index": {i+1}, "score": {s}}}' for i, s in enumerate(scs)
        )
        return f'{{"rubrics": [{rubrics}], "scores": [{scores_blob}]}}'

    rubric_only = '{"rubrics": [' + ", ".join(
        f'{{"theme": "{t}", "weight": {w}}}' for t, w in zip(themes, w_gen)
    ) + "]}"
    def _make():
        return RolloutPair(
            student_rubric  = parse_rubric_only(rubric_only),
            student_full_pos= parse_rubric_and_scores(_full_blob(w_pos_echo, s_pos)),
            student_full_neg= parse_rubric_and_scores(_full_blob(w_neg_echo, s_neg)),
        )

    # No lock: weights diverge → S⁺ and S⁻ both ≈ 1.0 → invalid → r_pref = 0.
    ro_a = _make()
    _, bd_no = compute_pair_reward(
        ro_a, GOLD_PATHS,
        embedder=DummyEmbedder(),
        cfg=BatchRewardConfig(lock_weights_to_generation=False),
    )
    assert math.isclose(ro_a.s_pos, 1.0, abs_tol=1e-6)
    assert math.isclose(ro_a.s_neg, 1.0, abs_tol=1e-6)
    assert not bd_no["valid"]                # adversarial weights tied them
    assert bd_no["r_pref"] == 0.0

    # With lock: both sides use uniform [0.2]*5 → S⁺ = 2.8, S⁻ = 2.2.
    ro_b = _make()
    _, bd_lock = compute_pair_reward(
        ro_b, GOLD_PATHS,
        embedder=DummyEmbedder(),
        cfg=BatchRewardConfig(lock_weights_to_generation=True),
    )
    assert math.isclose(ro_b.s_pos, 2.8, abs_tol=1e-6)
    assert math.isclose(ro_b.s_neg, 2.2, abs_tol=1e-6)
    assert bd_lock["valid"]
    assert bd_lock["r_pref"] > 0.55          # σ(+0.6) ≈ 0.646


def test_fairness_lock_recomputes_when_s_already_cached():
    """If a caller pre-set rollout.s_pos/s_neg (using stale weights), the lock
    must invalidate them so the post-lock aggregate is correct."""
    themes = ["Theme: VQ", "Theme: Char", "Theme: Env", "Theme: Act", "Theme: Cam"]
    w_gen = [0.2]*5
    w_echo = [1.0, 0.0, 0.0, 0.0, 0.0]   # wildly skewed echo
    scores = [4.0, 1.0, 1.0, 1.0, 1.0]

    def _make():
        rubric_only = '{"rubrics": [' + ", ".join(
            f'{{"theme": "{t}", "weight": {w}}}' for t, w in zip(themes, w_gen)
        ) + "]}"
        full = lambda: parse_rubric_and_scores(
            '{"rubrics": [' + ", ".join(
                f'{{"theme": "{t}", "weight": {w}}}' for t, w in zip(themes, w_echo)
            ) + '], "scores": [' + ", ".join(
                f'{{"rubric_index": {i+1}, "score": {s}}}' for i, s in enumerate(scores)
            ) + "]}"
        )
        return RolloutPair(
            student_rubric  = parse_rubric_only(rubric_only),
            student_full_pos= full(),
            student_full_neg= full(),
        )

    ro = _make()
    # Caller pre-cached stale scores using echo weights:
    ro.s_pos = 4.0     # = 1.0*4 + 0*... under echo weights
    ro.s_neg = 4.0
    _, bd = compute_pair_reward(
        ro, GOLD_PATHS,
        embedder=DummyEmbedder(),
        cfg=BatchRewardConfig(lock_weights_to_generation=True),
    )
    # Post-lock, S = mean(scores) = 1.6 on both sides (same pos/neg here).
    assert math.isclose(ro.s_pos, 1.6, abs_tol=1e-6)
    assert math.isclose(ro.s_neg, 1.6, abs_tol=1e-6)


def test_fairness_lock_default_is_on():
    """Sanity: BatchRewardConfig() defaults to lock=True."""
    cfg = BatchRewardConfig()
    assert cfg.lock_weights_to_generation is True


# ---------------------------------------------------------------------------
# β_format scheduler — see reward_fn.BetaFormatScheduler.
#
# Validates:
#   • linear hits exact endpoints + midpoint;
#   • cosine starts at β_0 and ends at β_min, monotone non-increasing;
#   • step is a hard cliff at decay_steps;
#   • constant ignores step;
#   • beyond decay_steps, β floors at β_min;
#   • the scheduler is built correctly from BatchRewardConfig.
# ---------------------------------------------------------------------------
def test_beta_scheduler_linear_endpoints_and_midpoint():
    s = BetaFormatScheduler(beta_0=0.5, beta_min=0.05, decay_steps=100,
                            schedule="linear")
    assert math.isclose(s.beta(0),   0.5,   abs_tol=1e-9)
    assert math.isclose(s.beta(50),  0.275, abs_tol=1e-9)
    assert math.isclose(s.beta(100), 0.05,  abs_tol=1e-9)
    # Beyond decay_steps the schedule floors at β_min.
    assert math.isclose(s.beta(500), 0.05,  abs_tol=1e-9)


def test_beta_scheduler_cosine_endpoints_and_monotone():
    s = BetaFormatScheduler(beta_0=0.5, beta_min=0.05, decay_steps=100,
                            schedule="cosine")
    assert math.isclose(s.beta(0),   0.5,   abs_tol=1e-9)
    assert math.isclose(s.beta(100), 0.05,  abs_tol=1e-9)
    # Cosine half-cycle is monotone non-increasing.
    prev = s.beta(0)
    for t in range(1, 101):
        cur = s.beta(t)
        assert cur <= prev + 1e-9
        prev = cur


def test_beta_scheduler_step_is_a_cliff():
    s = BetaFormatScheduler(beta_0=0.5, beta_min=0.05, decay_steps=100,
                            schedule="step")
    assert s.beta(0)   == 0.5
    assert s.beta(99)  == 0.5
    assert s.beta(100) == 0.05
    assert s.beta(500) == 0.05


def test_beta_scheduler_constant_ignores_step():
    s = BetaFormatScheduler(beta_0=0.5, beta_min=0.05, decay_steps=100,
                            schedule="constant")
    for t in (0, 1, 50, 100, 1000):
        assert s.beta(t) == 0.5


def test_beta_scheduler_unknown_schedule_falls_back_to_linear():
    s = BetaFormatScheduler(beta_0=0.5, beta_min=0.05, decay_steps=100,
                            schedule="nonsense")
    # Falls back to linear semantics, so midpoint matches linear midpoint.
    assert math.isclose(s.beta(50), 0.275, abs_tol=1e-9)


def test_beta_scheduler_handles_zero_decay_steps():
    """decay_steps=0 must not divide-by-zero — clamped to 1 internally."""
    s = BetaFormatScheduler(beta_0=0.5, beta_min=0.05, decay_steps=0,
                            schedule="linear")
    # With D=1 effectively, any step >= 1 is at β_min.
    assert math.isclose(s.beta(0), 0.5,  abs_tol=1e-9)
    assert math.isclose(s.beta(1), 0.05, abs_tol=1e-9)


def test_beta_scheduler_from_cfg_round_trips():
    cfg = BatchRewardConfig(
        beta_format=0.4, beta_format_min=0.02,
        beta_format_decay_steps=300, beta_format_schedule="cosine",
    )
    s = BetaFormatScheduler.from_cfg(cfg)
    assert s.beta_0      == 0.4
    assert s.beta_min    == 0.02
    assert s.decay_steps == 300
    assert s.schedule    == "cosine"


# ---------------------------------------------------------------------------
# β(t) feeds into the reward correctly.
# ---------------------------------------------------------------------------
def test_compute_pair_reward_uses_beta_format_t_when_provided():
    """Explicit ``beta_format_t`` overrides cfg.beta_format in the total."""
    ro = _rollout_with_scores([3, 3, 3, 3, 3], [2, 2, 2, 2, 2])
    cfg = BatchRewardConfig(beta_format=0.5)
    # Static β_format=0.5 baseline.
    r_static, bd_static = compute_pair_reward(
        ro, GOLD_PATHS, embedder=DummyEmbedder(), cfg=cfg,
    )
    assert bd_static["beta_format"] == 0.5
    # Override to β=0.0 — r_format component vanishes from the total,
    # but r_format itself is unchanged in the breakdown.
    ro2 = _rollout_with_scores([3]*5, [2]*5)
    r_zero, bd_zero = compute_pair_reward(
        ro2, GOLD_PATHS, embedder=DummyEmbedder(), cfg=cfg,
        beta_format_t=0.0,
    )
    assert bd_zero["beta_format"] == 0.0
    assert math.isclose(bd_zero["r_format"], bd_static["r_format"], abs_tol=1e-9)
    # Reward total drops by exactly β·r_format.
    expected_drop = bd_static["beta_format"] * bd_static["r_format"]
    assert math.isclose(r_static - r_zero, expected_drop, abs_tol=1e-6)


def test_compute_batch_rewards_propagates_beta_format_t():
    rollouts = [_rollout_with_scores([3]*5, [2]*5) for _ in range(4)]
    cfg = BatchRewardConfig(beta_format=0.5)
    _, bds = compute_batch_rewards(
        rollouts, GOLD_PATHS, embedder=DummyEmbedder(), cfg=cfg,
        beta_format_t=0.1,
    )
    for bd in bds:
        assert bd["beta_format"] == 0.1


def test_beta_format_default_when_not_passed():
    """Back-compat: callers that don't pass beta_format_t still get
    cfg.beta_format applied (no behaviour change)."""
    ro = _rollout_with_scores([3]*5, [2]*5)
    cfg = BatchRewardConfig(beta_format=0.42)
    _, bd = compute_pair_reward(
        ro, GOLD_PATHS, embedder=DummyEmbedder(), cfg=cfg,
    )
    assert bd["beta_format"] == 0.42
