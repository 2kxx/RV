"""Unit tests for ``rm_training.stage2_grpo.rubric_parser``.

Run with::

    cd /workspace/user_code/40063/jacuzzitang/projects/RewardVerse
    python -m pytest rm_training/tests/test_rubric_parser.py -v

These tests are pure-Python (no torch / GPU); they exercise the
LLM-output JSON tolerance and the weighted-mean aggregation.
"""

from __future__ import annotations

import math

try:
    import pytest                                    # noqa: F401  (optional)
except ImportError:                                  # pragma: no cover
    pytest = None                                    # type: ignore

from rm_training.stage2_grpo.rubric_parser import (
    format_reward,
    format_score,
    parse_rubric_and_scores,
    parse_rubric_only,
)


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------
GOOD_RUBRIC_JSON = """{
  "rubrics": [
    {"path": "Theme: VQ",   "weight": 0.30, "tips": ["sharpness", "stability"]},
    {"path": "Theme: Char", "weight": 0.20, "tips": ["fidelity"]},
    {"path": "Theme: Env",  "weight": 0.20, "tips": ["lighting"]},
    {"path": "Theme: Act",  "weight": 0.15, "tips": ["plausibility"]},
    {"path": "Theme: Cam",  "weight": 0.15, "tips": ["framing"]}
  ]
}"""

GOOD_FULL_JSON = """{
  "rubrics": [
    {"path": "Theme: VQ",   "weight": 0.30},
    {"path": "Theme: Char", "weight": 0.20},
    {"path": "Theme: Env",  "weight": 0.20},
    {"path": "Theme: Act",  "weight": 0.15},
    {"path": "Theme: Cam",  "weight": 0.15}
  ],
  "scores": [
    {"rubric_index": 1, "score": 3.5},
    {"rubric_index": 2, "score": 2.0},
    {"rubric_index": 3, "score": 3.0},
    {"rubric_index": 4, "score": 2.5},
    {"rubric_index": 5, "score": 4.0}
  ]
}"""

NOISY_FENCED_JSON = """Here is my answer:

```json
""" + GOOD_RUBRIC_JSON + """
```

Hope this helps."""

BROKEN_JSON = "{ this is not valid json at all"


# ---------------------------------------------------------------------------
# parse_rubric_only.
# ---------------------------------------------------------------------------
def test_parse_clean_rubric():
    p = parse_rubric_only(GOOD_RUBRIC_JSON)
    assert p.valid
    assert len(p.rubrics) == 5
    assert math.isclose(sum(r.weight for r in p.rubrics), 1.0, abs_tol=1e-3)


def test_parse_fenced_rubric_still_works():
    p = parse_rubric_only(NOISY_FENCED_JSON)
    assert p.valid
    assert len(p.rubrics) == 5


def test_parse_broken_returns_invalid_not_raise():
    p = parse_rubric_only(BROKEN_JSON)
    assert not p.valid
    assert p.rubrics == []
    assert "no_json_blob" in p.issues or "json_not_object" in p.issues


def test_parse_wrong_K_marks_invalid():
    j = """{"rubrics": [{"path": "p1", "weight": 1.0, "tips": []}]}"""
    p = parse_rubric_only(j)
    assert not p.valid
    assert any("wrong_K" in s for s in p.issues)


# ---------------------------------------------------------------------------
# parse_rubric_and_scores + format_score.
# ---------------------------------------------------------------------------
def test_parse_full_and_score():
    p = parse_rubric_and_scores(GOOD_FULL_JSON)
    assert p.valid
    assert len(p.scores) == 5
    s = format_score(p)
    expected = (0.30*3.5 + 0.20*2.0 + 0.20*3.0 + 0.15*2.5 + 0.15*4.0) / 1.0
    assert math.isclose(s, expected, abs_tol=1e-6)


def test_parse_full_clamps_out_of_range():
    j = """{
      "rubrics": [
        {"path": "p1", "weight": 1.0}
      ],
      "scores": [
        {"rubric_index": 1, "score": 9.0}
      ]
    }"""
    p = parse_rubric_and_scores(j)
    # K mismatch trips ``valid=False`` but the score still gets clamped.
    assert any("score_out_of_range" in s for s in p.issues)
    assert p.scores[0] == 5.0   # clamped to upper bound (Likert 1-5)


def test_format_score_returns_none_on_nan():
    j = """{
      "rubrics": [
        {"path": "p1", "weight": 1.0},
        {"path": "p2", "weight": 0.0}
      ],
      "scores": [
        {"rubric_index": 1, "score": 2.0}
      ]
    }"""
    p = parse_rubric_and_scores(j)
    assert format_score(p) is None    # missing index 2 → NaN → reject


# ---------------------------------------------------------------------------
# format_reward shape.
# ---------------------------------------------------------------------------
def test_format_reward_full_compliance():
    """rubric_json mode on a fully compliant rubric (K=5, Σw=1, all tips
    non-empty) → 1.0."""
    p = parse_rubric_only(GOOD_RUBRIC_JSON)
    r = format_reward(p, mode="rubric_json")
    assert math.isclose(r, 1.0, abs_tol=1e-6)


def test_format_reward_partial():
    """Same fixture, no kwargs → rubric_json by default → still 1.0."""
    p = parse_rubric_only(GOOD_RUBRIC_JSON)
    r = format_reward(p)
    assert math.isclose(r, 1.0, abs_tol=1e-6)


def test_format_reward_zero_on_bad_json():
    p = parse_rubric_only(BROKEN_JSON)
    assert format_reward(p, mode="rubric_json") == 0.0


def test_format_reward_penalises_empty_tips():
    """Anti-cheat: every rubric must carry non-empty tips. Empty arrays
    forfeit the +0.2 tips component → 0.8, not 1.0."""
    j = """{
      "rubrics": [
        {"path": "Theme: A", "weight": 0.20, "tips": []},
        {"path": "Theme: B", "weight": 0.20, "tips": []},
        {"path": "Theme: C", "weight": 0.20, "tips": []},
        {"path": "Theme: D", "weight": 0.20, "tips": []},
        {"path": "Theme: E", "weight": 0.20, "tips": []}
      ]
    }"""
    p = parse_rubric_only(j)
    r = format_reward(p, mode="rubric_json")
    assert math.isclose(r, 0.8, abs_tol=1e-6)


def test_format_reward_score_scaffold_full():
    """score_scaffold mode: 5 finite scores in [1,5] with non-trivial
    spread → 1.0."""
    p = parse_rubric_and_scores(GOOD_FULL_JSON)
    r = format_reward(p, mode="score_scaffold")
    assert math.isclose(r, 1.0, abs_tol=1e-6)


def test_format_reward_score_scaffold_no_spread():
    """All-equal scores forfeit the spread bonus → 0.75 (length 0.25 +
    finite 0.25 + range 0.25; spread fails)."""
    from rm_training.stage2_grpo.rubric_parser import build_full_from_scores
    rubric = parse_rubric_only(GOOD_RUBRIC_JSON).rubrics
    full = build_full_from_scores(rubric, [3.0]*5)
    r = format_reward(full, mode="score_scaffold")
    assert math.isclose(r, 0.75, abs_tol=1e-6)


def test_format_reward_score_scaffold_out_of_range():
    """Any score outside [1,5] forfeits the range bonus AND the spread
    sub-bonus that depends on it → 0.5 (length 0.25 + finite 0.25)."""
    from rm_training.stage2_grpo.rubric_parser import build_full_from_scores
    rubric = parse_rubric_only(GOOD_RUBRIC_JSON).rubrics
    # 5.5 is finite but outside the Likert range; build_full clamps it to
    # 5.0, so we have to bypass build_full to exercise this path.
    from rm_training.stage2_grpo.rubric_parser import ParsedFull
    p = ParsedFull(rubrics=rubric, scores=[3.0, 3.5, 5.5, 2.0, 4.0])
    r = format_reward(p, mode="score_scaffold")
    assert math.isclose(r, 0.5, abs_tol=1e-6)


def test_format_reward_score_scaffold_legacy_alias():
    """``want_scores=True`` should still route to score_scaffold."""
    p = parse_rubric_and_scores(GOOD_FULL_JSON)
    r = format_reward(p, want_scores=True)
    assert math.isclose(r, 1.0, abs_tol=1e-6)
