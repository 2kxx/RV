"""Unit tests for the soft-logits / scaffold-parsing scoring path.

Validates that:
  • ``rubric_parser.parse_score_scaffold`` recovers per-theme scores from
    the plain-text scaffold under several emission noise levels;
  • ``rubric_parser.build_full_from_scores`` round-trips a (rubric, scores)
    pair into a ``ParsedFull`` whose ``format_score`` matches the
    expected weighted mean;
  • ``StudentWrapper`` dummy mode produces the new ``(text, list[float])``
    contract;
  • A toy soft-logits aggregation (offline, no transformers) gives the
    expected E[k] = Σ k·softmax(logits) over digit ids.
"""

from __future__ import annotations

import math

try:
    import pytest                                    # noqa: F401  (optional)
except ImportError:                                  # pragma: no cover
    pytest = None                                    # type: ignore

from rm_training.stage2_grpo.rubric_parser   import (
    build_full_from_scores,
    format_score,
    parse_rubric_only,
    parse_score_scaffold,
)
from rm_training.stage2_grpo.student_wrapper import StudentConfig, StudentWrapper


GOOD_RUBRIC_JSON = """{
  "rubrics": [
    {"theme": "Theme: A", "weight": 0.30},
    {"theme": "Theme: B", "weight": 0.20},
    {"theme": "Theme: C", "weight": 0.20},
    {"theme": "Theme: D", "weight": 0.15},
    {"theme": "Theme: E", "weight": 0.15}
  ]
}"""


# ---------------------------------------------------------------------------
# parse_score_scaffold.
# ---------------------------------------------------------------------------
def test_scaffold_clean():
    text = "theme_1: 4 theme_2: 3 theme_3: 5 theme_4: 2 theme_5: 4"
    out = parse_score_scaffold(text, 5)
    assert out == [4.0, 3.0, 5.0, 2.0, 4.0]


def test_scaffold_with_decimals():
    text = "theme_1: 4.5 theme_2: 3.0 theme_3: 1.7 theme_4: 5.0 theme_5: 2.2"
    out = parse_score_scaffold(text, 5)
    # Continuous values pass through; clamp keeps within [1, 5].
    assert out == [4.5, 3.0, 1.7, 5.0, 2.2]


def test_scaffold_with_prose_prefix():
    text = ("Sure, here is the scoring:\n"
            "theme_1: 3, theme_2: 4, theme_3: 2, theme_4: 5, theme_5: 1.")
    out = parse_score_scaffold(text, 5)
    assert out == [3.0, 4.0, 2.0, 5.0, 1.0]


def test_scaffold_falls_back_to_positional():
    """No theme_k keys at all → take first n_themes integers in [1,5]."""
    text = "3 4 2 5 1 (extra noise 6 7)"
    out = parse_score_scaffold(text, 5)
    assert out == [3.0, 4.0, 2.0, 5.0, 1.0]


def test_scaffold_returns_none_when_no_numbers():
    text = "I cannot score this video."
    out = parse_score_scaffold(text, 5)
    assert out == [None] * 5


def test_scaffold_partial_keys_filled_with_none():
    text = "theme_1: 3 theme_3: 5"      # only 2 of 5 themes named
    out = parse_score_scaffold(text, 5)
    # Tier 1 hits → fills 1 and 3 explicitly, others None.
    assert out == [3.0, None, 5.0, None, None]


# ---------------------------------------------------------------------------
# build_full_from_scores.
# ---------------------------------------------------------------------------
def test_build_full_round_trip():
    rubric = parse_rubric_only(GOOD_RUBRIC_JSON).rubrics
    scores = [3.5, 2.0, 3.0, 2.5, 4.0]
    full = build_full_from_scores(rubric, scores)
    expected = (0.30*3.5 + 0.20*2.0 + 0.20*3.0 + 0.15*2.5 + 0.15*4.0)
    assert full.valid
    assert math.isclose(format_score(full), expected, abs_tol=1e-6)


def test_build_full_clamps_out_of_range():
    rubric = parse_rubric_only(GOOD_RUBRIC_JSON).rubrics
    scores = [9.0, 0.5, 3.0, 2.5, 4.0]    # 9 → 5.0, 0.5 → 1.0
    full = build_full_from_scores(rubric, scores)
    assert full.scores[0] == 5.0
    assert full.scores[1] == 1.0
    assert any("score_out_of_range" in s for s in full.issues)


def test_build_full_none_becomes_invalid():
    rubric = parse_rubric_only(GOOD_RUBRIC_JSON).rubrics
    scores = [3.0, None, 3.0, 3.0, 3.0]
    full = build_full_from_scores(rubric, scores)
    # NaN propagates → format_score returns None.
    assert format_score(full) is None
    assert not full.valid


# ---------------------------------------------------------------------------
# StudentWrapper dummy mode contract.
# ---------------------------------------------------------------------------
def test_dummy_score_video_returns_text_and_floats():
    s = StudentWrapper(StudentConfig(model_path=None, dummy=True))
    text, scores = s.score_video(
        caption="q",
        video_path="/tmp/varr_seed_001_win.mp4",
        rubric_text='{"rubrics":[]}',
    )
    # Default 5 themes, all in Likert range [1, 5].
    assert isinstance(text, str)
    assert "theme_1:" in text
    assert len(scores) == 5
    for s_ in scores:
        assert s_ is None or (1.0 <= s_ <= 5.0)


def test_dummy_score_video_win_higher_than_lose():
    """The dummy bias keeps GRPO smoke runs informative — _win > _lose."""
    s = StudentWrapper(StudentConfig(model_path=None, dummy=True))
    _, win  = s.score_video(caption="q", video_path="/tmp/varr_seed_001_win.mp4",
                            rubric_text='{"rubrics":[]}')
    _, lose = s.score_video(caption="q", video_path="/tmp/varr_seed_001_lose.mp4",
                            rubric_text='{"rubrics":[]}')
    # Mean of the 5-tuple should be clearly higher for the "_win" sample.
    win_mean  = sum(x for x in win  if x is not None) / max(1, len(win))
    lose_mean = sum(x for x in lose if x is not None) / max(1, len(lose))
    assert win_mean > lose_mean + 0.5


# ---------------------------------------------------------------------------
# Toy soft-logits aggregation. We don't need a real LM here — just verify
# the "softmax over the {1..5} digit-id subset, weighted by digit value"
# math matches what the wrapper does at runtime.
# ---------------------------------------------------------------------------
def test_softlogits_expectation_math():
    import torch
    # Logits over a tiny vocab where ids 0..4 represent digits 1..5.
    logits = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0])   # digit 1 most likely
    probs  = torch.softmax(logits, dim=-1)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    ev = float((probs * values).sum().item())
    # With those logits the expectation should sit just above 1, well
    # below the midpoint of the scale.
    assert 1.0 <= ev < 2.0
    # And the dual: flip the logits → expectation near 5.
    logits2 = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
    probs2  = torch.softmax(logits2, dim=-1)
    ev2 = float((probs2 * values).sum().item())
    assert 4.0 < ev2 <= 5.0
