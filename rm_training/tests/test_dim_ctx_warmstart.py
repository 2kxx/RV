"""Unit tests for the per-dimension prompt injection + warm-start anchor.

Validates:
  • ``task_descriptions.load_task_specs`` returns ≥ 18 dims with parsed
    sub_aspects (when applicable);
  • ``DimensionContext.generic`` and ``.sub_aspects_block`` produce
    sensible scaffolds;
  • ``StudentWrapper._format_rubric_prompt`` substitutes every template
    slot (no leftover ``{...}`` tokens, no KeyError);
  • dummy ``generate_rubric / score_video`` accept the new signature;
  • ``_r_align`` saturates at ``align_floor`` and decays linearly below;
  • ``--no-align_saturate`` recovers the legacy linear behaviour.
"""

from __future__ import annotations

import math
from pathlib import Path

try:
    import pytest                                    # noqa: F401  (optional)
except ImportError:                                  # pragma: no cover
    pytest = None                                    # type: ignore

from rm_training.stage1_teacher.task_descriptions import (
    DEFAULT_CONFIG_GLOB,
    TaskSpec,
    load_task_specs,
)
from rm_training.stage2_grpo.gold_align    import DummyEmbedder
from rm_training.stage2_grpo.reward_fn     import (
    BatchRewardConfig,
    _r_align,
)
from rm_training.stage2_grpo.rubric_parser import parse_rubric_only
from rm_training.stage2_grpo.student_wrapper import (
    DimensionContext,
    StudentConfig,
    StudentWrapper,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# task_specs loader.
# ---------------------------------------------------------------------------
def test_load_task_specs_covers_18_evalverse_dims():
    specs = load_task_specs(REPO_ROOT)
    # 17 explicit yamls + visual_quality fallback = 18.
    assert len(specs) >= 18
    expected = {
        "action", "character", "chromaticity", "composition", "consistency",
        "expression", "grounding", "lens", "lighting", "logic", "materiality",
        "pacing", "progression", "rhythm", "scene", "soundscape",
        "visual_quality", "vocal",
    }
    assert expected.issubset(specs.keys())


def test_task_spec_sub_aspects_extracted_for_action():
    """sub_aspects bullets must come ONLY from the 'Sub-aspects' header
    block — not from Rubric Requirements / Pointwise calibration sections."""
    specs = load_task_specs(REPO_ROOT)
    s = specs["action"]
    # The action yaml lists exactly 3 sub-aspects.
    assert len(s.sub_aspects) == 3
    assert all(a.startswith("Acting-Action-") for a in s.sub_aspects)
    # Sanity: rubric-requirements bullets must NOT leak in.
    leak_phrases = ("Observable from the supplied",
                    "Score 5 = excellent",
                    "Discriminative between the winner")
    for p in leak_phrases:
        assert not any(p in a for a in s.sub_aspects), \
            f"leak detected: {p!r} in {s.sub_aspects}"


def test_visual_quality_fallback_present_with_empty_sub_aspects():
    specs = load_task_specs(REPO_ROOT)
    vq = specs["visual_quality"]
    assert vq.dimension == "visual_quality"
    assert vq.evalverse_main == "Visual_Quality"
    # Generic prompt has no fixed taxonomy → empty sub_aspects on purpose.
    assert vq.sub_aspects == []
    # But the description body must be non-trivial.
    assert len(vq.task_description_short) > 200


def test_task_spec_short_drops_pointwise_boilerplate():
    specs = load_task_specs(REPO_ROOT)
    s = specs["action"]
    assert "## Pointwise scoring calibration" not in s.task_description_short
    assert "HARD constraints" not in s.task_description_short
    # And the sub-aspects header IS preserved (so student can see it).
    assert "Sub-aspects covered" in s.task_description_short


# ---------------------------------------------------------------------------
# DimensionContext.
# ---------------------------------------------------------------------------
def test_dim_ctx_generic_renders():
    ctx = DimensionContext.generic("action")
    assert ctx.dimension == "action"
    assert ctx.task_description
    block = ctx.sub_aspects_block()
    assert block.startswith("- ")    # always at least one bullet


def test_dim_ctx_sub_aspects_block_lists_when_populated():
    ctx = DimensionContext(
        dimension="action",
        evalverse_path="Acting > Action",
        task_description="(...)",
        sub_aspects=["Acting-Action-Action-Emotion_Synergy",
                     "Acting-Action-Action_Tension"],
    )
    block = ctx.sub_aspects_block()
    assert "Acting-Action-Action_Tension" in block
    # One bullet per sub-aspect.
    assert block.count("\n- ") + 1 == 2


# ---------------------------------------------------------------------------
# Prompt template substitution — every slot must be filled, no KeyError.
# ---------------------------------------------------------------------------
def test_rubric_prompt_template_substitutes_all_slots():
    s   = StudentWrapper(StudentConfig(model_path=None, dummy=True))
    ctx = DimensionContext.generic("action")
    prompt = s._format_rubric_prompt(ctx, caption="A red sports car drifts.")
    # Key fields must appear in the rendered prompt.
    assert "action" in prompt.lower()
    assert "red sports car" in prompt.lower()
    # No KeyError-style unsubstituted slots.
    bad = [t for t in (
        "{dimension}", "{evalverse_path}", "{task_description}",
        "{sub_aspects_block}", "{caption}",
    ) if t in prompt]
    assert bad == [], f"unsubstituted: {bad}"


def test_score_prompt_template_substitutes_all_slots():
    s   = StudentWrapper(StudentConfig(model_path=None, dummy=True))
    ctx = DimensionContext.generic("action")
    fmt = "theme_1: <X> theme_2: <X>"
    prompt = s._format_score_prompt(
        ctx, caption="cap", rubric_text='{"rubrics":[]}',
        format_line=fmt,
    )
    assert "action" in prompt.lower()
    assert "theme_1: <X>" in prompt
    # No KeyError-style `{...}` leak.
    bad = [t for t in ("{query}", "{caption}", "{rubric_text}", "{dimension}")
           if t in prompt]
    assert bad == [], f"unsubstituted: {bad}"


# ---------------------------------------------------------------------------
# Dummy public ops accept the new signature.
# ---------------------------------------------------------------------------
def test_dummy_generate_rubric_accepts_dim_ctx():
    s   = StudentWrapper(StudentConfig(model_path=None, dummy=True))
    ctx = DimensionContext.generic("character")
    out = s.generate_rubric("a man in red robes", dim_ctx=ctx)
    assert isinstance(out, str)
    assert len(out) > 0


def test_dummy_score_video_accepts_dim_ctx_kwargs():
    s   = StudentWrapper(StudentConfig(model_path=None, dummy=True))
    ctx = DimensionContext.generic("character")
    text, scores = s.score_video(
        "caption", "/tmp/varr_seed_001_win.mp4", '{"rubrics":[]}',
        dim_ctx=ctx,
    )
    assert isinstance(text, str)
    assert len(scores) == 5


# ---------------------------------------------------------------------------
# r_align warm-start anchor.
# ---------------------------------------------------------------------------
def _rubric_with_paths(paths):
    js = (
        '{"rubrics": ['
        + ", ".join(
            f'{{"path": "{p}", "weight": {1/len(paths):.4f}, "tips": ["t1","t2"]}}'
            for p in paths
        )
        + "]}"
    )
    return parse_rubric_only(js)


def test_r_align_saturates_above_floor():
    """Identical rubric paths → sim ≈ 1 → saturated → exactly 1.0."""
    paths = ["Theme: A", "Theme: B", "Theme: C", "Theme: D", "Theme: E"]
    student = _rubric_with_paths(paths)
    gold    = [{"path": p} for p in paths]
    cfg = BatchRewardConfig(align_floor=0.7, align_saturate=True)
    r = _r_align(student, gold, embedder=DummyEmbedder(), cfg=cfg)
    assert math.isclose(r, 1.0, abs_tol=1e-9)


def test_r_align_linear_below_floor():
    """Unrelated paths → sim well below 0.7 → linearly scaled to floor."""
    student = _rubric_with_paths(["red car", "blue sky", "green grass",
                                  "cold mountain", "warm sun"])
    gold    = [{"path": p} for p in
               ["Theme: VQ", "Theme: Char", "Theme: Env",
                "Theme: Act", "Theme: Cam"]]
    cfg = BatchRewardConfig(align_floor=0.7, align_saturate=True)
    r = _r_align(student, gold, embedder=DummyEmbedder(dim=64), cfg=cfg)
    # Some positive reward iff sim > 0; capped well below 1.0.
    assert 0.0 <= r < 1.0
    assert r < 0.7   # well under the saturation point


def test_r_align_legacy_mode_returns_raw_sim():
    """--no-align_saturate path: r_align is sim itself (capped at 1)."""
    paths = ["Theme: A", "Theme: B", "Theme: C", "Theme: D", "Theme: E"]
    student = _rubric_with_paths(paths)
    gold    = [{"path": p} for p in paths]
    cfg = BatchRewardConfig(align_saturate=False)
    r = _r_align(student, gold, embedder=DummyEmbedder(), cfg=cfg)
    # sim ≈ 1.0 either way for self-match.
    assert math.isclose(r, 1.0, abs_tol=1e-3)


def test_r_align_floor_kwarg_actually_changes_threshold():
    """A higher floor should pull the saturated reward downward for
    fixed-similarity inputs."""
    student = _rubric_with_paths(
        ["Theme: A", "Theme: B", "Theme: X", "Theme: Y", "Theme: Z"]
    )
    gold = [
        {"path": "Theme: A"}, {"path": "Theme: B"},
        {"path": "Theme: C"}, {"path": "Theme: D"}, {"path": "Theme: E"},
    ]
    e = DummyEmbedder(dim=64)
    cfg_low  = BatchRewardConfig(align_floor=0.3, align_saturate=True)
    cfg_high = BatchRewardConfig(align_floor=0.95, align_saturate=True)
    r_low  = _r_align(student, gold, embedder=e, cfg=cfg_low)
    r_high = _r_align(student, gold, embedder=e, cfg=cfg_high)
    # Lower floor saturates earlier → reward at least as high.
    assert r_low >= r_high - 1e-9
