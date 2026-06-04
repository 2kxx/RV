"""rm_training.stage2_grpo.rubric_parser
=====================================
Robust JSON parser that turns the **student's** raw text output into a
structured rubric + per-criterion score record. Designed to tolerate
the usual messy-LLM symptoms: stray markdown fences, leading/trailing
prose, single-quoted JSON, slightly broken weight floats, etc.

Expected canonical schema for the student
-----------------------------------------
The student is prompted to emit a JSON object of the form::

    {
      "rubrics": [
        {"path": "<theme: ...>",
         "weight": 0.2,
         "tips": ["...", "..."]            # optional, free-text
        },
        ...   # K=5 entries, weights ≈ sum to 1.0
      ],
      "scores":  [
        {"rubric_index": 1, "score": 2.5},
        ...   # one per rubrics entry
      ]
    }

When ``scores`` is absent (rubric-only generation step), only the
``rubrics`` field is required.

Public API
----------
- ``parse_rubric_only(text) -> ParsedRubric``
- ``parse_rubric_and_scores(text) -> ParsedFull``
- ``format_score(parsed) -> float | None``    weighted average c_k

All functions return ``None`` on hard failure rather than raising, so
the GRPO reward function can convert the failure into a low ``r_format``.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

__all__ = [
    "ParsedRubric",
    "ParsedFull",
    "parse_rubric_only",
    "parse_rubric_and_scores",
    "parse_score_scaffold",
    "build_full_from_scores",
    "format_score",
    "format_reward",
]

# ---------------------------------------------------------------------------
# Data classes.
# ---------------------------------------------------------------------------
@dataclass
class RubricEntry:
    path: str
    weight: float
    tips: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"path": self.path, "weight": self.weight, "tips": list(self.tips)}


@dataclass
class ParsedRubric:
    rubrics: list[RubricEntry]
    raw_text: str = ""
    valid: bool = False
    issues: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rubrics)


@dataclass
class ParsedFull(ParsedRubric):
    scores: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 1. Pre-processing: strip code-fences, extract first {...} block.
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_FIRST_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

def _extract_json_blob(text: str) -> str | None:
    """Try, in order:
    1. The first ```json ... ``` fenced code block.
    2. The first {...} run (greedy across newlines).
    """
    if not text:
        return None
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _FIRST_OBJ_RE.search(text)
    if m:
        return m.group(0).strip()
    return None


def _safe_json_loads(blob: str) -> dict | None:
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # Soft fix #1: single quotes -> double quotes, but only inside
        # the obvious key/value runs to avoid mangling string contents.
        # We're conservative: if the simple swap parses, take it; else fail.
        try:
            return json.loads(blob.replace("'", '"'))
        except json.JSONDecodeError:
            return None


# ---------------------------------------------------------------------------
# 2. Rubric-only parsing.
# ---------------------------------------------------------------------------
_EXPECTED_K = 5  # Number of rubric entries the student is asked to emit.
_W_TOLERANCE = 0.05  # |Σ weights − 1| ≤ this is OK.


def parse_rubric_only(text: str) -> ParsedRubric:
    """Parse a student emission that contains ONLY the rubrics block.

    Returns a ParsedRubric whose ``valid`` flag captures the strict
    contract expected by ``r_format`` (see ``format_reward``).
    """
    issues: list[str] = []
    rubrics: list[RubricEntry] = []
    out = ParsedRubric(rubrics=rubrics, raw_text=text or "", issues=issues)

    blob = _extract_json_blob(text or "")
    if blob is None:
        issues.append("no_json_blob")
        return out

    obj = _safe_json_loads(blob)
    if not isinstance(obj, dict):
        issues.append("json_not_object")
        return out

    raw_rubrics = obj.get("rubrics")
    if not isinstance(raw_rubrics, list) or not raw_rubrics:
        issues.append("rubrics_field_missing_or_empty")
        return out

    for r in raw_rubrics:
        if not isinstance(r, dict):
            continue
        path = (r.get("path") or "").strip()
        if not path:
            continue
        try:
            weight = float(r.get("weight"))
        except (TypeError, ValueError):
            issues.append(f"bad_weight:{path[:30]}")
            weight = 0.0
        tips = r.get("tips") or []
        if not isinstance(tips, list):
            tips = [str(tips)]
        rubrics.append(RubricEntry(path=path, weight=weight,
                                    tips=[str(t) for t in tips]))

    if len(rubrics) != _EXPECTED_K:
        issues.append(f"wrong_K={len(rubrics)} (expected {_EXPECTED_K})")

    w_sum = sum(r.weight for r in rubrics)
    if abs(w_sum - 1.0) > _W_TOLERANCE:
        issues.append(f"weight_sum={w_sum:.3f}")

    out.valid = (len(issues) == 0)
    return out


# ---------------------------------------------------------------------------
# 3. Full (rubrics + scores) parsing.
# ---------------------------------------------------------------------------
# Likert 1-5 scale (matches stage0 D5 + standard VBench/Q-Bench/VideoScore-2).
# Soft-logits mode produces continuous floats in [1.0, 5.0]; integer_text
# mode produces ints in {1,2,3,4,5}. Both share the same downstream code.
_SCORE_LO, _SCORE_HI = 1.0, 5.0


def parse_rubric_and_scores(text: str) -> ParsedFull:
    """Parse a student emission carrying both rubrics and per-criterion scores.

    The scores list is **aligned by ``rubric_index`` (1-based)**. Missing
    indices yield NaN entries to keep length parity with rubrics; this lets
    downstream code detect partial failures without crashing.
    """
    base = parse_rubric_only(text)
    out = ParsedFull(
        rubrics=base.rubrics,
        raw_text=base.raw_text,
        valid=False,                # tighten below
        issues=list(base.issues),
        scores=[],
    )

    blob = _extract_json_blob(text or "")
    obj = _safe_json_loads(blob) if blob else None
    if not isinstance(obj, dict):
        out.issues.append("scores_no_object")
        return out

    raw_scores = obj.get("scores")
    if not isinstance(raw_scores, list):
        out.issues.append("scores_field_missing")
        return out

    by_idx: dict[int, float] = {}
    for s in raw_scores:
        if not isinstance(s, dict):
            continue
        try:
            idx = int(s.get("rubric_index"))
            val = float(s.get("score"))
        except (TypeError, ValueError):
            continue
        by_idx[idx] = val

    scores: list[float] = []
    for i in range(1, len(out.rubrics) + 1):
        v = by_idx.get(i)
        if v is None or not math.isfinite(v):
            scores.append(float("nan"))
            out.issues.append(f"missing_score_idx={i}")
            continue
        if v < _SCORE_LO or v > _SCORE_HI:
            out.issues.append(f"score_out_of_range:{v}")
            v = max(_SCORE_LO, min(_SCORE_HI, v))
        scores.append(v)
    out.scores = scores

    out.valid = (
        base.valid
        and len(scores) == len(out.rubrics)
        and all(math.isfinite(v) for v in scores)
    )
    return out


# ---------------------------------------------------------------------------
# 3b. Plain-text score scaffold parser (used when score_method=integer_text
#     and the student emits the stage0-D5 plain-text format
#     "theme_1: <X> theme_2: <X> ..." instead of JSON).
# ---------------------------------------------------------------------------
def parse_score_scaffold(text: str, n_themes: int) -> list[float | None]:
    """Extract per-theme scores from a ``theme_k: <X>`` plain-text scaffold.

    Tolerates the usual 7B noise:
      • leading prose ("Sure, here are the scores:");
      • trailing punctuation ("theme_1: 3, theme_2: 2");
      • mixed integer / decimal ("theme_1: 3.5");
      • missing themes (filled with None — caller decides what to do).

    Falls back to a positional digit scrape when explicit ``theme_k:``
    keys aren't found, so a fully derailed emission like
    "3 4 2 5 3" still yields ``[3, 4, 2, 5, 3]``.
    """
    if not text or n_themes <= 0:
        return [None] * max(n_themes, 0)

    # Tier 1: explicit "theme_<k>: <number>" keys.
    kv: dict[str, str] = {}
    for m in re.finditer(
        r'(?:^|[^a-zA-Z_])theme[_\s]*(\d+)\s*[:=]\s*([-+]?\d*\.?\d+)',
        text,
        flags=re.IGNORECASE,
    ):
        kv[m.group(1)] = m.group(2)
    if kv:
        out = [None] * n_themes
        for k in range(1, n_themes + 1):
            v = kv.get(str(k))
            if v is None:
                continue
            try:
                fv = float(v)
            except ValueError:
                continue
            out[k - 1] = max(_SCORE_LO, min(_SCORE_HI, fv))
        if any(x is not None for x in out):
            return out

    # Tier 2: positional digit scrape (any int/float in [1, 5]).
    nums: list[float] = []
    for tok in re.findall(r"[-+]?\d*\.?\d+", text):
        try:
            v = float(tok)
        except ValueError:
            continue
        if _SCORE_LO - 0.5 <= v <= _SCORE_HI + 0.5:
            nums.append(max(_SCORE_LO, min(_SCORE_HI, v)))
    if len(nums) >= n_themes:
        return nums[:n_themes]

    return [None] * n_themes


# ---------------------------------------------------------------------------
# 3c. Build a ParsedFull directly from a rubric + score list — the shape
#     the soft-logits scoring path produces. Caller doesn't need to JSON-
#     encode → re-parse.
# ---------------------------------------------------------------------------
def build_full_from_scores(
    rubric_entries: list,
    scores:         list[float | None],
    *,
    raw_text:       str = "",
) -> "ParsedFull":
    """Assemble a ParsedFull whose ``rubrics`` are the locked generation
    rubric and whose ``scores`` come from the score forward (text-parsed
    or logits-derived). Mirrors the post-condition of
    ``parse_rubric_and_scores`` so reward_fn can stay unchanged.

    None entries in ``scores`` are converted to NaN so ``format_score``'s
    "any NaN → None" guard still triggers (consistent with the old path).
    """
    issues: list[str] = []
    cleaned: list[float] = []
    for i, s in enumerate(scores):
        if s is None or not (isinstance(s, (int, float))) or not math.isfinite(float(s)):
            cleaned.append(float("nan"))
            issues.append(f"missing_score_idx={i+1}")
            continue
        v = float(s)
        if v < _SCORE_LO or v > _SCORE_HI:
            issues.append(f"score_out_of_range:{v}")
            v = max(_SCORE_LO, min(_SCORE_HI, v))
        cleaned.append(v)
    valid = (
        bool(rubric_entries)
        and len(cleaned) == len(rubric_entries)
        and all(math.isfinite(v) for v in cleaned)
    )
    return ParsedFull(
        rubrics=list(rubric_entries),
        raw_text=raw_text,
        valid=valid,
        issues=issues,
        scores=cleaned,
    )


# ---------------------------------------------------------------------------
# 4. Aggregation: weighted average c_k.
# ---------------------------------------------------------------------------
def format_score(parsed: ParsedFull) -> float | None:
    """Compute  S = Σ w_k · c_k / Σ w_k.

    Returns None if any score is NaN or all weights are zero.
    """
    if not parsed.rubrics or not parsed.scores:
        return None
    if len(parsed.rubrics) != len(parsed.scores):
        return None
    w_sum = sum(r.weight for r in parsed.rubrics)
    if w_sum <= 0:
        return None
    num = 0.0
    for r, s in zip(parsed.rubrics, parsed.scores):
        if not math.isfinite(s):
            return None
        num += r.weight * s
    return num / w_sum


# ---------------------------------------------------------------------------
# 5. r_format reward: 0..1 scalar capturing schema compliance.
#
# Two distinct evaluation modes — rubric-gen and score-forward emissions
# satisfy different schemas, so blending them on one rubric (the old
# ``want_scores`` flag) over- and under-counts in different places:
#
#   • For the score forward, ``parsed.rubrics`` is now ALWAYS populated
#     by ``build_full_from_scores`` from the locked rubric-gen output —
#     so the K=5 / Σw=1 sub-rewards are 1.0 by construction and tell us
#     nothing about whether the score forward did its job. Hence
#     ``mode="score_scaffold"`` only evaluates the scores list itself.
#
#   • For the rubric-gen forward, the scaffold doesn't apply (no scores
#     are emitted there). ``mode="rubric_json"`` evaluates the JSON
#     schema + tips field. Tips weight rewards a non-empty tips array
#     (instead of the old "want_scores=False → +0.2 freebie" stub) so
#     the student can't game the format reward by emitting
#     ``"tips": []`` for every rubric.
#
# Public surface
# --------------
# - ``format_reward(parsed, mode="rubric_json")`` — for rubric-gen output
# - ``format_reward(parsed, mode="score_scaffold")`` — for score forward
# - ``format_reward(parsed, want_scores=...)`` — DEPRECATED alias kept
#   for backward compatibility; True → score_scaffold, False → rubric_json.
# ---------------------------------------------------------------------------
_SCAFFOLD_MIN_STD = 0.1   # score forward must show meaningful spread


def _format_reward_rubric_json(parsed: ParsedRubric | ParsedFull) -> float:
    """0..1 scalar for the rubric-gen JSON emission."""
    if not parsed.rubrics:
        return 0.0
    r = 0.0
    # +0.4: parses to a dict with rubrics ≥ 1.
    r += 0.4
    # +0.2: exact K=5 rubrics.
    if len(parsed.rubrics) == _EXPECTED_K:
        r += 0.2
    # +0.2: weights sum to 1 ± 0.05.
    w_sum = sum(x.weight for x in parsed.rubrics)
    if abs(w_sum - 1.0) <= _W_TOLERANCE:
        r += 0.2
    # +0.2: every rubric carries non-empty tips (anti-cheat: forbid
    # ``"tips": []`` shortcuts that would otherwise satisfy the schema).
    if parsed.rubrics and all(
        isinstance(rb.tips, list) and any(str(t).strip() for t in rb.tips)
        for rb in parsed.rubrics
    ):
        r += 0.2
    return min(1.0, r)


def _format_reward_score_scaffold(parsed: ParsedFull) -> float:
    """0..1 scalar for the score forward's plain-text scaffold emission.

    Granular: rewards length, finiteness, range, and inter-theme spread.
    Each component is 0.25 so a fully-broken emission scores 0 and a
    perfect one scores 1.
    """
    sc = list(getattr(parsed, "scores", None) or [])
    K  = len(parsed.rubrics) if parsed.rubrics else _EXPECTED_K
    if not sc:
        return 0.0
    r = 0.0
    # +0.25: scores list length matches rubric count (or K=5 if the
    #        rubric-gen step degenerated to <5 entries).
    if len(sc) == K:
        r += 0.25
    # +0.25: every score is finite (no NaN / None / inf leakage).
    if all(isinstance(v, (int, float)) and math.isfinite(v) for v in sc):
        r += 0.25
        # +0.25: every score is within the Likert range [1, 5].
        if all(_SCORE_LO <= float(v) <= _SCORE_HI for v in sc):
            r += 0.25
            # +0.25: meaningful spread across themes (sample std > 0.1).
            if len(sc) >= 2:
                mean = sum(float(v) for v in sc) / len(sc)
                var  = sum((float(v) - mean) ** 2 for v in sc) / len(sc)
                if (var ** 0.5) > _SCAFFOLD_MIN_STD:
                    r += 0.25
            else:
                # Degenerate K=1 scaffold — nothing to measure spread on.
                r += 0.25
    return min(1.0, r)


def format_reward(parsed: ParsedRubric | ParsedFull,
                  *, mode: str | None = None,
                  want_scores: bool | None = None) -> float:
    """Granular schema-compliance reward for GRPO.

    Pass ``mode="rubric_json"`` for rubric-gen output and
    ``mode="score_scaffold"`` for score-forward output. The legacy
    ``want_scores`` flag remains as a deprecated alias.
    """
    if mode is None:
        # Back-compat shim: True → score_scaffold, False → rubric_json,
        # default (no kwarg) → rubric_json (the rubric-only case).
        if want_scores is None:
            mode = "rubric_json"
        else:
            mode = "score_scaffold" if want_scores else "rubric_json"

    if mode == "rubric_json":
        return _format_reward_rubric_json(parsed)
    if mode == "score_scaffold":
        return _format_reward_score_scaffold(parsed)
    raise ValueError(f"format_reward: unknown mode={mode!r}")
