#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rm_training.stage0_diagnosis.exp0_base_diagnosis
================================================
Stage ⓪ — Base-model "physical exam" before any RL training.

Adapted from the original Exp-0 RGPO gating script for RewardVerse:
  • paths / data schema match RewardVerse's existing v_arr_mvp pair files
    (use ``build_pair_data.py`` first to materialise them);
  • dimension list matches RewardVerse's 18-axis taxonomy;
  • frame-count fields use a multi-strategy fallback so the script
    survives qwen-vl-utils version drift (``nframes`` vs ``max_frames``);
  • CLI overrides for every CONFIG path so a future GPU node run only
    needs to edit one or two flags.

Four diagnostics (all zero-shot, no training):
  D1  Pairwise judgement baseline                 → gating headline number
  D2  Frame-sampling stress test (8/16/32/64)     → smallest acc-saturating N
  D3  Per-dimension probe                         → strong/weak heatmap
  D4  Pointwise per-theme rubric scoring          → 'pointwise + agg'
      ── for each pair, score each video on every rubric theme
         independently (model sees ONE video at a time), then aggregate
         the per-theme scores into a scalar S(video). Pair preference is
         decided by S(v_pos) vs S(v_neg). This bypasses positional bias
         entirely and is architecturally identical to the Stage ②/③
         pointwise reward path.
  D5  Pointwise + full-rubric JSON multi-score    → deployable variant
      ── same architecture as D4, but the FULL rubric (all themes, all
         tips) is inlined into a single forward per video and the model
         emits a JSON object with one score per theme. Forward count
         drops from N_themes per video to 1 per video, matching the
         budget Stage ②/③ training/inference will demand. Per-theme
         breakdown is preserved via a 3-tier JSON parser fallback so
         downstream variance / gap diagnostics remain available.

Goal:  if D1 acc < 0.60, the base model can't even see relative quality →
       RL is hopeless. >0.70 = PASS, 0.65-0.70 = GRAY (rubric helps),
       <0.65 = FAIL.

Three quality knobs baked in:
  • ``debias_swap`` — judge each pair (pos, neg) and (neg, pos); only count
    when both orderings agree. Exposes positional bias.
  • Two reported accuracies — ``accuracy_on_decided`` (only count agreed
    pairs) and ``accuracy_conservative`` (uncertain == wrong).
  • Data-driven — point CONFIG at JSONL files; no hard-coded cases.

Usage
-----
::

    # 1) Prepare pair data from v_arr_mvp (one-time):
    python rm_training/stage0_diagnosis/build_pair_data.py --require_local

    # 2) Run the full physical exam (needs a GPU node):
    python rm_training/stage0_diagnosis/exp0_base_diagnosis.py \\
        --diagnosis all \\
        --model_path /group/40063/jacuzzitang/models/Qwen2.5-VL-7B-Instruct

    # Single diagnostic only:
    python rm_training/stage0_diagnosis/exp0_base_diagnosis.py --diagnosis d1 \\
        --max_samples 50

Dependencies (already installed in the project env):
  transformers>=4.53, torch, qwen-vl-utils, decord, av, accelerate
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

# Make ``rm_training.*`` importable when run as a plain script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# CONFIG — defaults; every value can be overridden via CLI.
# ============================================================
CONFIG = {
    # Base model — local cluster path is preferred; HF id is a fallback.
    "model_path": os.environ.get(
        "RM_MODEL_PATH",
        "/group/40063/jacuzzitang/models/Qwen2.5-VL-7B-Instruct",
    ),
    "device": "cuda",
    "dtype":  "bfloat16",

    # Data — produced by ``build_pair_data.py``.
    "pair_data":     str(ROOT / "data" / "rm_train" / "exp0" / "exp0_pairs.jsonl"),
    "dim_pair_data": str(ROOT / "data" / "rm_train" / "exp0" / "exp0_dim_pairs.jsonl"),

    # Frame sampling.
    #   strategy="nframes"      — force exactly N equispaced frames (legacy)
    #   strategy="fps"          — qwen-vl-utils picks frames at fps Hz, capped
    #                              at max_frames (matches Qwen2.5-VL official
    #                              README + VideoScore2 default)
    #   strategy="auto_default" — don't pass any frame fields; let
    #                              qwen-vl-utils use its built-in defaults
    "frame_strategy":   "fps",        # ← default switched away from nframes
    "fps":              2.0,           # used when strategy in {fps}
    "max_frames":       64,            # cap so long videos don't OOM
    "default_nframes":  16,            # used only when strategy == "nframes"
    "frame_sweep":      [8, 16, 32, 64],   # only consulted by D2

    # Output.
    "out_dir": str(ROOT / "runs" / "exp0_base_diagnosis"),

    # Eval control.
    "max_samples": None,   # None = full set; debug with e.g. 50
    "seed":        0,
    "debias_swap": True,

    # In-context gold-rubric ablation (default off).
    "rubric_mode":        "none",        # {none, gold_dim, gold_overall}
    "rubric_dir":         str(ROOT / "rubric_pipeline" / "rubrics"),
    "overall_rubric_dim": "visual_quality",

    # D5 scoring method.
    #   "integer_text" — model emits a JSON of int 1-5 per theme; we
    #                    parse the text. Original D5 path. Coarse, but
    #                    no special generate kwargs.
    #   "soft_logits"  — model emits the SAME plain-text scaffold but
    #                    we read per-position logits and compute the
    #                    expected score E[k]=Σ k·softmax(logits_k)
    #                    for k=1..5. Yields continuous floats in [1,5],
    #                    eliminates parse_fail, and gives much higher
    #                    var_pos / lower uncertain-rate. Requires a
    #                    single forward with output_scores=True.
    "score_method":       "soft_logits",   # {integer_text, soft_logits}
}


# ============================================================
# DATA FORMAT
# ------------------------------------------------------------
# pair_data (exp0_pairs.jsonl) — produced by build_pair_data.py:
# {"id": "varr_vq_000",
#  "instruction": "... full prompt ...",
#  "video_pos":   "/path/to/winner.mp4",
#  "video_neg":   "/path/to/loser.mp4",
#  "dimension":   "visual_quality",
#  "url_pos":     "https://...mp4",   # for traceability
#  "url_neg":     "https://...mp4"}
#
# dim_pair_data is the same schema; D3 reads ``dimension`` to bucket.
# ============================================================


# ============================================================
# RewardVerse 18-axis taxonomy (matches data/v_arr_mvp/<dim>_mvp/).
# Strong/weak split helps decide which dims rubric must shore up.
# ============================================================
DIMENSIONS = [
    "visual_quality",   # baseline visual fidelity
    "consistency",      # temporal / cross-shot consistency
    "grounding",        # prompt instruction grounding
    "character",        # subject identity / count
    "action",           # action plausibility
    "lens",             # camera framing / language
    "lighting",         # lighting quality
    "composition",      # framing composition
    "scene",            # scene structure
    "chromaticity",     # colour grading
    "expression",       # facial expression / emotion
    "materiality",      # material rendering / physical realism
    "logic",            # logical / narrative coherence
    "pacing",           # editing pace / shot length
    "progression",      # narrative progression
    "rhythm",           # temporal rhythm
    "soundscape",       # ambient sound
    "vocal",            # speech / dialogue
]


# ============================================================
# Rubric loader (in-context gold-rubric ablation).
#
# Mode semantics:
#   * "none"         — no rubric injected; baseline pairwise judgement
#                      (matches the original D1/D2/D3 paths).
#   * "gold_dim"     — inject the per-dim gold rubric for the pair's
#                      ``dimension`` field; D3 uses this most naturally.
#                      Falls back to no-rubric if the file is missing.
#   * "gold_overall" — inject the visual_quality v2 rubric (or whatever
#                      ``--overall_rubric_dim`` points at) regardless of
#                      the pair's own dimension. Useful for D1/D2 where
#                      we want one consistent rubric over the whole set.
# ============================================================
def _rubric_path(rubric_dir: Path, dimension: str) -> Path:
    """Return the canonical rubric .txt path for a given dimension.

    ``visual_quality`` is special-cased: its v2 rubric file is named
    ``v_arr_mvp_vq_v2_prompt_pair.txt`` rather than ``v_arr_mvp_visual_quality_v2.txt``
    (a quirk of the historical rubric authoring round).
    """
    if dimension == "visual_quality":
        return rubric_dir / "v_arr_mvp_vq_v2_prompt_pair.txt"
    return rubric_dir / f"v_arr_mvp_{dimension}_v2.txt"


_RUBRIC_CACHE: dict[str, str] = {}
_RUBRIC_THEMES_CACHE: dict[str, list[dict]] = {}


def _parse_rubric_themes(text: str) -> list[dict]:
    """Parse a v_arr_mvp_<dim>_v2.txt rubric into a list of themes.

    The rubric files follow this exact shape (see
    ``v_arr_mvp_visual_quality_v2.txt`` etc.):

        Rubric 1:
        Theme: Visual Quality and Temporal Consistency
        - Tip1: ...
        - Tip2: ...
        - Tip3: ...
        - Tip4: ...

        Rubric 2:
        Theme: Character Fidelity and Appearance
        - Tip1: ...
        ...

    Returns a list of ``{"theme": str, "tips": list[str]}`` dicts in
    file order. Robust to:
      * 3-tip OR 4-tip themes,
      * blank lines between blocks,
      * a missing trailing newline,
      * the ``Rubric N:`` header being optional (some files only have
        ``Theme: ...`` lines).
    """
    themes: list[dict] = []
    cur: dict | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        # Start of a new block — either ``Rubric N:`` or ``Theme: ...``.
        if line.lower().startswith("rubric ") and line.rstrip().endswith(":"):
            # Pure header; flush previous, start fresh.
            if cur and cur.get("theme"):
                themes.append(cur)
            cur = {"theme": None, "tips": []}
            continue
        if line.startswith("Theme:"):
            # Some rubric files omit the ``Rubric N:`` header.
            if cur is None:
                cur = {"theme": None, "tips": []}
            if cur.get("theme"):
                # Already had a theme in the same block → flush & restart.
                themes.append(cur)
                cur = {"theme": None, "tips": []}
            cur["theme"] = line[len("Theme:"):].strip()
            continue
        # Tip line: ``- Tip1: ...`` or ``- ...``.
        stripped = line.lstrip()
        if stripped.startswith("-"):
            if cur is None:
                cur = {"theme": None, "tips": []}
            cur["tips"].append(stripped.lstrip("-").strip())
            continue
        # Anything else (rare) — append to last tip as continuation.
        if cur is not None and cur["tips"]:
            cur["tips"][-1] += " " + stripped
    if cur and cur.get("theme"):
        themes.append(cur)
    return themes


def _load_rubric_themes(rubric_dir: Path, dimension: str) -> list[dict]:
    """Like ``_load_rubric`` but returns parsed themes (cached).

    Returns ``[]`` if the rubric file is missing or unparseable so the
    caller can fall back gracefully.
    """
    key = f"{rubric_dir}::{dimension}"
    if key in _RUBRIC_THEMES_CACHE:
        return _RUBRIC_THEMES_CACHE[key]
    fp = _rubric_path(rubric_dir, dimension)
    if not fp.is_file():
        _RUBRIC_THEMES_CACHE[key] = []
        return []
    try:
        text = fp.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"[rubric/themes] failed to read {fp}: {e}", file=sys.stderr)
        _RUBRIC_THEMES_CACHE[key] = []
        return []
    themes = _parse_rubric_themes(text)
    _RUBRIC_THEMES_CACHE[key] = themes
    return themes


def _load_rubric(rubric_dir: Path, dimension: str, max_chars: int = 4000) -> str | None:
    """Read the per-dim rubric .txt and return its content trimmed to ``max_chars``.

    Returns ``None`` if the file is missing (the caller should then fall
    back to no-rubric judgement so the experiment still produces a number).
    Caches per (dim, dir) so the file is only read once per process.
    """
    key = f"{rubric_dir}::{dimension}"
    if key in _RUBRIC_CACHE:
        return _RUBRIC_CACHE[key] or None
    fp = _rubric_path(rubric_dir, dimension)
    if not fp.is_file():
        _RUBRIC_CACHE[key] = ""
        return None
    try:
        text = fp.read_text(encoding="utf-8").strip()
    except Exception as e:
        print(f"[rubric] failed to read {fp}: {e}", file=sys.stderr)
        _RUBRIC_CACHE[key] = ""
        return None
    if len(text) > max_chars:
        # Keep the first max_chars; the rubric file leads with the most
        # important themes, so head-truncation is safe.
        text = text[:max_chars] + "\n[...truncated...]"
    _RUBRIC_CACHE[key] = text
    return text


# ============================================================
# Frame-spec helper: qwen-vl-utils 0.0.x and 0.1.x disagree on which
# field name controls frame count. We try ``nframes`` first (the
# original script's choice), and if downstream raises a
# TypeError/ValueError we retry with ``max_frames`` + ``fps``.
# ============================================================
# ============================================================
# Frame-spec helper. Three strategies are supported (see CONFIG):
#   * ``nframes``      — fixed N equispaced frames (legacy).
#   * ``fps``          — sample at fps Hz, cap at max_frames
#                        (Qwen2.5-VL + VideoScore2 official default).
#   * ``auto_default`` — don't pass any frame field; defer entirely
#                        to qwen-vl-utils' built-in defaults.
#
# When ``strategy="nframes"`` we additionally retry once with
# ``max_frames`` + ``fps`` if the first call raises a TypeError /
# ValueError (the qwen-vl-utils 0.0.x ↔ 0.1.x API rename).
# ============================================================
def _video_block(video_path: str,
                 nframes:    int,
                 *,
                 mode:       str   = "nframes",
                 fps:        float = 2.0,
                 max_frames: int   = 64) -> dict:
    """Build one Qwen2.5-VL chat-template video content block.

    Args
    ----
    nframes
        Honoured only when ``mode="nframes"``. Otherwise informational.
    mode
        One of {``"nframes"``, ``"fps"``, ``"max_frames"``, ``"auto_default"``}.
        ``"max_frames"`` is the legacy fallback used by the version-drift
        retry path; it derives ``fps`` from ``nframes`` to approximate the
        same effective frame count.
    fps, max_frames
        Used in mode ``"fps"`` to fix the official Qwen2.5-VL preset
        (fps=2.0 + max_frames=64 by default).
    """
    if mode == "nframes":
        return {"type": "video", "video": video_path, "nframes": nframes}
    if mode == "fps":
        return {
            "type": "video",
            "video": video_path,
            "fps": float(fps),
            "max_frames": int(max_frames),
        }
    if mode == "auto_default":
        return {"type": "video", "video": video_path}
    # Legacy fallback used by version-drift retry: derive fps from nframes
    # so the effective frame count is similar to what nframes would emit.
    return {
        "type": "video",
        "video": video_path,
        "fps": max(1.0, float(nframes) / 8.0),
        "max_frames": int(nframes),
    }


# ============================================================
# MODEL BACKEND
# ============================================================
class QwenVLJudge:
    """Wrap Qwen2.5-VL for video pairwise judgement (with optional dimension)."""

    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.cfg   = cfg
        self.torch = torch

        if not Path(cfg["model_path"]).is_dir() and not cfg["model_path"].startswith(
            ("Qwen/", "https://")
        ):
            print(f"[judge] WARNING: model_path={cfg['model_path']!r} does not exist "
                  f"as a local dir; transformers will try to download from HF.",
                  file=sys.stderr)

        dtype = getattr(torch, cfg["dtype"])
        print(f"[judge] loading {cfg['model_path']!r}  dtype={cfg['dtype']}",
              file=sys.stderr)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg["model_path"], torch_dtype=dtype, device_map=cfg["device"]
        ).eval()
        self.processor = AutoProcessor.from_pretrained(cfg["model_path"])

        # ──────────────────────────────────────────────────────────────
        # Cache the token ids for "1".."5" used by the soft-logits
        # scoring path (D5 score_method="soft_logits").
        #
        # Qwen2.5 BPE tokenizer does NOT always encode a bare digit as a
        # single token when called with add_special_tokens=False (it may
        # prepend a BOS or merge with a preceding space). We therefore
        # try three progressively more robust strategies and accept the
        # first one that yields 5 *distinct* ids:
        #
        #   Strategy A: convert_tokens_to_ids("1".."5")
        #               — direct vocab lookup, no encoding side-effects.
        #   Strategy B: encode("score: k")[-1] for k in 1..5
        #               — take the last token in a numeric context.
        #   Strategy C: encode(str(k))[-1] for k in 1..5
        #               — last token of bare digit (ignores any prefix).
        #
        # We only require that the 5 ids are *mutually distinct* (so the
        # logit read-out is unambiguous). We do NOT require single-token
        # encoding of the bare string.
        # ──────────────────────────────────────────────────────────────
        try:
            tok = self.processor.tokenizer
        except AttributeError:
            tok = None
        self._score_token_ids: list[int] | None = None
        self._score_token_ids_with_space: list[int] | None = None
        self._soft_logits_ok: bool = False
        if tok is not None:
            def _try_get_ids(strategy: str) -> list[int] | None:
                ids: list[int] = []
                try:
                    for k in range(1, 6):
                        if strategy == "A":
                            tid = tok.convert_tokens_to_ids(str(k))
                            # convert_tokens_to_ids returns unk_token_id
                            # when the token is not in vocab.
                            if tid == tok.unk_token_id:
                                return None
                        elif strategy == "B":
                            enc = tok.encode(
                                "score: " + str(k),
                                add_special_tokens=False,
                            )
                            tid = enc[-1]
                        else:  # strategy C
                            enc = tok.encode(
                                str(k), add_special_tokens=False,
                            )
                            tid = enc[-1]
                        ids.append(tid)
                except Exception:
                    return None
                # Accept only if all 5 ids are distinct.
                return ids if len(set(ids)) == 5 else None

            ids_bare: list[int] | None = None
            for strat in ("A", "B", "C"):
                ids_bare = _try_get_ids(strat)
                if ids_bare is not None:
                    print(f"[judge] soft_logits token-id strategy={strat}: "
                          f"digit_ids={ids_bare}", file=sys.stderr)
                    break

            if ids_bare is not None:
                # Also build the "space + digit" variant for hit-detection
                # in the generated sequence (Qwen sometimes emits " 1"
                # rather than "1" after a colon).
                ids_space: list[int] = []
                for k in range(1, 6):
                    enc = tok.encode(" " + str(k), add_special_tokens=False)
                    ids_space.append(enc[-1])
                self._score_token_ids            = ids_bare
                self._score_token_ids_with_space = ids_space

                # ──────────────────────────────────────────────────────
                # Collect ALL last-token variants that represent each
                # integer k ∈ {1..5} when emitted right after a colon
                # in the scaffold. We probe several plausible prefixes:
                #
                #   "k", " k", ": k", ":k", "\nk", " : k"
                #
                # and take the LAST token id in each encoding. The union
                # is what we treat as "this position decodes integer k".
                # We also build a reverse map id -> integer value, used
                # both for hit-detection and for soft-logits aggregation
                # (so that probability mass on " 3" and on "3" both
                # count toward the value 3, not get split / dropped).
                # ──────────────────────────────────────────────────────
                digit_id_groups: list[list[int]] = [[] for _ in range(5)]
                _probe_prefixes = ["", " ", ": ", ":", "\n", " : "]
                for k_idx in range(5):
                    for pref in _probe_prefixes:
                        try:
                            enc = tok.encode(
                                pref + str(k_idx + 1),
                                add_special_tokens=False,
                            )
                        except Exception:
                            continue
                        if enc:
                            digit_id_groups[k_idx].append(enc[-1])
                # Deduplicate while preserving order.
                for k_idx in range(5):
                    seen: set[int] = set()
                    uniq: list[int] = []
                    for tid in digit_id_groups[k_idx]:
                        if tid not in seen:
                            seen.add(tid)
                            uniq.append(tid)
                    digit_id_groups[k_idx] = uniq

                # Build id->value map. If the SAME id appears in two
                # different digit groups (extremely unlikely but possible
                # for weird BPE merges), the smaller k wins — but we
                # also drop that id from the larger group's softmax
                # contribution to avoid double counting.
                id_to_value: dict[int, int] = {}
                for k_idx, group in enumerate(digit_id_groups):
                    for tid in group:
                        if tid not in id_to_value:
                            id_to_value[tid] = k_idx + 1
                # Final per-digit id list, restricted to entries whose
                # id_to_value resolves to that exact digit (drops dupes).
                clean_groups: list[list[int]] = [[] for _ in range(5)]
                for k_idx, group in enumerate(digit_id_groups):
                    for tid in group:
                        if id_to_value.get(tid) == k_idx + 1:
                            clean_groups[k_idx].append(tid)

                # Also collect colon-ish token ids: any token whose
                # decoded form ends with ':' (e.g. ":", " :", "_:",
                # "theme_1:", etc.). We use these as anchors during
                # hit-detection to require that a digit token is
                # immediately preceded by a colon-terminated token.
                colon_ids: set[int] = set()
                # Cap the scan to vocab to keep startup cheap; for
                # tokenizers >150k this is still <1s.
                try:
                    vocab_size = tok.vocab_size
                except Exception:
                    vocab_size = 0
                if vocab_size and vocab_size < 200_000:
                    for tid in range(vocab_size):
                        try:
                            s = tok.decode([tid])
                        except Exception:
                            continue
                        if s and s.rstrip(" ").endswith(":"):
                            colon_ids.add(tid)

                self._score_digit_id_groups: list[list[int]] = clean_groups
                self._score_id_to_value: dict[int, int]      = id_to_value
                self._score_colon_ids: set[int]              = colon_ids
                self._soft_logits_ok                         = True

                # One-line debug summary.
                grp_sizes = [len(g) for g in clean_groups]
                print(f"[judge] soft_logits ready  "
                      f"space_digit_ids={ids_space}  "
                      f"digit_group_sizes={grp_sizes}  "
                      f"colon_ids={len(colon_ids)}", file=sys.stderr)
            else:
                print("[judge] WARNING: could not resolve distinct token ids "
                      "for '1'..'5'; soft_logits will fall back to "
                      "integer_text at runtime.", file=sys.stderr)

    # ------------------------------------------------------------------
    # Prompt construction (criterion swaps based on whether a single
    # dimension was asked for).
    # ------------------------------------------------------------------
    @staticmethod
    def _build_messages(instruction: str, video_a: str, video_b: str,
                        nframes: int, dimension: str | None,
                        *, frame_mode: str = "nframes",
                        rubric_text: str | None = None,
                        fps:        float = 2.0,
                        max_frames: int   = 64) -> list:
        if dimension is None:
            criterion = (
                "which video better satisfies the generation prompt overall, "
                "considering instruction fidelity AND visual quality"
            )
        else:
            criterion = (f"which video is better SPECIFICALLY on the dimension "
                         f"of '{dimension}'")
        if rubric_text:
            rubric_block = (
                "\nReference rubric (judge by these criteria, weighted as given):\n"
                f"{rubric_text}\n"
            )
        else:
            rubric_block = ""
        question = (
            f"Generation prompt: {instruction}\n"
            f"You are shown two generated videos: Video A (first) and Video B (second).\n"
            f"{rubric_block}"
            f"Judge {criterion}.\n"
            f"Answer with EXACTLY one token: 'A' or 'B'. Do not explain."
        )
        return [{
            "role": "user",
            "content": [
                _video_block(video_a, nframes, mode=frame_mode,
                             fps=fps, max_frames=max_frames),
                _video_block(video_b, nframes, mode=frame_mode,
                             fps=fps, max_frames=max_frames),
                {"type": "text", "text": question},
            ],
        }]

    # ------------------------------------------------------------------
    # Single forward; returns 'A' / 'B' / None.
    # The frame_mode fallback is opt-in via env RM_FRAME_MODE.
    # When rubric_mode != "none", a per-dim or fixed-overall rubric is
    # injected into the user prompt; on missing rubric file we silently
    # fall back to no-rubric judgement so the experiment still produces
    # a number for every pair.
    # ------------------------------------------------------------------
    def judge_pair(self, instruction: str, video_a: str, video_b: str,
                   nframes: int, dimension: str | None = None) -> str | None:
        from qwen_vl_utils import process_vision_info

        # ── Decide whether to inject a rubric into the prompt.
        rubric_text: str | None = None
        rmode = self.cfg.get("rubric_mode", "none")
        rdir  = self.cfg.get("rubric_dir")
        if rmode != "none" and rdir:
            rubric_dir = Path(rdir)
            if rmode == "gold_overall":
                rubric_text = _load_rubric(rubric_dir,
                                           self.cfg.get("overall_rubric_dim",
                                                        "visual_quality"))
            elif rmode == "gold_dim" and dimension:
                rubric_text = _load_rubric(rubric_dir, dimension)

        # ── Choose primary frame_mode from cfg (CLI-driven), with the
        # legacy ``max_frames`` retry as a defensive fallback only when
        # the primary is the original ``nframes`` path.
        primary = self.cfg.get("frame_strategy", "nframes")
        modes_to_try = [os.environ.get("RM_FRAME_MODE", primary)]
        if modes_to_try[0] == "nframes":
            modes_to_try.append("max_frames")
        fps_val   = float(self.cfg.get("fps", 2.0))
        maxf_val  = int(self.cfg.get("max_frames", 64))

        last_err: Exception | None = None
        for mode in modes_to_try:
            messages = self._build_messages(
                instruction, video_a, video_b, nframes, dimension,
                frame_mode=mode,
                rubric_text=rubric_text,
                fps=fps_val,
                max_frames=maxf_val,
            )
            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.cfg["device"])
                with self.torch.no_grad():
                    gen = self.model.generate(
                        **inputs, max_new_tokens=4, do_sample=False,
                    )
                trimmed = gen[:, inputs.input_ids.shape[1]:]
                out = self.processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
                )[0].strip().upper()
                if out.startswith("A"):
                    return "A"
                if out.startswith("B"):
                    return "B"
                return None
            except (TypeError, ValueError) as e:
                # Likely the qwen-vl-utils field-name mismatch. Try next mode.
                last_err = e
                continue
        if last_err is not None:
            print(f"[judge] all frame_mode attempts failed: {last_err!r}", file=sys.stderr)
        return None


    # ------------------------------------------------------------------
    # D4 helper — pointwise per-theme score (integer 1-5) for a SINGLE video.
    #
    # Architectural note: this is the inference-time twin of Stage ②/③
    # cross-scoring. The model sees exactly one video and a single theme
    # (theme name + 3-4 tips) and is asked for an integer 1-5 score
    # (Likert-style, mirroring VideoScore-2 / Q-Bench / VBench). We use
    # greedy decoding with a tight max_new_tokens budget and parse the
    # first integer out of the output; if parsing fails we return None
    # and the caller increments score_fail_count.
    # ------------------------------------------------------------------
    def score_video_on_theme(
        self,
        instruction: str,
        video_path: str,
        theme:      str,
        tips:       list[str],
        dimension:  str | None = None,
        *,
        max_new_tokens: int = 8,
    ) -> float | None:
        from qwen_vl_utils import process_vision_info

        primary    = self.cfg.get("frame_strategy", "nframes")
        modes_to_try = [os.environ.get("RM_FRAME_MODE", primary)]
        if modes_to_try[0] == "nframes":
            modes_to_try.append("max_frames")
        fps_val   = float(self.cfg.get("fps", 2.0))
        maxf_val  = int(self.cfg.get("max_frames", 64))
        nframes   = int(self.cfg.get("default_nframes", 16))

        tips_block = "\n".join(f"- {t}" for t in tips) if tips else "(no tips)"
        dim_clause = (
            f"This evaluation focuses on the dimension '{dimension}'.\n"
            if dimension else ""
        )
        question = (
            f"Generation prompt: {instruction}\n"
            f"You are shown ONE generated video. Rate it on the following\n"
            f"single rubric theme:\n"
            f"Theme: {theme}\n"
            f"Criteria:\n{tips_block}\n"
            f"{dim_clause}"
            f"Output a SINGLE integer score from 1 to 5, where:\n"
            f"  1 = very poor, 2 = poor, 3 = fair, 4 = good, 5 = excellent.\n"
            f"Respond with ONLY the digit (one character), no words, no units."
        )

        last_err: Exception | None = None
        for mode in modes_to_try:
            messages = [{
                "role": "user",
                "content": [
                    _video_block(video_path, nframes, mode=mode,
                                 fps=fps_val, max_frames=maxf_val),
                    {"type": "text", "text": question},
                ],
            }]
            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.cfg["device"])
                with self.torch.no_grad():
                    gen = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                trimmed = gen[:, inputs.input_ids.shape[1]:]
                out = self.processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                # Parse first integer in [1, 5]. Accept floats like
                # "3" / "3.0" / "3.5" defensively (some 7B emissions),
                # then round to the nearest integer in {1..5}.
                import re
                m = re.search(r"[-+]?\d*\.?\d+", out)
                if not m:
                    return None
                try:
                    val = float(m.group(0))
                except ValueError:
                    return None
                # Reject clearly broken outputs like "42" or "-3".
                if val < 0.0 or val > 6.0:
                    return None
                # Round to nearest integer, then clamp to [1, 5].
                ival = int(round(val))
                if ival < 1:
                    ival = 1
                if ival > 5:
                    ival = 5
                return float(ival)
            except (TypeError, ValueError) as e:
                last_err = e
                continue
        if last_err is not None:
            print(f"[score] all frame_mode attempts failed: {last_err!r}",
                  file=sys.stderr)
        return None

    # ------------------------------------------------------------------
    # D5 helper — pointwise multi-theme JSON score for a SINGLE video.
    #
    # One forward pass consumes the FULL rubric (all themes, all tips)
    # plus the video, and the model emits a JSON object with one score
    # per theme. This collapses D4's per-theme N-forwards into a single
    # forward (5x speedup) while preserving the per-theme breakdown that
    # makes downstream paper analysis tractable.
    #
    # Robustness: 3-tier parsing fallback (strict JSON → regex key/value
    # → ordered int scrape) so 7B's tendency to emit "Score: 3/5" or
    # "3.5" style noise does not destroy the run (parser rounds to int).
    #
    # Returns:
    #   list[float | None] of length len(themes); None at position k
    #   means that theme's score could not be recovered.
    # ------------------------------------------------------------------
    def score_video_multi_theme(
        self,
        instruction: str,
        video_path:  str,
        themes:      list[dict],
        dimension:   str | None = None,
        *,
        max_new_tokens: int = 96,
    ) -> tuple[list[float | None], str]:
        """Score one video on all rubric themes in a single forward.

        Returns ``(scores, raw_output)`` where ``scores[k]`` is an integer
        in {1,2,3,4,5} (stored as float for downstream arithmetic) or
        ``None``, and ``raw_output`` is the model's raw text (kept for
        debugging — caller may discard).
        """
        from qwen_vl_utils import process_vision_info

        if not themes:
            return ([], "")

        primary    = self.cfg.get("frame_strategy", "nframes")
        modes_to_try = [os.environ.get("RM_FRAME_MODE", primary)]
        if modes_to_try[0] == "nframes":
            modes_to_try.append("max_frames")
        fps_val   = float(self.cfg.get("fps", 2.0))
        maxf_val  = int(self.cfg.get("max_frames", 64))
        nframes   = int(self.cfg.get("default_nframes", 16))

        # --- Build the rubric block. Compact + numbered so the model can
        # ground each JSON key back to a theme.
        rubric_lines: list[str] = []
        for k, th in enumerate(themes, start=1):
            tips = th.get("tips") or []
            tips_compact = " ".join(
                f"({i+1}) {t}" for i, t in enumerate(tips)
            ) or "(no tips)"
            rubric_lines.append(
                f'theme_{k} ("{th.get("theme", "").strip()}"): {tips_compact}'
            )
        rubric_block = "\n".join(rubric_lines)

        keys_spec = ", ".join(f'"theme_{k}": <integer 1-5>'
                              for k in range(1, len(themes) + 1))
        dim_clause = (
            f"This evaluation focuses on the dimension '{dimension}'.\n"
            if dimension else ""
        )
        question = (
            f"Generation prompt: {instruction}\n"
            f"You are shown ONE generated video. Rate it on EACH of the\n"
            f"following {len(themes)} rubric themes independently:\n"
            f"{rubric_block}\n"
            f"{dim_clause}"
            f"For each theme, output an INTEGER score from 1 to 5, where:\n"
            f"  1 = very poor, 2 = poor, 3 = fair, 4 = good, 5 = excellent.\n"
            f"Use the criteria above as the scoring guide.\n"
            f"Respond with ONLY a single JSON object on one line, EXACTLY:\n"
            f"{{{keys_spec}}}\n"
            f"All values MUST be integers in {{1,2,3,4,5}}. Do not add\n"
            f"explanations, units, decimals, or extra keys."
        )

        last_err: Exception | None = None
        for mode in modes_to_try:
            messages = [{
                "role": "user",
                "content": [
                    _video_block(video_path, nframes, mode=mode,
                                 fps=fps_val, max_frames=maxf_val),
                    {"type": "text", "text": question},
                ],
            }]
            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.cfg["device"])
                with self.torch.no_grad():
                    gen = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                trimmed = gen[:, inputs.input_ids.shape[1]:]
                out = self.processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                scores = _parse_multi_theme_scores(out, len(themes))
                return (scores, out)
            except (TypeError, ValueError) as e:
                last_err = e
                continue
        if last_err is not None:
            print(f"[score-multi] all frame_mode attempts failed: {last_err!r}",
                  file=sys.stderr)
        return ([None] * len(themes), "")

    # ------------------------------------------------------------------
    # D5 soft-logits variant — multi-theme, ONE forward, continuous
    # floats in [1.0, 5.0] from per-position softmax expectation.
    #
    # Method
    # ------
    # We constrain the output to a deterministic plain-text scaffold:
    #
    #     theme_1: <X> theme_2: <X> theme_3: <X> ...
    #
    # During greedy generation, the digit immediately following each
    # ``theme_k:`` colon is one of {1,2,3,4,5} (or `` 1``..`` 5`` with a
    # leading space, depending on how the tokenizer merges whitespace).
    # We turn on ``output_scores=True`` so we can read the next-token
    # logits at the step that produced that digit, restrict to the 5
    # candidate ids, take softmax, and report E[k]=Σ k·p_k.
    #
    # This converts the integer-粒度 1-5 signal into a *truly continuous*
    # float, eliminating the 0.5 random-tie problem AND parse failures
    # (no text parsing happens). Slightly more expensive than greedy
    # multi-theme integer because we need the scores tensor, but still
    # ONE forward per video.
    #
    # Fallback path: if soft-logits readout fails for theme k (e.g. the
    # generation derailed and ``theme_{k}:`` never appears, or the next
    # token is not in the score-id set), we leave that slot as None and
    # the caller's downstream parser stays bounded.
    # ------------------------------------------------------------------
    def score_video_multi_theme_softlogits(
        self,
        instruction: str,
        video_path:  str,
        themes:      list[dict],
        dimension:   str | None = None,
        *,
        max_new_tokens: int = 96,
    ) -> tuple[list[float | None], str]:
        """Soft-logits multi-theme score; same return shape as the
        integer_text twin but each entry is a continuous float in
        [1.0, 5.0]. ``raw_output`` is the decoded greedy string for
        debugging; the actual scores come from logits, not from text.
        """
        from qwen_vl_utils import process_vision_info
        import torch

        if not themes:
            return ([], "")

        # If the tokenizer cannot emit each of "1".."5" as a single
        # token, soft_logits is unsafe — fall back to integer_text path.
        if not getattr(self, "_soft_logits_ok", False):
            print("[score-multi-soft] tokenizer not soft_logits-ready; "
                  "falling back to integer_text path.", file=sys.stderr)
            return self.score_video_multi_theme(
                instruction, video_path, themes, dimension,
                max_new_tokens=max_new_tokens,
            )

        primary    = self.cfg.get("frame_strategy", "nframes")
        modes_to_try = [os.environ.get("RM_FRAME_MODE", primary)]
        if modes_to_try[0] == "nframes":
            modes_to_try.append("max_frames")
        fps_val   = float(self.cfg.get("fps", 2.0))
        maxf_val  = int(self.cfg.get("max_frames", 64))
        nframes   = int(self.cfg.get("default_nframes", 16))

        # --- Compact rubric block, identical to integer_text path so
        # the model sees the same context and only the *output format*
        # differs. This keeps comparisons apples-to-apples in ablation.
        rubric_lines: list[str] = []
        for k, th in enumerate(themes, start=1):
            tips = th.get("tips") or []
            tips_compact = " ".join(
                f"({i+1}) {t}" for i, t in enumerate(tips)
            ) or "(no tips)"
            rubric_lines.append(
                f'theme_{k} ("{th.get("theme", "").strip()}"): {tips_compact}'
            )
        rubric_block = "\n".join(rubric_lines)

        # Plain-text scaffold (NOT JSON). Keeping the colon + single
        # digit grammar makes per-theme score-token localisation
        # deterministic.
        format_line = " ".join(
            f"theme_{k}: <X>" for k in range(1, len(themes) + 1)
        )
        dim_clause = (
            f"This evaluation focuses on the dimension '{dimension}'.\n"
            if dimension else ""
        )
        question = (
            f"Generation prompt: {instruction}\n"
            f"You are shown ONE generated video. Rate it on EACH of the\n"
            f"following {len(themes)} rubric themes independently:\n"
            f"{rubric_block}\n"
            f"{dim_clause}"
            f"For each theme, output an INTEGER score from 1 to 5, where:\n"
            f"  1 = very poor, 2 = poor, 3 = fair, 4 = good, 5 = excellent.\n"
            f"Respond with EXACTLY this one-line format, replacing each\n"
            f"<X> with a single digit 1-5. No commas, no extra text:\n"
            f"{format_line}"
        )

        score_ids_bare  = self._score_token_ids                # [id('1')..id('5')]
        score_ids_space = self._score_token_ids_with_space     # [id(' 1')..id(' 5')]
        # Full per-digit id groups (bare + leading-space + colon-fused
        # variants). Hit-detection scans these; aggregation re-uses them
        # so probability mass on any variant of digit k contributes to
        # the value k (no leakage to other digits, no split between
        # "3" and " 3").
        digit_groups: list[list[int]] = getattr(
            self, "_score_digit_id_groups",
            [[score_ids_bare[k], score_ids_space[k]] for k in range(5)],
        )
        id_to_value: dict[int, int] = getattr(
            self, "_score_id_to_value",
            {tid: (k + 1) for k, tid in enumerate(score_ids_bare)}
            | {tid: (k + 1) for k, tid in enumerate(score_ids_space)},
        )
        score_ids_set: set[int] = set(id_to_value.keys())
        colon_ids:     set[int] = getattr(self, "_score_colon_ids", set())
        # Flat list of all digit ids and a parallel value list, used to
        # build the softmax in one shot.
        flat_digit_ids:    list[int] = []
        flat_digit_values: list[int] = []
        for k_idx, group in enumerate(digit_groups):
            for tid in group:
                flat_digit_ids.append(tid)
                flat_digit_values.append(k_idx + 1)
        # Tensor-friendly views (built lazily once we have a logits dev).
        flat_digit_ids_t:    "torch.Tensor | None" = None
        flat_digit_values_t: "torch.Tensor | None" = None

        # One-time debug dump: only fire on the first call per process,
        # so we can eyeball the token decoding without flooding logs.
        _debug_dump = not getattr(self, "_softlogits_debug_done", False)

        last_err: Exception | None = None
        for mode in modes_to_try:
            messages = [{
                "role": "user",
                "content": [
                    _video_block(video_path, nframes, mode=mode,
                                 fps=fps_val, max_frames=maxf_val),
                    {"type": "text", "text": question},
                ],
            }]
            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.cfg["device"])
                with self.torch.no_grad():
                    gen_out = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                # ``sequences`` includes the prompt; trim it.
                trimmed = gen_out.sequences[:, inputs.input_ids.shape[1]:]
                gen_ids = trimmed[0].tolist()                   # list[int], length = T
                # ``scores`` is a tuple of length T; scores[t] is the
                # next-token logits AT step t (i.e. the distribution
                # used to pick gen_ids[t]).
                step_scores = gen_out.scores                    # tuple of [B, V]
                # Decode the greedy string for debug + raw_output.
                out_text = self.processor.batch_decode(
                    trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()

                # ── Locate the score-token positions inside gen_ids.
                #
                # CRITICAL: previous version used "any token whose id is
                # in the digit set" as a hit. That misfired badly because
                # the scaffold itself contains the substrings 'theme_1'..
                # 'theme_5', and BPE often splits them into '..._' + '1',
                # so the digit IDs of theme indices were misread as the
                # score digits. We now require an explicit colon anchor:
                # a token is accepted as a score position only if it is
                # a digit id AND the immediately preceding (or one-before
                # if the previous is whitespace) token decodes to a
                # string that ends with ':'. This matches the scaffold
                # "theme_k: <X>" deterministically.
                T = len(gen_ids)
                hit_positions: list[int] = []
                # Build a small helper: is position `i`'s token id a
                # colon-terminated token?
                def _ends_with_colon(tid: int) -> bool:
                    if tid in colon_ids:
                        return True
                    # Fallback decode (cheap, only runs on a handful of
                    # candidate positions in practice).
                    try:
                        s = self.processor.tokenizer.decode([tid])
                    except Exception:
                        return False
                    return bool(s) and s.rstrip(" ").endswith(":")

                for pos in range(T):
                    tid = gen_ids[pos]
                    if tid not in score_ids_set:
                        continue
                    # Look back for the colon anchor. Skip at most one
                    # whitespace-only token between colon and digit.
                    anchor_ok = False
                    for back in (1, 2):
                        j = pos - back
                        if j < 0:
                            break
                        prev_tid = gen_ids[j]
                        if _ends_with_colon(prev_tid):
                            anchor_ok = True
                            break
                        # Allow a single whitespace token in between.
                        try:
                            prev_str = self.processor.tokenizer.decode(
                                [prev_tid]
                            )
                        except Exception:
                            prev_str = ""
                        if back == 1 and prev_str.strip() == "":
                            continue
                        break
                    if not anchor_ok:
                        continue
                    hit_positions.append(pos)
                    if len(hit_positions) >= len(themes):
                        break

                # Build the flat-ids tensors lazily on the right device.
                if flat_digit_ids_t is None:
                    dev = step_scores[0].device if step_scores else self.cfg["device"]
                    flat_digit_ids_t    = torch.tensor(
                        flat_digit_ids, device=dev, dtype=torch.long,
                    )
                    flat_digit_values_t = torch.tensor(
                        flat_digit_values, device=dev, dtype=torch.float32,
                    )

                scores_out: list[float | None] = []
                per_pos_debug: list[dict] = []
                for k in range(len(themes)):
                    if k >= len(hit_positions):
                        scores_out.append(None)
                        continue
                    pos = hit_positions[k]
                    if pos >= len(step_scores):
                        scores_out.append(None)
                        continue
                    logits = step_scores[pos][0]                  # [V]
                    sel    = logits.index_select(0, flat_digit_ids_t).float()
                    # Softmax over ALL digit-id variants (bare + space
                    # + colon-fused), so the probability mass on " 3",
                    # "3", ":3", … all routes to integer value 3.
                    probs    = torch.softmax(sel, dim=-1)
                    expected = float((probs * flat_digit_values_t).sum().item())
                    if expected < 1.0:
                        expected = 1.0
                    if expected > 5.0:
                        expected = 5.0
                    scores_out.append(expected)

                    if _debug_dump:
                        # Per-digit aggregated probability mass (sum over
                        # all id variants of that digit).
                        agg = [0.0] * 5
                        for j, v in enumerate(flat_digit_values):
                            agg[v - 1] += float(probs[j].item())
                        per_pos_debug.append({
                            "theme_idx": k + 1,
                            "pos": pos,
                            "expected": round(expected, 4),
                            "p_per_digit": [round(x, 4) for x in agg],
                        })

                if _debug_dump:
                    try:
                        tok = self.processor.tokenizer
                        head_n = min(40, T)
                        head_dump = [
                            {"i": i,
                             "id": gen_ids[i],
                             "tok": tok.decode([gen_ids[i]])}
                            for i in range(head_n)
                        ]
                    except Exception:
                        head_dump = []
                    print("[score-multi-soft][DEBUG first-call]",
                          file=sys.stderr)
                    print(f"  raw_text   = {out_text!r}", file=sys.stderr)
                    print(f"  T(gen)     = {T}  hits={hit_positions}",
                          file=sys.stderr)
                    print(f"  head_tokens(<=40):", file=sys.stderr)
                    for row in head_dump:
                        print(f"    {row}", file=sys.stderr)
                    print(f"  per_theme  = {per_pos_debug}",
                          file=sys.stderr)
                    self._softlogits_debug_done = True

                return (scores_out, out_text)
            except (TypeError, ValueError) as e:
                last_err = e
                continue
        if last_err is not None:
            print(f"[score-multi-soft] all frame_mode attempts failed: "
                  f"{last_err!r}", file=sys.stderr)
        return ([None] * len(themes), "")


# ============================================================
# Multi-theme JSON score parser — 3-tier fallback.
#
#   tier 1: strict ``json.loads`` over the first {...} substring
#   tier 2: regex over ``"theme_N": <number>`` key/value pairs
#   tier 3: ordered ``re.findall(r"[-+]?\d*\.?\d+", text)`` and take
#           the first ``n_themes`` numbers in declaration order
#
# Each parsed score is sanity-clamped to integer {1,2,3,4,5}; any score
# outside the [0, 6] hard band becomes ``None`` (almost certainly garbage).
# Floats from 7B emissions (e.g. "3.5") are rounded to nearest integer.
# ============================================================
def _parse_multi_theme_scores(text: str, n_themes: int) -> list[float | None]:
    import re

    if not text or n_themes <= 0:
        return [None] * n_themes

    def _clamp(x: float) -> float | None:
        if x < 0.0 or x > 6.0:
            return None
        ival = int(round(x))
        if ival < 1:
            ival = 1
        if ival > 5:
            ival = 5
        return float(ival)

    # ── Tier 1: strict JSON. Slice out the FIRST {...} block to ignore
    # leading "Sure, here is the JSON:" prefixes that 7B sometimes adds.
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        blob = text[start:end + 1]
        try:
            obj = json.loads(blob)
            if isinstance(obj, dict):
                out: list[float | None] = []
                ok = True
                for k in range(1, n_themes + 1):
                    v = obj.get(f"theme_{k}")
                    if v is None:
                        ok = False
                        break
                    try:
                        out.append(_clamp(float(v)))
                    except (TypeError, ValueError):
                        ok = False
                        break
                if ok and len(out) == n_themes:
                    return out
        except (json.JSONDecodeError, ValueError):
            pass

    # ── Tier 2: regex over key/value pairs (handles trailing commas,
    # missing braces, single quotes, etc.).
    kv = dict(
        re.findall(
            r'["\']?theme[_\s]*(\d+)["\']?\s*[:=]\s*([-+]?\d*\.?\d+)',
            text,
            flags=re.IGNORECASE,
        )
    )
    if kv:
        out2: list[float | None] = []
        for k in range(1, n_themes + 1):
            v = kv.get(str(k))
            if v is None:
                out2.append(None)
            else:
                try:
                    out2.append(_clamp(float(v)))
                except ValueError:
                    out2.append(None)
        if any(s is not None for s in out2):
            return out2

    # ── Tier 3: ordered float scrape — last-resort positional binding.
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if len(nums) >= n_themes:
        out3: list[float | None] = []
        for v in nums[:n_themes]:
            try:
                out3.append(_clamp(float(v)))
            except ValueError:
                out3.append(None)
        return out3

    return [None] * n_themes


# ============================================================
# Per-pair eval (pos,neg) → bool|None with optional debias swap.
# ============================================================
def eval_one_pair(judge: QwenVLJudge,
                  instruction: str,
                  v_pos: str, v_neg: str,
                  nframes: int,
                  dimension: str | None = None,
                  debias_swap: bool = True) -> bool | None:
    """Returns True/False/None (None ⇒ uncertain due to position flip).

    Order 1 — pos=A, neg=B → expect 'A'.
    Order 2 — neg=A, pos=B → expect 'B'.
    Both must agree to count as 'decided'.
    """
    c1 = judge.judge_pair(instruction, v_pos, v_neg, nframes, dimension)
    if not debias_swap:
        if c1 is None:
            return None
        return c1 == "A"
    c2 = judge.judge_pair(instruction, v_neg, v_pos, nframes, dimension)
    if c1 is None or c2 is None:
        return None
    correct1 = (c1 == "A")
    correct2 = (c2 == "B")
    if correct1 == correct2:
        return correct1
    return None


def load_jsonl(path: str, max_samples: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]
    return rows


def summarize(results: list) -> dict:
    n = len(results)
    decided = [r for r in results if r is not None]
    n_dec = len(decided)
    acc = (sum(decided) / n_dec) if n_dec else 0.0
    return {
        "n_total":             n,
        "n_decided":           n_dec,
        "uncertain_rate":      round(1 - n_dec / n, 4) if n else 0.0,
        "accuracy_on_decided": round(acc, 4),
        "accuracy_conservative": round(sum(decided) / n, 4) if n else 0.0,
    }


# ============================================================
# D1: Pairwise judgement baseline.
# ============================================================
def diagnosis_1(judge: QwenVLJudge, cfg: dict) -> dict:
    print("\n========== D1: Pairwise judgement baseline ==========", file=sys.stderr)
    data = load_jsonl(cfg["pair_data"], cfg["max_samples"])
    nframes = cfg["default_nframes"]
    results = []
    for i, ex in enumerate(data):
        r = eval_one_pair(
            judge, ex["instruction"], ex["video_pos"], ex["video_neg"],
            nframes, dimension=None, debias_swap=cfg["debias_swap"],
        )
        results.append(r)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(data)}] running acc="
                  f"{summarize(results)['accuracy_on_decided']}", file=sys.stderr)
    stats = summarize(results)
    stats["nframes"] = nframes
    acc = stats["accuracy_on_decided"]
    stats["gating"] = (
        "PASS (>0.70)"            if acc > 0.70
        else "GRAY (0.65-0.70, rubric can compensate)" if acc >= 0.65
        else "FAIL (<0.65, do not enter RL)"
    )
    print(f"  => D1: {json.dumps(stats, ensure_ascii=False)}", file=sys.stderr)
    return stats


# ============================================================
# D2: Frame-sampling stress test.
# ============================================================
def diagnosis_2(judge: QwenVLJudge, cfg: dict) -> dict:
    print("\n========== D2: Frame-sampling stress test ==========", file=sys.stderr)
    data = load_jsonl(cfg["pair_data"], cfg["max_samples"])
    out: dict = {}
    for nf in cfg["frame_sweep"]:
        results = []
        for ex in data:
            r = eval_one_pair(
                judge, ex["instruction"], ex["video_pos"], ex["video_neg"],
                nf, dimension=None, debias_swap=cfg["debias_swap"],
            )
            results.append(r)
        stats = summarize(results)
        out[f"nframes_{nf}"] = stats
        print(f"  nframes={nf:>3}: acc={stats['accuracy_on_decided']} "
              f"uncertain={stats['uncertain_rate']}", file=sys.stderr)
    accs = {nf: out[f"nframes_{nf}"]["accuracy_on_decided"] for nf in cfg["frame_sweep"]}
    out["recommendation"] = (
        f"pick the smallest nframes whose acc is within 0.01 of the max; "
        f"measured accs={accs}"
    )
    print(f"  => D2: {out['recommendation']}", file=sys.stderr)
    return out


# ============================================================
# D4: Pointwise per-theme rubric scoring → programmatic aggregation.
#
# For every (v_pos, v_neg, dimension) triple:
#   1) Load the rubric for ``dimension`` and parse its themes.
#   2) For each theme T_k, score v_pos and v_neg independently
#      (model sees ONE video at a time) → s_pos_k, s_neg_k integers in {1..5}.
#   3) Aggregate with uniform weights:
#         S(v) = mean_k s_v_k    (parse failures dropped per video)
#   4) Decide: True if S(v_pos) > S(v_neg) + tau,
#              False if S(v_neg) > S(v_pos) + tau,
#              None if |S_pos - S_neg| <= tau (tie / undecided).
#
# Reported metrics per dim:
#   accuracy_on_decided / accuracy_conservative / uncertain_rate
#   parse_fail_rate     — % of theme-scoring forwards that returned None
#   score_variance_pos  — Var(s_pos_k) averaged across pairs (degeneracy probe)
#   mean_score_gap      — mean(S_pos - S_neg) across decided pairs
#
# Why this matters for the paper:
#   * It is the zero-shot inference-time twin of Stage ②/③ pointwise
#     reward, so the gain from training is directly measurable.
#   * It bypasses positional bias entirely (no swap needed).
#   * It exposes whether 7B can use the rubric structurally (good acc)
#     vs. just rote-scoring everything 2.5 (low score_variance).
# ============================================================
def _aggregate_pointwise(per_theme_scores: list[float | None]) -> float | None:
    """Mean of non-None per-theme scores; None if all themes failed to parse."""
    valid = [s for s in per_theme_scores if s is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def eval_one_pair_pointwise(
    judge:        QwenVLJudge,
    instruction:  str,
    v_pos:        str,
    v_neg:        str,
    themes:       list[dict],
    dimension:    str | None,
    tau:          float,
    *,
    max_new_tokens: int = 8,
) -> dict:
    """Score one pair pointwise; return a record with diagnostic detail."""
    s_pos_k: list[float | None] = []
    s_neg_k: list[float | None] = []
    for th in themes:
        s_pos_k.append(
            judge.score_video_on_theme(
                instruction, v_pos, th["theme"], th["tips"], dimension,
                max_new_tokens=max_new_tokens,
            )
        )
        s_neg_k.append(
            judge.score_video_on_theme(
                instruction, v_neg, th["theme"], th["tips"], dimension,
                max_new_tokens=max_new_tokens,
            )
        )
    S_pos = _aggregate_pointwise(s_pos_k)
    S_neg = _aggregate_pointwise(s_neg_k)
    correct: bool | None
    if S_pos is None or S_neg is None:
        correct = None
    elif S_pos > S_neg + tau:
        correct = True
    elif S_neg > S_pos + tau:
        correct = False
    else:
        correct = None  # tie within tau
    n_attempts = len(themes) * 2
    n_fail = sum(s is None for s in s_pos_k) + sum(s is None for s in s_neg_k)
    return {
        "correct":    correct,
        "S_pos":      S_pos,
        "S_neg":      S_neg,
        "s_pos_k":    s_pos_k,
        "s_neg_k":    s_neg_k,
        "n_themes":   len(themes),
        "n_attempts": n_attempts,
        "n_fail":     n_fail,
    }


def _summarize_pointwise(records: list[dict]) -> dict:
    n = len(records)
    decided      = [r for r in records if r["correct"] is not None]
    correct_dec  = [r for r in decided if r["correct"]]
    n_dec        = len(decided)
    acc_decided  = (len(correct_dec) / n_dec) if n_dec else 0.0
    acc_conserv  = (len(correct_dec) / n)     if n     else 0.0
    n_attempts   = sum(r["n_attempts"] for r in records)
    n_fail       = sum(r["n_fail"]     for r in records)
    parse_fail_rate = (n_fail / n_attempts) if n_attempts else 0.0
    # Score-degeneracy probe: variance of per-theme scores within v_pos.
    import statistics as _stats
    pos_vars: list[float] = []
    for r in records:
        valid = [s for s in r["s_pos_k"] if s is not None]
        if len(valid) >= 2:
            pos_vars.append(_stats.pvariance(valid))
    score_var_pos = (sum(pos_vars) / len(pos_vars)) if pos_vars else 0.0
    # Mean signed gap S_pos - S_neg on decided pairs.
    gaps = [
        r["S_pos"] - r["S_neg"]
        for r in decided
        if r["S_pos"] is not None and r["S_neg"] is not None
    ]
    mean_gap = (sum(gaps) / len(gaps)) if gaps else 0.0
    return {
        "n_total":               n,
        "n_decided":             n_dec,
        "uncertain_rate":        round(1 - n_dec / n, 4) if n else 0.0,
        "accuracy_on_decided":   round(acc_decided, 4),
        "accuracy_conservative": round(acc_conserv, 4),
        "parse_fail_rate":       round(parse_fail_rate, 4),
        "score_variance_pos":    round(score_var_pos, 4),
        "mean_score_gap":        round(mean_gap, 4),
    }


def diagnosis_4(judge: QwenVLJudge, cfg: dict) -> dict:
    print("\n========== D4: Pointwise per-theme rubric scoring ==========",
          file=sys.stderr)
    data = load_jsonl(cfg["dim_pair_data"], cfg["max_samples"])
    rubric_dir = Path(cfg.get("rubric_dir", str(ROOT / "rubric_pipeline" / "rubrics")))
    tau = float(cfg.get("score_tau", 0.0))
    max_new_tokens = int(cfg.get("score_max_new_tokens", 8))

    by_dim: dict[str, list[dict]] = defaultdict(list)
    skipped_no_rubric: dict[str, int] = defaultdict(int)
    total = len(data)
    for i, ex in enumerate(data):
        dim = ex.get("dimension")
        if not dim:
            continue
        themes = _load_rubric_themes(rubric_dir, dim)
        if not themes:
            skipped_no_rubric[dim] += 1
            continue
        rec = eval_one_pair_pointwise(
            judge, ex["instruction"], ex["video_pos"], ex["video_neg"],
            themes, dim, tau, max_new_tokens=max_new_tokens,
        )
        rec["id"] = ex.get("id")
        by_dim[dim].append(rec)
        if (i + 1) % 20 == 0:
            # Running aggregate over all dims so far.
            flat = [r for lst in by_dim.values() for r in lst]
            stat = _summarize_pointwise(flat)
            print(
                f"  [{i+1}/{total}] running pointwise acc(decided)={stat['accuracy_on_decided']} "
                f"acc(conserv)={stat['accuracy_conservative']} "
                f"parse_fail={stat['parse_fail_rate']} "
                f"var_pos={stat['score_variance_pos']}",
                file=sys.stderr,
            )

    heatmap: dict[str, dict] = {}
    for dim in DIMENSIONS:
        if dim in by_dim:
            heatmap[dim] = _summarize_pointwise(by_dim[dim])
    for dim, lst in by_dim.items():
        if dim not in heatmap:
            heatmap[dim] = _summarize_pointwise(lst)

    flat_all = [r for lst in by_dim.values() for r in lst]
    overall = _summarize_pointwise(flat_all)

    ranked = sorted(
        heatmap.items(),
        key=lambda kv: kv[1]["accuracy_conservative"],
        reverse=True,
    )
    print("  Per-dimension pointwise heatmap (sorted by conservative acc):",
          file=sys.stderr)
    for dim, st in ranked:
        flag = ("strong"               if st["accuracy_conservative"] >= 0.40
                else "weak (rubric ⊕)" if st["accuracy_conservative"] <  0.20
                else "medium")
        print(
            f"    {dim:<22s} conserv={st['accuracy_conservative']:.3f}  "
            f"decided={st['accuracy_on_decided']:.3f}  "
            f"parse_fail={st['parse_fail_rate']:.3f}  "
            f"var_pos={st['score_variance_pos']:.3f}  "
            f"gap={st['mean_score_gap']:+.3f}  "
            f"n={st['n_decided']:>3}/{st['n_total']:<3}  [{flag}]",
            file=sys.stderr,
        )
    print(
        f"  OVERALL: acc(decided)={overall['accuracy_on_decided']} "
        f"acc(conserv)={overall['accuracy_conservative']} "
        f"uncertain={overall['uncertain_rate']} "
        f"parse_fail={overall['parse_fail_rate']} "
        f"var_pos={overall['score_variance_pos']} "
        f"gap={overall['mean_score_gap']:+.3f}",
        file=sys.stderr,
    )
    return {
        "overall":            overall,
        "heatmap":            heatmap,
        "ranked":             [d for d, _ in ranked],
        "tau":                tau,
        "score_max_new_tokens": max_new_tokens,
        "skipped_no_rubric":  dict(skipped_no_rubric),
    }


# ============================================================
# D5: Pointwise + full-rubric JSON multi-theme scoring.
#
# Architectural relation to D4:
#   D4 issues N_themes forwards per video, each scoring ONE theme to a
#   single float. That is robust per-theme but multiplies forward count
#   by N (10x for 5 themes, two videos), which makes Stage ②/③
#   reward-time inference impractical (~8M forwards / 1k RL steps).
#
#   D5 collapses this into ONE forward per video by inlining the FULL
#   rubric and asking the model to emit a JSON object with one score
#   per theme. Forward count drops to 2 per pair, matching the budget
#   we will need at Stage ②/③ deployment. The per-theme breakdown is
#   preserved so all downstream diagnostics (variance probe, gap, etc.)
#   from D4 carry over verbatim.
#
# For every (v_pos, v_neg, dimension) triple:
#   1) Load the rubric for ``dimension`` and parse its themes.
#   2) ONE forward each on v_pos and v_neg → JSON {"theme_1": ..., ...}
#      → list[float | None] of length n_themes (3-tier parser).
#   3) Aggregate: S(v) = mean over non-None per-theme scores.
#   4) Decide: True if S_pos > S_neg + tau; False if S_neg > S_pos + tau;
#              None if |S_pos - S_neg| <= tau (tie / undecided).
#
# Reported metrics per dim are identical to D4 (same _summarize_pointwise),
# so the two diagnostics are directly comparable in the report JSON.
# ============================================================
def eval_one_pair_multi(
    judge:        QwenVLJudge,
    instruction:  str,
    v_pos:        str,
    v_neg:        str,
    themes:       list[dict],
    dimension:    str | None,
    tau:          float,
    *,
    max_new_tokens: int = 96,
    score_method:   str = "integer_text",
) -> dict:
    """Score one pair via 2 forwards (one per video, full rubric).

    ``score_method``:
      - ``"integer_text"`` (legacy): emit JSON of int 1-5 per theme,
        parse text. Discrete scores in {1,2,3,4,5}.
      - ``"soft_logits"``: same prompt scaffold but read per-position
        logits and compute E[k]=Σ k·softmax(logits_k). Continuous
        floats in [1.0, 5.0]; eliminates parse_fail and tie-rounding.

    Returns a record with the same shape as ``eval_one_pair_pointwise``
    so ``_summarize_pointwise`` works unchanged.
    """
    if score_method == "soft_logits":
        s_pos_k, raw_pos = judge.score_video_multi_theme_softlogits(
            instruction, v_pos, themes, dimension,
            max_new_tokens=max_new_tokens,
        )
        s_neg_k, raw_neg = judge.score_video_multi_theme_softlogits(
            instruction, v_neg, themes, dimension,
            max_new_tokens=max_new_tokens,
        )
    else:
        s_pos_k, raw_pos = judge.score_video_multi_theme(
            instruction, v_pos, themes, dimension,
            max_new_tokens=max_new_tokens,
        )
        s_neg_k, raw_neg = judge.score_video_multi_theme(
            instruction, v_neg, themes, dimension,
            max_new_tokens=max_new_tokens,
        )
    S_pos = _aggregate_pointwise(s_pos_k)
    S_neg = _aggregate_pointwise(s_neg_k)
    correct: bool | None
    if S_pos is None or S_neg is None:
        correct = None
    elif S_pos > S_neg + tau:
        correct = True
    elif S_neg > S_pos + tau:
        correct = False
    else:
        correct = None  # tie within tau
    n_attempts = len(themes) * 2
    n_fail = sum(s is None for s in s_pos_k) + sum(s is None for s in s_neg_k)
    return {
        "correct":    correct,
        "S_pos":      S_pos,
        "S_neg":      S_neg,
        "s_pos_k":    s_pos_k,
        "s_neg_k":    s_neg_k,
        "n_themes":   len(themes),
        "n_attempts": n_attempts,
        "n_fail":     n_fail,
        # Optional raw outputs kept off the record by default to keep
        # the JSON compact; uncomment if debugging parser fallout:
        # "raw_pos": raw_pos, "raw_neg": raw_neg,
    }


def diagnosis_5(judge: QwenVLJudge, cfg: dict) -> dict:
    print("\n========== D5: Pointwise full-rubric multi-score ==========",
          file=sys.stderr)
    data = load_jsonl(cfg["dim_pair_data"], cfg["max_samples"])
    rubric_dir = Path(cfg.get("rubric_dir", str(ROOT / "rubric_pipeline" / "rubrics")))
    tau = float(cfg.get("score_tau", 0.0))
    max_new_tokens = int(cfg.get("multi_score_max_new_tokens", 96))
    score_method = str(cfg.get("score_method", "integer_text"))
    if score_method not in ("integer_text", "soft_logits"):
        print(f"[D5] WARNING: unknown score_method={score_method!r}; "
              f"falling back to integer_text.", file=sys.stderr)
        score_method = "integer_text"
    print(f"[D5] score_method={score_method}  tau={tau}  "
          f"max_new_tokens={max_new_tokens}", file=sys.stderr)

    by_dim: dict[str, list[dict]] = defaultdict(list)
    skipped_no_rubric: dict[str, int] = defaultdict(int)
    total = len(data)
    for i, ex in enumerate(data):
        dim = ex.get("dimension")
        if not dim:
            continue
        themes = _load_rubric_themes(rubric_dir, dim)
        if not themes:
            skipped_no_rubric[dim] += 1
            continue
        rec = eval_one_pair_multi(
            judge, ex["instruction"], ex["video_pos"], ex["video_neg"],
            themes, dim, tau, max_new_tokens=max_new_tokens,
            score_method=score_method,
        )
        rec["id"] = ex.get("id")
        by_dim[dim].append(rec)
        if (i + 1) % 20 == 0:
            flat = [r for lst in by_dim.values() for r in lst]
            stat = _summarize_pointwise(flat)
            print(
                f"  [{i+1}/{total}] running multi-score acc(decided)={stat['accuracy_on_decided']} "
                f"acc(conserv)={stat['accuracy_conservative']} "
                f"parse_fail={stat['parse_fail_rate']} "
                f"var_pos={stat['score_variance_pos']}",
                file=sys.stderr,
            )

    heatmap: dict[str, dict] = {}
    for dim in DIMENSIONS:
        if dim in by_dim:
            heatmap[dim] = _summarize_pointwise(by_dim[dim])
    for dim, lst in by_dim.items():
        if dim not in heatmap:
            heatmap[dim] = _summarize_pointwise(lst)

    flat_all = [r for lst in by_dim.values() for r in lst]
    overall = _summarize_pointwise(flat_all)

    ranked = sorted(
        heatmap.items(),
        key=lambda kv: kv[1]["accuracy_conservative"],
        reverse=True,
    )
    print("  Per-dimension multi-score heatmap (sorted by conservative acc):",
          file=sys.stderr)
    for dim, st in ranked:
        flag = ("strong"               if st["accuracy_conservative"] >= 0.40
                else "weak (rubric ⊕)" if st["accuracy_conservative"] <  0.20
                else "medium")
        print(
            f"    {dim:<22s} conserv={st['accuracy_conservative']:.3f}  "
            f"decided={st['accuracy_on_decided']:.3f}  "
            f"parse_fail={st['parse_fail_rate']:.3f}  "
            f"var_pos={st['score_variance_pos']:.3f}  "
            f"gap={st['mean_score_gap']:+.3f}  "
            f"n={st['n_decided']:>3}/{st['n_total']:<3}  [{flag}]",
            file=sys.stderr,
        )
    print(
        f"  OVERALL: acc(decided)={overall['accuracy_on_decided']} "
        f"acc(conserv)={overall['accuracy_conservative']} "
        f"uncertain={overall['uncertain_rate']} "
        f"parse_fail={overall['parse_fail_rate']} "
        f"var_pos={overall['score_variance_pos']} "
        f"gap={overall['mean_score_gap']:+.3f}",
        file=sys.stderr,
    )
    return {
        "overall":                  overall,
        "heatmap":                  heatmap,
        "ranked":                   [d for d, _ in ranked],
        "tau":                      tau,
        "multi_score_max_new_tokens": max_new_tokens,
        "score_method":             score_method,
        "skipped_no_rubric":        dict(skipped_no_rubric),
    }


# ============================================================
# D3: Per-dimension probe → strong/weak heatmap.
# ============================================================
def diagnosis_3(judge: QwenVLJudge, cfg: dict) -> dict:
    print("\n========== D3: Per-dimension probe ==========", file=sys.stderr)
    data = load_jsonl(cfg["dim_pair_data"], cfg["max_samples"])
    nframes = cfg["default_nframes"]
    by_dim: dict[str, list] = defaultdict(list)
    for ex in data:
        dim = ex.get("dimension")
        if not dim:
            continue
        r = eval_one_pair(
            judge, ex["instruction"], ex["video_pos"], ex["video_neg"],
            nframes, dimension=dim, debias_swap=cfg["debias_swap"],
        )
        by_dim[dim].append(r)
    heatmap: dict[str, dict] = {}
    # Iterate in DIMENSIONS order (canonical taxonomy) so output is stable.
    for dim in DIMENSIONS:
        if dim in by_dim:
            heatmap[dim] = summarize(by_dim[dim])
    # Also surface any dim present in data but not in our taxonomy.
    for dim, lst in by_dim.items():
        if dim not in heatmap:
            heatmap[dim] = summarize(lst)
    # ── Three rankings, each highlighting a different lens.
    #
    # `ranked` (the canonical one consumed by downstream code) is now
    # sorted by `accuracy_conservative` — i.e. the realistic
    # deployability score that treats positional-bias-induced
    # `uncertain` as wrong. This fixes a prior bug where strong/weak
    # labels were assigned by `accuracy_on_decided`, which can read
    # 1.000 even when only 2/30 pairs are decided.
    ranked = sorted(
        heatmap.items(),
        key=lambda kv: kv[1]["accuracy_conservative"],
        reverse=True,
    )
    ranked_by_decided = sorted(
        heatmap.items(),
        key=lambda kv: kv[1]["accuracy_on_decided"],
        reverse=True,
    )
    ranked_by_uncertain = sorted(
        heatmap.items(),
        key=lambda kv: kv[1]["uncertain_rate"],
    )
    print("  Per-dimension heatmap "
          "(sorted by conservative acc, descending):", file=sys.stderr)
    for dim, st in ranked:
        cons = st["accuracy_conservative"]
        ad   = st["accuracy_on_decided"]
        u    = st["uncertain_rate"]
        # Thresholds calibrated against the realistic conservative range
        # observed in this codebase (best ~0.42, worst 0.00). A "strong"
        # dim is one the base model can actually use after positional
        # debiasing; "weak" is one rubric must shore up most.
        flag = ("strong"               if cons >= 0.40
                else "weak (rubric ⊕)" if cons <  0.20
                else "medium")
        print(
            f"    {dim:<22s} conserv={cons:.3f}  decided={ad:.3f}  "
            f"uncert={u:.3f}  n={st['n_decided']:>3}/{st['n_total']:<3}  [{flag}]",
            file=sys.stderr,
        )
    return {
        "heatmap":             heatmap,
        "ranked":              [d for d, _ in ranked],              # by conservative
        "ranked_by_decided":   [d for d, _ in ranked_by_decided],   # legacy lens
        "ranked_by_uncertain": [d for d, _ in ranked_by_uncertain],
        "ranking_key":         "accuracy_conservative",
    }


# ============================================================
# CLI overlay onto CONFIG.
# ============================================================
def overlay_cli_into_config(cfg: dict, args: argparse.Namespace) -> dict:
    out = dict(cfg)
    for k in ("model_path", "device", "dtype", "pair_data",
              "dim_pair_data", "out_dir"):
        v = getattr(args, k, None)
        if v is not None:
            out[k] = v
    if args.max_samples is not None:
        out["max_samples"] = args.max_samples
    if args.no_debias:
        out["debias_swap"] = False
    if args.default_nframes is not None:
        out["default_nframes"] = args.default_nframes
    if args.frame_sweep:
        out["frame_sweep"] = [int(x) for x in args.frame_sweep.split(",")]
    if args.frame_strategy:
        out["frame_strategy"] = args.frame_strategy
    if args.fps is not None:
        out["fps"] = args.fps
    if args.max_frames is not None:
        out["max_frames"] = args.max_frames
    if args.rubric_mode:
        out["rubric_mode"] = args.rubric_mode
    if args.rubric_dir:
        out["rubric_dir"] = args.rubric_dir
    if args.overall_rubric_dim:
        out["overall_rubric_dim"] = args.overall_rubric_dim
    # D4-only knobs.
    if getattr(args, "score_tau", None) is not None:
        out["score_tau"] = args.score_tau
    if getattr(args, "score_max_new_tokens", None) is not None:
        out["score_max_new_tokens"] = args.score_max_new_tokens
    # D5-only knob (multi-theme JSON forward budget).
    if getattr(args, "multi_score_max_new_tokens", None) is not None:
        out["multi_score_max_new_tokens"] = args.multi_score_max_new_tokens
    if getattr(args, "score_method", None) is not None:
        out["score_method"] = args.score_method
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    # NOTE: D2 (frame-sweep stress test) is no longer included in
    # ``--diagnosis all``; it must be requested explicitly via
    # ``--diagnosis d2``. Default "all" now means D1 + D3 only.
    # ``--diagnosis all`` now runs D1 + D3 + D5 — the deployable
    # configuration: 2 forwards/pair pointwise via JSON multi-theme.
    # D2 (frame sweep) and D4 (per-theme N-forwards) remain opt-in for
    # ablation runs only.
    p.add_argument("--diagnosis",
                   choices=["all", "d1", "d2", "d3", "d4", "d5"],
                   default="all")
    # Model.
    p.add_argument("--model_path", default=None)
    p.add_argument("--device",     default=None)
    p.add_argument("--dtype",      default=None)
    # Data.
    p.add_argument("--pair_data",     default=None)
    p.add_argument("--dim_pair_data", default=None)
    # Output.
    p.add_argument("--out_dir",       default=None)
    # Eval knobs.
    p.add_argument("--max_samples",   type=int, default=None)
    # Frame strategy (the primary lever as of this revision).
    p.add_argument("--frame_strategy",
                   choices=["nframes", "fps", "auto_default"],
                   default=None,
                   help="how to specify frames to qwen-vl-utils. "
                        "Default (from CONFIG) is 'fps' to match Qwen2.5-VL's "
                        "official README + VideoScore2 conventions.")
    p.add_argument("--fps",           type=float, default=None,
                   help="frames per second when --frame_strategy=fps (default 2.0)")
    p.add_argument("--max_frames",    type=int,   default=None,
                   help="hard cap on frames per video for fps mode (default 64)")
    # Legacy nframes knobs (only used when --frame_strategy=nframes or for D2).
    p.add_argument("--default_nframes", type=int, default=None)
    p.add_argument("--frame_sweep",   default=None,
                   help="comma-separated list, e.g. 8,16,32,64")
    p.add_argument("--no_debias", action="store_true",
                   help="disable position-bias debiasing (single-shot judgement)")
    # In-context gold-rubric ablation.
    p.add_argument("--rubric_mode",
                   choices=["none", "gold_dim", "gold_overall"],
                   default=None,
                   help="in-context rubric ablation: "
                        "none = baseline (default); "
                        "gold_dim = inject per-dim gold rubric (uses pair's "
                        "`dimension` field); "
                        "gold_overall = inject one fixed rubric for all pairs "
                        "(controlled by --overall_rubric_dim)")
    p.add_argument("--rubric_dir", default=str(ROOT / "rubric_pipeline" / "rubrics"),
                   help="directory of v_arr_mvp_<dim>_v2.txt rubric files")
    p.add_argument("--overall_rubric_dim", default="visual_quality",
                   help="which rubric to use under --rubric_mode gold_overall")
    # D4 knobs.
    p.add_argument("--score_tau", type=float, default=0.0,
                   help="D4 only: ties within |S_pos - S_neg| <= tau are "
                        "counted as undecided. Default 0.0 (any non-zero gap "
                        "decides). Use 0.05-0.1 to require a meaningful gap.")
    p.add_argument("--score_max_new_tokens", type=int, default=8,
                   help="D4 only: max tokens for the per-theme score forward.")
    # D5 knob (the JSON multi-theme path needs a larger budget).
    p.add_argument("--multi_score_max_new_tokens", type=int, default=96,
                   help="D5 only: max tokens for the single multi-theme JSON "
                        "forward. ~96 fits 5 themes comfortably; bump to 160 "
                        "if rubrics grow to 8+ themes.")
    # D5 score-method switch.
    p.add_argument("--score_method",
                   choices=["integer_text", "soft_logits"],
                   default=None,
                   help="D5 only: how to turn the per-theme score into a "
                        "number. 'integer_text' parses the model's text "
                        "output as int 1-5 (legacy). 'soft_logits' (default "
                        "in CONFIG) reads per-position logits and reports "
                        "E[k]=Σ k·softmax(logits_k) as a continuous float "
                        "in [1.0, 5.0]; eliminates parse_fail, ties, and "
                        "the 1-5 quantisation floor on var_pos.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    cfg = overlay_cli_into_config(CONFIG, args)
    random.seed(cfg["seed"])
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)

    # Sanity print.
    print("=" * 60, file=sys.stderr)
    print("Exp-0 base-model diagnosis (RewardVerse Stage ⓪)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    for k in ("model_path", "device", "dtype",
              "pair_data", "dim_pair_data",
              "default_nframes", "frame_sweep",
              "max_samples", "debias_swap", "out_dir"):
        print(f"  {k:<18s} = {cfg[k]}", file=sys.stderr)

    print("\nLoading Qwen2.5-VL ...", file=sys.stderr)
    judge = QwenVLJudge(cfg)

    # Sanity print AFTER cfg is finalised so frame strategy is visible.
    print("=" * 60, file=sys.stderr)
    print("Exp-0 base-model diagnosis (RewardVerse Stage ⓪)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    for k in ("model_path", "device", "dtype",
              "pair_data", "dim_pair_data",
              "frame_strategy", "fps", "max_frames",
              "default_nframes", "frame_sweep",
              "max_samples", "debias_swap",
              "rubric_mode", "rubric_dir",
              "out_dir"):
        if k in cfg:
            print(f"  {k:<18s} = {cfg[k]}", file=sys.stderr)

    report: dict = {"config": {k: v for k, v in cfg.items()}}
    out_path = Path(cfg["out_dir"]) / "exp0_report.json"

    def _checkpoint(tag: str) -> None:
        """Atomically dump `report` so far to `out_path`.
        Called after every diagnostic completes so that:
          • running the script under `nohup` and losing ssh is safe;
          • a crash in D2 still preserves D1 numbers;
          • the user can `cat` the report mid-run to peek at D1.
        """
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        tmp.replace(out_path)
        print(f"[checkpoint] {tag} dumped -> {out_path}", file=sys.stderr)

    if args.diagnosis in ("all", "d1"):
        report["D1"] = diagnosis_1(judge, cfg)
        _checkpoint("D1")
    # NOTE: D2 (frame-sweep) is OFF by default — we now use --frame_strategy=fps
    # (Qwen2.5-VL official preset). D2 must be requested explicitly via
    # ``--diagnosis d2`` if you still want the nframes 8/16/32/64 sweep.
    if args.diagnosis == "d2":
        report["D2"] = diagnosis_2(judge, cfg)
        _checkpoint("D2")
    if args.diagnosis in ("all", "d3"):
        report["D3"] = diagnosis_3(judge, cfg)
        _checkpoint("D3")
    # D4 (per-theme, N forwards/video) is opt-in only — kept for the
    # ablation table; the deployable D5 path replaces it in --diagnosis all.
    if args.diagnosis == "d4":
        report["D4"] = diagnosis_4(judge, cfg)
        _checkpoint("D4")
    if args.diagnosis in ("all", "d5"):
        report["D5"] = diagnosis_5(judge, cfg)
        _checkpoint("D5")

    print(f"\nReport written: {out_path}", file=sys.stderr)

    if "D1" in report:
        print("\n=============== GATING VERDICT ===============", file=sys.stderr)
        print(f"  D1 pairwise acc = {report['D1']['accuracy_on_decided']} "
              f"=> {report['D1']['gating']}", file=sys.stderr)
        print("  PASS  → enter Stage ① (teacher) + Stage ② (GRPO).", file=sys.stderr)
        print("  GRAY  → still safe to enter; let rubric reinforce weak dims (see D3).",
              file=sys.stderr)
        print("  FAIL  → do NOT enter RL; first fix base model / framing / data.",
              file=sys.stderr)


if __name__ == "__main__":
    main()
