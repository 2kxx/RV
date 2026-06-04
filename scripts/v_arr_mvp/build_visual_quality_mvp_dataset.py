#!/usr/bin/env python3
"""Build the Visual-Quality MVP rubric dataset (30 pairs, 4 input variants).

Pipeline:

  1. Load EvalVerse human-MoS JSONL (default
     ``merged_evaluate_results_with_path_part1.jsonl``).
  2. Within each record (= same prompt), enumerate ALL ordered (winner, loser)
     method pairs that satisfy the *strict* gating rule:

         winner.<sub>  >  loser.<sub>      for sub in {
             Aesthetics-Visual_Quality-Physics,
             Aesthetics-Visual_Quality-Rendering_Quality,
             Aesthetics-Visual_Quality-Temporal_Consistency,
         }

     i.e. the winner must dominate the loser on *every* VQ sub-dim.
  3. Rank candidates by the mean VQ-MoS gap and pick 30 prompt-disjoint pairs
     (one per prompt, deterministic with ``--seed``).
  4. Translate each Chinese prompt into English via the OpenAI-compatible
     endpoint (``--base_url`` + ``OPENAI_API_KEY``). Translations are cached.
  5. Emit four JSON files matching the MVP rubric-input variants:

       v1_prompt_ywin.json   ->  Prompt + y+   (pointwise, exemplar rubric)
       v2_prompt_pair.json   ->  Prompt + y+ + y-  (ARR canonical)
       v3_ywin_only.json     ->  y+ only       (pointwise, no prompt)
       v4_pair_only.json     ->  y+ + y-       (pairwise, no prompt)

     Field shapes match exactly what
     ``scripts/v_arr_mvp/learn_visual_quality_rubric_variants.py`` consumes.

Usage::

    export OPENAI_API_KEY=...     # Gemini / OpenAI-compat key
    python scripts/v_arr_mvp/build_visual_quality_mvp_dataset.py \
        --input merged_evaluate_results_with_path_part1.jsonl \
        --output_dir data/v_arr_mvp/visual_quality_mvp \
        --max_pairs 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Visual Quality sub-dimensions (EvalVerse Aesthetics > Visual_Quality).
# ---------------------------------------------------------------------------

VQ_SUB_DIMS: tuple[str, ...] = (
    "Aesthetics-Visual_Quality-Physics",
    "Aesthetics-Visual_Quality-Rendering_Quality",
    "Aesthetics-Visual_Quality-Temporal_Consistency",
)

# Prompt placeholders for the no-prompt variants (must match
# learn_visual_quality_rubric_variants.py exactly).
NO_PROMPT_QUERY_POINTWISE = (
    "(no prompt supplied; judge intrinsic visual quality from the video only)"
)
NO_PROMPT_QUERY_PAIRWISE = (
    "(no prompt supplied; rank the two videos purely by intrinsic visual quality)"
)

# ---------------------------------------------------------------------------
# Loading + filtering.
# ---------------------------------------------------------------------------


def _load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                print(f"[warn] {path}:{ln}: {exc}", file=sys.stderr)
    return rows


def _resolve_input_paths(patterns: list[str]) -> list[str]:
    """Expand glob patterns / comma-separated lists into a sorted unique list."""
    import glob

    expanded: list[str] = []
    seen: set[str] = set()
    for raw in patterns:
        for piece in str(raw).split(","):
            piece = piece.strip()
            if not piece:
                continue
            matches = sorted(glob.glob(piece)) if any(c in piece for c in "*?[") else [piece]
            if not matches:
                print(f"[warn] no files matched pattern: {piece}", file=sys.stderr)
                continue
            for m in matches:
                if m not in seen:
                    seen.add(m)
                    expanded.append(m)
    return expanded


def _load_jsonl_many(paths: list[str]) -> tuple[list[dict], dict[str, int]]:
    all_rows: list[dict] = []
    per_file: dict[str, int] = {}
    for p in paths:
        rows = _load_jsonl(p)
        per_file[p] = len(rows)
        all_rows.extend(rows)
    return all_rows, per_file


def _vq_sub_scores(eval_result: dict) -> dict[str, float] | None:
    """Return {sub: score} only if all 3 VQ sub-dims have numeric scores."""
    out: dict[str, float] = {}
    for k in VQ_SUB_DIMS:
        cell = eval_result.get(k)
        if not isinstance(cell, dict):
            return None
        v = cell.get("average_score")
        if not isinstance(v, (int, float)):
            return None
        out[k] = float(v)
    return out


def _strictly_dominates(winner: dict[str, float], loser: dict[str, float]) -> bool:
    return all(winner[k] > loser[k] for k in VQ_SUB_DIMS)


def _video_url(entry: dict, prefer: str) -> str | None:
    if prefer == "cos_url":
        return entry.get("cos_url") or entry.get("path")
    return entry.get("path") or entry.get("cos_url")


def mine_strict_vq_pairs(
    records: list[dict],
    task_filter: str | None,
    video_source: str,
    max_pairs: int,
    seed: int,
    relax_to_mean_if_short: bool = False,
) -> tuple[list[dict], dict]:
    """One pair per prompt; winner dominates loser on ALL 3 VQ sub-dims.

    If ``relax_to_mean_if_short`` is True and the number of strictly-dominating
    pairs is below ``max_pairs``, top-up by adding prompts where the winner has
    a *strictly larger mean* of the 3 VQ sub-dims (with no sub-dim where the
    winner is *worse*). Strictly-dominating pairs always come first.
    """
    strict_candidates: list[dict] = []
    relaxed_candidates: list[dict] = []
    stats = {
        "loaded": len(records),
        "skipped_wrong_task": 0,
        "skipped_no_evaluate_data": 0,
        "skipped_no_two_scored_methods": 0,
        "skipped_no_dominating_pair": 0,
        "kept_prompts_strict": 0,
        "kept_prompts_relaxed": 0,
    }

    for rec in records:
        if not isinstance(rec.get("evaluate_data"), list):
            stats["skipped_no_evaluate_data"] += 1
            continue
        task = rec.get("task_name")
        if task_filter and task != task_filter:
            stats["skipped_wrong_task"] += 1
            continue

        scored: list[dict] = []
        for ent in rec["evaluate_data"]:
            if not isinstance(ent, dict):
                continue
            subs = _vq_sub_scores(ent.get("evaluate_result") or {})
            if subs is None:
                continue
            url = _video_url(ent, video_source)
            if not url:
                continue
            scored.append(
                {
                    "method": ent.get("method") or "unknown",
                    "url": url,
                    "subs": subs,
                    "mean": sum(subs.values()) / len(subs),
                },
            )

        if len(scored) < 2:
            stats["skipped_no_two_scored_methods"] += 1
            continue

        # Find the (winner, loser) pair with the largest mean-gap that ALSO
        # strictly dominates on every sub-dim.
        best_strict: tuple[dict, dict, float] | None = None
        best_relaxed: tuple[dict, dict, float] | None = None
        for i, w in enumerate(scored):
            for j, l in enumerate(scored):
                if i == j:
                    continue
                gap = w["mean"] - l["mean"]
                if _strictly_dominates(w["subs"], l["subs"]):
                    if best_strict is None or gap > best_strict[2]:
                        best_strict = (w, l, gap)
                elif (
                    relax_to_mean_if_short
                    and gap > 0
                    # Winner must not be *worse* on any sub-dim (>= on all).
                    and all(w["subs"][k] >= l["subs"][k] for k in VQ_SUB_DIMS)
                ):
                    if best_relaxed is None or gap > best_relaxed[2]:
                        best_relaxed = (w, l, gap)

        chosen = best_strict or (best_relaxed if relax_to_mean_if_short else None)
        if chosen is None:
            stats["skipped_no_dominating_pair"] += 1
            continue

        winner, loser, gap = chosen
        is_strict = chosen is best_strict
        prompt_zh = rec.get("instruction_prompt") or rec.get("prompt") or ""
        item = {
            "prompt_zh": prompt_zh,
            "winner_url": winner["url"],
            "loser_url": loser["url"],
            "winner_method": winner["method"],
            "loser_method": loser["method"],
            "winner_subs": winner["subs"],
            "loser_subs": loser["subs"],
            "winner_mean": winner["mean"],
            "loser_mean": loser["mean"],
            "score_gap": gap,
            "task_name": task,
            "evalverse_ref_path": rec.get("path"),
            "evalverse_ref_cos_url": rec.get("path_cos_url"),
            "selection_rule": "strict_all_subdims" if is_strict else "relaxed_mean_with_no_loss",
        }
        if is_strict:
            strict_candidates.append(item)
            stats["kept_prompts_strict"] += 1
        else:
            relaxed_candidates.append(item)
            stats["kept_prompts_relaxed"] += 1

    # Strict pairs first (sorted by gap desc), then relaxed top-ups.
    strict_candidates.sort(
        key=lambda c: (-c["score_gap"], str(c.get("evalverse_ref_path") or "")),
    )
    relaxed_candidates.sort(
        key=lambda c: (-c["score_gap"], str(c.get("evalverse_ref_path") or "")),
    )
    selected = (strict_candidates + relaxed_candidates)[:max_pairs]
    stats["final_pairs"] = len(selected)
    stats["final_pairs_strict"] = sum(
        1 for s in selected if s["selection_rule"] == "strict_all_subdims"
    )
    stats["final_pairs_relaxed"] = sum(
        1 for s in selected if s["selection_rule"] == "relaxed_mean_with_no_loss"
    )
    return selected, stats


# ---------------------------------------------------------------------------
# Mid-range standalone mining (to populate the 3-region of the label scale).
# ---------------------------------------------------------------------------

def _normalize_url_key(url: str | None) -> str:
    """Lower-cased, query-stripped URL key for de-dup against pair winners/losers."""
    if not url:
        return ""
    u = str(url).split("?")[0].strip().lower()
    return u

def mine_mid_samples(
    records: list[dict],
    used_urls: set[str],
    task_filter: str | None,
    video_source: str,
    n_mid: int,
    n_low: int,
    seed: int,
) -> list[dict]:
    """Mine standalone single-video samples whose VQ-mean lies in the
    mid range (~[0.55, 0.70)) and the low range ([0.0, 0.40)).

    These fill the 1.0-3.0 region of the label scale that strictly-dominating
    winner/loser pairs alone cannot cover (winner_mean ~ 0.96, loser_mean
    ~ 0.45 only). For each sample we keep the prompt + this single video, plus
    its sub-scores, so downstream label_score = 1 + 4*mean falls into the
    desired region.

    Args:
        used_urls: URLs already used by the strict-pair list (winner/loser).
        n_mid: max number of [0.55, 0.70) samples to keep.
        n_low: max number of [0.0, 0.40) samples to keep.
    """
    import random
    rng = random.Random(seed)

    mid_pool: list[dict] = []
    low_pool: list[dict] = []

    for rec in records:
        if not isinstance(rec.get("evaluate_data"), list):
            continue
        task = rec.get("task_name")
        if task_filter and task != task_filter:
            continue
        prompt_zh = rec.get("instruction_prompt") or rec.get("prompt") or ""
        for ent in rec["evaluate_data"]:
            if not isinstance(ent, dict):
                continue
            subs = _vq_sub_scores(ent.get("evaluate_result") or {})
            if subs is None:
                continue
            url = _video_url(ent, video_source)
            if not url:
                continue
            if _normalize_url_key(url) in used_urls:
                continue
            mean = sum(subs.values()) / len(subs)
            item = {
                "prompt_zh": prompt_zh,
                "url": url,
                "method": ent.get("method") or "unknown",
                "subs": subs,
                "mean": mean,
                "task_name": task,
                "evalverse_ref_path": rec.get("path"),
            }
            if 0.55 <= mean < 0.70:
                mid_pool.append(item)
            elif 0.0 <= mean < 0.40:
                low_pool.append(item)

    rng.shuffle(mid_pool)
    rng.shuffle(low_pool)
    chosen = mid_pool[:n_mid] + low_pool[:n_low]
    chosen.sort(key=lambda x: x["mean"])  # stable order
    return chosen

# ---------------------------------------------------------------------------
# Translation (Chinese prompt -> English).
# ---------------------------------------------------------------------------


_TRANSLATE_SYSTEM = (
    "You are a precise translator. Translate the user's Chinese text into "
    "natural, vivid English suitable for a text-to-video generation prompt. "
    "Preserve all visual details, camera language, lighting and motion cues. "
    "Output ONLY the English translation as plain text, with no quotes, no "
    "labels, and no commentary."
)


def _looks_english(text: str) -> bool:
    if not text:
        return False
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return ascii_chars / max(len(text), 1) >= 0.95


def translate_prompts(
    prompts: list[str],
    base_url: str,
    model: str,
    api_key: str,
    cache_path: Path | None = None,
    max_retries: int = 3,
) -> list[str]:
    cache: dict[str, str] = {}
    if cache_path and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    # Lazy client: only construct when we actually have a non-cache-hit prompt
    # to translate. This lets users run a fully cache-hit build without a valid
    # OPENAI_API_KEY (e.g. when manual translations have been pre-populated
    # via scripts/v_arr_mvp/_inject_shared_test_translations.py).
    client = None
    out: list[str] = []
    for idx, zh in enumerate(prompts):
        if not zh:
            out.append("")
            continue
        if _looks_english(zh):
            out.append(zh)
            continue
        if zh in cache and cache[zh]:
            out.append(cache[zh])
            continue

        if client is None:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "openai package is required: pip install openai>=1.0",
                ) from exc
            if not api_key:
                print(
                    f"[error] cannot translate prompt #{idx}: cache miss and no "
                    f"OPENAI_API_KEY available. Run "
                    f"scripts/v_arr_mvp/_inject_shared_test_translations.py to "
                    f"pre-populate manual translations.",
                    file=sys.stderr,
                )
                # Fall back to original Chinese to avoid crashing the build.
                out.append(zh)
                continue
            client = OpenAI(base_url=base_url, api_key=api_key)

        en = ""
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _TRANSLATE_SYSTEM},
                        {"role": "user", "content": zh},
                    ],
                    temperature=0.2,
                )
                en = (resp.choices[0].message.content or "").strip()
                if en:
                    break
            except Exception as exc:
                last_err = exc
                time.sleep(1.5 * (attempt + 1))
        if not en:
            print(f"[warn] translation failed for prompt #{idx}: {last_err}",
                  file=sys.stderr)
            en = zh  # fall back to the original Chinese
        cache[zh] = en
        out.append(en)
        if cache_path:
            cache_path.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    return out


# ---------------------------------------------------------------------------
# Variant emission.
# ---------------------------------------------------------------------------


def _to_label_score(mean: float) -> float:
    """Map the 0-1 EvalVerse VQ MoS into the 1-5 ARR scale used downstream.

    Returns a continuous float in [1.0, 5.0] (rounded to 2 decimals). The
    downstream metrics (PLCC/SRCC/MAE) consume the raw float; we no longer
    snap to integers because that collapses the 4 vs 5 boundary on a tightly
    clustered winner distribution (winner_mean ~ 0.96 -> all 5s).
    """
    return round(max(1.0, min(5.0, 1.0 + 4.0 * float(mean))), 2)


def _normalize_task_type(task_name: str | None) -> str:
    if not task_name:
        return "t2v"
    return "t2v" if task_name == "t2v" else "r2v"


def _stratified_sample_pointwise(
    pairs_en: list[dict],
    mid_samples: list[dict] | None,
    target_count: int,
    seed: int = 0,
    excluded_urls: set[str] | None = None,
) -> list[tuple[str, dict]]:
    """Build a unified single-video candidate pool from (winner, loser) pairs
    and standalone mid/low samples, then stratified-sample ``target_count``
    items so that the resulting label distribution is spread as evenly as
    possible across the five 1-5 score bands.

    Returns a list of (role, item_dict) tuples where ``role`` is one of
    {"winner", "loser", "mid"} and ``item_dict`` carries the keys consumed by
    the per-variant builders below (``prompt_en``, ``url`` / ``winner_url``,
    ``mean``, plus the original metadata payload).

    Each pair contributes at most ONE side (winner or loser) to avoid letting
    the same prompt show up twice in v1/v3 -- which would also be unfair
    against v2/v4 (those use 30 pairs = 30 distinct prompts).

    ``excluded_urls`` (A1, 2026-05): hard-exclude these URLs from the pool.
    Used to keep v1/v3 (pointwise training) strictly disjoint from the shared
    pointwise test set, even when both draw from the same pair winners/losers.
    """
    import random
    rng = random.Random(seed)
    excluded_urls = excluded_urls or set()

    # Build a flat candidate pool. For each pair we emit BOTH the winner and
    # loser as candidates, but tag them with ``pair_idx`` so the sampler can
    # later enforce "at most one side per pair".
    pool: list[dict] = []
    for i, p in enumerate(pairs_en):
        pool.append({
            "role": "winner",
            "pair_idx": i,
            "prompt_en": p["prompt_en"],
            "url": p["winner_url"],
            "mean": p["winner_mean"],
            "src": p,
        })
        pool.append({
            "role": "loser",
            "pair_idx": i,
            "prompt_en": p["prompt_en"],
            "url": p["loser_url"],
            "mean": p["loser_mean"],
            "src": p,
        })
    for m in (mid_samples or []):
        pool.append({
            "role": "mid",
            "pair_idx": None,
            "prompt_en": m["prompt_en"],
            "url": m["url"],
            "mean": m["mean"],
            "src": m,
        })

    # Five label bands (1..5) defined on the 0-1 mean scale so each band is
    # exactly 0.2 wide. The last band is closed at 1.0.
    bands: list[tuple[float, float]] = [
        (0.0, 0.2),  # ~ label [1.0, 1.8)
        (0.2, 0.4),  # ~ label [1.8, 2.6)
        (0.4, 0.6),  # ~ label [2.6, 3.4)
        (0.6, 0.8),  # ~ label [3.4, 4.2)
        (0.8, 1.001),  # ~ label [4.2, 5.0]
    ]
    band_pools: list[list[dict]] = [[] for _ in bands]
    for c in pool:
        for bi, (lo, hi) in enumerate(bands):
            if lo <= c["mean"] < hi:
                band_pools[bi].append(c)
                break
    for bp in band_pools:
        rng.shuffle(bp)

    # Round-robin draw across bands so the result spans all 5 bands as evenly
    # as feasible. We enforce uniqueness by URL and "at most one side per pair".
    chosen: list[dict] = []
    used_urls: set[str] = set()
    used_pairs: set[int] = set()
    while len(chosen) < target_count:
        progress = False
        for bp in band_pools:
            if len(chosen) >= target_count:
                break
            while bp:
                cand = bp.pop()
                key = (cand["url"] or "").split("?")[0].lower()
                if key in used_urls:
                    continue
                if key in excluded_urls:
                    continue
                pi = cand["pair_idx"]
                if pi is not None and pi in used_pairs:
                    continue
                used_urls.add(key)
                if pi is not None:
                    used_pairs.add(pi)
                chosen.append(cand)
                progress = True
                break
        if not progress:
            break  # all bands exhausted

    # Stable ordering by mean (low -> high) so the resulting JSON is easier
    # to eyeball.
    chosen.sort(key=lambda x: x["mean"])
    return [(c["role"], c) for c in chosen]


def _build_v1_prompt_ywin(
    pairs_en: list[dict],
    mid_samples: list[dict] | None = None,
    max_samples: int = 30,
    seed: int = 0,
    excluded_urls: set[str] | None = None,
) -> list[dict]:
    """Variant 1: Prompt + single video (pointwise).

    To keep v1 the same SIZE as v2/v4 (30 items) while still spanning the
    full 1-5 label range we draw 30 single-video samples via stratified
    sampling over a unified pool of {pair winners, pair losers, standalone
    mid/low samples}, with at most one side per pair.
    """
    selected = _stratified_sample_pointwise(
        pairs_en, mid_samples, max_samples, seed,
        excluded_urls=excluded_urls,
    )
    out: list[dict] = []
    for cnt, (role, c) in enumerate(selected):
        if role == "mid":
            metadata = _mid_meta(c["src"], mode="v1_prompt_ywin")
        else:
            metadata = {**_shared_meta(c["src"], mode="v1_prompt_ywin"), "role": role}
        out.append({
            "id": f"varr_vq_{cnt:03d}",
            "query": c["prompt_en"],
            "response": c["url"],
            "label_score": _to_label_score(c["mean"]),
            "dimension": "visual_quality",
            "task_type": _normalize_task_type(c["src"].get("task_name")),
            "metadata": metadata,
        })
    return out


def _build_v2_prompt_pair(pairs_en: list[dict]) -> list[dict]:
    """Variant 2: Prompt + y+ + y-  (ARR canonical pairwise)."""
    out: list[dict] = []
    for i, p in enumerate(pairs_en):
        out.append(
            {
                "id": f"varr_vq_{i:03d}",
                "query": p["prompt_en"],
                "response": [p["winner_url"], p["loser_url"]],
                "label_rank": [1, 2],
                "task_type": _normalize_task_type(p.get("task_name")),
                "metadata": _shared_meta(p, mode="v2_prompt_pair"),
            },
        )
    return out


def _build_v3_ywin_only(
    pairs_en: list[dict],
    mid_samples: list[dict] | None = None,
    max_samples: int = 30,
    seed: int = 0,
    excluded_urls: set[str] | None = None,
) -> list[dict]:
    """Variant 3: single video, NO prompt (pointwise).

    Uses the exact same stratified pool as v1 so v1 vs v3 only differs in
    whether the prompt is shown to the judge (the canonical RQ for v1 vs v3).
    """
    selected = _stratified_sample_pointwise(
        pairs_en, mid_samples, max_samples, seed,
        excluded_urls=excluded_urls,
    )
    out: list[dict] = []
    for cnt, (role, c) in enumerate(selected):
        if role == "mid":
            metadata = _mid_meta(c["src"], mode="v3_ywin_only")
        else:
            metadata = {**_shared_meta(c["src"], mode="v3_ywin_only"), "role": role}
        out.append({
            "id": f"varr_vq_{cnt:03d}",
            "query": NO_PROMPT_QUERY_POINTWISE,
            "response": c["url"],
            "label_score": _to_label_score(c["mean"]),
            "dimension": "visual_quality",
            "task_type": _normalize_task_type(c["src"].get("task_name")),
            "metadata": metadata,
        })
    return out


def _build_v4_pair_only(pairs_en: list[dict]) -> list[dict]:
    """Variant 4: y+ + y-  (pairwise, no prompt)."""
    out: list[dict] = []
    for i, p in enumerate(pairs_en):
        out.append(
            {
                "id": f"varr_vq_{i:03d}",
                "query": NO_PROMPT_QUERY_PAIRWISE,
                "response": [p["winner_url"], p["loser_url"]],
                "label_rank": [1, 2],
                "task_type": _normalize_task_type(p.get("task_name")),
                "metadata": _shared_meta(p, mode="v4_pair_only"),
            },
        )
    return out


def _mine_shared_test_candidates(
    records: list[dict],
    excluded_urls: set[str],
    task_filter: str | None,
    video_source: str,
    target_count: int,
    seed: int = 0,
) -> list[dict]:
    """Pick candidate (prompt, single_video, label) tuples for the shared
    pointwise test set, *without* translating prompts.

    Translation is deferred to the caller so that all prompt-batches
    (pairs + mid_samples + shared_test) can share a single cache file and a
    single LLM call.

    Strategy:
      * Stratify videos by VQ-mean across 5 bands [0,0.2)..[0.8,1.0].
      * Round-robin across bands so the result spans the full 1-5 scale.
      * Exclude URLs in ``excluded_urls`` (typically the v2/v4 winner+loser
        pairs already chosen for the rubric-generation seeds).
      * One video per prompt to avoid leaky duplicates within the test set.

    To maximise label coverage at the low end of the scale (where source data
    is extremely scarce: only 8 videos with VQ-mean < 0.4), this function is
    called BEFORE ``mine_mid_samples`` so it gets first pick of band 1/2.
    """
    import random
    rng = random.Random(seed)

    pool: list[dict] = []
    for rec in records:
        if not isinstance(rec.get("evaluate_data"), list):
            continue
        task = rec.get("task_name")
        if task_filter and task != task_filter:
            continue
        prompt_zh = rec.get("instruction_prompt") or rec.get("prompt") or ""
        for ent in rec["evaluate_data"]:
            if not isinstance(ent, dict):
                continue
            subs = _vq_sub_scores(ent.get("evaluate_result") or {})
            if subs is None:
                continue
            url = _video_url(ent, video_source)
            if not url:
                continue
            if _normalize_url_key(url) in excluded_urls:
                continue
            mean = sum(subs.values()) / len(subs)
            pool.append({
                "prompt_zh": prompt_zh,
                "url": url,
                "method": ent.get("method") or "unknown",
                "subs": subs,
                "mean": mean,
                "task_name": task,
                "evalverse_ref_path": rec.get("path"),
            })

    bands: list[tuple[float, float]] = [
        (0.0, 0.2),
        (0.2, 0.4),
        (0.4, 0.6),
        (0.6, 0.8),
        (0.8, 1.001),
    ]
    band_pools: list[list[dict]] = [[] for _ in bands]
    for c in pool:
        for bi, (lo, hi) in enumerate(bands):
            if lo <= c["mean"] < hi:
                band_pools[bi].append(c)
                break
    for bp in band_pools:
        rng.shuffle(bp)

    chosen: list[dict] = []
    used_urls: set[str] = set()
    used_test_prompts: set[str] = set()
    while len(chosen) < target_count:
        progress = False
        for bp in band_pools:
            if len(chosen) >= target_count:
                break
            while bp:
                cand = bp.pop()
                key = _normalize_url_key(cand["url"])
                if key in used_urls:
                    continue
                pkey = (cand["prompt_zh"] or "").strip()
                if pkey and pkey in used_test_prompts:
                    continue
                used_urls.add(key)
                if pkey:
                    used_test_prompts.add(pkey)
                chosen.append(cand)
                progress = True
                break
        if not progress:
            break
    chosen.sort(key=lambda x: x["mean"])
    return chosen


def _assemble_shared_test_payload(
    candidates_en: list[dict],
) -> list[dict]:
    """Turn translated shared-test candidates into the final dataset payload."""
    out: list[dict] = []
    for i, c in enumerate(candidates_en):
        out.append({
            "id": f"varr_vq_test_{i:03d}",
            "query": c["prompt_en"],
            "response": c["url"],
            "label_score": _to_label_score(c["mean"]),
            "dimension": "visual_quality",
            "task_type": _normalize_task_type(c["task_name"]),
            "metadata": {
                "dimension": "visual_quality",
                "synthesis_mode": "shared_test_pointwise",
                "variant": "shared_test",
                "method": c["method"],
                "score": c["mean"],
                "sub_scores": c["subs"],
                "task_name": c["task_name"],
                "evalverse_ref_path": c["evalverse_ref_path"],
                "prompt_zh": c["prompt_zh"],
            },
        })
    return out


def _shared_meta(p: dict, mode: str) -> dict:
    return {
        "dimension": "visual_quality",
        "synthesis_mode": "strict_vq_subdim_dominance",
        "variant": mode,
        "winner_method": p["winner_method"],
        "loser_method": p["loser_method"],
        "winner_score": p["winner_mean"],
        "loser_score": p["loser_mean"],
        "score_gap": p["score_gap"],
        "winner_sub_scores": p["winner_subs"],
        "loser_sub_scores": p["loser_subs"],
        "task_name": p.get("task_name"),
        "evalverse_ref_path": p.get("evalverse_ref_path"),
        "prompt_zh": p.get("prompt_zh", ""),
    }

def _mid_meta(m: dict, mode: str) -> dict:
    return {
        "dimension": "visual_quality",
        "synthesis_mode": "mid_range_standalone",
        "variant": mode,
        "role": "mid",
        "method": m.get("method"),
        "score": m["mean"],
        "sub_scores": m.get("subs"),
        "task_name": m.get("task_name"),
        "evalverse_ref_path": m.get("evalverse_ref_path"),
        "prompt_zh": m.get("prompt_zh", ""),
    }


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the 30-pair Visual-Quality MVP dataset (4 variants).",
    )
    ap.add_argument("--input", action="append", default=None,
                    help="EvalVerse human-MoS JSONL. Repeat or comma-separate "
                         "to combine multiple files; glob patterns supported "
                         "(e.g. 'merged_evaluate_results_with_path_part*.jsonl').")
    ap.add_argument("--output_dir", default="data/v_arr_mvp/visual_quality_mvp",
                    help="Directory for the 4 variant JSON files + manifest.")
    ap.add_argument("--max_pairs", type=int, default=30,
                    help="Number of (prompt, winner, loser) triples to keep.")
    ap.add_argument("--task_filter", default="",
                    help="Restrict to this task_name (empty = no filter; "
                         "keep t2v + all *_r2v variants).")
    ap.add_argument("--video_source", default="path", choices=("path", "cos_url"),
                    help="Use local 'path' or remote 'cos_url' for video URLs.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--relax_if_short", action="store_true", default=True,
                    help="If <max_pairs strictly-dominating pairs are found, "
                         "top-up with pairs where winner has a strictly larger "
                         "VQ-mean and is not worse on any sub-dim (default on).")
    ap.add_argument("--strict_only", dest="relax_if_short", action="store_false",
                    help="Disable relaxation; keep only strictly-dominating pairs.")

    # Mid-range standalone samples (populate the 3.0 region of the label scale).
    ap.add_argument("--n_mid", type=int, default=20,
                    help="Number of standalone mid-range (mean in [0.55, 0.70)) "
                         "samples to mine into v1/v3 to fill the 3-region of "
                         "the label scale (default: 20).")
    ap.add_argument("--n_low", type=int, default=8,
                    help="Number of standalone low-range (mean in [0.0, 0.40)) "
                         "samples to mine into v1/v3 to densify the 1-2 region "
                         "(default: 8). Combined with winner/loser pairs this "
                         "yields a fuller 1-5 distribution for PLCC/SRCC.")

    # Shared pointwise test set (used by ALL 4 variants for head-to-head eval).
    ap.add_argument("--shared_test_count", type=int, default=30,
                    help="Number of (prompt, single_video, label_score) items "
                         "to emit into shared_test_pointwise.json. The set is "
                         "5-band stratified across the 0-1 VQ-mean and is "
                         "disjoint from every URL used by v1-v4 (default: 30).")

    # Translation knobs.
    ap.add_argument("--translate", action="store_true", default=True,
                    help="Translate Chinese prompts to English (default: on).")
    ap.add_argument("--no_translate", dest="translate", action="store_false",
                    help="Skip translation (keep prompts as-is).")
    ap.add_argument("--base_url",
                    default=os.environ.get(
                        "BASE_URL",
                        "https://generativelanguage.googleapis.com/v1beta/openai/",
                    ),
                    help="OpenAI-compatible endpoint for translation.")
    ap.add_argument("--model", default=os.environ.get("MODEL", "gemini-2.5-pro"),
                    help="Translation model.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Mine strictly-dominating pairs.
    input_patterns = args.input or ["merged_evaluate_results_with_path_part*.jsonl"]
    input_paths = _resolve_input_paths(input_patterns)
    if not input_paths:
        print("[error] No input JSONL files resolved.", file=sys.stderr)
        sys.exit(2)
    print(f"[1/3] mining strict VQ-dominance pairs from {len(input_paths)} file(s):")
    for p in input_paths:
        print(f"        {p}")
    records, per_file = _load_jsonl_many(input_paths)
    print(f"  total records loaded             : {len(records)}")
    for p, n in per_file.items():
        print(f"    {p:<60s} {n}")
    pairs, stats = mine_strict_vq_pairs(
        records,
        task_filter=args.task_filter or None,
        video_source=args.video_source,
        max_pairs=args.max_pairs,
        seed=args.seed,
        relax_to_mean_if_short=args.relax_if_short,
    )
    for k, v in stats.items():
        print(f"  {k:<36s}: {v}")
    if not pairs:
        print("[error] No pairs survived filtering; aborting.", file=sys.stderr)
        sys.exit(3)

    # 1b) Mine the SHARED pointwise test set FIRST so it gets first pick of
    # the (extremely scarce) low-VQ samples (band 1: 2 videos; band 2: 6
    # videos -- and ALL 8 of those happen to be losers in the strict-dominance
    # pairs picked above). Without this priority, mine_mid_samples below would
    # consume what little remains and the shared test set would degenerate to
    # a 3-band span [0.4, 1.0], shrinking the label range used by PLCC/SRCC.
    #
    # A1 design choice: the shared test is allowed to REUSE pair-LOSER URLs.
    # Rationale: v2/v4 use these losers in a LISTWISE training task ("rank
    # winner > loser"); the shared test re-uses the same loser clip for a
    # POINTWISE evaluation ("absolute 1-5 score for this single clip"). The
    # tasks are functionally different so this is not a leak, and it is the
    # only way to break the data scarcity floor at the low end of the scale.
    # Pair WINNERS are still excluded so the test set never reuses a clip on
    # which a variant has already been told "this is the good one".
    pair_winner_urls: set[str] = {
        _normalize_url_key(p.get("winner_url")) for p in pairs
    }
    pair_loser_urls: set[str] = {
        _normalize_url_key(p.get("loser_url")) for p in pairs
    }
    shared_candidates = _mine_shared_test_candidates(
        records,
        excluded_urls=pair_winner_urls,  # losers ALLOWED into shared test
        task_filter=args.task_filter or None,
        video_source=args.video_source,
        target_count=args.shared_test_count,
        seed=args.seed,
    )
    shared_excluded_urls: set[str] = {
        _normalize_url_key(c["url"]) for c in shared_candidates
    }
    print(f"  shared test candidates picked      : {len(shared_candidates)} "
          f"(pair-winners excluded; pair-losers ALLOWED to break low-band scarcity)")

    # 1c) Mine standalone mid-range and low-range samples for v1/v3 only.
    # v1/v3 are POINTWISE training seeds that share their task type with the
    # shared test set, so we strictly exclude every URL the shared test has
    # already picked (in addition to all pair URLs).
    used_urls: set[str] = pair_winner_urls | pair_loser_urls | shared_excluded_urls
    mid_samples = mine_mid_samples(
        records,
        used_urls=used_urls,
        task_filter=args.task_filter or None,
        video_source=args.video_source,
        n_mid=args.n_mid,
        n_low=args.n_low,
        seed=args.seed,
    )
    n_mid_kept = sum(1 for m in mid_samples if 0.55 <= m["mean"] < 0.70)
    n_low_kept = sum(1 for m in mid_samples if m["mean"] < 0.40)
    print(f"  mid-range standalone samples mined: {n_mid_kept} (mid) + "
          f"{n_low_kept} (low) = {len(mid_samples)} total")

    # 2) Translate Chinese prompts -> English (single batched pass for all
    # three sources: pairs + mid_samples + shared_test).
    pair_prompts_zh   = [p["prompt_zh"] for p in pairs]
    mid_prompts_zh    = [m["prompt_zh"] for m in mid_samples]
    shared_prompts_zh = [c["prompt_zh"] for c in shared_candidates]
    all_prompts_zh    = pair_prompts_zh + mid_prompts_zh + shared_prompts_zh
    print(f"[2/3] translating {len(all_prompts_zh)} prompts "
          f"({len(pair_prompts_zh)} pair + {len(mid_prompts_zh)} mid + "
          f"{len(shared_prompts_zh)} shared) (translate={args.translate})")
    if args.translate:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            # No API key: fall back to cache-only translation. Any prompt that
            # is a cache miss will be silently kept in Chinese (with a warning
            # printed by translate_prompts). This lets users build the dataset
            # offline once they've pre-populated _prompt_translations.json via
            # scripts/v_arr_mvp/_inject_shared_test_translations.py.
            print("[info] OPENAI_API_KEY not set; running in CACHE-ONLY translation mode.",
                  file=sys.stderr)
        cache_path = out_dir / "_prompt_translations.json"
        en_all = translate_prompts(
            all_prompts_zh,
            base_url=args.base_url,
            model=args.model,
            api_key=api_key,
            cache_path=cache_path,
        )
    else:
        en_all = list(all_prompts_zh)

    en_pairs   = en_all[:len(pairs)]
    en_mid     = en_all[len(pairs):len(pairs) + len(mid_samples)]
    en_shared  = en_all[len(pairs) + len(mid_samples):]
    pairs_en = []
    for p, en in zip(pairs, en_pairs):
        pe = dict(p)
        pe["prompt_en"] = en
        pairs_en.append(pe)
    mid_en: list[dict] = []
    for m, en in zip(mid_samples, en_mid):
        me = dict(m)
        me["prompt_en"] = en
        mid_en.append(me)
    shared_candidates_en: list[dict] = []
    for c, en in zip(shared_candidates, en_shared):
        ce = dict(c)
        ce["prompt_en"] = en
        shared_candidates_en.append(ce)

    # 3) Emit the 4 variant JSONs + manifest.
    print(f"[3/3] writing 4 variant JSON files into {out_dir}")
    variants = {
        "v1_prompt_ywin.json": _build_v1_prompt_ywin(
            pairs_en, mid_samples=mid_en, max_samples=args.max_pairs,
            seed=args.seed, excluded_urls=shared_excluded_urls,
        ),
        "v2_prompt_pair.json": _build_v2_prompt_pair(pairs_en),
        "v3_ywin_only.json":   _build_v3_ywin_only(
            pairs_en, mid_samples=mid_en, max_samples=args.max_pairs,
            seed=args.seed, excluded_urls=shared_excluded_urls,
        ),
        "v4_pair_only.json":   _build_v4_pair_only(pairs_en),
    }
    for fname, payload in variants.items():
        path = out_dir / fname
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {len(payload):>3d} samples -> {path}")

    # 3b) Build the SHARED pointwise test set used by all 4 variants.
    # Note: candidates were already mined BEFORE mid_samples so they got first
    # pick of band 1/2 (where source data is extremely scarce).
    shared_test_path = out_dir / "shared_test_pointwise.json"
    print(f"[3b/3] assembling shared pointwise test set "
          f"({len(shared_candidates_en)} items, 5-band stratified, disjoint from v1-v4)")
    shared_test_payload = _assemble_shared_test_payload(shared_candidates_en)
    shared_test_path.write_text(
        json.dumps(shared_test_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # Diagnostic print: actual band distribution.
    import collections as _coll
    _band_dist = _coll.Counter()
    for it in shared_test_payload:
        m = it["metadata"]["score"]
        for bi, (lo, hi) in enumerate([(0.0,0.2),(0.2,0.4),(0.4,0.6),(0.6,0.8),(0.8,1.001)]):
            if lo <= m < hi:
                _band_dist[bi+1] += 1
                break
    print(f"  wrote {len(shared_test_payload):>3d} samples -> {shared_test_path}")
    print(f"  shared test 5-band distribution (band:count): {dict(sorted(_band_dist.items()))}")
    if shared_test_payload:
        _ls = [it["label_score"] for it in shared_test_payload]
        print(f"  shared test label_score: min={min(_ls):.2f} max={max(_ls):.2f} mean={sum(_ls)/len(_ls):.2f}")

    manifest = {
        "input_patterns": input_patterns,
        "input_paths": input_paths,
        "input_records_per_file": per_file,
        "max_pairs": args.max_pairs,
        "task_filter": args.task_filter,
        "video_source": args.video_source,
        "relax_if_short": args.relax_if_short,
        "n_mid_requested": args.n_mid,
        "n_low_requested": args.n_low,
        "n_mid_kept": n_mid_kept,
        "n_low_kept": n_low_kept,
        "translate": args.translate,
        "translation_model": args.model if args.translate else None,
        "stats": stats,
        "variants": {
            "v1_prompt_ywin": "Prompt + y+ + y- + mid (pointwise, full-spectrum labels)",
            "v2_prompt_pair": "Prompt + y+ + y-   (ARR canonical pairwise)",
            "v3_ywin_only":   "y+ + y- + mid (pointwise, no prompt, full-spectrum labels)",
            "v4_pair_only":   "y+ + y- (pairwise, no prompt)",
            "shared_test_pointwise": (
                f"Shared pointwise (prompt + single video) test set, "
                f"{args.shared_test_count} items, 5-band stratified across 0-1 "
                f"VQ-mean, disjoint from v1-v4 URLs. Used by ALL variants for "
                f"head-to-head PLCC/SRCC/MAE comparison."
            ),
        },
        "shared_test_count": args.shared_test_count,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  wrote manifest -> {out_dir / 'manifest.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
