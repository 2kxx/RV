"""RewardVerse MVP — per-dimension v2_prompt_pair seed-set builder.

Given an EvalVerse second-level dimension (slug from
``scripts/v_arr_mvp/dimension_registry.py``), this script mines 30
``(prompt, winner_video, loser_video)`` triples ranked by the dimension's
own sub-keys, translates the Chinese prompts to English (using the
existing on-disk cache populated for the visual_quality MVP), and writes
out a single ``v2_prompt_pair.json`` ready to be fed to the v_arr_mvp
rubric-generation pipeline.

This is a **dimension-aware** wrapper around the helpers in
``build_visual_quality_mvp_dataset.py``. It does NOT touch the
visual_quality MVP outputs (those live under
``data/v_arr_mvp/visual_quality_mvp/``) — outputs from this script land
in ``data/v_arr_mvp/<dimension>_mvp/`` so existing v_arr_mvp v1-v4
results stay frozen.

Usage::

    python scripts/v_arr_mvp/build_dim_seed_v2.py \\
        --dimension character \\
        --output_dir data/v_arr_mvp/character_mvp

The output schema matches what
``build_visual_quality_mvp_dataset.py::_build_v2_prompt_pair`` produces,
i.e. each item is::

    {
        "id": "varr_<dim>_000",
        "query": "<English prompt>" | NO_PROMPT_QUERY_PAIRWISE,
        "response": [winner_url, loser_url],
        "label_rank": [1, 2],
        "task_type": "t2v" | ...,
        "metadata": {dimension, sub_keys, winner_*, loser_*, ...},
    }
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Re-use helpers from the visual_quality builder so behaviour stays in lock-step.
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
import build_visual_quality_mvp_dataset as bvq  # type: ignore  # noqa: E402
from dimension_registry import (                # type: ignore  # noqa: E402
    DIMENSIONS_BY_NAME,
    DimensionSpec,
    list_dimension_names,
)


# ---------------------------------------------------------------------------
# Per-dimension sub-score extraction.
# ---------------------------------------------------------------------------

def _dim_sub_scores(eval_result: dict, dim: DimensionSpec) -> dict[str, float] | None:
    """Pull only the sub-scores belonging to ``dim`` from one
    ``evaluate_result`` block. Returns None if any sub-key is missing or
    not numeric — the entry is then ineligible for this dimension.
    """
    out: dict[str, float] = {}
    for k in dim.sub_keys:
        v = eval_result.get(k)
        if not isinstance(v, dict):
            return None
        s = v.get("average_score")
        if not isinstance(s, (int, float)):
            return None
        out[k] = float(s)
    return out


def _strictly_dominates(winner: dict[str, float], loser: dict[str, float],
                        sub_keys: tuple[str, ...]) -> bool:
    """winner > loser on EVERY sub-key (strict)."""
    return all(winner[k] > loser[k] for k in sub_keys)


def _mean_dominates(winner: dict[str, float], loser: dict[str, float],
                    sub_keys: tuple[str, ...]) -> bool:
    """winner.mean > loser.mean AND winner is no-worse on every sub-key.

    Used as the relax fallback when a dimension does not have enough strict
    pairs (typically when sub-key count is high, e.g. expression has 4 subs
    and lens has 4 subs — strict dominance becomes very rare).
    """
    if not (sum(winner.values()) > sum(loser.values())):
        return False
    return all(winner[k] >= loser[k] for k in sub_keys)


def _majority_dominates(winner: dict[str, float], loser: dict[str, float],
                        sub_keys: tuple[str, ...],
                        min_gap: float = 0.10) -> bool:
    """winner is strictly better on a strict majority of sub-keys, AND wins
    or ties on every remaining sub-key, AND the overall mean gap is at
    least ``min_gap``.

    This is "_mean_dominates" plus an explicit majority-strict-win
    requirement: catches cases where winner ties with loser on most subs
    but is strictly better on a clear plurality. Slightly stronger signal
    than mean-dominance but weaker than full strict-dominance.
    """
    gap = (sum(winner.values()) - sum(loser.values())) / len(sub_keys)
    if gap < min_gap:
        return False
    if not all(winner[k] >= loser[k] for k in sub_keys):
        return False
    wins = sum(1 for k in sub_keys if winner[k] > loser[k])
    return wins > len(sub_keys) / 2


def _rough_majority_dominates(winner: dict[str, float], loser: dict[str, float],
                              sub_keys: tuple[str, ...],
                              min_gap: float = 0.05) -> bool:
    """Loosest fallback (1B, 2026-05-21): winner is strictly better on a
    strict majority of sub-keys AND the overall mean gap is at least
    ``min_gap``. Loser is allowed to strictly outperform winner on the
    remaining minority of sub-keys.

    Only used when none of strict / mean / majority can fill the budget for
    a high-arity dimension. The 0.05 mean-gap floor protects against
    selecting genuinely tied pairs while still admitting many more pairs
    than majority_dominates does.
    """
    gap = (sum(winner.values()) - sum(loser.values())) / len(sub_keys)
    if gap < min_gap:
        return False
    wins = sum(1 for k in sub_keys if winner[k] > loser[k])
    return wins > len(sub_keys) / 2


def mine_dim_pairs(
    records: list[dict],
    dim: DimensionSpec,
    task_filter: str | None,
    video_source: str,
    max_pairs: int,
    seed: int,
    relax_to_mean_if_short: bool = True,
    max_pairs_per_prompt: int = 1,
    min_score_gap: float = 0.0,
) -> tuple[list[dict], dict]:
    """Mine ``max_pairs`` ranked pairs for the given dimension.

    Each record's ``evaluate_data`` is scanned: any two entries whose
    sub-scores (restricted to ``dim.sub_keys``) form a strict dominance
    relation become a candidate pair. By default we keep *one pair per
    prompt* (the pair with the largest sub-mean gap) so the seed set has
    30 distinct prompts, mirroring the visual_quality MVP convention.

    Parameters
    ----------
    max_pairs_per_prompt : int, default 1
        For low-coverage dimensions (logic / rhythm), the number of
        unique multi-method prompts is < 30 even though each such prompt
        has C(M,2) candidate pairs. Setting this > 1 lets one prompt
        contribute up to K pairs, ranked by gap. The K pairs from one
        prompt always have distinct (winner_url, loser_url) sets within
        the same kind tier, and across tiers the pool dedups by url-pair
        too. Pure quality knob — the strict / relaxed / majority /
        rough_majority cascade still happens, just with K pairs/prompt
        instead of 1.
    min_score_gap : float, default 0.0
        Hard floor on ``winner_mean - loser_mean``; pairs with a smaller
        gap are dropped. 0.0 = no floor (legacy behaviour). A value of
        ~0.05 keeps the BT signal interpretable while still admitting
        most relaxed/majority pairs.

    If fewer than ``max_pairs`` strict pairs are found and
    ``relax_to_mean_if_short`` is True, we top-up with mean-dominance
    pairs (winner has strictly larger sub-mean and is no-worse on every
    sub-key). Strict pairs always come first.
    """
    rng = random.Random(seed)

    # Per-tier per-prompt buckets: tier -> list of pair-dicts. Within a
    # tier, pairs are kept across all prompts; max_pairs_per_prompt limits
    # how many a single prompt contributes BEFORE we cross-tier merge.
    strict_pairs:         list[dict] = []
    mean_pairs:           list[dict] = []
    majority_pairs:       list[dict] = []
    rough_majority_pairs: list[dict] = []
    stats: dict = {
        "loaded": 0,
        "skipped_wrong_task": 0,
        "skipped_no_evaluate_data": 0,
        "skipped_no_two_scored_methods": 0,
        "skipped_no_dominating_pair": 0,
        "skipped_below_min_gap": 0,
        "kept_prompts_strict": 0,
        "kept_prompts_relaxed": 0,
        "kept_prompts_majority": 0,
        "kept_prompts_rough_majority": 0,
        "final_pairs": 0,
        "final_pairs_strict": 0,
        "final_pairs_relaxed": 0,
        "final_pairs_majority": 0,
        "final_pairs_rough_majority": 0,
        "dimension": dim.name,
        "n_sub_keys": len(dim.sub_keys),
        "max_pairs_per_prompt": max_pairs_per_prompt,
        "min_score_gap": min_score_gap,
    }

    for rec in records:
        stats["loaded"] += 1
        if task_filter and rec.get("task_name") != task_filter:
            stats["skipped_wrong_task"] += 1
            continue
        ed = rec.get("evaluate_data")
        if not isinstance(ed, list):
            stats["skipped_no_evaluate_data"] += 1
            continue

        # Score every entry on this dimension.
        scored: list[dict] = []
        for ent in ed:
            if not isinstance(ent, dict):
                continue
            subs = _dim_sub_scores(ent.get("evaluate_result") or {}, dim)
            if subs is None:
                continue
            url = bvq._video_url(ent, video_source)
            if not url:
                continue
            mean = sum(subs.values()) / len(subs)
            scored.append({
                "method": ent.get("method") or "unknown",
                "url": url,
                "subs": subs,
                "mean": mean,
            })
        if len(scored) < 2:
            stats["skipped_no_two_scored_methods"] += 1
            continue

        # Per-prompt: collect ALL ordered (winner, loser) candidate pairs
        # bucketed by tier (strict / mean / majority / rough_majority),
        # ordered by gap descending. Then keep the top
        # ``max_pairs_per_prompt`` per tier (typically 1; raise for
        # low-coverage dims like logic / rhythm). Tiers are ordered so
        # that the same pair never appears in more than one tier.
        prompt_strict:   list[tuple[float, dict, dict]] = []
        prompt_mean:     list[tuple[float, dict, dict]] = []
        prompt_majority: list[tuple[float, dict, dict]] = []
        prompt_rough:    list[tuple[float, dict, dict]] = []
        for i in range(len(scored)):
            for j in range(len(scored)):
                if i == j:
                    continue
                w, l = scored[i], scored[j]
                gap = w["mean"] - l["mean"]
                if gap <= 0:
                    continue
                if gap < min_score_gap:
                    stats["skipped_below_min_gap"] += 1
                    continue
                if _strictly_dominates(w["subs"], l["subs"], dim.sub_keys):
                    prompt_strict.append((gap, w, l))
                elif _mean_dominates(w["subs"], l["subs"], dim.sub_keys):
                    prompt_mean.append((gap, w, l))
                elif _majority_dominates(w["subs"], l["subs"], dim.sub_keys):
                    prompt_majority.append((gap, w, l))
                elif _rough_majority_dominates(w["subs"], l["subs"], dim.sub_keys):
                    prompt_rough.append((gap, w, l))

        prompt_strict.sort(  key=lambda t: t[0], reverse=True)
        prompt_mean.sort(    key=lambda t: t[0], reverse=True)
        prompt_majority.sort(key=lambda t: t[0], reverse=True)
        prompt_rough.sort(   key=lambda t: t[0], reverse=True)

        prompt_zh = rec.get("instruction_prompt") or rec.get("prompt") or ""
        ref_path = rec.get("path")
        task = rec.get("task_name")

        def _make_pair(w: dict, l: dict, gap: float, kind: str) -> dict:
            return {
                "prompt_zh": prompt_zh,
                "winner_url": w["url"],
                "loser_url": l["url"],
                "winner_method": w["method"],
                "loser_method": l["method"],
                "winner_subs": w["subs"],
                "loser_subs": l["subs"],
                "winner_mean": w["mean"],
                "loser_mean": l["mean"],
                "score_gap": gap,
                "task_name": task,
                "evalverse_ref_path": ref_path,
                "_kind": kind,
            }

        # Within a single prompt, taking K pairs from the same tier may
        # repeat one video. We dedup by ``(winner_url, loser_url)`` set
        # — different orderings of the same two videos still count as
        # one — so a high-arity prompt with one stand-out method doesn't
        # pollute the pool with K near-duplicate entries.
        seen_pair_keys: set[frozenset[str]] = set()

        def _take(pairs_sorted, kind: str, dst: list[dict]) -> int:
            """Append at most ``max_pairs_per_prompt`` deduped pairs of
            ``kind`` from ``pairs_sorted`` into ``dst``. Returns how many
            were appended (used by the kept_prompts_* stats below)."""
            taken = 0
            for gap, w, l in pairs_sorted:
                if taken >= max_pairs_per_prompt:
                    break
                key = frozenset((w["url"], l["url"]))
                if key in seen_pair_keys:
                    continue
                seen_pair_keys.add(key)
                dst.append(_make_pair(w, l, gap, kind))
                taken += 1
            return taken

        n_strict = _take(prompt_strict,   "strict",        strict_pairs)
        n_mean   = _take(prompt_mean,     "relaxed",       mean_pairs)
        n_maj    = _take(prompt_majority, "majority",      majority_pairs)
        n_rough  = _take(prompt_rough,    "rough_majority", rough_majority_pairs)

        if n_strict > 0:
            stats["kept_prompts_strict"] += 1
        elif n_mean > 0:
            stats["kept_prompts_relaxed"] += 1
        elif n_maj > 0:
            stats["kept_prompts_majority"] += 1
        elif n_rough > 0:
            stats["kept_prompts_rough_majority"] += 1
        else:
            stats["skipped_no_dominating_pair"] += 1

    # Shuffle each pool (deterministic) so the chosen ``max_pairs`` are
    # not biased toward a particular source-file order.
    rng.shuffle(strict_pairs)
    rng.shuffle(mean_pairs)
    rng.shuffle(majority_pairs)
    rng.shuffle(rough_majority_pairs)

    # Cross-tier dedup safety net: if max_pairs_per_prompt > 1, the same
    # ``frozenset(winner_url, loser_url)`` cannot land in two tiers
    # within one prompt (the per-prompt set ``seen_pair_keys`` prevents
    # it), but it CAN land in two prompts if the same two videos were
    # judged on different prompts. Keep the first occurrence by tier
    # priority (strict > relaxed > majority > rough).
    cross_seen: set[frozenset[str]] = set()

    def _filter_dedup(pool: list[dict]) -> list[dict]:
        out = []
        for p in pool:
            k = frozenset((p["winner_url"], p["loser_url"]))
            if k in cross_seen:
                continue
            cross_seen.add(k)
            out.append(p)
        return out

    strict_pairs        = _filter_dedup(strict_pairs)
    mean_pairs          = _filter_dedup(mean_pairs)
    majority_pairs      = _filter_dedup(majority_pairs)
    rough_majority_pairs = _filter_dedup(rough_majority_pairs)

    if len(strict_pairs) >= max_pairs:
        chosen = strict_pairs[:max_pairs]
    elif relax_to_mean_if_short:
        # Cascade: strict -> mean -> majority -> rough_majority, in priority order.
        chosen = list(strict_pairs)
        if len(chosen) < max_pairs:
            chosen += mean_pairs[: max_pairs - len(chosen)]
        if len(chosen) < max_pairs:
            chosen += majority_pairs[: max_pairs - len(chosen)]
        if len(chosen) < max_pairs:
            chosen += rough_majority_pairs[: max_pairs - len(chosen)]
    else:
        chosen = strict_pairs

    stats["final_pairs"] = len(chosen)
    stats["final_pairs_strict"]         = sum(1 for p in chosen if p["_kind"] == "strict")
    stats["final_pairs_relaxed"]        = sum(1 for p in chosen if p["_kind"] == "relaxed")
    stats["final_pairs_majority"]       = sum(1 for p in chosen if p["_kind"] == "majority")
    stats["final_pairs_rough_majority"] = sum(1 for p in chosen if p["_kind"] == "rough_majority")
    return chosen, stats


# ---------------------------------------------------------------------------
# v2_prompt_pair JSON assembly.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Pointwise test mining (T2, 2026-05-21).
# ---------------------------------------------------------------------------

def _to_label_score_01_to_15(mean: float) -> float:
    """Map an EvalVerse 0-1 sub-score mean into the 1-5 ARR scale, mirroring
    ``build_visual_quality_mvp_dataset._to_label_score`` so T1 (visual_quality
    shared_test_pointwise.json, label_score=1+4*mean) and T2 (this function's
    output, label_score=1+4*mean) live on the SAME scale and can be plotted /
    compared row-wise without any rescaling.
    """
    return round(max(1.0, min(5.0, 1.0 + 4.0 * float(mean))), 2)


def mine_dim_test_pointwise(
    records: list[dict],
    dim: DimensionSpec,
    task_filter: str | None,
    video_source: str,
    target_count: int,
    seed: int,
    excluded_urls: set[str] | None = None,
    max_per_record: int = 1,
) -> tuple[list[dict], dict]:
    """Mine ``target_count`` independent pointwise test samples for ``dim``.

    Each candidate = one (prompt, generated_video, dimension_sub_score_mean)
    triple drawn from a single ``evaluate_data`` entry. We then 5-band
    stratify on the [0, 1] sub-score mean so the resulting test set spans
    the full 1.0-5.0 label range (matches T1 visual_quality conventions).

    De-dup rules (in priority order):
      * URL must be unique across the test set.
      * URL must NOT appear in ``excluded_urls`` (caller passes the
        v2_prompt_pair winner+loser URLs to keep training and test
        strictly disjoint).
      * At most ``max_per_record`` entries per source record. Default 1
        (legacy: each record contributes a single band-representative
        sample, preventing one high-arity record from dominating). For
        target_count > number_of_multi_method_prompts (typical for our
        16-dim 1k-row test set) raise to 4-5 so a record can contribute
        multiple distinct (method, score) tuples to fill the budget.
    """
    rng = random.Random(seed)
    excluded_urls = set(excluded_urls or set())

    # 5 bands on the [0, 1] mean axis; matches the VQ MVP convention so
    # T1 and T2 share the same band boundaries.
    bands: list[tuple[float, float]] = [
        (0.0, 0.2),    # ~ label [1.0, 1.8)
        (0.2, 0.4),    # ~ label [1.8, 2.6)
        (0.4, 0.6),    # ~ label [2.6, 3.4)
        (0.6, 0.8),    # ~ label [3.4, 4.2)
        (0.8, 1.001),  # ~ label [4.2, 5.0]
    ]
    band_pools: list[list[dict]] = [[] for _ in bands]

    stats: dict = {
        "loaded": 0,
        "skipped_wrong_task": 0,
        "skipped_no_evaluate_data": 0,
        "skipped_no_scored_method": 0,
        "skipped_excluded_url": 0,
        "skipped_no_url": 0,
        "candidates_per_band": [0, 0, 0, 0, 0],
        "final_per_band":      [0, 0, 0, 0, 0],
        "final_count":         0,
        "target_count":        target_count,
        "dimension":           dim.name,
        "n_sub_keys":          len(dim.sub_keys),
    }

    for rec_idx, rec in enumerate(records):
        stats["loaded"] += 1
        if task_filter and rec.get("task_name") != task_filter:
            stats["skipped_wrong_task"] += 1
            continue
        ed = rec.get("evaluate_data")
        if not isinstance(ed, list):
            stats["skipped_no_evaluate_data"] += 1
            continue

        prompt_zh = rec.get("instruction_prompt") or ""
        task_name = rec.get("task_name")
        ref_path  = rec.get("path") or rec.get("path_cos_url") or ""

        any_scored = False
        for ent in ed:
            if not isinstance(ent, dict):
                continue
            subs = _dim_sub_scores(ent.get("evaluate_result") or {}, dim)
            if subs is None:
                continue
            url = bvq._video_url(ent, video_source)
            if not url:
                stats["skipped_no_url"] += 1
                continue
            if url in excluded_urls:
                stats["skipped_excluded_url"] += 1
                continue
            any_scored = True
            mean = sum(subs.values()) / len(subs)
            cand = {
                "rec_idx":  rec_idx,
                "method":   ent.get("method") or "unknown",
                "url":      url,
                "subs":     subs,
                "mean":     mean,
                "prompt_zh": prompt_zh,
                "task_name": task_name,
                "evalverse_ref_path": ref_path,
            }
            for bi, (lo, hi) in enumerate(bands):
                if lo <= mean < hi:
                    band_pools[bi].append(cand)
                    stats["candidates_per_band"][bi] += 1
                    break
        if not any_scored:
            stats["skipped_no_scored_method"] += 1

    # Round-robin draw across bands so the result spans all 5 bands as
    # evenly as feasible. URL uniqueness enforced; further, at most
    # ``max_per_record`` candidates per (rec_idx) -- prevents a single
    # high-arity record from eating the entire band quota when set to 1
    # (legacy), but lets it contribute K samples when target_count
    # exceeds the number of unique multi-method prompts.
    for bp in band_pools:
        rng.shuffle(bp)
    used_urls: set[str] = set()
    rec_counts: dict[int, int] = {}
    chosen: list[dict] = []
    cursors = [0] * len(bands)
    while len(chosen) < target_count:
        progressed = False
        for bi in range(len(bands)):
            if len(chosen) >= target_count:
                break
            while cursors[bi] < len(band_pools[bi]):
                c = band_pools[bi][cursors[bi]]
                cursors[bi] += 1
                if c["url"] in used_urls:
                    continue
                if rec_counts.get(c["rec_idx"], 0) >= max_per_record:
                    continue
                used_urls.add(c["url"])
                rec_counts[c["rec_idx"]] = rec_counts.get(c["rec_idx"], 0) + 1
                chosen.append(c)
                stats["final_per_band"][bi] += 1
                progressed = True
                break
        if not progressed:
            break

    stats["final_count"] = len(chosen)
    return chosen, stats


def build_test_pointwise_payload(
    cands_en: list[dict],
    dim: DimensionSpec,
) -> list[dict]:
    """Assemble the T2 ``test_pointwise_<dim>.json`` payload.

    Schema is identical to the T1 ``shared_test_pointwise.json`` so the same
    pointwise judger can consume it without branching.
    """
    out: list[dict] = []
    for i, c in enumerate(cands_en):
        # No-prompt dimensions: replace the query with the canonical
        # "judge purely from video" placeholder so Gemini gets no hint.
        query = (
            c["prompt_en"] if dim.expose_prompt
            else bvq.NO_PROMPT_QUERY_POINTWISE
        )
        out.append({
            "id":          f"varr_{dim.name}_test_{i:03d}",
            "query":       query,
            "response":    c["url"],
            "label_score": _to_label_score_01_to_15(c["mean"]),
            "dimension":   dim.name,
            "task_type":   bvq._normalize_task_type(c.get("task_name")),
            "metadata": {
                "dimension":         dim.name,
                "evalverse_aspect":  dim.aspect,
                "evalverse_main":    dim.main,
                "evalverse_sub_keys": list(dim.sub_keys),
                "synthesis_mode":    "test_pointwise_dim",
                "variant":           "test_pointwise",
                "method":            c["method"],
                "score":             c["mean"],
                "sub_scores":        c["subs"],
                "task_name":         c.get("task_name"),
                "evalverse_ref_path": c.get("evalverse_ref_path"),
                "prompt_zh":         c.get("prompt_zh", ""),
            },
        })
    return out


def _shared_meta_dim(p: dict, dim: DimensionSpec, mode: str) -> dict:
    if p["_kind"] == "strict":
        synth = "strict_subdim_dominance"
    elif p["_kind"] == "relaxed":
        synth = "relaxed_mean_dominance"
    elif p["_kind"] == "majority":
        synth = "majority_subdim_dominance"
    else:  # rough_majority
        synth = "rough_majority_subdim_dominance"
    return {
        "dimension": dim.name,
        "evalverse_aspect": dim.aspect,
        "evalverse_main": dim.main,
        "evalverse_sub_keys": list(dim.sub_keys),
        "synthesis_mode": synth,
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


def build_v2_prompt_pair(
    pairs_en: list[dict],
    dim: DimensionSpec,
) -> list[dict]:
    """Assemble the listwise v2_prompt_pair payload."""
    out: list[dict] = []
    for i, p in enumerate(pairs_en):
        # No-prompt dimensions: replace the query with the canonical
        # "judge purely from video" placeholder so Gemini doesn't get any
        # textual hint about the intended content.
        query = (
            p["prompt_en"] if dim.expose_prompt
            else bvq.NO_PROMPT_QUERY_PAIRWISE
        )
        out.append({
            "id": f"varr_{dim.name}_{i:03d}",
            "query": query,
            "response": [p["winner_url"], p["loser_url"]],
            "label_rank": [1, 2],
            "task_type": bvq._normalize_task_type(p.get("task_name")),
            "metadata": _shared_meta_dim(p, dim, mode="v2_prompt_pair"),
        })
    return out


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dimension", required=True,
                    help=f"one of: {','.join(list_dimension_names())}")
    ap.add_argument("--mode", default="pair",
                    choices=("pair", "test_pointwise"),
                    help="'pair' (default) emits the v2_prompt_pair listwise "
                         "seed used for rubric generation. 'test_pointwise' "
                         "emits a per-dim 5-band-stratified pointwise test "
                         "set (T2) with label_score = 1+4*sub_mean (matches "
                         "the visual_quality T1 shared_test scale).")
    ap.add_argument("--target_count", type=int, default=30,
                    help="number of test samples for --mode test_pointwise "
                         "(default: 30 to match T1).")
    ap.add_argument("--max_per_record", type=int, default=1,
                    help="--mode test_pointwise: max number of test "
                         "samples mined from a SINGLE source record. "
                         "Default 1 (legacy: prevents one high-arity "
                         "record from dominating). For target_count > "
                         "the number of unique multi-method prompts "
                         "(e.g. 1k-row tests on logic/rhythm where only "
                         "~25 prompts exist), raise to 4-5.")
    ap.add_argument("--exclude_pair_dataset", default=None,
                    help="path to a v2_prompt_pair.json whose winner/loser "
                         "URLs should be EXCLUDED from the test set, to keep "
                         "rubric-generation training and test strictly "
                         "disjoint. Default: data/v_arr_mvp/<dim>_mvp/"
                         "v2_prompt_pair.json (auto-detected).")
    ap.add_argument("--input", action="append", default=None,
                    help="glob pattern(s) for source jsonl. "
                         "Default: merged_evaluate_results_with_path_part*.jsonl")
    ap.add_argument("--output_dir", default=None,
                    help="output directory (default: data/v_arr_mvp/<dim>_mvp)")
    ap.add_argument("--max_pairs", type=int, default=30,
                    help="number of (winner, loser) pairs to emit (default: 30)")
    ap.add_argument("--task_filter", default="",
                    help="optional task_name filter (e.g. text_to_video). "
                         "Default '' = no filter (mirrors VQ MVP behaviour).")
    ap.add_argument("--video_source", default="path",
                    choices=("path", "cos_url"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--relax_if_short", action="store_true", default=True)
    ap.add_argument("--strict_only", dest="relax_if_short", action="store_false",
                    help="abort if fewer than --max_pairs strict pairs exist")
    ap.add_argument("--max_pairs_per_prompt", type=int, default=1,
                    help="max number of (winner, loser) pairs to keep "
                         "FROM A SINGLE prompt. Default 1 (one pair per "
                         "prompt — the visual_quality MVP convention). "
                         "For low-coverage dims like logic / rhythm where "
                         "the number of unique multi-method prompts is < "
                         "max_pairs, raise this to 2-3 so each prompt can "
                         "contribute multiple top-gap pairs (deduped by "
                         "winner/loser URL pair).")
    ap.add_argument("--min_score_gap", type=float, default=0.0,
                    help="hard floor on (winner_mean - loser_mean). 0.0 = "
                         "no floor (legacy). 0.05 keeps the BT signal "
                         "interpretable while still admitting most "
                         "relaxed/majority pairs.")
    ap.add_argument("--translate", action="store_true", default=True)
    ap.add_argument("--no_translate", dest="translate", action="store_false")
    ap.add_argument("--base_url",
                    default=os.environ.get(
                        "OPENAI_BASE_URL",
                        "http://v2.open.venus.oa.com/llmproxy"))
    ap.add_argument("--model", default=os.environ.get("MODEL", "gemini-2.5-pro"))
    ap.add_argument(
        "--shared_translation_cache",
        default="data/v_arr_mvp/visual_quality_mvp/_prompt_translations.json",
        help="reuse the translation cache populated by the VQ MVP. "
             "Cache MISSes will be silently kept in Chinese unless "
             "OPENAI_API_KEY is set.",
    )
    args = ap.parse_args()

    # Resolve dimension.
    dim_name = args.dimension.lower().strip()
    if dim_name not in DIMENSIONS_BY_NAME:
        print(f"[error] unknown dimension {dim_name!r}; valid: "
              f"{list_dimension_names()}", file=sys.stderr)
        sys.exit(2)
    dim: DimensionSpec = DIMENSIONS_BY_NAME[dim_name]

    out_dir = Path(args.output_dir or f"data/v_arr_mvp/{dim.name}_mvp")
    out_dir.mkdir(parents=True, exist_ok=True)

    input_patterns = args.input or ["merged_evaluate_results_with_path_part*.jsonl"]
    input_paths = bvq._resolve_input_paths(input_patterns)
    if not input_paths:
        print("[error] no input jsonl files resolved.", file=sys.stderr)
        sys.exit(2)

    records, per_file = bvq._load_jsonl_many(input_paths)
    print(f"  total records loaded                : {len(records)}")

    # ----- T2 mode: emit per-dim test_pointwise instead of v2_prompt_pair -----
    if args.mode == "test_pointwise":
        excluded_urls: set[str] = set()
        exclude_path = args.exclude_pair_dataset or str(out_dir / "v2_prompt_pair.json")
        if Path(exclude_path).is_file():
            try:
                pair_payload = json.loads(Path(exclude_path).read_text(encoding="utf-8"))
                for it in pair_payload:
                    resp = it.get("response")
                    if isinstance(resp, list):
                        for u in resp:
                            if isinstance(u, str):
                                excluded_urls.add(u)
                    elif isinstance(resp, str):
                        excluded_urls.add(resp)
                print(f"[1/3] excluding {len(excluded_urls)} URLs from "
                      f"{exclude_path} to keep T2 disjoint from training pairs")
            except Exception as e:
                print(f"[warn] failed to parse {exclude_path}: {e}; "
                      "proceeding without URL exclusion.")
        else:
            print(f"[1/3] no pair-dataset to exclude (looked at {exclude_path}); "
                  "T2 will be mined from the full pool.")

        cands, stats = mine_dim_test_pointwise(
            records=records,
            dim=dim,
            task_filter=args.task_filter or None,
            video_source=args.video_source,
            target_count=args.target_count,
            seed=args.seed,
            excluded_urls=excluded_urls,
            max_per_record=args.max_per_record,
        )
        for k, v in stats.items():
            print(f"  {k:<30s}: {v}")
        if not cands:
            print(f"[error] no {dim.name} test_pointwise candidates survived "
                  "filtering; aborting.", file=sys.stderr)
            sys.exit(3)

        # Translate prompts for prompt-exposing dims; placeholder otherwise.
        print(f"[2/3] translating {len(cands)} prompts (translate={args.translate}; "
              f"expose_prompt={dim.expose_prompt})")
        if args.translate and dim.expose_prompt:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                print("[info] OPENAI_API_KEY not set; running in CACHE-ONLY mode.",
                      file=sys.stderr)
            cache_path = Path(args.shared_translation_cache)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            en_prompts = bvq.translate_prompts(
                [c["prompt_zh"] for c in cands],
                base_url=args.base_url,
                model=args.model,
                api_key=api_key,
                cache_path=cache_path,
            )
        else:
            # Even for non-expose dims we keep prompt_en synced (stored in
            # metadata only, so downstream debugging still has English text).
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if api_key and dim.expose_prompt is False:
                # Best-effort translate for metadata only; cache misses fall
                # back to the Chinese original.
                pass
            en_prompts = [c.get("prompt_zh", "") for c in cands]

        cands_en: list[dict] = []
        for c, en in zip(cands, en_prompts):
            ce = dict(c)
            ce["prompt_en"] = en
            cands_en.append(ce)

        print(f"[3/3] writing test_pointwise_{dim.name}.json into {out_dir}")
        payload = build_test_pointwise_payload(cands_en, dim)
        out_path = out_dir / f"test_pointwise_{dim.name}.json"
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {len(payload):>3d} samples -> {out_path}")

        # T2 manifest.
        manifest = {
            "dimension":         dim.name,
            "evalverse_aspect":  dim.aspect,
            "evalverse_main":    dim.main,
            "evalverse_sub_keys": list(dim.sub_keys),
            "expose_prompt":     dim.expose_prompt,
            "needs_audio":       dim.needs_audio,
            "mode":              "test_pointwise",
            "input_patterns":    input_patterns,
            "input_paths":       input_paths,
            "target_count":      args.target_count,
            "task_filter":       args.task_filter,
            "video_source":      args.video_source,
            "translate":         args.translate,
            "shared_translation_cache": str(args.shared_translation_cache),
            "model":             args.model,
            "seed":              args.seed,
            "exclude_pair_dataset": exclude_path,
            "excluded_urls":     len(excluded_urls),
            "stats":             stats,
        }
        (out_dir / f"test_pointwise_{dim.name}.manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"  wrote manifest -> {out_dir/f'test_pointwise_{dim.name}.manifest.json'}")
        print("Done.")
        return

    # ----- Default mode: v2_prompt_pair (rubric-generation seed) -----
    print(f"[1/3] mining {dim.name} pairs from {len(input_paths)} file(s); "
          f"sub_keys={len(dim.sub_keys)}; expose_prompt={dim.expose_prompt}; "
          f"needs_audio={dim.needs_audio}")
    pairs, stats = mine_dim_pairs(
        records=records,
        dim=dim,
        task_filter=args.task_filter or None,
        video_source=args.video_source,
        max_pairs=args.max_pairs,
        seed=args.seed,
        relax_to_mean_if_short=args.relax_if_short,
        max_pairs_per_prompt=args.max_pairs_per_prompt,
        min_score_gap=args.min_score_gap,
    )
    for k, v in stats.items():
        print(f"  {k:<36s}: {v}")
    if not pairs:
        print(f"[error] no {dim.name} pairs survived filtering; aborting.",
              file=sys.stderr)
        sys.exit(3)

    # ----- Translation (cache-only when no API key) -----
    print(f"[2/3] translating {len(pairs)} prompts (translate={args.translate}; "
          f"shared cache={args.shared_translation_cache})")
    if args.translate:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("[info] OPENAI_API_KEY not set; running in CACHE-ONLY mode.",
                  file=sys.stderr)
        cache_path = Path(args.shared_translation_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        en_pairs = bvq.translate_prompts(
            [p["prompt_zh"] for p in pairs],
            base_url=args.base_url,
            model=args.model,
            api_key=api_key,
            cache_path=cache_path,
        )
    else:
        en_pairs = [p["prompt_zh"] for p in pairs]

    pairs_en: list[dict] = []
    for p, en in zip(pairs, en_pairs):
        pe = dict(p)
        pe["prompt_en"] = en
        pairs_en.append(pe)

    # ----- Emit v2_prompt_pair.json -----
    print(f"[3/3] writing v2_prompt_pair.json into {out_dir}")
    payload = build_v2_prompt_pair(pairs_en, dim)
    out_path = out_dir / "v2_prompt_pair.json"
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  wrote {len(payload):>3d} samples -> {out_path}")

    # Manifest with provenance + sanity stats.
    manifest = {
        "dimension": dim.name,
        "evalverse_aspect": dim.aspect,
        "evalverse_main": dim.main,
        "evalverse_sub_keys": list(dim.sub_keys),
        "expose_prompt": dim.expose_prompt,
        "needs_audio": dim.needs_audio,
        "input_patterns": input_patterns,
        "input_paths": input_paths,
        "input_records_per_file": per_file,
        "max_pairs": args.max_pairs,
        "max_pairs_per_prompt": args.max_pairs_per_prompt,
        "min_score_gap": args.min_score_gap,
        "relax_if_short": args.relax_if_short,
        "task_filter": args.task_filter,
        "video_source": args.video_source,
        "translate": args.translate,
        "shared_translation_cache": str(args.shared_translation_cache),
        "model": args.model,
        "seed": args.seed,
        "stats": stats,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  wrote manifest -> {out_dir/'manifest.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
