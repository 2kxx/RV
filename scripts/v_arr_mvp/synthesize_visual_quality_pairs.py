#!/usr/bin/env python3
"""Synthesize Visual Quality preference pairs from a per-video JSONL.

Supports both upstream schemas:

  (A) "nested" (EvalVerse merged_output_slim_with_new_path.jsonl):

      {"input":  {"instruction_prompt": "...", "task_name": "t2v",
                  "ref_type": "...", "path": "...", "path_cos_url": "...",
                  "ref_cos_urls": [...], "multi_shot_prompt_details": [...]},
       "output": [{"method": "seedance-2.0", "path": "...", "cos_url": "..."},
                  {"method": "univideo",     "path": "...", "cos_url": "..."},
                  ...]}

  (B) "evaluate_data" (real EvalVerse human-MoS annotations, current schema
      for merged_evaluate_results_with_path_part*.jsonl):

      {"path": "<ref>.mp4", "path_cos_url": "...", "task_name": "t2v",
       "instruction_prompt": "...",
       "evaluate_data": [
         {"method": "ltx2", "path": "...", "cos_url": "...",
          "evaluate_result": {
            "Aesthetics-Visual_Quality-Physics":              {"average_score": 0.96, ...},
            "Aesthetics-Visual_Quality-Rendering_Quality":    {"average_score": 0.90, ...},
            "Aesthetics-Visual_Quality-Temporal_Consistency": {"average_score": 0.70, ...},
            ...
          }}, ...
       ]}

  (C) "flat" (one generated video per line, with per-video human VQ score):

      {"prompt": "...", "video_path": "...", "human_dims": {"Visual Quality": 4},
       ...}

Pair selection modes (`--mode`):

  - `auto` (default)
      Inspect the first record; pick `mos_aesthetics_vq` for schema B,
      `method_rank` for schema A, `score` for schema C.

  - `mos_aesthetics_vq`
      For schema (B): per video, the Visual Quality MoS is the mean of the
      three EvalVerse sub-dim `average_score` values
      (`Aesthetics-Visual_Quality-Physics`,
      `Aesthetics-Visual_Quality-Rendering_Quality`,
      `Aesthetics-Visual_Quality-Temporal_Consistency`), skipping nulls. Within
      each record, pair the highest-MoS method (Vw) against the lowest-MoS
      method (Vl), gated by `--min_score_gap` (default 0.10 on the 0-1 scale).

  - `method_rank`
      For schema (A): use the EvalVerse Tab. 3 Visual Quality win-ratios as a
      method-quality table. Within each record, pick the highest-rank method
      as y+ and the lowest-rank as y-, gated by `--min_method_gap` on the
      win-ratio difference (default 0.20).

  - `score`
      For schema (C): legacy path. Group by prompt, pair best vs worst
      per-video VQ score, gated by `--min_score_gap` (default 1.0 on 1-5).

Output: ARR canonical pairwise JSON ready for
`learn_visual_quality_rubric_variants.py`.

Usage:
    # Real EvalVerse JSONL, method-rank mode, T2V only, 200 pairs.
    python scripts/v_arr_mvp/synthesize_visual_quality_pairs.py \
        --input merged_output_slim_with_new_path.jsonl \
        --output data/v_arr_mvp/visual_quality_pairs.json \
        --mode method_rank \
        --task_name_filter t2v \
        --max_total_pairs 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# EvalVerse Tab. 3 Visual Quality win-ratios (SIGGRAPH Asia 2026 submission).
# Higher = better Visual Quality across the benchmark. Methods are matched
# case-insensitively after lowercase + dash/underscore normalisation.
# ---------------------------------------------------------------------------

_VQ_RANK_TABLE: dict[str, float] = {
    # name (normalised)         win-ratio (Visual Quality row)
    "seedance-2.0":                       0.71,
    "happy_horse_1.0":                    0.68,
    "ltx2":                               0.67,
    "kling-v3-omni":                      0.66,
    "hailuo-2.3":                         0.53,
    "hunyuan1.5":                         0.53,
    "holocine_full":                      0.44,
    "wan2.2_14b":                         0.33,
    "viduq2-pro":                         0.31,
    "univideo":                           0.28,
    "multishot_master_14b":               0.05,
}

# Common spelling aliases observed in the JSONL.
_METHOD_ALIASES: dict[str, str] = {
    "seedance2.0": "seedance-2.0",
    "seedance_2.0": "seedance-2.0",
    "happy-horse-1.0": "happy_horse_1.0",
    "happy_horse": "happy_horse_1.0",
    "kling_v3_omni": "kling-v3-omni",
    "kling": "kling-v3-omni",
    "hailuo2.3": "hailuo-2.3",
    "hunyuan-1.5": "hunyuan1.5",
    "wan2.2": "wan2.2_14b",
    "wan2.2-14b": "wan2.2_14b",
    "vidu-q2-pro": "viduq2-pro",
    "vidu_q2_pro": "viduq2-pro",
    "multishot-master-14b": "multishot_master_14b",
    "multishotmaster": "multishot_master_14b",
}


def _norm_method(name: str | None) -> str:
    if not name:
        return ""
    key = name.strip().lower().replace(" ", "_")
    key = _METHOD_ALIASES.get(key, key)
    return key


# ---------------------------------------------------------------------------
# Flat-schema field aliases (preserved for backward compatibility).
# ---------------------------------------------------------------------------

PROMPT_KEYS = ("prompt", "caption", "text", "query", "prompt_text", "instruction_prompt")
VIDEO_KEYS = ("new_path", "video_path", "video", "path", "url")
SCORE_KEYS_DIRECT = (
    "visual_quality",
    "vq_score",
    "quality_score",
    "visual_quality_score",
)
DIM_LABELS = ("Visual Quality", "visual_quality", "visual quality", "VQ", "vq")


def _first_non_empty(d: dict, keys: tuple[str, ...]) -> Any:
    for k in keys:
        v = d.get(k)
        if v is not None and v != "":
            return v
    return None


def _extract_vq_score(record: dict) -> float | None:
    for k in SCORE_KEYS_DIRECT:
        if isinstance(record.get(k), (int, float)):
            return float(record[k])
    for container_key in ("human_dims", "dimension_scores", "scores", "dim_scores"):
        sub = record.get(container_key)
        if isinstance(sub, dict):
            for label in DIM_LABELS:
                if isinstance(sub.get(label), (int, float)):
                    return float(sub[label])
            for k, v in sub.items():
                if str(k).strip().lower().replace(" ", "_") == "visual_quality" and isinstance(
                    v, (int, float),
                ):
                    return float(v)
    return None


# ---------------------------------------------------------------------------
# Loading.
# ---------------------------------------------------------------------------


def _load_records(path: str) -> list[dict]:
    records: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] skipping malformed line {line_no}: {exc}", file=sys.stderr)
    return records


def _is_method_rank_schema(record: dict) -> bool:
    return isinstance(record.get("input"), dict) and isinstance(record.get("output"), list)


def _is_evaluate_data_schema(record: dict) -> bool:
    return (
        isinstance(record.get("evaluate_data"), list)
        and "evaluate_data" in record
        and any(isinstance(ent, dict) and "evaluate_result" in ent for ent in record["evaluate_data"])
    )


def _detect_mode(records: list[dict]) -> str:
    head = records[0] if records else {}
    if _is_evaluate_data_schema(head):
        return "mos_aesthetics_vq"
    if _is_method_rank_schema(head):
        return "method_rank"
    return "score"


# ---------------------------------------------------------------------------
# Visual Quality MoS extraction (EvalVerse Aesthetics > Visual Quality).
# ---------------------------------------------------------------------------

VQ_SUB_DIM_KEYS: tuple[str, ...] = (
    "Aesthetics-Visual_Quality-Physics",
    "Aesthetics-Visual_Quality-Rendering_Quality",
    "Aesthetics-Visual_Quality-Temporal_Consistency",
)


def _visual_quality_mos(evaluate_result: dict) -> tuple[float | None, dict[str, float | None]]:
    """Mean of the three Aesthetics > Visual Quality sub-dim `average_score` values.

    Sub-dims with `average_score` of `null` (e.g. all-`-1` `scores_list`) are
    skipped. Returns (mean_or_None, {sub_dim: score}).
    """
    sub_scores: dict[str, float | None] = {}
    vals: list[float] = []
    for k in VQ_SUB_DIM_KEYS:
        cell = evaluate_result.get(k)
        if isinstance(cell, dict):
            v = cell.get("average_score")
            if isinstance(v, (int, float)):
                sub_scores[k] = float(v)
                vals.append(float(v))
                continue
        sub_scores[k] = None
    if not vals:
        return None, sub_scores
    return sum(vals) / len(vals), sub_scores


# ---------------------------------------------------------------------------
# Method-rank mode (nested schema).
# ---------------------------------------------------------------------------


def _output_video_path(out: dict, video_source: str) -> str | None:
    if video_source == "cos_url":
        return out.get("cos_url") or out.get("path")
    # default: prefer local path; fall back to cos_url
    return out.get("path") or out.get("cos_url")


def _mine_method_rank_pairs(
    records: list[dict],
    task_name_filter: str | None,
    min_method_gap: float,
    max_pairs_per_record: int,
    max_total_pairs: int,
    video_source: str,
    seed: int,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    pairs: list[dict] = []
    stats = {
        "loaded": len(records),
        "nested_records": 0,
        "skipped_wrong_task": 0,
        "skipped_lt_two_methods": 0,
        "skipped_unknown_methods": 0,
        "skipped_no_gap": 0,
        "skipped_no_video_path": 0,
        "records_used": 0,
        "final_pairs": 0,
    }

    for rec in records:
        if not _is_method_rank_schema(rec):
            continue
        stats["nested_records"] += 1
        inp = rec["input"]
        outs = rec.get("output", []) or []
        task = inp.get("task_name")
        if task_name_filter and task != task_name_filter:
            stats["skipped_wrong_task"] += 1
            continue
        if len(outs) < 2:
            stats["skipped_lt_two_methods"] += 1
            continue

        ranked: list[tuple[float, dict]] = []
        for o in outs:
            method = _norm_method(o.get("method"))
            wr = _VQ_RANK_TABLE.get(method)
            if wr is None:
                continue
            ranked.append((wr, {"raw": o, "method": method}))
        if len(ranked) < 2:
            stats["skipped_unknown_methods"] += 1
            continue
        ranked.sort(key=lambda x: x[0])
        worst_wr, worst = ranked[0]
        best_wr, best = ranked[-1]
        if best_wr - worst_wr < min_method_gap:
            stats["skipped_no_gap"] += 1
            continue
        win_path = _output_video_path(best["raw"], video_source)
        lose_path = _output_video_path(worst["raw"], video_source)
        if not win_path or not lose_path:
            stats["skipped_no_video_path"] += 1
            continue

        prompt = inp.get("instruction_prompt") or inp.get("prompt") or ""
        chosen_pairs = [
            {
                "win": best,
                "win_wr": best_wr,
                "lose": worst,
                "lose_wr": worst_wr,
            }
        ]
        if max_pairs_per_record > 1 and len(ranked) >= 4:
            mid_best_wr, mid_best = ranked[-2]
            mid_worst_wr, mid_worst = ranked[1]
            if mid_best_wr - mid_worst_wr >= min_method_gap and len(chosen_pairs) < max_pairs_per_record:
                chosen_pairs.append(
                    {
                        "win": mid_best,
                        "win_wr": mid_best_wr,
                        "lose": mid_worst,
                        "lose_wr": mid_worst_wr,
                    },
                )

        for choice in chosen_pairs:
            wp = _output_video_path(choice["win"]["raw"], video_source)
            lp = _output_video_path(choice["lose"]["raw"], video_source)
            if not wp or not lp:
                continue
            pairs.append(
                {
                    "query": prompt,
                    "response": [wp, lp],
                    "label_rank": [1, 2],
                    "task_type": "t2v" if task == "t2v" else "r2v",
                    "metadata": {
                        "dimension": "visual_quality",
                        "synthesis_mode": "method_rank",
                        "winner_method": choice["win"]["method"],
                        "loser_method": choice["lose"]["method"],
                        "winner_score": float(choice["win_wr"]),
                        "loser_score": float(choice["lose_wr"]),
                        "score_gap": float(choice["win_wr"] - choice["lose_wr"]),
                        "task_name": task,
                        "ref_type": inp.get("ref_type"),
                        "evalverse_input_path": inp.get("path"),
                    },
                },
            )
        stats["records_used"] += 1

    rng.shuffle(pairs)
    pairs = pairs[:max_total_pairs]
    stats["final_pairs"] = len(pairs)
    return pairs, stats


# ---------------------------------------------------------------------------
# Score mode (flat schema, legacy path).
# ---------------------------------------------------------------------------


def _mine_score_pairs(
    records: list[dict],
    min_score_gap: float,
    max_pairs_per_prompt: int,
    max_total_pairs: int,
    seed: int,
) -> tuple[list[dict], dict]:
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    stats = {
        "loaded": len(records),
        "skipped_no_prompt": 0,
        "skipped_no_video": 0,
        "skipped_no_score": 0,
        "unique_prompts": 0,
        "prompts_no_gap": 0,
        "prompts_used": 0,
        "final_pairs": 0,
    }
    for r in records:
        if _is_method_rank_schema(r) or _is_evaluate_data_schema(r):
            stats["skipped_no_score"] += 1
            continue
        prompt = _first_non_empty(r, PROMPT_KEYS)
        video = _first_non_empty(r, VIDEO_KEYS)
        score = _extract_vq_score(r)
        if not prompt:
            stats["skipped_no_prompt"] += 1
            continue
        if not video:
            stats["skipped_no_video"] += 1
            continue
        if score is None:
            stats["skipped_no_score"] += 1
            continue
        by_prompt[str(prompt)].append({"video": str(video), "score": float(score), "raw": r})

    stats["unique_prompts"] = len(by_prompt)
    rng = random.Random(seed)
    pairs: list[dict] = []
    for prompt, items in by_prompt.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda x: x["score"])
        worst, best = items[0], items[-1]
        if best["score"] - worst["score"] < min_score_gap:
            stats["prompts_no_gap"] += 1
            continue
        chosen = [(best, worst)]
        if max_pairs_per_prompt > 1 and len(items) >= 4:
            mid_best = items[-2]
            mid_worst = items[1]
            if mid_best["score"] - mid_worst["score"] >= min_score_gap:
                chosen.append((mid_best, mid_worst))
                chosen = chosen[:max_pairs_per_prompt]
        for win, lose in chosen:
            pairs.append(
                {
                    "query": prompt,
                    "response": [win["video"], lose["video"]],
                    "label_rank": [1, 2],
                    "task_type": "t2v",
                    "metadata": {
                        "dimension": "visual_quality",
                        "synthesis_mode": "score",
                        "winner_score": win["score"],
                        "loser_score": lose["score"],
                        "score_gap": win["score"] - lose["score"],
                    },
                },
            )
        stats["prompts_used"] += 1
    rng.shuffle(pairs)
    pairs = pairs[:max_total_pairs]
    stats["final_pairs"] = len(pairs)
    return pairs, stats


# ---------------------------------------------------------------------------
# MoS mode (EvalVerse human-annotation schema, evaluate_data).
# ---------------------------------------------------------------------------


def _mine_mos_aesthetics_vq_pairs(
    records: list[dict],
    task_name_filter: str | None,
    min_score_gap: float,
    max_pairs_per_record: int,
    max_total_pairs: int,
    video_source: str,
    seed: int,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    pairs: list[dict] = []
    stats = {
        "loaded": len(records),
        "evaluate_data_records": 0,
        "skipped_wrong_task": 0,
        "skipped_lt_two_methods_with_scores": 0,
        "skipped_no_gap": 0,
        "skipped_no_video_path": 0,
        "records_used": 0,
        "final_pairs": 0,
    }

    for rec in records:
        if not _is_evaluate_data_schema(rec):
            continue
        stats["evaluate_data_records"] += 1
        task = rec.get("task_name")
        if task_name_filter and task != task_name_filter:
            stats["skipped_wrong_task"] += 1
            continue

        scored: list[dict] = []
        for ent in rec.get("evaluate_data", []) or []:
            mos, sub_scores = _visual_quality_mos(ent.get("evaluate_result", {}) or {})
            if mos is None:
                continue
            video_path = ent.get("path") if video_source == "path" else ent.get("cos_url")
            if not video_path:
                video_path = ent.get("path") or ent.get("cos_url")
            if not video_path:
                continue
            scored.append(
                {
                    "method": ent.get("method") or "unknown",
                    "path": video_path,
                    "mos": float(mos),
                    "sub_scores": sub_scores,
                    "raw": ent,
                },
            )

        if len(scored) < 2:
            stats["skipped_lt_two_methods_with_scores"] += 1
            continue
        scored.sort(key=lambda x: x["mos"])
        worst = scored[0]
        best = scored[-1]
        if best["mos"] - worst["mos"] < min_score_gap:
            stats["skipped_no_gap"] += 1
            continue

        chosen_pairs: list[tuple[dict, dict]] = [(best, worst)]
        if max_pairs_per_record > 1 and len(scored) >= 4:
            mid_worst = scored[1]
            mid_best = scored[-2]
            if mid_best["mos"] - mid_worst["mos"] >= min_score_gap:
                chosen_pairs.append((mid_best, mid_worst))
                chosen_pairs = chosen_pairs[:max_pairs_per_record]

        prompt = rec.get("instruction_prompt") or rec.get("prompt") or ""
        for win, lose in chosen_pairs:
            pairs.append(
                {
                    "query": prompt,
                    "response": [win["path"], lose["path"]],
                    "label_rank": [1, 2],
                    "task_type": "t2v",  # rubric learning is task-agnostic for VQ
                    "metadata": {
                        "dimension": "visual_quality",
                        "synthesis_mode": "mos_aesthetics_vq",
                        "winner_method": win["method"],
                        "loser_method": lose["method"],
                        "winner_score": win["mos"],
                        "loser_score": lose["mos"],
                        "score_gap": win["mos"] - lose["mos"],
                        "winner_sub_scores": win["sub_scores"],
                        "loser_sub_scores": lose["sub_scores"],
                        "task_name": task,
                        "evalverse_ref_path": rec.get("path"),
                    },
                },
            )
        stats["records_used"] += 1

    rng.shuffle(pairs)
    pairs = pairs[:max_total_pairs]
    stats["final_pairs"] = len(pairs)
    return pairs, stats


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine Visual Quality preference pairs (auto-detect schema; MoS / method-rank / score).",
    )
    parser.add_argument("--input", required=True,
                        help="JSONL file (e.g. merged_evaluate_results_with_path_part1.jsonl).")
    parser.add_argument("--output", required=True,
                        help="Output ARR canonical pairwise JSON.")
    parser.add_argument("--mode", default="auto",
                        choices=("auto", "mos_aesthetics_vq", "method_rank", "score"),
                        help="Pair selection mode. 'auto' inspects the first record.")
    parser.add_argument("--task_name_filter", default="",
                        help="Only mine pairs where task_name matches (set to empty = no filter).")
    parser.add_argument("--min_score_gap", type=float, default=0.10,
                        help="(mos_aesthetics_vq / score) Minimum Vw - Vl score gap (default 0.10).")
    parser.add_argument("--min_method_gap", type=float, default=0.20,
                        help="(method_rank) Minimum win-ratio gap between Vw and Vl methods.")
    parser.add_argument("--max_pairs_per_record", type=int, default=2,
                        help="Up to N pairs per source record (default 2 for MoS / method-rank).")
    parser.add_argument("--max_pairs_per_prompt", type=int, default=2,
                        help="(score) Up to N pairs per prompt group.")
    parser.add_argument("--max_total_pairs", type=int, default=30,
                        help="Cap on the synthesized dataset size (default 30).")
    parser.add_argument("--video_source", default="path",
                        choices=("path", "cos_url"),
                        help="Use local 'path' or remote 'cos_url' as the video URL.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for the global shuffle.")
    args = parser.parse_args()

    records = _load_records(args.input)
    mode = args.mode if args.mode != "auto" else _detect_mode(records)

    if mode == "mos_aesthetics_vq":
        pairs, stats = _mine_mos_aesthetics_vq_pairs(
            records,
            args.task_name_filter or None,
            args.min_score_gap,
            args.max_pairs_per_record,
            args.max_total_pairs,
            args.video_source,
            args.seed,
        )
    elif mode == "method_rank":
        pairs, stats = _mine_method_rank_pairs(
            records,
            args.task_name_filter or None,
            args.min_method_gap,
            args.max_pairs_per_record,
            args.max_total_pairs,
            args.video_source,
            args.seed,
        )
    else:
        pairs, stats = _mine_score_pairs(
            records,
            args.min_score_gap,
            args.max_pairs_per_prompt,
            args.max_total_pairs,
            args.seed,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    stats_path = output_path.with_suffix(".stats.json")
    stats_payload = {"mode": mode, **stats}
    stats_path.write_text(
        json.dumps(stats_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Mode     : {mode}{' (auto-detected)' if args.mode == 'auto' else ''}")
    for k, v in stats.items():
        print(f"  {k:<36s}: {v}")
    print(f"Wrote    : {output_path}")
    print(f"Stats    : {stats_path}")


if __name__ == "__main__":
    main()
