#!/usr/bin/env python3
"""RewardVerse MVP baseline: pairwise zero-shot video judge (NO rubric).

Used to answer RQ4: does Auto-Rubric beat a naive zero-shot pairwise judge?

Usage:
    python scripts/v_arr_mvp/zero_shot_judge.py \
        --test_dataset examples/v_arr_mvp_test_pairwise.json \
        --output_json /tmp/v_arr_mvp_zero_shot.json \
        --base_url https://generativelanguage.googleapis.com/v1beta/openai/ \
        --model gemini-2.5-pro
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rubric_pipeline.models.openai_chat_model import OpenAIChatModel
from rubric_pipeline.models.schema.oai.message import ChatMessage
from rubric_pipeline.utils.media import (
    add_videos_to_messages,
    extract_video_paths,
    is_video_task,
)


_ZERO_SHOT_PROMPT = """
You are an impartial video-quality judge. You will see exactly two generated
video clips for the same text prompt, labelled Video 1 and Video 2 (in the
original order they were provided).

Pick which video is overall better, considering both appearance fidelity to the
prompt and motion plausibility. Do NOT inspect or invent any taxonomy; answer
purely from what you observe.

Prompt: {query}

Return STRICT JSON in this exact shape:
{{"rank": [r_for_video_1, r_for_video_2],
  "reason": "<one short sentence>"}}

Rank values must be 1 and 2 with no ties. rank=1 means better, rank=2 means worse.
""".strip()


def _parse_rank(raw: str) -> list[int] | None:
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    rank = parsed.get("rank")
    if not isinstance(rank, list) or len(rank) != 2:
        return None
    try:
        rank = [int(x) for x in rank]
    except (TypeError, ValueError):
        return None
    if sorted(rank) != [1, 2]:
        return None
    return rank


async def judge_one(
    model: OpenAIChatModel,
    item: dict,
    video_strategy: str,
) -> dict:
    query = item.get("query") or item.get("prompt") or ""
    sample_id = item.get("id") or (item.get("metadata") or {}).get("id")
    paths = extract_video_paths(item, item.get("task_type", "t2v")) if is_video_task(item.get("task_type", "t2v")) else []
    if not paths and isinstance(item.get("response"), list):
        paths = [str(p) for p in item["response"]]
    if len(paths) != 2:
        return {
            "id": sample_id,
            "error": (
                f"expected exactly 2 video paths, got {len(paths)} "
                "(this dataset is likely pointwise; use pointwise_zero_shot_judge.py instead)"
            ),
        }

    user_text = _ZERO_SHOT_PROMPT.format(query=query)
    messages = [
        {"role": "system", "content": "Output strictly valid JSON. No commentary outside JSON."},
        {"role": "user", "content": user_text},
    ]
    messages = add_videos_to_messages(
        messages,
        paths,
        item.get("task_type", "t2v"),
        has_base=False,
        strategy=video_strategy,  # type: ignore[arg-type]
    )
    try:
        response = await model.achat(messages=messages)
    except Exception as exc:
        return {"id": sample_id, "error": str(exc)}

    raw = getattr(response, "content", "") or (response.get("content", "") if isinstance(response, dict) else "")
    rank = _parse_rank(raw)
    return {
        "id": sample_id,
        "query": query,
        "pred_rank": rank,
        "label_rank": item.get("label_rank"),
        "raw": raw,
    }


async def main_async(args: argparse.Namespace) -> None:
    samples = json.loads(Path(args.test_dataset).read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        samples = [samples]

    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = OpenAIChatModel(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        temperature=0,
    )

    semaphore = asyncio.Semaphore(args.concurrency_limit)
    results: list[dict] = []

    async def _one(item: dict) -> None:
        async with semaphore:
            result = await judge_one(model, item, args.video_strategy)
            results.append(result)

    await asyncio.gather(*[_one(s) for s in samples])

    correct = 0
    parseable = 0
    for r in results:
        if r.get("pred_rank") is None:
            continue
        parseable += 1
        if r["pred_rank"] == r["label_rank"]:
            correct += 1

    report = {
        "test_dataset": args.test_dataset,
        "model": args.model,
        "total": len(samples),
        "parseable": parseable,
        "correct": correct,
        "accuracy": correct / parseable if parseable else 0.0,
        "raw_accuracy": correct / len(samples) if samples else 0.0,
        "predictions": results,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Zero-shot judge: {correct}/{parseable} parseable ({report['accuracy']:.2%}). "
        f"Raw acc (over total): {report['raw_accuracy']:.2%}",
    )
    print(f"Wrote report to {args.output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RewardVerse MVP zero-shot pairwise baseline.")
    parser.add_argument("--test_dataset", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument(
        "--base_url",
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    # video_strategy=None means: fall back to the ARR_VIDEO_INPUT_STRATEGY env
    # variable (defaults to native_video + venus upload, matching the rubric
    # judger). Pass --video_strategy uniform_frames only if you have local mp4
    # files that decord can read.
    parser.add_argument("--video_strategy", default=None)
    parser.add_argument("--concurrency_limit", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
