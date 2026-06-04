#!/usr/bin/env python3
"""RewardVerse MVP baseline: pointwise zero-shot video judge (NO rubric).

Counterpart of `zero_shot_judge.py` for pointwise datasets, used by variants
v1_prompt_ywin and v3_ywin_only where each sample contains a *single* video and
a `label_score` (1-5).

Usage:
    python scripts/v_arr_mvp/pointwise_zero_shot_judge.py \
        --test_dataset data/v_arr_mvp/visual_quality_mvp/v1_prompt_ywin.json \
        --output_json runs/v_arr_mvp/v1_prompt_ywin/zero_shot.json \
        --base_url http://v2.open.venus.oa.com/llmproxy \
        --model gemini-3.1-pro
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
from rubric_pipeline.utils.media import (
    add_videos_to_messages,
    extract_video_paths,
    is_video_task,
)

def _ranks_with_ties(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks

def _compute_metrics(preds, labels):
    if not preds or not labels or len(preds) != len(labels):
        return None, None, None
    n = len(preds)
    mae = sum(abs(p - l) for p, l in zip(preds, labels)) / n
    plcc = None
    if n >= 2:
        mp = sum(preds) / n; ml = sum(labels) / n
        num = sum((p - mp) * (l - ml) for p, l in zip(preds, labels))
        dp = sum((p - mp) ** 2 for p in preds) ** 0.5
        dl = sum((l - ml) ** 2 for l in labels) ** 0.5
        if dp > 1e-12 and dl > 1e-12:
            plcc = num / (dp * dl)
    srcc = None
    if n >= 2:
        rp = _ranks_with_ties(preds); rl = _ranks_with_ties(labels)
        mp = sum(rp) / n; ml = sum(rl) / n
        num = sum((a - mp) * (b - ml) for a, b in zip(rp, rl))
        dp = sum((a - mp) ** 2 for a in rp) ** 0.5
        dl = sum((b - ml) ** 2 for b in rl) ** 0.5
        if dp > 1e-12 and dl > 1e-12:
            srcc = num / (dp * dl)
    return plcc, srcc, mae

_ZERO_SHOT_PROMPT = """
You are an impartial video-quality judge. You will see exactly ONE generated
video clip for the text prompt below. Score the visual quality on a
continuous 1.0-5.0 scale considering rendering quality, physics plausibility,
and temporal consistency. Do NOT inspect or invent any taxonomy; answer purely
from what you observe.

Scoring rubric (use the WHOLE 1.0-5.0 range, do not bias to 3.0; one decimal
place is fine, e.g. 4.3 / 2.7):
  5.0 = excellent overall visual quality, no obvious artifacts
  4.0 = good, minor issues only
  3.0 = average, noticeable but acceptable issues
  2.0 = poor, multiple obvious issues
  1.0 = very poor, severe artifacts / broken physics / heavy flicker

Prompt: {query}

Return STRICT JSON in this exact shape:
{{"score": <float in [1.0, 5.0]>, "reason": "<one short sentence>"}}
""".strip()

def _parse_score(raw: str) -> float | None:
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    score = parsed.get("score")
    try:
        score_f = float(score)
    except (TypeError, ValueError):
        return None
    if not 1.0 <= score_f <= 5.0:
        # Tolerate small overshoots from the model.
        if score_f < 1.0 - 0.5 or score_f > 5.0 + 0.5:
            return None
        score_f = max(1.0, min(5.0, score_f))
    return round(score_f, 3)


async def judge_one(
    model: OpenAIChatModel,
    item: dict,
    video_strategy: str | None,
) -> dict:
    sample_id = item.get("id") or (item.get("metadata") or {}).get("id")
    query = item.get("query") or item.get("prompt") or ""
    task_type = item.get("task_type", "t2v")

    paths = extract_video_paths(item, task_type) if is_video_task(task_type) else []
    if not paths:
        resp = item.get("response")
        if isinstance(resp, str):
            paths = [resp]
        elif isinstance(resp, list) and resp:
            paths = [str(resp[0])]

    if len(paths) != 1:
        return {
            "id": sample_id,
            "error": (
                f"expected exactly 1 video path, got {len(paths)} "
                "(this dataset is likely pairwise; use zero_shot_judge.py instead)"
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
        task_type,
        has_base=False,
        strategy=video_strategy,  # type: ignore[arg-type]
    )

    try:
        response = await model.achat(messages=messages)
    except Exception as exc:
        return {"id": sample_id, "error": str(exc)}

    raw = getattr(response, "content", "") or (response.get("content", "") if isinstance(response, dict) else "")
    score = _parse_score(raw)
    return {
        "id": sample_id,
        "query": query,
        "pred_score": score,
        "label_score": item.get("label_score"),
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

    # Continuous-score metrics (PLCC / SRCC / MAE) on the parseable subset.
    preds: list[float] = []
    labels: list[float] = []
    for r in results:
        s = r.get("pred_score")
        if s is None:
            continue
        try:
            l = float(r.get("label_score"))
        except (TypeError, ValueError):
            continue
        preds.append(float(s))
        labels.append(l)

    plcc, srcc, mae = _compute_metrics(preds, labels)

    report = {
        "test_dataset": args.test_dataset,
        "model": args.model,
        "mode": "pointwise",
        "total": len(samples),
        "parseable": len(preds),
        "plcc": plcc,
        "srcc": srcc,
        "mae": mae,
        "predictions": results,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    plcc_str = f"{plcc:.4f}" if plcc is not None else "n/a"
    srcc_str = f"{srcc:.4f}" if srcc is not None else "n/a"
    mae_str  = f"{mae:.4f}"  if mae  is not None else "n/a"
    print(
        f"Pointwise zero-shot: parseable {len(preds)}/{len(samples)}; "
        f"PLCC={plcc_str}, SRCC={srcc_str}, MAE={mae_str}"
    )
    print(f"Wrote report to {args.output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RewardVerse MVP pointwise zero-shot baseline.")
    parser.add_argument("--test_dataset", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--model", default="gemini-3.1-pro")
    parser.add_argument(
        "--base_url",
        default="http://v2.open.venus.oa.com/llmproxy",
    )
    parser.add_argument("--video_strategy", default=None)
    parser.add_argument("--concurrency_limit", type=int, default=2)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()