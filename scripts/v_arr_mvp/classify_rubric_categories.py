#!/usr/bin/env python3
"""Classify generated rubrics into {spatial, temporal, instruction, quality}.

Reads a rubric text file (one rubric per line, or numbered list) and asks a VLM
(via OpenAI-compatible endpoint, by default Gemini) to assign one of four labels
per rubric. Reports the distribution and flags if temporal coverage < threshold.

Usage:
    python scripts/v_arr_mvp/classify_rubric_categories.py \
        --rubrics_path rubric_pipeline/rubrics/v_arr_mvp_t2v_v1.txt \
        --output_json /tmp/v_arr_mvp_rubric_categories.json \
        --temporal_threshold 0.30
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

CATEGORIES = ("spatial", "temporal", "instruction", "quality")

_CLASSIFY_INSTRUCTION = """
You will be shown one rubric used to judge generated videos. Classify it into
exactly one of four mutually exclusive categories:

- "spatial": criteria that can be checked per-frame (appearance, identity,
  object presence, colour, composition, anti-artifact).
- "temporal": criteria about motion, dynamics, or cross-frame phenomena
  (action plausibility, temporal consistency, motion direction / speed,
  flicker, camera trajectory).
- "instruction": criteria that check whether the video follows the textual
  prompt (prompt adherence on actions, attributes, counts, scene, style),
  when the criterion cannot be cleanly split into spatial/temporal.
- "quality": low-level perceptual quality (resolution, blur, compression
  artifacts, lighting realism) that is not specific to motion or to prompt
  following.

If two categories tie, prefer "temporal" over "spatial" over "instruction"
over "quality".

Return STRICT JSON in this exact shape:
{"category": "spatial" | "temporal" | "instruction" | "quality",
 "confidence": 0.0-1.0,
 "reason": "<one sentence justification>"}

Rubric:
\"\"\"
{rubric}
\"\"\"
""".strip()


def parse_rubrics(text: str) -> list[str]:
    """Split a rubric file into individual rubric strings.

    Supports several styles:
      - One rubric per line.
      - Numbered lines: "1. ...".
      - Categorized sections starting with "##".
    """
    rubrics: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        line = re.sub(r"^\s*[-*]\s*", "", line)
        line = re.sub(r"^\s*\d+[.)]\s*", "", line)
        line = re.sub(r"^\s*\[[^\]]+\]\s*", "", line)
        if len(line) < 8:
            continue
        rubrics.append(line)
    return rubrics


async def classify_one(model: OpenAIChatModel, rubric: str) -> dict:
    messages = [
        ChatMessage(
            role="system",
            content="You are a strict classifier. Output only valid JSON. No prose.",
        ),
        ChatMessage(
            role="user",
            content=_CLASSIFY_INSTRUCTION.replace("{rubric}", rubric),
        ),
    ]
    try:
        response = await model.achat(messages=[m.model_dump() for m in messages])
    except Exception as exc:  # pragma: no cover - depends on endpoint
        return {"category": "unknown", "confidence": 0.0, "reason": f"endpoint error: {exc}"}

    raw = ""
    if hasattr(response, "content"):
        raw = response.content or ""
    elif isinstance(response, dict):
        raw = response.get("content", "")
    elif isinstance(response, str):
        raw = response

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return {"category": "unknown", "confidence": 0.0, "reason": "no JSON in response"}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"category": "unknown", "confidence": 0.0, "reason": "invalid JSON"}
    cat = str(parsed.get("category", "unknown")).strip().lower()
    if cat not in CATEGORIES:
        cat = "unknown"
    parsed["category"] = cat
    return parsed


async def main_async(args: argparse.Namespace) -> None:
    rubrics_text = Path(args.rubrics_path).read_text(encoding="utf-8")
    rubrics = parse_rubrics(rubrics_text)
    if not rubrics:
        print("[error] No rubrics parsed from file.", file=sys.stderr)
        sys.exit(2)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = OpenAIChatModel(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        temperature=0,
    )

    semaphore = asyncio.Semaphore(args.concurrency_limit)
    results: list[dict] = []

    async def _one(idx: int, rubric: str) -> None:
        async with semaphore:
            classification = await classify_one(model, rubric)
            results.append({"index": idx, "rubric": rubric, **classification})

    await asyncio.gather(*[_one(i, r) for i, r in enumerate(rubrics)])
    results.sort(key=lambda r: r["index"])

    counts = {c: 0 for c in CATEGORIES}
    counts["unknown"] = 0
    for r in results:
        counts[r["category"]] = counts.get(r["category"], 0) + 1
    total = len(results)
    fractions = {c: (counts[c] / total if total else 0.0) for c in counts}

    report = {
        "rubrics_path": str(args.rubrics_path),
        "model": args.model,
        "total": total,
        "counts": counts,
        "fractions": fractions,
        "temporal_threshold": args.temporal_threshold,
        "temporal_threshold_passed": fractions.get("temporal", 0.0) >= args.temporal_threshold,
        "samples": results,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Total rubrics: {total}")
    for c in (*CATEGORIES, "unknown"):
        print(f"  {c:<11s}: {counts[c]:>3d} ({fractions[c]:.1%})")
    print(
        f"Temporal threshold ({args.temporal_threshold:.0%}): "
        f"{'PASS' if report['temporal_threshold_passed'] else 'FAIL'}",
    )
    print(f"Wrote report to {args.output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify RewardVerse rubrics into 4 categories.")
    parser.add_argument("--rubrics_path", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument(
        "--base_url",
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    parser.add_argument("--concurrency_limit", type=int, default=4)
    parser.add_argument("--temporal_threshold", type=float, default=0.30)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
