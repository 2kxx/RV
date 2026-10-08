"""Gemini-2.5-Pro pairwise judge with order-swap and stability filtering.

For each held-out prompt, compare arm-X-video vs. Base-video in both
input orderings (A/B and B/A). Only order-stable verdicts (both orders
agree on winner) count. Reports win-rate vs. Base for each arm.

Uses the OpenAI-compatible Gemini endpoint (matching the existing
rubric_pipeline setup) so we can reuse the same client and API key.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import random
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


PROMPT_TEMPLATE = """You are a strict, impartial video judge.

Text prompt:
{prompt}

Two videos have been generated for this prompt. Consider overall visual quality,
temporal consistency, motion smoothness, faithfulness to the prompt, and any
narrative coherence when relevant.

Which is better?

Reply with exactly ONE token from {{A, B, TIE}}. No explanation.
"""


async def _encode_video_bytes(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


async def one_verdict(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    video_a_path: str,
    video_b_path: str,
) -> str:
    va = await _encode_video_bytes(video_a_path)
    vb = await _encode_video_bytes(video_b_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT_TEMPLATE.format(prompt=prompt)},
                {"type": "text", "text": "Video A:"},
                {"type": "video", "video": {"data": va, "format": "mp4"}},
                {"type": "text", "text": "Video B:"},
                {"type": "video", "video": {"data": vb, "format": "mp4"}},
            ],
        }
    ]
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,   # type: ignore[arg-type]
        temperature=0.0,
        max_tokens=4,
    )
    txt = resp.choices[0].message.content or ""
    txt = txt.strip().upper()
    if "A" in txt and "B" not in txt:
        return "A"
    if "B" in txt and "A" not in txt:
        return "B"
    return "TIE"


async def _pair_winrate(
    *,
    arm_dir: Path,
    base_dir: Path,
    splits_path: Path,
    client: AsyncOpenAI,
    model: str,
    concurrency: int,
    max_prompts: int | None,
) -> dict:
    from scripts.downstream_grpo.src.prompt_pool import read_split

    prompts = read_split(splits_path)
    if max_prompts:
        prompts = prompts[:max_prompts]

    sem = asyncio.Semaphore(concurrency)
    rng = random.Random(0)

    async def _run_one(p) -> dict:
        arm_mp4 = arm_dir / f"{p.prompt_id}.mp4"
        base_mp4 = base_dir / f"{p.prompt_id}.mp4"
        if not arm_mp4.exists() or not base_mp4.exists():
            return {"prompt_id": p.prompt_id, "verdict_ab": None, "verdict_ba": None}

        async with sem:
            try:
                v_ab = await one_verdict(client, model, p.query, str(arm_mp4), str(base_mp4))
                v_ba = await one_verdict(client, model, p.query, str(base_mp4), str(arm_mp4))
            except Exception as exc:  # noqa: BLE001
                print(f"[gemini] {p.prompt_id} failed: {exc}")
                return {"prompt_id": p.prompt_id, "verdict_ab": None, "verdict_ba": None}
        return {"prompt_id": p.prompt_id, "verdict_ab": v_ab, "verdict_ba": v_ba, "dim": p.dimension}

    results = await asyncio.gather(*[_run_one(p) for p in prompts])

    stable_wins = 0
    stable_losses = 0
    stable_ties = 0
    unstable = 0
    for r in results:
        v_ab, v_ba = r["verdict_ab"], r["verdict_ba"]
        if v_ab is None or v_ba is None:
            continue
        arm_win_ab = v_ab == "A"
        arm_win_ba = v_ba == "B"
        arm_lose_ab = v_ab == "B"
        arm_lose_ba = v_ba == "A"
        if arm_win_ab and arm_win_ba:
            stable_wins += 1
        elif arm_lose_ab and arm_lose_ba:
            stable_losses += 1
        elif v_ab == "TIE" and v_ba == "TIE":
            stable_ties += 1
        else:
            unstable += 1

    total_stable = stable_wins + stable_losses + stable_ties
    win_rate = stable_wins / total_stable if total_stable else 0.0
    return {
        "n_prompts": len(results),
        "stable_wins": stable_wins,
        "stable_losses": stable_losses,
        "stable_ties": stable_ties,
        "unstable": unstable,
        "total_stable": total_stable,
        "win_rate_vs_base": win_rate,
        "records": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm_videos", required=True)
    ap.add_argument("--base_videos", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gemini-2.5-pro")
    ap.add_argument("--base_url", default="https://generativelanguage.googleapis.com/v1beta/openai/")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max_prompts", type=int, default=None)
    args = ap.parse_args()

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set OPENAI_API_KEY or GEMINI_API_KEY to a Gemini key")

    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url)

    result = asyncio.run(_pair_winrate(
        arm_dir=Path(args.arm_videos),
        base_dir=Path(args.base_videos),
        splits_path=Path(args.splits),
        client=client,
        model=args.model,
        concurrency=args.concurrency,
        max_prompts=args.max_prompts,
    ))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"[gemini] {args.arm_videos}: "
        f"win-rate vs Base = {result['win_rate_vs_base']:.3f} "
        f"({result['stable_wins']}/{result['total_stable']} stable pairs) "
        f"({result['unstable']} unstable filtered)"
    )


if __name__ == "__main__":
    main()
