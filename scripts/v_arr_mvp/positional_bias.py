#!/usr/bin/env python3
"""Measure positional bias of the RewardVerse judge.

For each pairwise sample, the judge is run twice:
    1) original (Video 1 = Vw, Video 2 = Vl)
    2) swapped  (Video 1 = Vl, Video 2 = Vw)

A judge is *consistent* when the swap flips its predicted rank exactly. We
report the consistency rate, the accuracy in each ordering, and a per-position
"left wins" rate (an indicator of left-position bias).

Works for both the Auto-Rubric Judger and any judge that conforms to its
`_evaluate_item` -> GraderRank interface.

Usage:
    python scripts/v_arr_mvp/positional_bias.py \
        --config_path rubric_pipeline/config/v_arr_mvp_t2v.yaml \
        --rubrics_file rubric_pipeline/rubrics/v_arr_mvp_t2v_v1.txt \
        --test_dataset examples/v_arr_mvp_test_pairwise.json \
        --base_url https://generativelanguage.googleapis.com/v1beta/openai/ \
        --output_json /tmp/v_arr_mvp_positional_bias.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judger import Judger
from rubric_pipeline.graders.schema import GraderRank


async def main_async(args: argparse.Namespace) -> None:
    samples = json.loads(Path(args.test_dataset).read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        samples = [samples]

    judger = await Judger(
        config_path=args.config_path,
        rubrics_file=args.rubrics_file,
        base_url=args.base_url,
    ).initialize()

    semaphore = asyncio.Semaphore(args.concurrency_limit)
    records: list[dict] = []

    async def _eval(item: dict) -> dict:
        async with semaphore:
            response = await judger._evaluate_item(item)
            if isinstance(response, GraderRank):
                return {"rank": list(response.rank), "reason": response.reason}
            return {"rank": None, "reason": str(response)}

    async def _one(item: dict) -> None:
        original = await _eval(item)
        if not isinstance(item.get("response"), list) or len(item["response"]) != 2:
            return
        flipped_item = dict(item)
        flipped_item["response"] = [item["response"][1], item["response"][0]]
        if isinstance(item.get("label_rank"), list) and len(item["label_rank"]) == 2:
            flipped_item["label_rank"] = [item["label_rank"][1], item["label_rank"][0]]
        flipped = await _eval(flipped_item)

        pred_orig = original.get("rank")
        pred_flip = flipped.get("rank")
        consistent = (
            isinstance(pred_orig, list)
            and isinstance(pred_flip, list)
            and pred_orig == [pred_flip[1], pred_flip[0]]
        )
        records.append(
            {
                "id": (item.get("metadata") or {}).get("id"),
                "label_rank": item.get("label_rank"),
                "pred_orig": pred_orig,
                "pred_flip": pred_flip,
                "consistent": bool(consistent),
                "orig_correct": pred_orig == item.get("label_rank"),
                "flip_correct": pred_flip == flipped_item.get("label_rank"),
                "orig_left_wins": pred_orig[0] == 1 if isinstance(pred_orig, list) else None,
                "flip_left_wins": pred_flip[0] == 1 if isinstance(pred_flip, list) else None,
            },
        )

    await asyncio.gather(*[_one(s) for s in samples])

    total = len(records)
    consistent = sum(1 for r in records if r["consistent"])
    orig_correct = sum(1 for r in records if r["orig_correct"])
    flip_correct = sum(1 for r in records if r["flip_correct"])
    left_wins_total = sum(
        1
        for r in records
        for v in (r["orig_left_wins"], r["flip_left_wins"])
        if v is True
    )
    decisions_total = sum(
        1
        for r in records
        for v in (r["orig_left_wins"], r["flip_left_wins"])
        if v is not None
    )

    report = {
        "config_path": args.config_path,
        "rubrics_file": args.rubrics_file,
        "test_dataset": args.test_dataset,
        "total": total,
        "consistent": consistent,
        "consistency_rate": consistent / total if total else 0.0,
        "orig_accuracy": orig_correct / total if total else 0.0,
        "flip_accuracy": flip_correct / total if total else 0.0,
        "left_wins_rate": left_wins_total / decisions_total if decisions_total else None,
        "records": records,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Positional bias: consistency={report['consistency_rate']:.2%}")
    print(f"  orig_accuracy={report['orig_accuracy']:.2%} flip_accuracy={report['flip_accuracy']:.2%}")
    if report["left_wins_rate"] is not None:
        print(f"  left_wins_rate={report['left_wins_rate']:.2%}  (0.5 = no positional bias)")
    print(f"Wrote report to {args.output_json}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RewardVerse MVP positional bias diagnostic.")
    parser.add_argument(
        "--config_path",
        default="rubric_pipeline/config/v_arr_mvp_t2v.yaml",
    )
    parser.add_argument("--rubrics_file", default=None)
    parser.add_argument("--test_dataset", required=True)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--concurrency_limit", type=int, default=2)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
