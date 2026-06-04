#!/usr/bin/env python3
"""EvalVerse multi-dimension evaluator with Bradley-Terry pairwise reward.

Reports:
- Pairwise accuracy under EvalVerse Win/Lose annotations (Bradley-Terry signal).
- Per-aspect and per-main-dimension pointwise accuracy.
- Optional weighted-sum aggregation across dimensions using
  rubric_pipeline.runner.aggregator.weighted_sum_aggregator.WeightedSumAggregator.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judger import Judger
from rubric_pipeline.datasets.evalverse_adapter import (
    EVALVERSE_ASPECTS,
    EVALVERSE_ASPECT_TO_MAIN_DIMENSIONS,
    EVALVERSE_MAIN_DIMENSIONS,
    aspect_of,
    filter_pairwise,
    filter_pointwise_by_dimension,
    load_evalverse_dataset,
)
from rubric_pipeline.graders.schema import GraderRank, GraderScore
from rubric_pipeline.runner.aggregator.weighted_sum_aggregator import WeightedSumAggregator


async def evaluate_pairwise(
    config_path: str,
    rubrics_file: str | None,
    samples: list[dict],
    base_url: str | None,
    concurrency_limit: int,
) -> dict[str, Any]:
    judger = await Judger(
        config_path=config_path,
        rubrics_file=rubrics_file,
        base_url=base_url,
    ).initialize()

    semaphore = asyncio.Semaphore(concurrency_limit)
    correct = 0
    total = 0
    flipped_consistency = 0
    flipped_total = 0
    reasons: list[dict[str, Any]] = []

    async def _eval_one(item: dict) -> None:
        nonlocal correct, total, flipped_consistency, flipped_total
        async with semaphore:
            result = await judger._evaluate_item(item)
            total += 1
            if not isinstance(result, GraderRank):
                return
            pred = list(result.rank)
            label = item.get("label_rank") or []
            if pred == label:
                correct += 1
            reasons.append(
                {
                    "id": (item.get("metadata") or {}).get("evalverse_id"),
                    "pred_rank": pred,
                    "label_rank": label,
                    "reason": result.reason,
                },
            )
            # Position-bias consistency check: flip candidate order and re-rank.
            if isinstance(item.get("response"), list) and len(item["response"]) == 2:
                flipped_item = dict(item)
                flipped_item["response"] = [item["response"][1], item["response"][0]]
                if item.get("label_rank"):
                    flipped_item["label_rank"] = [item["label_rank"][1], item["label_rank"][0]]
                flipped = await judger._evaluate_item(flipped_item)
                flipped_total += 1
                if isinstance(flipped, GraderRank):
                    pred_flipped = list(flipped.rank)
                    if pred == [pred_flipped[1], pred_flipped[0]]:
                        flipped_consistency += 1

    await asyncio.gather(*[_eval_one(item) for item in samples])
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "position_bias_consistency": flipped_consistency / flipped_total if flipped_total else None,
        "samples": reasons,
    }


async def evaluate_pointwise_one_dim(
    config_path: str,
    rubrics_file: str | None,
    samples: list[dict],
    dimension: str,
    base_url: str | None,
    concurrency_limit: int,
) -> dict[str, Any]:
    if not samples:
        return {"total": 0, "correct": 0, "accuracy": 0.0}

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    config = dict(config)
    config["eval_mode"] = "pointwise"
    parent_aspect = aspect_of(dimension) or "unknown"
    config["task_description"] = (
        f"{config.get('task_description', '')}\n"
        f"EvalVerse focus: aspect={parent_aspect}, main_dim={dimension}.\n"
        "Score only this dimension; apply Self-Reflection and Context-Aware Gating."
    ).strip()
    tmp_config = Path(config_path).parent / f"_tmp_pointwise_{dimension}.yaml"
    tmp_config.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    try:
        judger = await Judger(
            config_path=str(tmp_config),
            rubrics_file=rubrics_file,
            base_url=base_url,
        ).initialize()

        correct = 0
        total = 0
        semaphore = asyncio.Semaphore(concurrency_limit)

        async def _eval_one(item: dict) -> None:
            nonlocal correct, total
            async with semaphore:
                result = await judger._evaluate_item(item)
                total += 1
                if isinstance(result, GraderScore):
                    expected = item.get("label_score")
                    if expected is not None and int(round(result.score)) == int(round(float(expected))):
                        correct += 1

        await asyncio.gather(*[_eval_one(item) for item in samples])
    finally:
        tmp_config.unlink(missing_ok=True)

    return {
        "dimension": dimension,
        "aspect": parent_aspect,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
    }


async def main_async(args: argparse.Namespace) -> None:
    samples = load_evalverse_dataset(args.input)
    pairwise_samples = filter_pairwise(samples)
    report: dict[str, Any] = {"pairwise": None, "main_dims": {}, "aspects": {}}

    if pairwise_samples and not args.skip_pairwise:
        print(f"[pairwise] {len(pairwise_samples)} samples")
        report["pairwise"] = await evaluate_pairwise(
            args.config_path,
            args.rubrics_file,
            pairwise_samples,
            args.base_url,
            args.concurrency_limit,
        )
        print(
            f"  acc={report['pairwise']['accuracy']:.2%} "
            f"position_bias_consistency={report['pairwise']['position_bias_consistency']}",
        )

    if not args.skip_pointwise:
        target_dims = args.dimensions.split(",") if args.dimensions else EVALVERSE_MAIN_DIMENSIONS
        for dim in target_dims:
            dim_samples = filter_pointwise_by_dimension(samples, dim)
            if not dim_samples:
                continue
            print(f"[main-dim] {dim} -> {len(dim_samples)} samples")
            result = await evaluate_pointwise_one_dim(
                args.config_path,
                args.rubrics_file,
                dim_samples,
                dim,
                args.base_url,
                args.concurrency_limit,
            )
            report["main_dims"][dim] = result

        # Aspect-level rollup.
        for aspect, dims in EVALVERSE_ASPECT_TO_MAIN_DIMENSIONS.items():
            relevant = [report["main_dims"][d] for d in dims if d in report["main_dims"]]
            if not relevant:
                continue
            total = sum(r["total"] for r in relevant)
            correct = sum(r["correct"] for r in relevant)
            report["aspects"][aspect] = {
                "total": total,
                "correct": correct,
                "accuracy": correct / total if total else 0.0,
                "main_dims": [r["dimension"] for r in relevant],
            }
            print(f"[aspect] {aspect}: acc={report['aspects'][aspect]['accuracy']:.2%} ({correct}/{total})")

        if args.weights_json:
            weights = json.loads(Path(args.weights_json).read_text(encoding="utf-8"))
            aggregator = WeightedSumAggregator(name="evalverse_aggregate", weights=weights)
            report["aggregator"] = {"name": aggregator.name, "weights": weights}

    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Wrote EvalVerse multi-dim report to {args.report_json}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="EvalVerse pairwise (Bradley-Terry) + multi-dim pointwise evaluation.",
    )
    parser.add_argument("--input", required=True, help="EvalVerse raw JSON or canonical ARR JSON")
    parser.add_argument("--config_path", default="rubric_pipeline/config/qwen3vl_8B_instruct_t2v.yaml")
    parser.add_argument("--rubrics_file", default=None)
    parser.add_argument("--base_url", default=None)
    parser.add_argument(
        "--dimensions",
        default=None,
        help=(
            "Comma-separated EvalVerse main dimension names. "
            f"Defaults to all 18 main dimensions: {','.join(EVALVERSE_MAIN_DIMENSIONS)}"
        ),
    )
    parser.add_argument("--weights_json", default=None, help="Optional dimension weights JSON")
    parser.add_argument("--concurrency_limit", type=int, default=4)
    parser.add_argument("--report_json", default=None)
    parser.add_argument("--skip_pairwise", action="store_true")
    parser.add_argument("--skip_pointwise", action="store_true")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
