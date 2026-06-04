#!/usr/bin/env python3
"""Compare 4 visual_quality rubric variants by PLCC + SRCC on a small sample.

For each rubric file passed via `--rubric_files`, load it into the ARR Judger
in pointwise mode and score every test sample. Report per-rubric PLCC, SRCC,
and MAE against the human `label_score` so we can pick the best
rubric-generation input format (Prompt+y+, Prompt+y++y-, y+, y++y-).

The eval-side video input strategy is held fixed (`--strategy`, default
`native_video`) so this script only varies the rubric.

Usage:
    python scripts/v_arr_mvp/visual_quality_rubric_variant_compare.py \
        --test_dataset examples/your_evalverse_visual_quality_test.json \
        --rubric_files \
            rubric_pipeline/rubrics/visual_quality_v1_prompt_ywin.txt,\
            rubric_pipeline/rubrics/visual_quality_v2_prompt_pair.txt,\
            rubric_pipeline/rubrics/visual_quality_v3_ywin_only.txt,\
            rubric_pipeline/rubrics/visual_quality_v4_pair_only.txt \
        --strategy native_video \
        --limit 30 \
        --output_dir runs/v_arr_mvp/visual_quality_rubric_variants
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rubric_pipeline.datasets.evalverse_adapter import (  # noqa: E402
    filter_pointwise_by_dimension,
)
from visual_quality_mvp_evaluation import (  # noqa: E402
    _HAS_SCIPY,
    _STRATEGY_SPEC,
    _fmt,
    _load_test_data,
    _patch_yaml_for_strategy,
    _plcc,
    _srcc,
)


async def _evaluate_rubric(
    rubric_file: str,
    strategy: str,
    base_config: str,
    samples: list[dict],
    base_url: str | None,
    concurrency: int,
) -> dict[str, Any]:
    from judger import Judger  # noqa: E402 pylint: disable=import-outside-toplevel
    from rubric_pipeline.graders.schema import GraderScore  # noqa: E402

    if strategy not in _STRATEGY_SPEC:
        return {"rubric_file": rubric_file, "error": f"unknown strategy: {strategy}"}

    spec = _STRATEGY_SPEC[strategy]
    patched = _patch_yaml_for_strategy(base_config, strategy)
    old_frame = os.environ.get("ARR_VIDEO_FRAME_COUNT")
    if spec["frame_count"] is not None:
        os.environ["ARR_VIDEO_FRAME_COUNT"] = str(spec["frame_count"])

    try:
        judger = Judger(config_path=patched, rubrics_file=rubric_file, base_url=base_url)
        await judger.initialize()

        semaphore = asyncio.Semaphore(concurrency)

        async def _one(idx: int, item: dict) -> dict[str, Any]:
            async with semaphore:
                eid = (item.get("metadata") or {}).get("evalverse_id") or item.get("id") or str(idx)
                label = float(item.get("label_score", 0))
                try:
                    result = await judger._evaluate_item(item)  # pylint: disable=protected-access
                except Exception as exc:  # pragma: no cover - depends on VLM
                    return {"idx": idx, "id": eid, "pred": None, "label": label, "error": str(exc)}
                if not isinstance(result, GraderScore):
                    return {
                        "idx": idx,
                        "id": eid,
                        "pred": None,
                        "label": label,
                        "error": f"unexpected result: {type(result).__name__}",
                    }
                return {
                    "idx": idx,
                    "id": eid,
                    "pred": float(result.score),
                    "label": label,
                    "reason": result.reason,
                }

        records = await asyncio.gather(*[_one(i, s) for i, s in enumerate(samples)])
    finally:
        if old_frame is None:
            os.environ.pop("ARR_VIDEO_FRAME_COUNT", None)
        else:
            os.environ["ARR_VIDEO_FRAME_COUNT"] = old_frame
        Path(patched).unlink(missing_ok=True)

    parseable = [r for r in records if r.get("pred") is not None]
    plcc = srcc = mae = None
    if len(parseable) >= 2:
        preds = [r["pred"] for r in parseable]
        labels = [r["label"] for r in parseable]
        plcc = _plcc(preds, labels)
        srcc = _srcc(preds, labels)
        mae = sum(abs(p - l) for p, l in zip(preds, labels)) / len(parseable)

    return {
        "rubric_file": rubric_file,
        "strategy": strategy,
        "total": len(samples),
        "parseable": len(parseable),
        "PLCC": plcc,
        "SRCC": srcc,
        "MAE": mae,
        "records": records,
    }


def _variant_label_from_path(rubric_file: str) -> str:
    stem = Path(rubric_file).stem
    if stem.startswith("visual_quality_"):
        return stem[len("visual_quality_"):]
    return stem


def _write_markdown(
    output_path: Path,
    args: argparse.Namespace,
    n_samples: int,
    results: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Visual Quality MVP — rubric generation input variant comparison\n\n")
    lines.append(
        f"- Dimension: `{args.dimension}` (EvalVerse second-level under Aesthetics)\n",
    )
    lines.append(f"- Test sample size: {n_samples} (`--limit {args.limit}`)\n")
    lines.append(f"- Eval strategy (fixed): `{args.strategy}`\n")
    lines.append(
        f"- Correlation library: {'scipy.stats' if _HAS_SCIPY else 'in-script fallback'}\n\n",
    )
    lines.append("## Variants\n\n")
    lines.append(
        "| Tag | Inputs to rubric generator | Rubric file | parseable | PLCC | SRCC | MAE |\n",
    )
    lines.append("|---|---|---|---|---|---|---|\n")
    inputs_for_label = {
        "v1_prompt_ywin": "Prompt + y+",
        "v2_prompt_pair": "Prompt + y+ + y-",
        "v3_ywin_only": "y+",
        "v4_pair_only": "y+ + y-",
    }
    for r in results:
        label = _variant_label_from_path(r["rubric_file"])
        inputs = inputs_for_label.get(label, "—")
        if "error" in r:
            lines.append(
                f"| `{label}` | {inputs} | `{r['rubric_file']}` | error | n/a | n/a | n/a |\n",
            )
            continue
        lines.append(
            f"| `{label}` | {inputs} | `{r['rubric_file']}` | "
            f"{r['parseable']}/{r['total']} | "
            f"{_fmt(r['PLCC'])} | {_fmt(r['SRCC'])} | {_fmt(r['MAE'])} |\n",
        )

    best_plcc = max(
        (r for r in results if isinstance(r.get("PLCC"), float)),
        key=lambda r: r["PLCC"],
        default=None,
    )
    best_srcc = max(
        (r for r in results if isinstance(r.get("SRCC"), float)),
        key=lambda r: r["SRCC"],
        default=None,
    )
    lines.append("\n")
    if best_plcc:
        lines.append(
            f"**Best PLCC**: `{_variant_label_from_path(best_plcc['rubric_file'])}` "
            f"({best_plcc['PLCC']:.3f})\n",
        )
    if best_srcc:
        lines.append(
            f"**Best SRCC**: `{_variant_label_from_path(best_srcc['rubric_file'])}` "
            f"({best_srcc['SRCC']:.3f})\n",
        )
    lines.append(
        "\n> PLCC = Pearson correlation between ARR pointwise score and human "
        "label_score; SRCC = Spearman rank correlation. Higher is better. "
        "MAE = mean absolute error on the 1-5 scale.\n",
    )
    output_path.write_text("".join(lines), encoding="utf-8")


async def main_async(args: argparse.Namespace) -> None:
    samples = _load_test_data(args.test_dataset)
    samples = filter_pointwise_by_dimension(samples, args.dimension)
    if not samples:
        print(
            f"[error] No pointwise samples found for dimension={args.dimension}.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.seed is not None:
        random.seed(args.seed)
        random.shuffle(samples)
    samples = samples[: args.limit]
    print(f"[info] Selected {len(samples)} samples for dimension={args.dimension}")

    rubric_files = [p.strip() for p in args.rubric_files.split(",") if p.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []
    for rf in rubric_files:
        label = _variant_label_from_path(rf)
        print(f"[run] rubric={label}  file={rf}")
        if not Path(rf).exists():
            print(f"  [skip] missing rubric file: {rf}")
            all_results.append({"rubric_file": rf, "error": "rubric file missing"})
            continue
        if Path(rf).stat().st_size == 0:
            print(f"  [skip] empty rubric file: {rf}")
            all_results.append({"rubric_file": rf, "error": "empty rubric file"})
            continue
        try:
            res = await _evaluate_rubric(
                rf,
                args.strategy,
                args.config_path,
                samples,
                args.base_url,
                args.concurrency_limit,
            )
        except Exception as exc:  # pragma: no cover - depends on VLM
            print(f"  [error] {exc}")
            all_results.append({"rubric_file": rf, "error": str(exc)})
            continue
        all_results.append(res)
        print(
            f"  PLCC={_fmt(res['PLCC'])} SRCC={_fmt(res['SRCC'])} "
            f"MAE={_fmt(res['MAE'])} parseable={res['parseable']}/{res['total']}",
        )

    out_json = output_dir / f"{args.dimension}_rubric_variant_comparison.json"
    out_json.write_text(
        json.dumps(
            {
                "dimension": args.dimension,
                "limit": args.limit,
                "strategy": args.strategy,
                "test_dataset": args.test_dataset,
                "rubric_files": rubric_files,
                "results": all_results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    md_path = output_dir / f"{args.dimension}_rubric_variant_comparison.md"
    _write_markdown(md_path, args, len(samples), all_results)
    print(f"Wrote {out_json}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare 4 visual_quality rubric variants by PLCC/SRCC.",
    )
    parser.add_argument("--test_dataset", required=True,
                        help="EvalVerse raw JSON (human_dims) or canonical pointwise.")
    parser.add_argument("--rubric_files", required=True,
                        help="Comma-separated rubric file paths (one per variant).")
    parser.add_argument("--config_path",
                        default="rubric_pipeline/config/qwen3vl_8B_instruct_t2v.yaml")
    parser.add_argument("--dimension", default="visual_quality")
    parser.add_argument("--strategy", default="native_video",
                        help="Fixed video input strategy for the eval pass.")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--concurrency_limit", type=int, default=2)
    parser.add_argument(
        "--output_dir",
        default="runs/v_arr_mvp/visual_quality_rubric_variants",
    )
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
