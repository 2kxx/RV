#!/usr/bin/env python3
"""Visual Quality MVP: compare video input strategies via PLCC + SRCC.

This script answers the question "which input strategy gives the strongest
correlation with the EvalVerse Visual Quality human scores on a small sample?"

For each strategy in `--strategies`:
  1. Materialise a temporary ARR YAML with `video_input_strategy` patched.
  2. Set `ARR_VIDEO_FRAME_COUNT` for `uniform_frames` variants.
  3. Build a pointwise ARR Judger (rubric loaded from `--rubrics_file`).
  4. Score every sample in the subsampled test set.
  5. Compute PLCC (Pearson) and SRCC (Spearman) against the human
     `label_score`, plus MAE for context.

Supported strategy labels: native_video, frames_4, frames_8, frames_16.

Usage:
    python scripts/v_arr_mvp/visual_quality_mvp_evaluation.py \
        --test_dataset examples/evalverse_visual_quality_seed.json \
        --config_path rubric_pipeline/config/qwen3vl_8B_instruct_t2v.yaml \
        --rubrics_file rubric_pipeline/rubrics/visual_quality.txt \
        --strategies native_video,frames_4,frames_8,frames_16 \
        --limit 30 \
        --base_url https://generativelanguage.googleapis.com/v1beta/openai/ \
        --output_dir runs/v_arr_mvp/visual_quality
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Light imports only at module load time. The heavy ARR judging stack
# (`judger`, `rubric_pipeline.graders.schema.GraderScore`) is imported lazily
# inside `_evaluate_strategy` so the script's helpers and --help remain usable
# without `bash env_setup.sh`.
from rubric_pipeline.datasets.evalverse_adapter import (  # noqa: E402
    filter_pointwise_by_dimension,
    load_evalverse_dataset,
)

try:
    from scipy.stats import pearsonr as _scipy_pearsonr
    from scipy.stats import spearmanr as _scipy_spearmanr

    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - tested via the fallback path
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Correlation helpers (scipy when available, pure-Python fallback otherwise).
# ---------------------------------------------------------------------------


def _pearsonr_pure(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sxx = sum((xi - mx) ** 2 for xi in x)
    syy = sum((yi - my) ** 2 for yi in y)
    if sxx == 0 or syy == 0:
        return 0.0
    return sxy / (sxx * syy) ** 0.5


def _ranks(arr: list[float]) -> list[float]:
    """Average ranks for ties."""
    n = len(arr)
    order = sorted(range(n), key=lambda i: arr[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearmanr_pure(x: list[float], y: list[float]) -> float:
    return _pearsonr_pure(_ranks(x), _ranks(y))


def _plcc(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    if _HAS_SCIPY:
        try:
            r, _ = _scipy_pearsonr(x, y)
            return float(r) if r == r else None  # NaN check
        except Exception:
            return None
    return _pearsonr_pure(x, y)


def _srcc(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    if _HAS_SCIPY:
        try:
            r, _ = _scipy_spearmanr(x, y)
            return float(r) if r == r else None
        except Exception:
            return None
    return _spearmanr_pure(x, y)


# ---------------------------------------------------------------------------
# Strategy specifications.
# ---------------------------------------------------------------------------


_STRATEGY_SPEC: dict[str, dict[str, Any]] = {
    "native_video": {"video_input_strategy": "native_video", "frame_count": None},
    "frames_4": {"video_input_strategy": "uniform_frames", "frame_count": 4},
    "frames_8": {"video_input_strategy": "uniform_frames", "frame_count": 8},
    "frames_16": {"video_input_strategy": "uniform_frames", "frame_count": 16},
}


def _patch_yaml_for_strategy(base_config: str, strategy: str) -> str:
    spec = _STRATEGY_SPEC[strategy]
    base = yaml.safe_load(Path(base_config).read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError(f"Unexpected YAML payload from {base_config}")
    patched = dict(base)
    patched["video_input_strategy"] = spec["video_input_strategy"]
    patched["eval_mode"] = "pointwise"
    patched.setdefault("min_score", 1)
    patched.setdefault("max_score", 5)
    tmp = Path(base_config).parent / f"_tmp_visual_quality_{strategy}.yaml"
    tmp.write_text(yaml.safe_dump(patched, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(tmp)


# ---------------------------------------------------------------------------
# Evaluation loop.
# ---------------------------------------------------------------------------


async def _evaluate_strategy(
    strategy: str,
    base_config: str,
    rubrics_file: str,
    samples: list[dict],
    base_url: str | None,
    concurrency: int,
) -> dict[str, Any]:
    if strategy not in _STRATEGY_SPEC:
        return {"strategy": strategy, "error": "unknown strategy"}

    # Heavy imports kept local so that --help and the helper smoke tests do
    # not require the full ARR stack to be installed.
    from judger import Judger  # noqa: E402 pylint: disable=import-outside-toplevel
    from rubric_pipeline.graders.schema import GraderScore  # noqa: E402

    spec = _STRATEGY_SPEC[strategy]
    patched_yaml = _patch_yaml_for_strategy(base_config, strategy)

    old_frame = os.environ.get("ARR_VIDEO_FRAME_COUNT")
    if spec["frame_count"] is not None:
        os.environ["ARR_VIDEO_FRAME_COUNT"] = str(spec["frame_count"])

    try:
        judger = Judger(config_path=patched_yaml, rubrics_file=rubrics_file, base_url=base_url)
        await judger.initialize()

        semaphore = asyncio.Semaphore(concurrency)

        async def _one(idx: int, item: dict) -> dict[str, Any]:
            async with semaphore:
                evalverse_id = (item.get("metadata") or {}).get("evalverse_id") or item.get("id") or str(idx)
                try:
                    result = await judger._evaluate_item(item)  # pylint: disable=protected-access
                except Exception as exc:  # pragma: no cover - depends on VLM
                    return {
                        "idx": idx,
                        "id": evalverse_id,
                        "pred": None,
                        "label": float(item.get("label_score", 0)),
                        "error": str(exc),
                    }
                if not isinstance(result, GraderScore):
                    return {
                        "idx": idx,
                        "id": evalverse_id,
                        "pred": None,
                        "label": float(item.get("label_score", 0)),
                        "error": f"unexpected result type: {type(result).__name__}",
                    }
                return {
                    "idx": idx,
                    "id": evalverse_id,
                    "pred": float(result.score),
                    "label": float(item.get("label_score", 0)),
                    "reason": result.reason,
                }

        records = await asyncio.gather(*[_one(i, s) for i, s in enumerate(samples)])
    finally:
        if old_frame is None:
            os.environ.pop("ARR_VIDEO_FRAME_COUNT", None)
        else:
            os.environ["ARR_VIDEO_FRAME_COUNT"] = old_frame
        Path(patched_yaml).unlink(missing_ok=True)

    parseable = [r for r in records if r.get("pred") is not None]
    plcc = srcc = mae = None
    if len(parseable) >= 2:
        preds = [r["pred"] for r in parseable]
        labels = [r["label"] for r in parseable]
        plcc = _plcc(preds, labels)
        srcc = _srcc(preds, labels)
        mae = sum(abs(p - l) for p, l in zip(preds, labels)) / len(parseable)

    return {
        "strategy": strategy,
        "video_input_strategy": spec["video_input_strategy"],
        "frame_count": spec["frame_count"],
        "total": len(samples),
        "parseable": len(parseable),
        "PLCC": plcc,
        "SRCC": srcc,
        "MAE": mae,
        "records": records,
    }


# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------


def _load_test_data(path: str) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = [raw]
    # If the input is canonical ARR pointwise (already has `dimension` and
    # `label_score`), keep it as-is. Otherwise pass it through the EvalVerse
    # adapter to expand `human_dims`.
    needs_expansion = any(
        ("human_dims" in r) or ("dimension_scores" in r) or ("scores" in r)
        for r in raw
        if isinstance(r, dict)
    ) and not any(
        ("dimension" in r and "label_score" in r) for r in raw if isinstance(r, dict)
    )
    if not needs_expansion:
        return list(raw)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(raw, tmp)
        tmp_path = tmp.name
    try:
        return load_evalverse_dataset(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_markdown(
    output_path: Path,
    args: argparse.Namespace,
    n_samples: int,
    results: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Visual Quality MVP — input strategy comparison\n")
    lines.append(
        f"- EvalVerse dimension: `{args.dimension}` "
        f"(second-level under Aesthetics; sub-dims temporal_consistency / "
        f"rendering_quality / physics)\n"
    )
    lines.append(f"- Sample size: {n_samples} (cap `--limit={args.limit}`)\n")
    lines.append(f"- Rubric file: `{args.rubrics_file}`\n")
    lines.append(f"- Base config: `{args.config_path}`\n")
    lines.append(
        f"- Correlation library: {'scipy.stats' if _HAS_SCIPY else 'in-script fallback'}\n\n",
    )

    lines.append("## Per-strategy results\n\n")
    lines.append("| Strategy | input_type | frames | parseable | PLCC | SRCC | MAE |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for r in results:
        if "error" in r:
            lines.append(f"| {r['strategy']} | error | - | - | n/a | n/a | n/a |\n")
            continue
        lines.append(
            f"| {r['strategy']} | {r['video_input_strategy']} | "
            f"{r['frame_count'] if r['frame_count'] is not None else '-'} | "
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
        lines.append(f"**Best PLCC**: `{best_plcc['strategy']}` ({best_plcc['PLCC']:.3f})\n")
    if best_srcc:
        lines.append(f"**Best SRCC**: `{best_srcc['strategy']}` ({best_srcc['SRCC']:.3f})\n")

    lines.append(
        "\n> PLCC = Pearson linear correlation between ARR predicted score and human "
        "label_score; SRCC = Spearman rank correlation. Higher is better.\n",
    )
    output_path.write_text("".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> None:
    samples = _load_test_data(args.test_dataset)
    samples = filter_pointwise_by_dimension(samples, args.dimension)
    if not samples:
        print(
            f"[error] No pointwise samples found for dimension={args.dimension}. "
            "Make sure the seed contains `human_dims` (raw EvalVerse) or canonical "
            "`dimension` + `label_score` entries.",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.seed is not None:
        random.seed(args.seed)
        random.shuffle(samples)
    samples = samples[: args.limit]
    print(f"[info] Selected {len(samples)} samples for dimension={args.dimension}")

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for strategy in strategies:
        print(f"[run] strategy={strategy}")
        if strategy not in _STRATEGY_SPEC:
            print(f"  [skip] unknown strategy: {strategy}")
            continue
        try:
            res = await _evaluate_strategy(
                strategy,
                args.config_path,
                args.rubrics_file,
                samples,
                args.base_url,
                args.concurrency_limit,
            )
        except Exception as exc:  # pragma: no cover - depends on VLM
            print(f"  [error] {exc}")
            results.append({"strategy": strategy, "error": str(exc)})
            continue
        results.append(res)
        print(
            f"  PLCC={_fmt(res['PLCC'])} SRCC={_fmt(res['SRCC'])} "
            f"MAE={_fmt(res['MAE'])} parseable={res['parseable']}/{res['total']}",
        )

    out_json = output_dir / f"{args.dimension}_strategy_comparison.json"
    out_json.write_text(
        json.dumps(
            {
                "dimension": args.dimension,
                "limit": args.limit,
                "test_dataset": args.test_dataset,
                "config_path": args.config_path,
                "rubrics_file": args.rubrics_file,
                "strategies": strategies,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    md_path = output_dir / f"{args.dimension}_strategy_comparison.md"
    _write_markdown(md_path, args, len(samples), results)

    print(f"Wrote {out_json}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual Quality MVP: PLCC/SRCC across input strategies.")
    parser.add_argument("--test_dataset", required=True,
                        help="EvalVerse raw JSON (with human_dims) or canonical ARR pointwise JSON.")
    parser.add_argument("--config_path",
                        default="rubric_pipeline/config/qwen3vl_8B_instruct_t2v.yaml",
                        help="Base ARR YAML; eval_mode and video_input_strategy are patched per run.")
    parser.add_argument("--rubrics_file", required=True,
                        help="Rubric file for the target dimension "
                             "(produced by generate_per_dim_rubrics.py).")
    parser.add_argument("--dimension", default="visual_quality",
                        help="Target EvalVerse dimension (default: visual_quality).")
    parser.add_argument("--strategies",
                        default="native_video,frames_4,frames_8,frames_16",
                        help="Comma-separated strategy labels.")
    parser.add_argument("--limit", type=int, default=30,
                        help="Maximum number of test samples (the script subsamples randomly).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for subsampling. Set to a fixed integer for reproducibility.")
    parser.add_argument("--base_url", default=None,
                        help="Override the ARR YAML base_url (OpenAI-compatible endpoint).")
    parser.add_argument("--concurrency_limit", type=int, default=2)
    parser.add_argument("--output_dir", default="runs/v_arr_mvp/visual_quality")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
