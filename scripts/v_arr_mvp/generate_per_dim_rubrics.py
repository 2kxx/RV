#!/usr/bin/env python3
"""Generate one ARR rubric text file per evaluation dimension.

Reads a pointwise-style seed dataset (each sample carries a `dimension` and
`label_score`), groups samples by dimension, then runs ARR's
generate-verify-refine loop in pointwise mode for each group. The verified
rubric text is written to `<output_dir>/<dimension>.txt`, ready to be loaded by
the EvalVerseEvaluator-side `ARRPointwiseEvaluator`.

Supported inputs:
- ARR canonical pointwise samples (each sample has `dimension` + `label_score`).
- EvalVerse raw records (passed through `load_evalverse_dataset`, which
  expands `human_dims` into per-dimension pointwise samples).
- A flat seed (no `dimension` field) when `--single_dimension <name>` is given:
  the whole seed is treated as one dimension and one rubric file is produced.

Usage:
    python scripts/v_arr_mvp/generate_per_dim_rubrics.py \
        --seed_dataset examples/evalverse_raw_pointwise.json \
        --config_path rubric_pipeline/config/qwen3vl_8B_instruct_t2v.yaml \
        --output_dir rubric_pipeline/rubrics

    python scripts/v_arr_mvp/generate_per_dim_rubrics.py \
        --seed_dataset examples/v_arr_mvp_seed_pairwise.json \
        --config_path rubric_pipeline/config/v_arr_mvp_t2v.yaml \
        --output_dir rubric_pipeline/rubrics \
        --single_dimension v_arr_overall
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from judger import Judger  # noqa: E402
from rubric_pipeline.datasets.evalverse_adapter import load_evalverse_dataset  # noqa: E402


def _load_raw(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw = json.loads(p.read_text(encoding="utf-8"))
    return raw if isinstance(raw, list) else [raw]


def _try_evalverse_expand(samples: list[dict]) -> list[dict]:
    """Best-effort: expand EvalVerse `human_dims` records into pointwise samples."""
    if not samples:
        return samples
    head = samples[0]
    if "human_dims" in head or "dimension_scores" in head:
        # EvalVerse-style. Re-route through the adapter so dimensions get expanded
        # and aliased into canonical names.
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump(samples, tmp)
            tmp_path = tmp.name
        try:
            return load_evalverse_dataset(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    return samples


def _group_by_dimension(
    samples: list[dict],
    single_dimension: str | None,
) -> dict[str, list[dict]]:
    if single_dimension:
        return {single_dimension: list(samples)}
    grouped: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        dim = s.get("dimension")
        if not dim:
            continue
        grouped[str(dim)].append(s)
    return grouped


def _patch_yaml_to_pointwise(base_config_path: str, target_dim: str) -> str:
    """Materialise a temp YAML that forces eval_mode=pointwise for this dim."""
    base = yaml.safe_load(Path(base_config_path).read_text(encoding="utf-8"))
    if not isinstance(base, Mapping):
        raise ValueError(f"Unexpected YAML payload from {base_config_path}")
    patched = dict(base)
    patched["eval_mode"] = "pointwise"
    patched.setdefault("min_score", 1)
    patched.setdefault("max_score", 5)
    extra_desc = (
        f"\nFocus on the evaluation dimension `{target_dim}`. Generate rubrics that "
        "scope to this dimension only; do not duplicate criteria that belong to "
        "other dimensions."
    )
    patched["task_description"] = (str(patched.get("task_description") or "") + extra_desc).strip()
    tmp_path = Path(base_config_path).parent / f"_tmp_pointwise_{target_dim}.yaml"
    tmp_path.write_text(yaml.safe_dump(patched, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(tmp_path)


async def _generate_one(
    dim: str,
    dim_samples: list[dict],
    base_config_path: str,
    output_dir: Path,
    base_url: str | None,
) -> tuple[str, bool, str]:
    seed_path = output_dir / f"_seed_{dim}.json"
    seed_path.write_text(json.dumps(dim_samples, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_config = _patch_yaml_to_pointwise(base_config_path, dim)
    try:
        judger = Judger(
            config_path=tmp_config,
            seed_dataset=str(seed_path),
            base_url=base_url,
        )
        await judger.initialize()
        rubric_text = judger.rubrics or ""
        out_path = output_dir / f"{dim}.txt"
        out_path.write_text(rubric_text, encoding="utf-8")
        return dim, True, str(out_path)
    except Exception as exc:  # pragma: no cover - depends on LLM endpoint
        return dim, False, f"ERROR: {exc}"
    finally:
        Path(tmp_config).unlink(missing_ok=True)
        # Keep seed JSON for reproducibility; comment out next line to remove.
        # seed_path.unlink(missing_ok=True)


async def main_async(args: argparse.Namespace) -> None:
    raw = _load_raw(args.seed_dataset)
    expanded = _try_evalverse_expand(raw)
    grouped = _group_by_dimension(expanded, args.single_dimension)
    if not grouped:
        print(
            "[error] No dimensions found. Pass --single_dimension or use a seed "
            "with `dimension` or `human_dims` fields.",
            file=sys.stderr,
        )
        sys.exit(2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for dim, dim_samples in sorted(grouped.items()):
        if args.dimensions and dim not in args.dimensions:
            continue
        if len(dim_samples) < args.min_samples_per_dim:
            print(f"[skip] {dim}: only {len(dim_samples)} sample(s); need >= {args.min_samples_per_dim}")
            summary.append({"dimension": dim, "status": "skipped", "samples": len(dim_samples)})
            continue
        print(f"[gen] {dim}: {len(dim_samples)} sample(s)")
        dim_name, ok, info = await _generate_one(
            dim,
            dim_samples,
            args.config_path,
            output_dir,
            args.base_url,
        )
        summary.append(
            {
                "dimension": dim_name,
                "status": "ok" if ok else "error",
                "info": info,
                "samples": len(dim_samples),
            },
        )

    summary_path = output_dir / "_generation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote rubric files to {output_dir}/<dimension>.txt")
    print(f"Summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one ARR rubric file per dimension.")
    parser.add_argument("--seed_dataset", required=True)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_dir", default="rubric_pipeline/rubrics")
    parser.add_argument("--base_url", default=None)
    parser.add_argument("--single_dimension", default=None,
                        help="Treat the whole seed as one dimension and emit one rubric file.")
    parser.add_argument("--dimensions", default=None,
                        help="Comma-separated dimension allowlist.")
    parser.add_argument("--min_samples_per_dim", type=int, default=2)
    args = parser.parse_args()
    args.dimensions = args.dimensions.split(",") if args.dimensions else None
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
