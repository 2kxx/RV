#!/usr/bin/env python3
"""Learn 4 visual_quality rubric variants from synthesized pairwise data.

Variants under test:

| Tag                 | Inputs to rubric generator        | Generator mode |
| ------------------- | --------------------------------- | -------------- |
| v1_prompt_ywin      | prompt + y+ (winner only)         | pointwise      |
| v2_prompt_pair      | prompt + y+ + y-                  | pairwise       |
| v3_ywin_only        | y+ only (no prompt)               | pointwise      |
| v4_pair_only        | y+ + y- (no prompt)               | pairwise       |

For each variant the script:
  1. Builds a per-variant seed JSON (under `--output_dir`).
  2. Patches the base YAML (eval_mode + optional task_description override).
  3. Runs `Judger.initialize()` to drive ARR's generate-verify-refine loop.
  4. Persists `judger.rubrics` to `visual_quality_<tag>.txt` in `--output_dir`.

Outputs are designed to feed `visual_quality_rubric_variant_compare.py`.

Usage:
    python scripts/v_arr_mvp/learn_visual_quality_rubric_variants.py \
        --input data/v_arr_mvp/visual_quality_pairs.json \
        --config_path rubric_pipeline/config/qwen3vl_8B_instruct_t2v.yaml \
        --output_dir rubric_pipeline/rubrics \
        --base_url https://generativelanguage.googleapis.com/v1beta/openai/
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


# ---------------------------------------------------------------------------
# Variant catalogue.
# ---------------------------------------------------------------------------


_NO_PROMPT_QUERY_POINTWISE = (
    "(no prompt supplied; judge intrinsic visual quality from the video only)"
)
_NO_PROMPT_QUERY_PAIRWISE = (
    "(no prompt supplied; rank the two videos purely by intrinsic visual quality)"
)


_VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "v1_prompt_ywin": {
        "label": "Prompt + y+",
        "eval_mode": "pointwise",
        "use_prompt": True,
        "pair_mode": "pointwise_ywin",
    },
    "v2_prompt_pair": {
        "label": "Prompt + y+ + y-",
        "eval_mode": "pairwise",
        "use_prompt": True,
        "pair_mode": "pairwise",
    },
    "v3_ywin_only": {
        "label": "y+",
        "eval_mode": "pointwise",
        "use_prompt": False,
        "pair_mode": "pointwise_ywin",
        "task_description_override": (
            "Focus only on the intrinsic visual quality of the video clip: "
            "temporal consistency across frames, rendering fidelity (resolution, "
            "anti-artifact, anti-flicker), and physical plausibility of motion. "
            "Do NOT score prompt adherence or instruction following; the prompt "
            "is intentionally omitted in this variant."
        ),
    },
    "v4_pair_only": {
        "label": "y+ + y-",
        "eval_mode": "pairwise",
        "use_prompt": False,
        "pair_mode": "pairwise",
        "task_description_override": (
            "Compare the two videos and rank them purely on intrinsic visual "
            "quality: temporal consistency, rendering fidelity, and physical "
            "plausibility of motion. Do NOT consider prompt adherence; the "
            "prompt is intentionally omitted in this variant."
        ),
    },
}


# ---------------------------------------------------------------------------
# Seed construction.
# ---------------------------------------------------------------------------


def _pointwise_ywin_samples(pairs: list[dict], use_prompt: bool) -> list[dict]:
    """One pointwise sample per pair, using only the winner video (y+).

    label_score = winner's Visual Quality score (from synthesis metadata).
    """
    out: list[dict] = []
    for p in pairs:
        if not isinstance(p.get("response"), list) or len(p["response"]) < 1:
            continue
        winner_path = str(p["response"][0])
        meta = p.get("metadata") or {}
        winner_score = meta.get("winner_score")
        if winner_score is None:
            continue
        query = p.get("query", "") if use_prompt else _NO_PROMPT_QUERY_POINTWISE
        out.append(
            {
                "query": query,
                "response": winner_path,
                "label_score": int(round(float(winner_score))),
                "dimension": "visual_quality",
                "task_type": "t2v",
                "metadata": dict(meta),
            },
        )
    return out


def build_pointwise_ywin_samples(pairs: list[dict], use_prompt: bool) -> list[dict]:
    """Public alias used by tests/smoke."""
    return _pointwise_ywin_samples(pairs, use_prompt)


def _pairwise_samples(pairs: list[dict], use_prompt: bool) -> list[dict]:
    """Pass the canonical pair through, optionally blanking the prompt."""
    out: list[dict] = []
    for p in pairs:
        if not isinstance(p.get("response"), list) or len(p["response"]) < 2:
            continue
        item = dict(p)
        item["task_type"] = "t2v"
        if not use_prompt:
            item["query"] = _NO_PROMPT_QUERY_PAIRWISE
        out.append(item)
    return out


def build_pairwise_samples(pairs: list[dict], use_prompt: bool) -> list[dict]:
    return _pairwise_samples(pairs, use_prompt)


def _build_seed_for_variant(pairs: list[dict], variant: str) -> list[dict]:
    spec = _VARIANT_SPECS[variant]
    if spec["pair_mode"] == "pointwise_ywin":
        return _pointwise_ywin_samples(pairs, spec["use_prompt"])
    return _pairwise_samples(pairs, spec["use_prompt"])


# ---------------------------------------------------------------------------
# YAML patching.
# ---------------------------------------------------------------------------


def patch_yaml_for_variant(base_config: str, variant: str) -> str:
    spec = _VARIANT_SPECS[variant]
    base = yaml.safe_load(Path(base_config).read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError(f"Unexpected YAML payload from {base_config}")
    patched = dict(base)
    patched["eval_mode"] = spec["eval_mode"]
    patched.setdefault("min_score", 1)
    patched.setdefault("max_score", 5)
    # Force-set task_description for no-prompt variants; otherwise leave it alone.
    if spec.get("task_description_override"):
        patched["task_description"] = spec["task_description_override"]
    tmp = Path(base_config).parent / f"_tmp_variant_{variant}.yaml"
    tmp.write_text(yaml.safe_dump(patched, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return str(tmp)


# ---------------------------------------------------------------------------
# Rubric learning.
# ---------------------------------------------------------------------------


async def _learn_one_variant(
    variant: str,
    pairs: list[dict],
    base_config: str,
    output_dir: Path,
    base_url: str | None,
) -> dict[str, Any]:
    from judger import Judger  # lazy import (heavy stack)

    spec = _VARIANT_SPECS[variant]
    seed = _build_seed_for_variant(pairs, variant)
    if not seed:
        rubric_path = output_dir / f"visual_quality_{variant}.txt"
        rubric_path.write_text("", encoding="utf-8")
        return {
            "variant": variant,
            "label": spec["label"],
            "status": "empty_seed",
            "samples": 0,
            "rubric_path": str(rubric_path),
        }

    seed_path = output_dir / f"_seed_{variant}.json"
    seed_path.write_text(json.dumps(seed, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_yaml = patch_yaml_for_variant(base_config, variant)

    rubric_text = ""
    error: str | None = None
    try:
        judger = Judger(
            config_path=tmp_yaml,
            seed_dataset=str(seed_path),
            base_url=base_url,
        )
        await judger.initialize()
        rubric_text = judger.rubrics or ""
    except Exception as exc:  # pragma: no cover - depends on VLM
        error = str(exc)
    finally:
        Path(tmp_yaml).unlink(missing_ok=True)

    rubric_path = output_dir / f"visual_quality_{variant}.txt"
    rubric_path.write_text(rubric_text, encoding="utf-8")
    return {
        "variant": variant,
        "label": spec["label"],
        "eval_mode": spec["eval_mode"],
        "use_prompt": spec["use_prompt"],
        "samples": len(seed),
        "rubric_path": str(rubric_path),
        "seed_path": str(seed_path),
        "status": "error" if error else ("ok" if rubric_text else "empty_rubric"),
        "error": error,
        "rubric_chars": len(rubric_text),
    }


async def main_async(args: argparse.Namespace) -> None:
    pairs = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(pairs, list):
        pairs = [pairs]
    print(f"[info] Loaded {len(pairs)} pairwise records from {args.input}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = args.variants.split(",") if args.variants else list(_VARIANT_SPECS.keys())
    summary: list[dict[str, Any]] = []
    for variant in variants:
        if variant not in _VARIANT_SPECS:
            print(f"[warn] unknown variant: {variant}, skipping")
            continue
        spec = _VARIANT_SPECS[variant]
        print(f"[run] variant={variant}  ({spec['label']})")
        res = await _learn_one_variant(
            variant,
            pairs,
            args.config_path,
            output_dir,
            args.base_url,
        )
        print(
            f"  -> {res['status']}  rubric={res.get('rubric_path')}  "
            f"samples={res.get('samples')}  rubric_chars={res.get('rubric_chars', 0)}",
        )
        summary.append(res)

    summary_path = output_dir / "rubric_variant_learning_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Learn 4 visual_quality rubric variants from synthesized pairwise data.",
    )
    parser.add_argument("--input", required=True,
                        help="Canonical pairwise JSON from synthesize_visual_quality_pairs.py.")
    parser.add_argument("--config_path",
                        default="rubric_pipeline/config/qwen3vl_8B_instruct_t2v.yaml",
                        help="Base ARR YAML; eval_mode + task_description are patched per variant.")
    parser.add_argument("--output_dir", default="rubric_pipeline/rubrics",
                        help="Directory for per-variant seed JSON, rubric text, and summary.")
    parser.add_argument("--base_url", default=None,
                        help="Override the YAML base_url (OpenAI-compat endpoint).")
    parser.add_argument("--variants", default=None,
                        help="Comma-separated variant subset (default: run all 4).")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
