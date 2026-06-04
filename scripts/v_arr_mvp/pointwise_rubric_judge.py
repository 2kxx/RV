"""RewardVerse — pointwise rubric judge driver for one EvalVerse L2 dimension.

This script applies a *pre-learned* rubric (text file produced by the
auto-rubric pipeline) to a pointwise test dataset for a single
EvalVerse second-level dimension and reports PLCC / SRCC / MAE
against the GT ``label_score``.

Output JSON schema is **deliberately aligned** with the baseline
reward-model runners (``baseline_videoscore.py``, ``baseline_videoscore2.py``,
``baseline_unifiedreward.py``) so the SOTA-table builder can ingest
all four method JSONs with identical code paths.

Output schema::

    {
      "method": "RewardVerse (rubric, gemini-3.1-pro)",
      "dataset": "data/v_arr_mvp/<dim>_mvp/test_pointwise_<dim>.json",
      "dimension": "<dim>",
      "rubrics_file": "rubric_pipeline/rubrics/v_arr_mvp_<dim>_v2.txt",
      "config_path":  "rubric_pipeline/config/v_arr_mvp_<dim>_t2v.yaml",
      "total":        N,
      "parseable":    M,
      "failures":     N - M,
      "metrics": {
        "plcc":       <float>,
        "srcc":       <float>,
        "mae_zs":     <float>,
        "mae_affine": <float>,
        "n":          M,
      },
      "predictions": [
        {"id": ..., "query": ..., "label_score": ..., "pred_score": ..., "reason": ...},
        ...
      ],
    }

Usage::

    python scripts/v_arr_mvp/pointwise_rubric_judge.py \\
        --dataset data/v_arr_mvp/rhythm_mvp/test_pointwise_rhythm.json \\
        --output runs/v_arr_mvp_dim/rhythm/pointwise.json \\
        --concurrency 4

By default the dimension is inferred from the dataset path (same logic as
the baseline runners), and the rubric/yaml are looked up via the standard
``v_arr_mvp_<dim>_v2.txt`` / ``v_arr_mvp_<dim>_t2v.yaml`` convention. Override
with ``--dimension``, ``--rubrics_file``, ``--config_path`` if the layout
differs.

The script imports ``_compute_metrics`` and ``_infer_dimension_from_dataset``
from baseline_videoscore so the rubric main metric uses the **same z-score
MAE convention** as the baselines (PLCC and SRCC are scale-invariant; MAE
on raw rubric scores in [1, 5] would not be comparable to baseline raw
logits centered near 0).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Same env defaults the rubric pipeline uses end-to-end. We pin them so a
# fresh shell that did not source env_setup.sh still gets the same
# native-video / Gemini multimodal-url path that visual_quality_mvp used.
os.environ.setdefault("ARR_VIDEO_INPUT_STRATEGY", "native_video")
os.environ.setdefault("ARR_VIDEO_BLOCK_TYPE", "gemini_multimodal_url")

# Re-use the baseline metric helpers so all four methods report under the
# same z-score / affine MAE convention. PLCC and SRCC are scale-invariant
# and would be identical anyway, but importing keeps the single source of
# truth for any future schema additions (e.g. bootstrap CI).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_videoscore import (  # type: ignore  # noqa: E402
    _compute_metrics as _compute_metrics_v1,
    _infer_dimension_from_dataset,
)


# ---------------------------------------------------------------------------
# Per-dimension asset resolution.
# ---------------------------------------------------------------------------
# Convention used by generate_per_dim_rubrics.py / generate_dim_yamls.py:
#
#   rubric : rubric_pipeline/rubrics/v_arr_mvp_<dim>_v2.txt
#   yaml   : rubric_pipeline/config/v_arr_mvp_<dim>_t2v.yaml
#
# `visual_quality` is the historical exception: its rubric was named
#   v_arr_mvp_vq_v2_prompt_pair.txt
# and it has always reused the generic
#   v_arr_mvp_t2v.yaml
# config (visual_quality is treated as the canonical t2v judge in the
# original MVP pipeline; see scripts/v_arr_mvp/generate_dim_yamls.py
# header which explicitly excludes it).
# ---------------------------------------------------------------------------
def _default_rubrics_file(dim: str) -> Path:
    if dim == "visual_quality":
        return ROOT / "rubric_pipeline" / "rubrics" / "v_arr_mvp_vq_v2_prompt_pair.txt"
    return ROOT / "rubric_pipeline" / "rubrics" / f"v_arr_mvp_{dim}_v2.txt"


def _default_config_path(dim: str) -> Path:
    if dim == "visual_quality":
        return ROOT / "rubric_pipeline" / "config" / "v_arr_mvp_t2v.yaml"
    return ROOT / "rubric_pipeline" / "config" / f"v_arr_mvp_{dim}_t2v.yaml"


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
async def _amain(args: argparse.Namespace) -> int:
    # Heavy imports kept local — same pattern as visual_quality_mvp_evaluation
    # so the script's --help and dim-resolution logic remain usable on a
    # fresh shell without the full ARR stack imported eagerly.
    from judger import Judger  # type: ignore  # noqa: E402
    from rubric_pipeline.graders.schema import GraderScore  # type: ignore  # noqa: E402

    # ------------------------------------------------------------------
    # Resolve dim → assets.
    # ------------------------------------------------------------------
    dim = args.dimension or _infer_dimension_from_dataset(args.dataset)
    if not dim:
        print(f"[fatal] could not infer dimension from --dataset path "
              f"{args.dataset!r}; pass --dimension explicitly.",
              file=sys.stderr)
        return 2

    rubrics_file = Path(args.rubrics_file) if args.rubrics_file else _default_rubrics_file(dim)
    config_path  = Path(args.config_path)  if args.config_path  else _default_config_path(dim)

    if not rubrics_file.is_file():
        print(f"[fatal] rubrics file not found: {rubrics_file}", file=sys.stderr)
        return 2
    if not config_path.is_file():
        print(f"[fatal] config yaml not found: {config_path}", file=sys.stderr)
        return 2

    print(f"[rubric] dimension     = {dim}", flush=True)
    print(f"[rubric] dataset       = {args.dataset}", flush=True)
    print(f"[rubric] rubrics_file  = {rubrics_file}", flush=True)
    print(f"[rubric] config_path   = {config_path}", flush=True)
    print(f"[rubric] base_url      = {args.base_url or '<from yaml>'}", flush=True)
    print(f"[rubric] concurrency   = {args.concurrency}", flush=True)

    # ------------------------------------------------------------------
    # Load samples.
    # ------------------------------------------------------------------
    samples_raw = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if isinstance(samples_raw, dict):
        samples_raw = [samples_raw]
    if args.limit:
        samples_raw = samples_raw[: args.limit]
    if not samples_raw:
        print("[fatal] dataset has 0 samples after --limit", file=sys.stderr)
        return 2
    print(f"[rubric] {len(samples_raw)} samples", flush=True)

    # ------------------------------------------------------------------
    # Build Judger. Per-dim yamls are written with eval_mode: listwise
    # (because rubric *generation* is listwise), but we want pointwise
    # *evaluation* here. ``_apply_eval_mode`` flips it cleanly.
    # ------------------------------------------------------------------
    judger = Judger(
        config_path=str(config_path),
        rubrics_file=str(rubrics_file),
        base_url=args.base_url,
    )
    judger._apply_eval_mode("pointwise")  # noqa: SLF001 (public-ish helper)
    t0 = time.time()
    await judger.initialize()
    print(f"[rubric] judger initialized in {time.time()-t0:.1f}s "
          f"(eval_mode={judger.eval_mode})", flush=True)

    # ------------------------------------------------------------------
    # Score every sample with bounded concurrency.
    # ------------------------------------------------------------------
    semaphore = asyncio.Semaphore(args.concurrency)

    async def _one(idx: int, item: dict) -> dict:
        sample_id = item.get("id") or str(idx)
        label = item.get("label_score")
        try:
            label_f = float(label) if label is not None else None
        except (TypeError, ValueError):
            label_f = None
        async with semaphore:
            try:
                result = await judger._evaluate_item(item)  # noqa: SLF001
            except Exception as exc:  # pragma: no cover — depends on VLM
                return {
                    "id":          sample_id,
                    "query":       item.get("query") or item.get("prompt") or "",
                    "label_score": label_f,
                    "pred_score":  None,
                    "error":       str(exc),
                }
        if not isinstance(result, GraderScore):
            return {
                "id":          sample_id,
                "query":       item.get("query") or item.get("prompt") or "",
                "label_score": label_f,
                "pred_score":  None,
                "error":       f"unexpected result type: {type(result).__name__}",
            }
        try:
            pred = float(result.score)
        except (TypeError, ValueError) as exc:
            return {
                "id":          sample_id,
                "query":       item.get("query") or item.get("prompt") or "",
                "label_score": label_f,
                "pred_score":  None,
                "error":       f"non-float score: {result.score!r} ({exc})",
            }
        # Extract per-rubric verdicts so we can experiment with alternative
        # aggregation strategies (min / median / vote / weighted-mean / lp-norm)
        # without re-invoking the LLM. The grader stores the parsed JSON in
        # ``result.metadata``; per_rubric may live at the top level or inside
        # a nested ``metadata`` key depending on the prompt template version.
        meta = getattr(result, "metadata", None) or {}
        per_rubric = None
        if isinstance(meta, dict):
            per_rubric = meta.get("per_rubric")
            if per_rubric is None and isinstance(meta.get("metadata"), dict):
                per_rubric = meta["metadata"].get("per_rubric")
        return {
            "id":          sample_id,
            "query":       item.get("query") or item.get("prompt") or "",
            "label_score": label_f,
            "pred_score":  pred,
            "reason":      getattr(result, "reason", None),
            "per_rubric":  per_rubric,
        }

    t1 = time.time()
    records = await asyncio.gather(
        *[_one(i, s) for i, s in enumerate(samples_raw)],
    )
    print(f"[rubric] all samples scored in {time.time()-t1:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Metrics — same z-score / affine MAE convention as the baselines.
    # ------------------------------------------------------------------
    preds:  list[float] = []
    labels: list[float] = []
    for r in records:
        p = r.get("pred_score")
        l = r.get("label_score")
        if p is None or l is None:
            continue
        preds.append(float(p))
        labels.append(float(l))

    failures = sum(1 for r in records if r.get("pred_score") is None)
    plcc, srcc, mae_zs = _compute_metrics_v1(preds, labels, mae_mode="zscore")
    _, _, mae_aff       = _compute_metrics_v1(preds, labels, mae_mode="affine")
    _, _, mae_raw       = _compute_metrics_v1(preds, labels, mae_mode="raw")

    out = {
        "method":        "RewardVerse (rubric, gemini-3.1-pro)",
        "dataset":       args.dataset,
        "dimension":     dim,
        "rubrics_file":  str(rubrics_file),
        "config_path":   str(config_path),
        "total":         len(samples_raw),
        "parseable":     len(preds),
        "failures":      failures,
        "metrics": {
            "plcc":       plcc,
            "srcc":       srcc,
            "mae_zs":     mae_zs,
            "mae_affine": mae_aff,
            "mae_raw":    mae_raw,
            "n":          len(preds),
        },
        "predictions":   records,
    }

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    def _fmt(x: Any) -> str:
        return f"{x:.4f}" if isinstance(x, float) else "n/a"
    print(f"[rubric] PLCC={_fmt(plcc)} SRCC={_fmt(srcc)} "
          f"MAE_zs={_fmt(mae_zs)} MAE_aff={_fmt(mae_aff)} "
          f"N={len(preds)}/{len(samples_raw)} failures={failures}",
          flush=True)
    print(f"[rubric] wrote -> {out_path}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                   help="pointwise test JSON (must have label_score per item)")
    p.add_argument("--output",  required=True,
                   help="where to write the result JSON")
    p.add_argument("--dimension", default=None,
                   help="EvalVerse L2 axis name (e.g. 'rhythm'); inferred "
                        "from --dataset path if omitted")
    p.add_argument("--rubrics_file", default=None,
                   help="path to a pre-learned rubric .txt; defaults to the "
                        "v_arr_mvp_<dim>_v2.txt file in rubric_pipeline/rubrics/ "
                        "(visual_quality uses v_arr_mvp_vq_v2_prompt_pair.txt)")
    p.add_argument("--config_path", default=None,
                   help="path to the per-dim yaml; defaults to "
                        "v_arr_mvp_<dim>_t2v.yaml in rubric_pipeline/config/ "
                        "(visual_quality uses the generic v_arr_mvp_t2v.yaml)")
    p.add_argument("--base_url", default=None,
                   help="OpenAI-compatible base URL; falls back to whatever "
                        "the yaml sets (Tencent Venus llmproxy by default)")
    p.add_argument("--concurrency", type=int, default=4,
                   help="number of concurrent Gemini calls (default 4); "
                        "raise carefully — venus llmproxy has per-key rate "
                        "limits")
    p.add_argument("--limit", type=int, default=None,
                   help="if set, only score the first N samples (smoke test)")
    args = p.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("[fatal] OPENAI_API_KEY is not exported. Set your venus token "
              "(or upstream OpenAI-compatible token) before running.",
              file=sys.stderr)
        sys.exit(2)

    rc = asyncio.run(_amain(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
