#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RewardVerse — untrained Qwen2.5-VL-7B baselines on the 16-dim pointwise
test set, with per-dim PLCC / SRCC reporting that drops straight into the
paper main table next to the rubric (Gemini) main result.

Four ablation cells covered, all running on the SAME hold-out set
(``data/v_arr_mvp/<dim>_mvp/test_pointwise_<dim>.json`` for 16 dims;
``soundscape``/``vocal`` excluded — vision-only backbone, paper §3 scope):

    | mode | rubric    | scoring                              | paper label |
    |------|-----------|--------------------------------------|-------------|
    | A1   | none      | natural-language free-float (JSON)   | V1 row      |
    | A2   | none      | soft-logits expectation over {1..5}  | V2 row      |
    | B1   | gold-seed | per-theme NL float JSON, mean-agg    | V4 row      |
    | D6   | gold-seed | per-theme soft-logits, mean-agg      | V3 row      |

A1/A2/B1/D6 reuse the existing scoring functions in
``rm_training.stage0_diagnosis.scalar_drift_ablation`` (V1/V2/V4) and
``QwenVLJudge.score_video_multi_theme_softlogits`` (V3). This driver is
ONLY a thin per-dim wrapper that:

  1. instantiates ``QwenVLJudge`` once,
  2. iterates 16 dims × ~63 rows ≈ 1k forwards,
  3. dumps one ``runs/baselines/qwenvl_<mode>/<dim>.json`` per dim with
     the SAME schema the rubric judge / VideoScore baselines use, plus
  4. an aggregate ``runs/baselines/qwenvl_<mode>/_runall_summary.tsv``
     containing per-dim PLCC / SRCC / MAE_zs / N for the paper main
     table builder.

Per-dim file schema (matches ``pointwise_rubric_judge.py`` output so
the SOTA-table builder can ingest it under identical code paths)::

    {
      "method":     "Qwen2.5-VL-7B [A1] no-rubric NL free-float",
      "mode":       "A1" | "A2" | "B1" | "D6",
      "model":      "Qwen2.5-VL-7B-Instruct",
      "dataset":    "data/v_arr_mvp/<dim>_mvp/test_pointwise_<dim>.json",
      "dimension":  "<dim>",
      "total":      N,
      "parseable":  M,
      "failures":   N - M,
      "metrics": {
        "plcc": <float>, "srcc": <float>,
        "mae_zs": <float>, "mae_affine": <float>,
        "n": M,
      },
      "predictions": [
        {"id":..., "label_score":..., "pred_score":..., "raw":...},
        ...
      ]
    }

Usage::

    # Single mode, all 16 dims (~1.5–2.5 h on a single H800):
    python scripts/v_arr_mvp/baseline_qwenvl_pointwise.py --mode A1
    python scripts/v_arr_mvp/baseline_qwenvl_pointwise.py --mode A2
    python scripts/v_arr_mvp/baseline_qwenvl_pointwise.py --mode B1
    python scripts/v_arr_mvp/baseline_qwenvl_pointwise.py --mode D6

    # Single dim smoke test (e.g. cap to 5 rows):
    python scripts/v_arr_mvp/baseline_qwenvl_pointwise.py \\
        --mode A1 --dimensions character --max_per_dim 5

The summary TSV is written incrementally — re-running the script will
APPEND-OR-OVERWRITE rows for whatever dims you supply, so smoke runs
do not destroy existing full-run results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# QwenVLJudge + helpers from stage0_diagnosis (loads the model once).
from rm_training.stage0_diagnosis.exp0_base_diagnosis import (  # noqa: E402
    QwenVLJudge,
    _load_rubric_themes,
    _resolve_local_video,
    _resolve_test_pointwise_path,
    _prefetch_url,
)

# Variant scoring functions (V1=A1, V2=A2, V4=B1) live in scalar_drift_ablation.
from rm_training.stage0_diagnosis.scalar_drift_ablation import (  # noqa: E402
    DIMENSIONS_16,
    score_video_natural,                # V1 — NL free-float (no rubric)
    score_video_overall_softlogits,     # V2 — overall soft-logits (no rubric)
    score_video_multi_theme_natural,    # V4 — rubric + per-theme NL float
)

# Shared metric helper so all baselines report the same PLCC/SRCC/MAE
# convention as ``baseline_videoscore.py`` and ``pointwise_rubric_judge.py``.
from scripts.v_arr_mvp.baseline_videoscore import (  # noqa: E402
    _compute_metrics as _compute_metrics_v1,
)


# ---------------------------------------------------------------------------
# Mode metadata. Keep label strings exactly as referenced in the paper main
# table so build_main_baseline_table.py can join on ``mode``.
# ---------------------------------------------------------------------------
MODE_TABLE: "OrderedDict[str, dict[str, Any]]" = OrderedDict([
    ("A1", {
        "label":    "Qwen2.5-VL-7B [A1] no-rubric NL free-float",
        "rubric":   False,
        "score_fn_kind": "v1",
    }),
    ("A2", {
        "label":    "Qwen2.5-VL-7B [A2] no-rubric soft-logits",
        "rubric":   False,
        "score_fn_kind": "v2",
    }),
    ("B1", {
        "label":    "Qwen2.5-VL-7B [B1] gold-seed rubric + NL multi-theme",
        "rubric":   True,
        "score_fn_kind": "v4",
    }),
    ("D6", {
        "label":    "Qwen2.5-VL-7B [D6] gold-seed rubric + soft-logits multi-theme",
        "rubric":   True,
        "score_fn_kind": "v3",
    }),
])


# ---------------------------------------------------------------------------
# Per-mode dispatcher. Returns (pred_score | None, raw_text).
# ---------------------------------------------------------------------------
def _score_one_video(judge: QwenVLJudge,
                     mode: str,
                     instruction: str,
                     video_path: str,
                     themes: list[dict] | None) -> tuple[float | None, str]:
    kind = MODE_TABLE[mode]["score_fn_kind"]

    if kind == "v1":
        # V1 / A1 — no rubric, natural-language free-float.
        s, raw = score_video_natural(judge, instruction, video_path, themes=None)
        return (s, raw)

    if kind == "v2":
        # V2 / A2 — no rubric, overall soft-logits expectation.
        s, raw = score_video_overall_softlogits(judge, instruction, video_path)
        return (s, raw)

    if kind == "v4":
        # V4 / B1 — rubric, per-theme NL float JSON, mean aggregation.
        if not themes:
            return (None, "[B1] no rubric themes for this dim")
        scores, raw = score_video_multi_theme_natural(
            judge, instruction, video_path, themes,
            max_new_tokens=int(judge.cfg.get("multi_score_max_new_tokens", 96)),
        )
        valid = [s for s in scores if s is not None]
        if not valid:
            return (None, raw)
        return (sum(valid) / len(valid), raw)

    if kind == "v3":
        # V3 / D6 — rubric, per-theme soft-logits, mean aggregation.
        if not themes:
            return (None, "[D6] no rubric themes for this dim")
        scores, raw = judge.score_video_multi_theme_softlogits(
            instruction, video_path, themes, dimension=None,
            max_new_tokens=int(judge.cfg.get("multi_score_max_new_tokens", 96)),
        )
        valid = [s for s in scores if s is not None]
        if not valid:
            return (None, raw)
        return (sum(valid) / len(valid), raw)

    raise ValueError(f"unknown score_fn_kind={kind!r} for mode={mode!r}")


# ---------------------------------------------------------------------------
# Per-dim runner. Returns the per-dim JSON dict (also written to disk).
# ---------------------------------------------------------------------------
def run_one_dim(judge: QwenVLJudge,
                mode: str,
                dim: str,
                out_dir: Path,
                *,
                max_per_dim: int | None,
                prefetch_missing: bool,
                rubric_dir: Path) -> dict | None:
    fp = _resolve_test_pointwise_path(dim, ROOT)
    if fp is None:
        print(f"[{mode}][{dim}] no test_pointwise file — SKIP", file=sys.stderr)
        return None

    rubric_required = bool(MODE_TABLE[mode]["rubric"])
    themes = _load_rubric_themes(rubric_dir, dim) if rubric_required else None
    if rubric_required and not themes:
        print(f"[{mode}][{dim}] no rubric themes — SKIP", file=sys.stderr)
        return None

    try:
        rows = json.loads(fp.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[{mode}][{dim}] {fp.name} unreadable: {e}", file=sys.stderr)
        return None
    if max_per_dim is not None and max_per_dim > 0:
        rows = rows[: int(max_per_dim)]

    baseline_cache = ROOT / "runs" / "baselines" / "_video_cache"
    prefetch_cache = ROOT / "data" / "v_arr_mvp" / "cache_videos"

    predictions: list[dict] = []
    n_failures = 0
    n_no_video = 0
    t0 = time.time()
    for ri, r in enumerate(rows):
        url = r.get("response")
        label = r.get("label_score")
        if not isinstance(url, str) or label is None:
            continue
        local = _resolve_local_video(url, baseline_cache, prefetch_cache)
        if local is None and prefetch_missing and url.startswith(("http://", "https://")):
            import hashlib
            import urllib.parse
            parsed = urllib.parse.urlparse(url)
            base = os.path.basename(parsed.path) or "video.mp4"
            if not base.lower().endswith(".mp4"):
                base += ".mp4"
            digest = hashlib.sha1(url.encode()).hexdigest()[:10]
            dst = prefetch_cache / f"{digest}_{base}"
            if _prefetch_url(url, dst):
                local = str(dst)
        if local is None:
            n_no_video += 1
            continue

        instruction = r.get("query") or r.get("metadata", {}).get("prompt_zh") or ""
        pred, raw = _score_one_video(judge, mode, instruction, local, themes)
        if pred is None:
            n_failures += 1
        predictions.append({
            "id":          r.get("id"),
            "label_score": float(label),
            "pred_score":  None if pred is None else float(pred),
            "raw":         raw if isinstance(raw, str) else "",
        })
        if (ri + 1) % 20 == 0:
            print(f"  [{mode}][{dim}] {ri+1}/{len(rows)}  "
                  f"fail={n_failures}  novid={n_no_video}  "
                  f"elapsed={int(time.time() - t0)}s", file=sys.stderr)

    preds_paired = [(p["pred_score"], p["label_score"])
                    for p in predictions if p["pred_score"] is not None]
    if preds_paired:
        ps = [p for p, _ in preds_paired]
        ls = [l for _, l in preds_paired]
        plcc, srcc, mae_zs = _compute_metrics_v1(ps, ls, mae_mode="zscore")
        _, _, mae_aff       = _compute_metrics_v1(ps, ls, mae_mode="affine")
    else:
        plcc = srcc = mae_zs = mae_aff = None

    payload = {
        "method":      MODE_TABLE[mode]["label"],
        "mode":        mode,
        "model":       judge.cfg.get("model_path", "").split("/")[-1] or "Qwen2.5-VL-7B-Instruct",
        "dataset":     str(fp.relative_to(ROOT)),
        "dimension":   dim,
        "total":       len(predictions),
        "parseable":   len(preds_paired),
        "failures":    len(predictions) - len(preds_paired),
        "metrics": {
            "plcc":        plcc,
            "srcc":        srcc,
            "mae_zs":      mae_zs,
            "mae_affine":  mae_aff,
            "n":           len(preds_paired),
        },
        "predictions": predictions,
        "n_skipped_no_video": n_no_video,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_fp = out_dir / f"{dim}.json"
    out_fp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"[{mode}][{dim}] wrote {out_fp.relative_to(ROOT)}  "
          f"PLCC={_fmt(plcc)} SRCC={_fmt(srcc)} N={len(preds_paired)}/{len(predictions)} "
          f"failures={len(predictions) - len(preds_paired)}  "
          f"({int(time.time() - t0)}s)", file=sys.stderr)
    return payload


def _fmt(x):
    if x is None:
        return "n/a"
    return f"{float(x):+.4f}"


# ---------------------------------------------------------------------------
# Aggregator: emit a TSV with one row per dim, suitable for paper-table import.
# ---------------------------------------------------------------------------
def update_runall_summary(out_dir: Path) -> None:
    rows: list[dict] = []
    for fp in sorted(out_dir.glob("*.json")):
        if fp.name.startswith("_"):
            continue
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        m = obj.get("metrics") or {}
        rows.append({
            "dim":       obj.get("dimension", fp.stem),
            "plcc":      m.get("plcc"),
            "srcc":      m.get("srcc"),
            "mae_zs":    m.get("mae_zs"),
            "n":         m.get("n", 0),
            "failures":  obj.get("failures", 0),
            "total":     obj.get("total", 0),
        })

    canonical = [d for d in DIMENSIONS_16 if any(r["dim"] == d for r in rows)]
    by_dim = {r["dim"]: r for r in rows}

    tsv_fp = out_dir / "_runall_summary.tsv"
    with tsv_fp.open("w", encoding="utf-8") as f:
        f.write("dim\tplcc\tsrcc\tmae_zs\tn\tfailures\ttotal\n")
        plccs: list[float] = []
        srccs: list[float] = []
        for dim in canonical:
            r = by_dim[dim]
            f.write(
                f"{dim}\t"
                f"{_tsv_num(r['plcc'])}\t{_tsv_num(r['srcc'])}\t"
                f"{_tsv_num(r['mae_zs'])}\t"
                f"{r['n']}\t{r['failures']}\t{r['total']}\n"
            )
            if isinstance(r["plcc"], (int, float)):
                plccs.append(float(r["plcc"]))
            if isinstance(r["srcc"], (int, float)):
                srccs.append(float(r["srcc"]))
        if plccs:
            f.write(f"# mean over {len(plccs)} dims:\t"
                    f"{sum(plccs)/len(plccs):+.4f}\t"
                    f"{(sum(srccs)/len(srccs) if srccs else 0):+.4f}\n")
    print(f"[runall] -> {tsv_fp.relative_to(ROOT)}  "
          f"({len(canonical)} dims  meanPLCC={sum(plccs)/len(plccs):+.4f})"
          if plccs else f"[runall] -> {tsv_fp.relative_to(ROOT)} (no rows)",
          file=sys.stderr)


def _tsv_num(x) -> str:
    if x is None:
        return "nan"
    return f"{float(x):.4f}"


# ---------------------------------------------------------------------------
# Entry.
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=list(MODE_TABLE.keys()),
                   help="A1=no-rubric NL free-float, A2=no-rubric soft-logits, "
                        "B1=rubric NL multi-theme, D6=rubric soft-logits multi-theme.")
    p.add_argument("--dimensions", default=None,
                   help="Comma-separated subset of DIMENSIONS_16. "
                        "Default = all 16 (≈1k pointwise rows).")
    p.add_argument("--max_per_dim", type=int, default=None,
                   help="Cap rows per dim. Default=None (use full ~63/dim).")
    p.add_argument("--out_dir", default=None,
                   help="Override output dir. Default=runs/baselines/qwenvl_<mode>.")
    p.add_argument("--model_path", default=None,
                   help="Qwen2.5-VL checkpoint path (overrides RM_MODEL_PATH).")
    p.add_argument("--no_prefetch", action="store_true",
                   help="Disable URL→cache prefetch (assume videos already local).")
    p.add_argument("--multi_score_max_new_tokens", type=int, default=96)
    args = p.parse_args()

    dims = list(DIMENSIONS_16)
    if args.dimensions:
        wanted = [d.strip() for d in args.dimensions.split(",") if d.strip()]
        unknown = [d for d in wanted if d not in DIMENSIONS_16]
        if unknown:
            p.error(f"--dimensions includes non-16-dim slugs: {unknown}")
        dims = wanted

    out_dir = (Path(args.out_dir).expanduser()
               if args.out_dir
               else ROOT / "runs" / "baselines" / f"qwenvl_{args.mode}")

    cfg = {
        "model_path": args.model_path or os.environ.get(
            "RM_MODEL_PATH",
            "/group/40063/jacuzzitang/models/Qwen2.5-VL-7B-Instruct",
        ),
        "device":           "cuda",
        "dtype":            "bfloat16",
        "frame_strategy":   "fps",
        "fps":              2.0,
        "max_frames":       64,
        "default_nframes":  16,
        "rubric_dir":       str(ROOT / "rubric_pipeline" / "rubrics"),
        "multi_score_max_new_tokens": int(args.multi_score_max_new_tokens),
    }
    rubric_dir = Path(cfg["rubric_dir"])

    print(f"[boot] mode={args.mode}  dims={len(dims)}  out_dir={out_dir.relative_to(ROOT)}",
          file=sys.stderr)
    print(f"[boot] label='{MODE_TABLE[args.mode]['label']}'", file=sys.stderr)
    judge = QwenVLJudge(cfg)

    t_run = time.time()
    n_done = 0
    for dim in dims:
        try:
            payload = run_one_dim(
                judge, args.mode, dim, out_dir,
                max_per_dim=args.max_per_dim,
                prefetch_missing=not args.no_prefetch,
                rubric_dir=rubric_dir,
            )
            if payload is not None:
                n_done += 1
        except Exception as e:
            print(f"[{args.mode}][{dim}] EXC: {e!r}", file=sys.stderr)
        # Refresh aggregate after every dim — partial runs still leave a
        # readable TSV for the paper-table builder.
        update_runall_summary(out_dir)

    elapsed = int(time.time() - t_run)
    print(f"\n[done] mode={args.mode}  {n_done}/{len(dims)} dims  elapsed={elapsed}s",
          file=sys.stderr)


if __name__ == "__main__":
    main()
