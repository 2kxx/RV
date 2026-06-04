"""RewardVerse — pairwise rubric judge driver for one RewardVerse axis.

Mirrors pointwise_rubric_judge.py but feeds the LLM judger in *listwise*
(pairwise) mode. Each input row carries two videos and the judge ranks
them under the per-axis rubric; we then map the rank to {A, B, same}
and report pair accuracy against the GT label.

Why this script exists alongside the pointwise version
------------------------------------------------------
RewardVerse rubrics were *evolved* on pairwise seeds. The pointwise
delta-threshold path (run_external_vgrb.sh phase 1+2) is a fair
apples-to-apples comparison vs VS-v1.1/VS-2 (which only emit pointwise
floats). This script measures rubric's NATIVE strength: directly
prompting the judge to rank A vs B under the rubric — the regime the
rubric was actually trained for.

Input schema (listwise / pairwise)
----------------------------------
Each record must contain at minimum::

    {
      "id":       "<unique pair id>",
      "query":    "<text prompt>",
      "response": ["<video_a path>", "<video_b path>"],
      "label_rank": [1, 2]   # GT: 1 = winner, 2 = loser
                              # use [1, 1] for genuine ties (we'll
                              # treat the judge's tie predictions as
                              # `same`)
    }

The auto-built VGRB pairwise dataset (see prep_vgrb_pairwise_view.py)
populates ``label_rank`` from the GT ``Overall`` (or any chosen axis)
column: A->[1,2], B->[2,1], same->[1,1].

Output schema (deliberately a SUPERSET of pointwise output so
summarize_external_vgrb.py can ingest both)::

    {
      "method":     "RewardVerse (rubric, pairwise)",
      "axis":       "<vgrb axis label, e.g. Overall>",
      "rv_axis":    "<rubric axis used, e.g. visual_quality>",
      "total":      N,
      "parseable":  M,
      "summary": {
        "axis":       "<vgrb axis>",
        "tau":        null,                 # N/A for native pairwise
        "n_total":    M,
        "n_no_tie":   <int>,
        "acc":        <float>,              # 3-way A/B/same
        "acc_no_tie": <float>,              # subset GT in {A,B}
        "confusion": {"A": {...}, "B": {...}, "same": {...}},
      },
      "per_pair": [
        {"id": ..., "rank_pred": [1,2], "decision": "A",
         "gt": "A", "correct": true, "reason": "..."},
        ...
      ],
    }
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Same env defaults as pointwise_rubric_judge so a fresh shell still
# gets the native-video / Gemini multimodal-url path.
os.environ.setdefault("ARR_VIDEO_INPUT_STRATEGY", "native_video")
os.environ.setdefault("ARR_VIDEO_BLOCK_TYPE", "gemini_multimodal_url")

# Re-use the per-dim rubric/yaml resolver from the pointwise driver so
# the two scripts share a single source of truth for asset lookup.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pointwise_rubric_judge import (  # type: ignore  # noqa: E402
    _default_config_path,
    _default_rubrics_file,
    _infer_dimension_from_dataset,
)


# ---------------------------------------------------------------------------
# Decision logic.
# ---------------------------------------------------------------------------
def _rank_to_decision(rank) -> str:
    """Map a 2-element rank list to {A, B, same}.

    The judger's listwise prompt enforces strict ranks ([1,2] or [2,1])
    in normal cases. A degenerate [1,1] (or any other shape) collapses
    to ``same`` — the safest interpretation when the LLM cannot
    distinguish the two candidates under the current rubric.
    """
    if not isinstance(rank, (list, tuple)) or len(rank) != 2:
        return "same"
    a, b = rank[0], rank[1]
    try:
        a_i, b_i = int(a), int(b)
    except (TypeError, ValueError):
        return "same"
    if a_i < b_i:
        return "A"
    if a_i > b_i:
        return "B"
    return "same"


def _gt_label_to_rank(label: str) -> list[int] | None:
    """A/B/same -> label_rank list. Returns None on an unknown label."""
    if label == "A":
        return [1, 2]
    if label == "B":
        return [2, 1]
    if label == "same":
        return [1, 1]
    return None


# ---------------------------------------------------------------------------
# Async driver.
# ---------------------------------------------------------------------------
async def _amain(args: argparse.Namespace) -> int:
    # Heavy imports kept local so --help works on a fresh shell.
    from judger import Judger  # type: ignore  # noqa: E402
    from rubric_pipeline.graders.schema import GraderRank  # type: ignore  # noqa: E402

    dim = args.dimension or _infer_dimension_from_dataset(args.dataset)
    if not dim:
        print(f"[fatal] could not infer dimension from --dataset path "
              f"{args.dataset!r}; pass --dimension explicitly.",
              file=sys.stderr)
        return 2

    rubrics_file = (Path(args.rubrics_file) if args.rubrics_file
                    else _default_rubrics_file(dim))
    config_path  = (Path(args.config_path)  if args.config_path
                    else _default_config_path(dim))
    if not rubrics_file.is_file():
        print(f"[fatal] rubrics file not found: {rubrics_file}",
              file=sys.stderr)
        return 2
    if not config_path.is_file():
        print(f"[fatal] config yaml not found: {config_path}",
              file=sys.stderr)
        return 2

    print(f"[rubric/pw] dimension     = {dim}", flush=True)
    print(f"[rubric/pw] axis (vgrb)   = {args.axis}", flush=True)
    print(f"[rubric/pw] dataset       = {args.dataset}", flush=True)
    print(f"[rubric/pw] rubrics_file  = {rubrics_file}", flush=True)
    print(f"[rubric/pw] config_path   = {config_path}", flush=True)
    print(f"[rubric/pw] base_url      = {args.base_url or '<from yaml>'}",
          flush=True)
    print(f"[rubric/pw] concurrency   = {args.concurrency}", flush=True)

    samples_raw = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if isinstance(samples_raw, dict):
        samples_raw = [samples_raw]
    if args.limit:
        samples_raw = samples_raw[: args.limit]
    if not samples_raw:
        print("[fatal] dataset has 0 samples after --limit", file=sys.stderr)
        return 2
    print(f"[rubric/pw] {len(samples_raw)} pairs", flush=True)

    judger = Judger(
        config_path=str(config_path),
        rubrics_file=str(rubrics_file),
        base_url=args.base_url,
    )
    judger._apply_eval_mode("listwise")  # noqa: SLF001
    t0 = time.time()
    await judger.initialize()
    print(f"[rubric/pw] judger initialized in {time.time()-t0:.1f}s "
          f"(eval_mode={judger.eval_mode})", flush=True)

    semaphore = asyncio.Semaphore(args.concurrency)

    async def _one(idx: int, item: dict) -> dict:
        sample_id = item.get("id") or str(idx)
        gt_label  = (item.get("gt_label") or item.get("gt")
                     or _decode_gt_from_rank(item.get("label_rank")))
        async with semaphore:
            try:
                result = await judger._evaluate_item(item)  # noqa: SLF001
            except Exception as exc:  # pragma: no cover — depends on VLM
                return {
                    "id":        sample_id,
                    "rank_pred": None,
                    "decision":  None,
                    "gt":        gt_label,
                    "error":     str(exc),
                }
        if not isinstance(result, GraderRank):
            return {
                "id":        sample_id,
                "rank_pred": None,
                "decision":  None,
                "gt":        gt_label,
                "error":     f"unexpected result type: {type(result).__name__}",
            }
        rank = list(result.rank)
        decision = _rank_to_decision(rank)
        # B4 (2026-05): the listwise prompt asks the LLM to also emit a
        # ``metadata.per_rubric`` list (per-sub-rubric ``rubric_rank`` and
        # ``weight``) plus an ``aggregation`` trace. Surface those raw
        # signals so a downstream mini-aggregator can replay alternative
        # formulas (weighted mean / unweighted mean / votes / min-rank /
        # ...) without re-calling the VLM. ``per_rubric`` may live at
        # the top level OR nested inside ``metadata`` depending on
        # whether the structured-output parser unwrapped it.
        meta = getattr(result, "metadata", None) or {}
        if not isinstance(meta, dict):
            meta = {}
        per_rubric = meta.get("per_rubric")
        if per_rubric is None and isinstance(meta.get("metadata"), dict):
            per_rubric = meta["metadata"].get("per_rubric")
        gating = meta.get("gating_decisions")
        if gating is None and isinstance(meta.get("metadata"), dict):
            gating = meta["metadata"].get("gating_decisions")
        aggregation = meta.get("aggregation")
        if aggregation is None and isinstance(meta.get("metadata"), dict):
            aggregation = meta["metadata"].get("aggregation")
        return {
            "id":          sample_id,
            "rank_pred":   rank,
            "decision":    decision,
            "gt":          gt_label,
            "correct":     (decision == gt_label) if gt_label else None,
            "reason":      getattr(result, "reason", None),
            "per_rubric":  per_rubric,
            "gating":      gating,
            "aggregation": aggregation,
        }

    t1 = time.time()
    records = await asyncio.gather(
        *[_one(i, s) for i, s in enumerate(samples_raw)],
    )
    print(f"[rubric/pw] all pairs scored in {time.time()-t1:.1f}s",
          flush=True)

    # ------------------------------------------------------------------
    # Aggregate to a summary dict matching pairwise_eval.py's schema so
    # summarize_external_vgrb.py can ingest both pointwise+threshold and
    # native-pairwise outputs identically.
    # ------------------------------------------------------------------
    confusion: dict[str, Counter] = {
        "A":    Counter(),
        "B":    Counter(),
        "same": Counter(),
    }
    n_total = 0
    n_missing = 0
    correct = 0
    correct_no_tie = 0
    n_no_tie = 0
    for r in records:
        if r.get("decision") is None or r.get("gt") is None:
            n_missing += 1
            continue
        n_total += 1
        gt = r["gt"]
        dec = r["decision"]
        if gt in confusion:
            confusion[gt][dec] += 1
        if dec == gt:
            correct += 1
        if gt in ("A", "B"):
            n_no_tie += 1
            if dec == gt:
                correct_no_tie += 1
    acc        = correct / n_total if n_total else 0.0
    acc_no_tie = correct_no_tie / n_no_tie if n_no_tie else 0.0

    failures = sum(1 for r in records if r.get("rank_pred") is None)
    summary = {
        "axis":        args.axis,
        "tau":         None,
        "n_total":     n_total,
        "n_missing":   n_missing,
        "n_no_tie":    n_no_tie,
        "acc":         round(acc, 4),
        "acc_no_tie":  round(acc_no_tie, 4),
        "confusion":   {gt: dict(c) for gt, c in confusion.items()},
    }

    out = {
        "method":     "RewardVerse (rubric, pairwise)",
        "axis":       args.axis,
        "rv_axis":    dim,
        "dataset":    args.dataset,
        "rubrics_file": str(rubrics_file),
        "config_path":  str(config_path),
        "total":      len(samples_raw),
        "parseable":  len(samples_raw) - failures,
        "failures":   failures,
        "summary":    summary,
        "per_pair":   records,
    }

    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    print(f"[rubric/pw] axis={args.axis} acc={acc:.4f} "
          f"acc_no_tie={acc_no_tie:.4f} "
          f"(n={n_total}, no_tie={n_no_tie}, missing={n_missing}, "
          f"failures={failures})", flush=True)
    for gt in ("A", "B", "same"):
        print(f"            GT={gt:4s}  pred="
              f"A:{confusion[gt]['A']:3d}  "
              f"B:{confusion[gt]['B']:3d}  "
              f"same:{confusion[gt]['same']:3d}", flush=True)
    print(f"[rubric/pw] wrote -> {out_path}", flush=True)
    return 0


def _decode_gt_from_rank(label_rank) -> str | None:
    """Inverse of _gt_label_to_rank, used when the input dataset only
    carries label_rank (the canonical listwise schema) and not a string
    gt label."""
    if not isinstance(label_rank, (list, tuple)) or len(label_rank) != 2:
        return None
    try:
        a, b = int(label_rank[0]), int(label_rank[1])
    except (TypeError, ValueError):
        return None
    if a == 1 and b == 2:
        return "A"
    if a == 2 and b == 1:
        return "B"
    if a == b:
        return "same"
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                   help="pairwise dataset JSON (rows with response=[A,B] "
                        "and label_rank or gt_label)")
    p.add_argument("--output",  required=True,
                   help="output JSON path")
    p.add_argument("--axis", required=True,
                   help="VGRB axis label this run targets (VQ/MQ/TA/Overall) "
                        "— recorded in the output for downstream summary "
                        "scripts; does NOT change rubric assets (use "
                        "--dimension for that).")
    p.add_argument("--dimension", default=None,
                   help="RewardVerse axis whose rubric to apply "
                        "(visual_quality / consistency / action / character "
                        "/ scene). Default: inferred from --dataset path.")
    p.add_argument("--rubrics_file", default=None)
    p.add_argument("--config_path",  default=None)
    p.add_argument("--base_url", default=None)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    raise SystemExit(asyncio.run(_amain(args)))


if __name__ == "__main__":
    main()
