"""prep_vgrb_pointwise_view.py — explode VGRB pairs into pointwise rows.

VGRB ships pair records (video_a + video_b + 4 pairwise labels).
Every pointwise scorer (rubric / VS-v1.1 / VS-2) operates on a flat
list of single-video samples. This script materialises a 2N-row
pointwise view where every row carries a stable id of the form
``<pair_id>__A`` or ``<pair_id>__B`` so pairwise_eval.py can recover
the pair structure later.

The rows DO NOT carry a ``label_score`` (VGRB has no pointwise GT).
That's fine — every downstream scorer treats ``label_score`` as
optional metadata; PLCC/SRCC against label_score will simply be NaN
in the per-axis JSON, which is the desired behaviour: VGRB is a
pairwise benchmark and we only care about pair-acc downstream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                   help="vgrb_pairwise_200.json from fetch_external_benchmarks.py")
    p.add_argument("--output", required=True,
                   help="path to the pointwise view JSON")
    p.add_argument("--dedup", action="store_true",
                   help="emit each unique video path at most once "
                        "(saves LLM/inference cost when the same mp4 "
                        "appears in multiple pairs). When set, the "
                        "row id is the video path's filename.")
    args = p.parse_args()

    src = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(src, list) or not src:
        raise SystemExit(f"[fatal] {args.input} is empty or not a list")

    rows: list[dict] = []
    seen: set[str] = set()
    for rec in src:
        for side in ("A", "B"):
            video_path = rec[f"video_{side.lower()}"]
            row_id = (
                Path(video_path).stem if args.dedup
                else f"{rec['id']}__{side}"
            )
            if args.dedup and row_id in seen:
                continue
            seen.add(row_id)
            rows.append({
                "id":          row_id,
                "query":       rec["query"],
                "response":    video_path,
                "task_type":   "t2v",
                # mirror canonical RewardVerse pointwise schema; no GT.
                "metadata": {
                    "vgrb_pair_id": rec["id"],
                    "side":         side,
                    "model":        rec[f"{side}_model"],
                },
            })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[VGRB/pointwise-view] {len(rows)} rows "
          f"(dedup={args.dedup}) -> {out_path}")


if __name__ == "__main__":
    main()
