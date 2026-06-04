"""prep_vgrb_pairwise_view.py — turn VGRB pair records into a listwise dataset.

VGRB ships pair records with 4 separate axis labels (VQ / MQ / TA /
Overall). The judger's listwise mode wants ONE ``label_rank`` per row.
This script produces ONE view JSON per VGRB axis, where each row
carries::

    {
      "id":         "<pair id>",
      "query":      "<text prompt>",
      "response":   ["<video_a path>", "<video_b path>"],
      "label_rank": [1, 2] | [2, 1] | [1, 1],
      "gt_label":   "A" | "B" | "same",     # convenience copy
      "task_type":  "t2v",
      "metadata":   {...},
    }

Intended for pairwise_rubric_judge.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_VGRB_AXES = ("VQ", "MQ", "TA", "Overall")


def _gt_to_rank(gt: str) -> list[int] | None:
    if gt == "A":
        return [1, 2]
    if gt == "B":
        return [2, 1]
    if gt == "same":
        return [1, 1]
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                   help="vgrb_pairwise_200.json")
    p.add_argument("--out_dir", required=True,
                   help="output directory; one <axis>.json per VGRB axis")
    args = p.parse_args()

    src = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(src, list) or not src:
        raise SystemExit(f"[fatal] {args.input} is empty or not a list")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index: dict[str, dict] = {}
    for axis in _VGRB_AXES:
        rows: list[dict] = []
        skipped = 0
        for rec in src:
            gt = (rec.get("labels") or {}).get(axis)
            if gt is None:
                skipped += 1
                continue
            rk = _gt_to_rank(gt)
            if rk is None:
                skipped += 1
                continue
            rows.append({
                "id":         rec["id"],
                "query":      rec.get("query", ""),
                "response":   [rec["video_a"], rec["video_b"]],
                "label_rank": rk,
                "gt_label":   gt,
                "task_type":  "t2v",
                "metadata": {
                    "vgrb_axis": axis,
                    "A_model":   rec.get("A_model"),
                    "B_model":   rec.get("B_model"),
                },
            })
        if not rows:
            print(f"[warn] axis '{axis}': 0 rows; nothing written")
            continue
        out_path = out_dir / f"{axis}.json"
        out_path.write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        index[axis] = {"file": str(out_path), "n": len(rows),
                       "skipped": skipped}
        print(f"[VGRB/pairwise-view] axis={axis:8s}  n={len(rows)}  "
              f"skipped={skipped}  -> {out_path}")

    (out_dir / "_axis_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[VGRB/pairwise-view] index -> {out_dir / '_axis_index.json'}")


if __name__ == "__main__":
    main()
