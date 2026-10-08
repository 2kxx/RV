"""rm_training.stage3_deploy.merge_vgrb_dp
========================================
Merge per-rank ``rank_<NN>.json`` files from ``eval_on_vgrb_dp.py`` into
a single summary that matches the shape of the single-process
``eval_on_vgrb.py`` output.

Usage::

    python rm_training/stage3_deploy/merge_vgrb_dp.py \\
        --rank_dir runs/.../Overall_dp/ \\
        --output   runs/.../Overall.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def merge(rank_dir: Path, output: Path) -> None:
    rank_files = sorted(rank_dir.glob("rank_*.json"))
    if not rank_files:
        raise SystemExit(f"[merge] no rank_*.json under {rank_dir}")

    print(f"[merge] found {len(rank_files)} rank files under {rank_dir}",
          file=sys.stderr)

    per_pair: list[dict] = []
    confusion: dict[str, Counter] = {"A": Counter(), "B": Counter(), "same": Counter()}
    n_correct = 0
    n_no_tie = 0
    n_no_tie_correct = 0
    wall_max = 0.0
    wall_sum = 0.0

    head: dict | None = None  # shared metadata pulled from the first rank file
    n_total_dataset_seen: set[int] = set()
    seen_ids: set[str] = set()
    dup = 0

    for rf in rank_files:
        d = json.loads(rf.read_text(encoding="utf-8"))
        if head is None:
            head = {k: d.get(k) for k in (
                "axis", "dimension", "tau", "student_ckpt", "dataset",
                "world_size", "n_total_dataset",
            )}
        n_total_dataset_seen.add(int(d.get("n_total_dataset", -1)))
        wall_max = max(wall_max, float(d.get("wall_s", 0.0)))
        wall_sum += float(d.get("wall_s", 0.0))

        for row in d.get("per_pair", []):
            rid = row.get("id")
            if rid in seen_ids:
                dup += 1
                continue
            seen_ids.add(rid)

            decision = row.get("decision", "same")
            gt = row.get("gt", "same")
            correct = bool(row.get("correct", decision == gt))
            if correct:
                n_correct += 1
            if gt in ("A", "B"):
                n_no_tie += 1
                if correct:
                    n_no_tie_correct += 1
            confusion[gt][decision] += 1
            per_pair.append(row)

    # Restore original order for reproducibility, falling back to id when
    # orig_idx is missing (older rank files).
    per_pair.sort(key=lambda r: (r.get("orig_idx") if r.get("orig_idx") is not None else 1 << 30,
                                 r.get("id", "")))

    total = len(per_pair)
    acc = n_correct / total if total else 0.0
    acc_no_tie = n_no_tie_correct / n_no_tie if n_no_tie else 0.0

    if len(n_total_dataset_seen) > 1:
        print(f"[merge][warn] ranks disagree on n_total_dataset: "
              f"{n_total_dataset_seen}", file=sys.stderr)
    if dup:
        print(f"[merge][warn] dropped {dup} duplicate pair ids across ranks "
              f"(should be 0 with stride sharding)", file=sys.stderr)

    head = head or {}
    summary = {
        "method":         "RewardVerse student RM (pointwise → tau-decision) [DP merge]",
        "axis":           head.get("axis"),
        "dimension":      head.get("dimension"),
        "tau":            head.get("tau"),
        "world_size":     head.get("world_size"),
        "n_total":        total,
        "n_total_dataset": head.get("n_total_dataset"),
        "n_no_tie":       n_no_tie,
        "acc":            acc,
        "acc_no_tie":     acc_no_tie,
        "confusion":      {k: dict(v) for k, v in confusion.items()},
        "wall_s_max":     round(wall_max, 2),   # parallel wall-clock
        "wall_s_sum":     round(wall_sum, 2),   # cumulative GPU-seconds
        "student_ckpt":   head.get("student_ckpt"),
        "dataset":        head.get("dataset"),
        "n_rank_files":   len(rank_files),
        "n_duplicates":   dup,
    }
    out = {
        "summary":  summary,
        "per_pair": per_pair,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"[merge] wrote -> {output}  acc={acc:.3f}  acc_no_tie={acc_no_tie:.3f}  "
          f"n={total}  wall_max={wall_max:.1f}s",
          file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank_dir", required=True,
                   help="Directory containing rank_<NN>.json files.")
    p.add_argument("--output",   required=True)
    args = p.parse_args()
    merge(Path(args.rank_dir), Path(args.output))


if __name__ == "__main__":
    main()
