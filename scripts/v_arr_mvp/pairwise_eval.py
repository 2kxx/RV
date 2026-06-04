"""pairwise_eval.py — turn pointwise predictions into pairwise accuracy.

Given the VGRB 200-pair subset (each record carries video_a / video_b
plus the pairwise GT labels {VQ, MQ, TA, Overall} in {'A','B','same'})
and a *pointwise* prediction file (one float per video), this script
computes per-axis pairwise accuracy via a delta-threshold rule:

    delta = pred_A - pred_B
    if abs(delta) < tau:  decision = 'same'
    elif delta > 0:       decision = 'A'
    else:                 decision = 'B'

Reports:
  * acc           — fraction matching the GT label exactly.
  * acc_no_tie    — accuracy on the subset where GT is 'A' or 'B' only
                    (i.e. ignoring 'same' rows; useful because the
                    'same' decision is sensitive to tau).
  * confusion     — 3x3 matrix (rows=GT, cols=pred).

The pointwise prediction file is expected to be a list[dict] with
fields ``id`` and ``pred_score`` (the convention used by every
RewardVerse pointwise output: rubric_judge writes ``predict_score``,
videoscore / videoscore2 write ``pred_score``; we accept both).

Usage::

    python scripts/v_arr_mvp/pairwise_eval.py \\
        --vgrb_subset /group/.../vgrb_pairwise_200.json \\
        --pred_a /tmp/run_a.json \\
        --pred_b /tmp/run_b.json \\
        --axis VQ \\
        --tau 0.3 \\
        --output runs/.../vgrb_VQ_eval.json

Note: ``--pred_a`` / ``--pred_b`` may point at the SAME file when the
predictor scored the union of all videos (recommended). The script
will still build a {video_path -> pred} index by ``id`` field which
encodes the original VGRB pair index + side.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load_pred_index(pred_path: str) -> dict[str, float]:
    """{record_id -> predicted scalar}. Accepts both 'pred_score' and
    'predict_score' keys."""
    rows = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        # rubric_judge wraps results under 'predictions'
        rows = rows.get("predictions") or rows.get("results") or []
    out: dict[str, float] = {}
    for r in rows:
        rid = r.get("id")
        if rid is None:
            continue
        v = r.get("pred_score")
        if v is None:
            v = r.get("predict_score")
        if v is None:
            continue
        try:
            out[str(rid)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vgrb_subset", required=True,
                   help="vgrb_pairwise_200.json from fetch_external_benchmarks.py")
    p.add_argument("--pred", required=True,
                   help="pointwise prediction JSON whose ids match "
                        "<vgrb_id>__A / <vgrb_id>__B (see "
                        "prep_vgrb_pointwise_view.py)")
    p.add_argument("--axis", required=True,
                   choices=("VQ", "MQ", "TA", "Overall"),
                   help="VGRB pairwise axis to evaluate")
    p.add_argument("--tau", type=float, default=0.3,
                   help="absolute-delta threshold for 'same' decision")
    p.add_argument("--output", required=True,
                   help="where to write the evaluation JSON")
    args = p.parse_args()

    subset = json.loads(Path(args.vgrb_subset).read_text(encoding="utf-8"))
    preds  = _load_pred_index(args.pred)

    n_total = 0
    n_missing = 0
    correct = 0
    correct_no_tie = 0
    n_no_tie = 0
    confusion: dict[str, Counter] = {
        "A":    Counter(),
        "B":    Counter(),
        "same": Counter(),
    }
    per_pair: list[dict] = []

    for rec in subset:
        rid = rec["id"]
        gt_label = rec["labels"][args.axis]
        id_a = f"{rid}__A"
        id_b = f"{rid}__B"
        pa = preds.get(id_a)
        pb = preds.get(id_b)
        if pa is None or pb is None:
            n_missing += 1
            continue
        n_total += 1
        delta = pa - pb
        if abs(delta) < args.tau:
            decision = "same"
        elif delta > 0:
            decision = "A"
        else:
            decision = "B"
        confusion[gt_label][decision] += 1
        if decision == gt_label:
            correct += 1
        if gt_label in ("A", "B"):
            n_no_tie += 1
            if decision == gt_label:
                correct_no_tie += 1
        per_pair.append({
            "id":       rid,
            "pred_a":   round(pa, 4),
            "pred_b":   round(pb, 4),
            "delta":    round(delta, 4),
            "decision": decision,
            "gt":       gt_label,
            "correct":  decision == gt_label,
        })

    acc        = correct / n_total if n_total else 0.0
    acc_no_tie = correct_no_tie / n_no_tie if n_no_tie else 0.0

    summary = {
        "axis":        args.axis,
        "tau":         args.tau,
        "n_total":     n_total,
        "n_missing":   n_missing,
        "n_no_tie":    n_no_tie,
        "acc":         round(acc, 4),
        "acc_no_tie":  round(acc_no_tie, 4),
        "confusion": {gt: dict(cnts) for gt, cnts in confusion.items()},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "per_pair": per_pair},
                   indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[pairwise] axis={args.axis} tau={args.tau} "
          f"acc={acc:.4f} acc_no_tie={acc_no_tie:.4f} "
          f"(n={n_total}, no_tie={n_no_tie}, missing={n_missing})")
    for gt in ("A", "B", "same"):
        print(f"           GT={gt:4s}  pred="
              f"A:{confusion[gt]['A']:3d}  "
              f"B:{confusion[gt]['B']:3d}  "
              f"same:{confusion[gt]['same']:3d}")
    print(f"[pairwise] -> {out_path}")


if __name__ == "__main__":
    main()
