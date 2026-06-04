"""vgrb_headtohead.py — head-to-head pairwise-acc table for VGRB.

Given the VGRB pairwise GT and *several* pointwise prediction files
(rubric / VS-v1.1 / VS-2 / ...), produce a single table of pairwise
accuracy on a chosen axis (VQ / MQ / TA / Overall) under multiple
delta-thresholds tau.

The table reports both the raw `acc` (which on VGRB is dominated by
the 'same' class — VQ has 55.5% 'same' as a constant baseline) and
`acc_no_tie` (accuracy on the A/B-only subset, which is the true
discriminative metric).

Each prediction file may either be:
  * a flat list[dict] with `id` + `pred_score` (canonical), OR
  * the rich rubric/VS dict with `predictions: [...]` inside, OR
  * the VS-v1.1 dict whose top-level `pred_score` is overall-like —
    in that case pass `--field raw_sub_scores.visual_quality` etc. to
    pull the actual subscore.

Usage::

    python scripts/v_arr_mvp/vgrb_headtohead.py \\
        --vgrb_subset .../vgrb_pairwise_200.json \\
        --axis VQ \\
        --taus 0.05,0.1,0.2,0.3,0.5 \\
        --method "RewardVerse=runs/.../rv/visual_quality.json" \\
        --method "VS-v1.1(overall_pred)=runs/.../vs11/visual_quality.json" \\
        --method "VS-v1.1(VQ_subscore)=runs/.../vs11/visual_quality.json:raw_sub_scores.visual quality" \\
        --output runs/.../_vq_headtohead/table.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _walk(d: dict, dotted: str) -> Any:
    """Drill into a dotted path, e.g. 'raw_sub_scores.visual quality'."""
    cur: Any = d
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _load_pred_index(spec: str) -> dict[str, float]:
    """spec is either '<path>' or '<path>:<dotted_field>'.
    Default field is 'pred_score' (with 'predict_score' fallback)."""
    if ":" in spec and not spec[1:3] == ":\\":  # avoid Windows-style
        path, _, field = spec.partition(":")
    else:
        path, field = spec, ""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("predictions") or rows.get("results") or []
    out: dict[str, float] = {}
    for r in rows:
        rid = r.get("id")
        if rid is None:
            continue
        if field:
            v = _walk(r, field)
        else:
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


def _eval(subset: list[dict], preds: dict[str, float],
          axis: str, tau: float) -> dict:
    n_total = correct = n_no_tie = correct_no_tie = n_missing = 0
    confusion: dict[str, Counter] = {"A": Counter(), "B": Counter(),
                                     "same": Counter()}
    deltas: list[float] = []
    for rec in subset:
        gt = rec["labels"][axis]
        pa = preds.get(f"{rec['id']}__A")
        pb = preds.get(f"{rec['id']}__B")
        if pa is None or pb is None:
            n_missing += 1
            continue
        n_total += 1
        delta = pa - pb
        deltas.append(delta)
        if abs(delta) < tau:
            decision = "same"
        elif delta > 0:
            decision = "A"
        else:
            decision = "B"
        confusion[gt][decision] += 1
        if decision == gt:
            correct += 1
        if gt in ("A", "B"):
            n_no_tie += 1
            if decision == gt:
                correct_no_tie += 1
    pred_same_rate = sum(1 for d in deltas if abs(d) < tau) / len(deltas) \
        if deltas else 0.0
    return {
        "tau":           tau,
        "n_total":       n_total,
        "n_no_tie":      n_no_tie,
        "n_missing":     n_missing,
        "acc":           round(correct / n_total, 4) if n_total else None,
        "acc_no_tie":    round(correct_no_tie / n_no_tie, 4)
                          if n_no_tie else None,
        "pred_same_rate": round(pred_same_rate, 4),
        "confusion": {gt: dict(cnts) for gt, cnts in confusion.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vgrb_subset", required=True)
    p.add_argument("--axis", required=True,
                   choices=("VQ", "MQ", "TA", "Overall"))
    p.add_argument("--taus", default="0.05,0.1,0.2,0.3,0.5",
                   help="comma-separated list of tau values")
    p.add_argument("--method", action="append", required=True,
                   metavar="NAME=PATH[:FIELD]",
                   help="repeat for each method, e.g. "
                        "'RewardVerse=runs/.../vq.json' or "
                        "'VS-v1.1(VQ)=runs/.../vq.json:raw_sub_scores.visual quality'")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    subset = json.loads(Path(args.vgrb_subset).read_text(encoding="utf-8"))
    taus = [float(x) for x in args.taus.split(",") if x.strip()]

    # GT class distribution for context.
    gt_dist = Counter(rec["labels"][args.axis] for rec in subset)
    gt_same_rate = gt_dist["same"] / sum(gt_dist.values())
    constant_same_acc = gt_same_rate            # all-'same' baseline
    constant_majority_no_tie = max(
        gt_dist["A"], gt_dist["B"]
    ) / max(1, gt_dist["A"] + gt_dist["B"])     # majority A/B baseline

    table: dict[str, Any] = {
        "axis":                 args.axis,
        "n_pairs":              len(subset),
        "gt_distribution":      dict(gt_dist),
        "gt_same_rate":         round(gt_same_rate, 4),
        "baselines": {
            "all_same_acc":             round(constant_same_acc, 4),
            "majority_no_tie_acc":      round(constant_majority_no_tie, 4),
        },
        "methods": {},
    }

    print(f"\n[head-to-head]  axis={args.axis}  "
          f"GT={dict(gt_dist)}  same-rate={gt_same_rate:.4f}")
    print(f"  Baseline acc(all-same) = {constant_same_acc:.4f} "
          f"| majority no-tie = {constant_majority_no_tie:.4f}\n")

    # column widths
    name_w = max(len(n.split("=", 1)[0]) for n in args.method)
    name_w = max(name_w, 18)
    header = (f"  {'method':<{name_w}}  " +
              "  ".join(f"τ={t:<5}" for t in taus))
    print(header + "  ←  acc / acc_no_tie  (pred-same-rate)")
    print("  " + "-" * (len(header) - 2))

    for spec in args.method:
        name, _, path_spec = spec.partition("=")
        preds = _load_pred_index(path_spec)
        per_tau = []
        cells = []
        for t in taus:
            res = _eval(subset, preds, args.axis, t)
            per_tau.append(res)
            if res["acc"] is None:
                cells.append("    n/a    ")
            else:
                cells.append(
                    f"{res['acc']:.3f}/{res['acc_no_tie']:.3f}({res['pred_same_rate']:.2f})"
                )
        # second print: just numbers in matching columns
        print(f"  {name:<{name_w}}  " + "  ".join(cells))
        table["methods"][name] = {
            "pred_path":  path_spec,
            "n_pred":     len(preds),
            "per_tau":    per_tau,
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(f"\n[head-to-head] -> {out_path}")


if __name__ == "__main__":
    main()
