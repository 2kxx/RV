"""build_overall_pred.py — average pointwise preds across (VQ, MQ, TA) axes.

VGRB's 'Overall' label has no direct RewardVerse axis. We derive an
'Overall' pointwise prediction by averaging the predictions from the
three axis-aligned runs (visual_quality + action + character), giving
each video a single scalar that pairwise_eval.py can compare against
the GT 'Overall' label via the same delta-threshold rule.

Inputs are pointwise prediction JSONs whose rows carry ``id`` plus
``pred_score`` (or ``predict_score``). Output mirrors that shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _index(path: str) -> dict[str, float]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(rows, dict):
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
    p.add_argument("--pred_vq", required=True)
    p.add_argument("--pred_mq", required=True)
    p.add_argument("--pred_ta", required=True)
    p.add_argument("--output",  required=True)
    args = p.parse_args()

    vq = _index(args.pred_vq)
    mq = _index(args.pred_mq)
    ta = _index(args.pred_ta)

    common = sorted(set(vq) & set(mq) & set(ta))
    rows: list[dict] = []
    for rid in common:
        rows.append({
            "id":         rid,
            "pred_score": round((vq[rid] + mq[rid] + ta[rid]) / 3.0, 4),
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[overall] {len(rows)} rows (intersection of 3 axis preds) "
          f"-> {out_path}")


if __name__ == "__main__":
    main()
