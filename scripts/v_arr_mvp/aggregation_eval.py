"""aggregation_eval.py — explore per-rubric aggregation strategies on VGRB.

Inputs
------
* ``--vgrb_subset``  vgrb_pairwise_200.json  (same as pairwise_eval.py)
* ``--pred``         pointwise prediction JSON produced by
                     ``pointwise_rubric_judge.py`` AFTER the per_rubric patch.
                     Each prediction row must carry::
                       { id, pred_score, per_rubric: [
                           {rubric_index, verdict, evidence}, ... ] }

What it does
------------
For each video the LLM returned K (typically 5) per-rubric *verdicts*
v_k ∈ {1..5}. Pointwise pipeline currently reduces them via plain
``mean(v_k)``; this is the reduction that gets thresholded by ``tau``
to produce the pair decision A / B / same.

The Rubric framework was *trained* on pairwise comparisons, so the
per-rubric verdicts are arguably more meaningful than their mean. This
script therefore re-aggregates the same verdicts under several
strategies and compares the resulting pair accuracy:

  * mean           — current baseline (sanity check; should match
                     pairwise_eval.py output up to numerical noise).
  * min            — bottleneck rubric dominates the score.
  * median         — outlier-robust center.
  * weighted_mean  — per-rubric weights derived from how *informative*
                     that rubric was on this dataset (here: inverse
                     frequency of the modal verdict, computed on the
                     pred file itself — a cheap data-driven prior).
                     Falls back to plain mean if a rubric is missing.
  * lp4            — soft-min: score = 5 - ( mean( (5-v_k)^4 ) )^(1/4).
                     Penalises low rubrics more than mean, less than
                     hard min.
  * votes          — per-rubric A/B/same vote, then majority vote at
                     pair level. Each rubric independently compares
                     its own v_k(A) vs v_k(B) using a small per-vote
                     threshold ``tau_v`` (default 0 → strict).
                     Pair decision: argmax over {A, B, same} where the
                     ``same`` count must exceed ``theta_same`` votes
                     to win; otherwise pair-level tie → same.

For the first 5 (scalar) strategies we sweep ``--tau`` (delta-threshold
on aggregated score) just like pairwise_eval.py. For ``votes`` we
sweep ``--tau_v`` (per-rubric tie threshold) and ``--theta_same``
(min ``same`` votes to declare pair-level tie).

Output
------
A single JSON with::
  { axis, n_total, n_no_tie, n_missing,
    by_strategy: {
      "mean":          {"tau": 0.3, "acc": ..., "acc_no_tie": ..., "confusion": ...},
      "min":           {...},
      ...
      "votes":         {"tau_v": 0, "theta_same": 2, "acc": ..., ...},
    },
    sweeps: { "mean": [{tau, acc, acc_no_tie}, ...], ... }
  }

Plus a stdout markdown-style table summarising the best (acc_no_tie)
config per strategy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _load_pred_index(pred_path: str) -> dict[str, dict[str, Any]]:
    """{record_id -> {pred_score, verdicts: list[float]}}"""
    rows = json.loads(Path(pred_path).read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("predictions") or rows.get("results") or []
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        rid = r.get("id")
        if rid is None:
            continue
        verdicts: list[float] = []
        per = r.get("per_rubric")
        if isinstance(per, list):
            # sort by rubric_index when available so weighted_mean can
            # consistently align weights across A and B sides.
            per_sorted = sorted(
                per,
                key=lambda x: x.get("rubric_index", 0)
                if isinstance(x, dict) else 0,
            )
            for item in per_sorted:
                if not isinstance(item, dict):
                    continue
                v = item.get("verdict")
                try:
                    verdicts.append(float(v))
                except (TypeError, ValueError):
                    continue
        ps = r.get("pred_score")
        if ps is None:
            ps = r.get("predict_score")
        try:
            ps_f = float(ps) if ps is not None else None
        except (TypeError, ValueError):
            ps_f = None
        out[str(rid)] = {"pred_score": ps_f, "verdicts": verdicts}
    return out


# ---------------------------------------------------------------------------
# Aggregation strategies (scalar)
# ---------------------------------------------------------------------------

def agg_mean(vs: list[float]) -> float:
    return sum(vs) / len(vs) if vs else float("nan")


def agg_min(vs: list[float]) -> float:
    return min(vs) if vs else float("nan")


def agg_median(vs: list[float]) -> float:
    if not vs:
        return float("nan")
    s = sorted(vs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def agg_lp4(vs: list[float]) -> float:
    """Soft-min on the 1..5 scale: penalises low verdicts harder than
    mean, less than hard min. Operates on (5 - v) so lower verdict ->
    larger penalty term."""
    if not vs:
        return float("nan")
    p = 4.0
    pen = sum((5.0 - v) ** p for v in vs) / len(vs)
    return 5.0 - pen ** (1.0 / p)


def _build_rubric_weights(pred_index: dict[str, dict[str, Any]]) -> list[float]:
    """Cheap data-driven weights: weight rubric k by 1 / freq(modal verdict).
    Rubrics whose verdict is concentrated on a single value carry less
    discriminative power -> down-weighted. Returned list is aligned with
    rubric position (0..K-1); length K = max verdicts seen."""
    by_pos: dict[int, Counter] = defaultdict(Counter)
    K = 0
    for entry in pred_index.values():
        vs = entry["verdicts"]
        K = max(K, len(vs))
        for k, v in enumerate(vs):
            by_pos[k][v] += 1
    weights: list[float] = []
    for k in range(K):
        c = by_pos[k]
        total = sum(c.values()) or 1
        modal = max(c.values()) if c else total
        # weight ∝ entropy proxy; here use 1 - modal_freq.
        w = max(0.05, 1.0 - modal / total)
        weights.append(w)
    s = sum(weights) or 1.0
    return [w / s * len(weights) for w in weights]  # mean-normalised


def agg_weighted_mean(vs: list[float], weights: list[float]) -> float:
    if not vs:
        return float("nan")
    n = min(len(vs), len(weights))
    if n == 0:
        return float("nan")
    num = sum(vs[i] * weights[i] for i in range(n))
    den = sum(weights[:n]) or 1.0
    return num / den


SCALAR_STRATS: dict[str, Any] = {
    "mean":          lambda vs, ctx: agg_mean(vs),
    "min":           lambda vs, ctx: agg_min(vs),
    "median":        lambda vs, ctx: agg_median(vs),
    "lp4":           lambda vs, ctx: agg_lp4(vs),
    "weighted_mean": lambda vs, ctx: agg_weighted_mean(vs, ctx["weights"]),
}


# ---------------------------------------------------------------------------
# Pair-level decision helpers
# ---------------------------------------------------------------------------

def _decision_from_delta(pa: float, pb: float, tau: float) -> str:
    delta = pa - pb
    if abs(delta) < tau:
        return "same"
    return "A" if delta > 0 else "B"


def _eval_scalar(
    subset: list[dict],
    pred_index: dict[str, dict[str, Any]],
    score_fn,
    ctx: dict[str, Any],
    axis: str,
    tau: float,
) -> dict[str, Any]:
    n_total = n_missing = correct = correct_no_tie = n_no_tie = 0
    confusion: dict[str, Counter] = {"A": Counter(), "B": Counter(), "same": Counter()}
    for rec in subset:
        rid = rec["id"]
        gt = rec["labels"][axis]
        ea = pred_index.get(f"{rid}__A")
        eb = pred_index.get(f"{rid}__B")
        if ea is None or eb is None or not ea["verdicts"] or not eb["verdicts"]:
            n_missing += 1
            continue
        pa = score_fn(ea["verdicts"], ctx)
        pb = score_fn(eb["verdicts"], ctx)
        if pa != pa or pb != pb:  # nan
            n_missing += 1
            continue
        n_total += 1
        decision = _decision_from_delta(pa, pb, tau)
        confusion[gt][decision] += 1
        if decision == gt:
            correct += 1
        if gt in ("A", "B"):
            n_no_tie += 1
            if decision == gt:
                correct_no_tie += 1
    return {
        "tau":        tau,
        "n_total":    n_total,
        "n_missing":  n_missing,
        "n_no_tie":   n_no_tie,
        "acc":        round(correct / n_total, 4) if n_total else 0.0,
        "acc_no_tie": round(correct_no_tie / n_no_tie, 4) if n_no_tie else 0.0,
        "confusion":  {gt: dict(cnt) for gt, cnt in confusion.items()},
    }


def _eval_votes(
    subset: list[dict],
    pred_index: dict[str, dict[str, Any]],
    axis: str,
    tau_v: float,
    theta_same: int,
) -> dict[str, Any]:
    """Per-rubric votes -> pair decision."""
    n_total = n_missing = correct = correct_no_tie = n_no_tie = 0
    confusion: dict[str, Counter] = {"A": Counter(), "B": Counter(), "same": Counter()}
    for rec in subset:
        rid = rec["id"]
        gt = rec["labels"][axis]
        ea = pred_index.get(f"{rid}__A")
        eb = pred_index.get(f"{rid}__B")
        if ea is None or eb is None or not ea["verdicts"] or not eb["verdicts"]:
            n_missing += 1
            continue
        va, vb = ea["verdicts"], eb["verdicts"]
        K = min(len(va), len(vb))
        if K == 0:
            n_missing += 1
            continue
        votes = Counter()
        for k in range(K):
            d = va[k] - vb[k]
            if abs(d) <= tau_v:
                votes["same"] += 1
            elif d > 0:
                votes["A"] += 1
            else:
                votes["B"] += 1
        n_total += 1
        # decision rule: 'same' wins iff its count > theta_same AND >= max(A, B);
        # else argmax(A, B) breaks ties towards 'same'.
        a_cnt, b_cnt, s_cnt = votes["A"], votes["B"], votes["same"]
        if s_cnt >= theta_same and s_cnt >= max(a_cnt, b_cnt):
            decision = "same"
        elif a_cnt > b_cnt:
            decision = "A"
        elif b_cnt > a_cnt:
            decision = "B"
        else:
            decision = "same"
        confusion[gt][decision] += 1
        if decision == gt:
            correct += 1
        if gt in ("A", "B"):
            n_no_tie += 1
            if decision == gt:
                correct_no_tie += 1
    return {
        "tau_v":      tau_v,
        "theta_same": theta_same,
        "n_total":    n_total,
        "n_missing":  n_missing,
        "n_no_tie":   n_no_tie,
        "acc":        round(correct / n_total, 4) if n_total else 0.0,
        "acc_no_tie": round(correct_no_tie / n_no_tie, 4) if n_no_tie else 0.0,
        "confusion":  {gt: dict(cnt) for gt, cnt in confusion.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vgrb_subset", required=True)
    p.add_argument("--pred",        required=True,
                   help="pointwise prediction JSON enriched with per_rubric verdicts")
    p.add_argument("--axis",        required=True,
                   choices=("VQ", "MQ", "TA", "Overall"))
    p.add_argument("--tau_grid",    default="0.0,0.1,0.2,0.3,0.4,0.5,0.7,1.0",
                   help="comma-separated tau values for scalar strategies")
    p.add_argument("--tau_v_grid",  default="0,1",
                   help="comma-separated per-rubric tie thresholds for 'votes'")
    p.add_argument("--theta_same_grid", default="2,3",
                   help="comma-separated min 'same' votes for pair-level tie")
    p.add_argument("--output",      required=True)
    args = p.parse_args()

    subset = json.loads(Path(args.vgrb_subset).read_text(encoding="utf-8"))
    pred_index = _load_pred_index(args.pred)

    # --- preflight: report verdict coverage --------------------------------
    n_with = sum(1 for v in pred_index.values() if v["verdicts"])
    K_seen = max((len(v["verdicts"]) for v in pred_index.values()), default=0)
    print(f"[agg] loaded {len(pred_index)} pred rows, {n_with} carry per_rubric, "
          f"max K={K_seen}")

    weights = _build_rubric_weights(pred_index)
    print(f"[agg] rubric weights = {[round(w, 3) for w in weights]}")
    ctx = {"weights": weights}

    tau_grid = [float(x) for x in args.tau_grid.split(",") if x.strip()]
    tauv_grid = [float(x) for x in args.tau_v_grid.split(",") if x.strip()]
    theta_grid = [int(x) for x in args.theta_same_grid.split(",") if x.strip()]

    sweeps: dict[str, list[dict[str, Any]]] = {}
    by_strategy: dict[str, dict[str, Any]] = {}

    # --- scalar strategies --------------------------------------------------
    for name, fn in SCALAR_STRATS.items():
        sweep_rows: list[dict[str, Any]] = []
        for tau in tau_grid:
            r = _eval_scalar(subset, pred_index, fn, ctx, args.axis, tau)
            sweep_rows.append(r)
        sweeps[name] = sweep_rows
        # pick best by acc_no_tie; tie-break by acc.
        best = max(sweep_rows, key=lambda r: (r["acc_no_tie"], r["acc"]))
        by_strategy[name] = best

    # --- votes -------------------------------------------------------------
    sweep_votes: list[dict[str, Any]] = []
    for tv in tauv_grid:
        for th in theta_grid:
            r = _eval_votes(subset, pred_index, args.axis, tv, th)
            sweep_votes.append(r)
    sweeps["votes"] = sweep_votes
    by_strategy["votes"] = max(sweep_votes,
                               key=lambda r: (r["acc_no_tie"], r["acc"]))

    # --- write ------------------------------------------------------------
    out = {
        "axis":         args.axis,
        "pred_path":    args.pred,
        "n_pred_rows":  len(pred_index),
        "rubric_weights": weights,
        "by_strategy":  by_strategy,
        "sweeps":       sweeps,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    # --- stdout summary ---------------------------------------------------
    print(f"\n[agg] axis={args.axis}  best per strategy (by acc_no_tie):")
    print(f"  {'strategy':<14} {'acc':>6}  {'acc_no_tie':>10}  config")
    for name in ["mean", "min", "median", "weighted_mean", "lp4", "votes"]:
        b = by_strategy[name]
        if name == "votes":
            cfg = f"tau_v={b['tau_v']}, theta_same={b['theta_same']}"
        else:
            cfg = f"tau={b['tau']}"
        print(f"  {name:<14} {b['acc']:>6.4f}  {b['acc_no_tie']:>10.4f}  {cfg}")
    print(f"[agg] -> {out_path}")


if __name__ == "__main__":
    main()
