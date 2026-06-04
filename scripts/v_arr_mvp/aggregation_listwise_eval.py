"""aggregation_listwise_eval.py — clean main-line aggregation evaluator
for the listwise pairwise output (`pairwise_listwise_perrubric/{axis}.json`).

Scope
-----
Each pair (vgrb_xxxxx) carries `decision` (LLM's direct A/B/same call)
plus a `per_rubric` list whose entries record the LLM's per-rubric
preference as ``rubric_rank=[ra, rb] in {[1,2], [2,1], [1,1]}`` along
with a fixed weight. We:

1. Map each ``rubric_rank`` to a per-side score (1.0 / 1.5 / 2.0).
2. Aggregate the K=5 per-rubric scores into one scalar per side using
   3 strategies (see below).
3. Decide A/B/same with a tau threshold over ``pa - pb``.
4. Compare against ``listwise_direct`` (the LLM's own decision).

Aggregation strategies (deliberately small)
------------------------------------------
- ``mean``           — uniform average; the simplest baseline.
- ``weighted_mean``  — weighted by the LLM-emitted per-rubric weight;
                        useful when those weights are non-uniform.
- ``lp4``            — soft-min (4-norm) on (TOP - score); penalises a
                        weak rubric harder than mean while staying smooth.

Strategies that were tried and dropped
--------------------------------------
- ``min``     — produced 22%-44% pa==pb ties on this dataset and lost to
                ``mean`` on every axis once the boundary bug was fixed.
- ``median``  — K=5 is too small; positionally sensitive and dominated
                by ``mean``.
- ``votes``   — 2-level rank signal gives no headroom over ``mean``.

Metrics
-------
- ``acc``        — 3-way accuracy (counts ``same`` predictions).
- ``acc_no_tie`` — 2-way accuracy on GT in {A, B}.
- ``macro_acc``  — mean per-class recall over A/B/same; surfaces the
                    fact that almost no method actually predicts ``same``.
- ``tie_rate``   — share of pairs where ``pa == pb`` (tau-independent);
                    a guardrail against a method ranking high purely
                    because the boundary handling silently broke ties.

Output
------
JSON with::

    {
      "axis": ...,
      "n_pairs": 200, "n_with_rubric": 200, "max_K": 5,
      "rank_dist": {"1-2": ..., "2-1": ..., "1-1": ...},
      "listwise_direct": {acc, acc_no_tie, macro_acc, confusion},
      "by_strategy": {<name>: <best row>},
      "sweeps":      {<name>: [<row at each tau>]},
    }

A short markdown table is also printed to stdout.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def _load_listwise(path: str) -> list[dict[str, Any]]:
    """Return the per_pair list from a listwise pairwise output."""
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(blob, dict):
        return blob.get("per_pair") or blob.get("predictions") or []
    return blob


def _rank_to_scores(rk) -> tuple[float, float]:
    """Map a rubric_rank pair to per-side scores. Ties -> 1.5/1.5.
    Anything malformed (length<2, non-int) -> NaN/NaN to be filtered."""
    if not rk or len(rk) < 2:
        return float("nan"), float("nan")
    try:
        ra, rb = int(rk[0]), int(rk[1])
    except (TypeError, ValueError):
        return float("nan"), float("nan")
    # Defensive: out-of-protocol rank values (e.g. (0,0)) -> treat as tie.
    if ra not in (1, 2) or rb not in (1, 2):
        if ra == rb:
            return 1.5, 1.5
        return float("nan"), float("nan")
    if ra == rb:
        return 1.5, 1.5
    return (2.0, 1.0) if ra < rb else (1.0, 2.0)


def _extract_per_pair(rec: dict[str, Any]) -> dict[str, Any] | None:
    """Pull (scores_a, scores_b, weights) from one per_pair row.
    Returns None if no usable per_rubric entry."""
    per = rec.get("per_rubric")
    if not isinstance(per, list) or not per:
        return None
    per_sorted = sorted(
        per,
        key=lambda x: x.get("rubric_index", 0) if isinstance(x, dict) else 0,
    )
    sa: list[float] = []
    sb: list[float] = []
    ws: list[float] = []
    raw_ranks: list[tuple[int, int]] = []
    for item in per_sorted:
        if not isinstance(item, dict):
            continue
        rk = item.get("rubric_rank")
        s_a, s_b = _rank_to_scores(rk)
        if s_a != s_a or s_b != s_b:  # NaN
            continue
        try:
            w = float(item.get("weight", 0.0))
        except (TypeError, ValueError):
            w = 0.0
        sa.append(s_a)
        sb.append(s_b)
        ws.append(w if w > 0 else 0.0)
        raw_ranks.append((int(rk[0]), int(rk[1])))
    if not sa:
        return None
    return {"scores_a": sa, "scores_b": sb, "weights": ws, "ranks": raw_ranks}


# ---------------------------------------------------------------------------
# Aggregations  (only the 3 we kept after empirical screening)
# ---------------------------------------------------------------------------

def agg_mean(vs: list[float]) -> float:
    return sum(vs) / len(vs) if vs else float("nan")


def agg_lp4(vs: list[float]) -> float:
    """Soft-min on the (1.0, 2.0) scale: pen = mean((TOP - v)^4); score = TOP - pen^(1/4).
    Penalises a weak rubric harder than mean, smoother than hard min."""
    if not vs:
        return float("nan")
    p = 4.0
    top = 2.0
    pen = sum((top - v) ** p for v in vs) / len(vs)
    return top - pen ** (1.0 / p)


def agg_weighted_mean(vs: list[float], weights: list[float]) -> float:
    """Weighted mean using LLM-emitted per-rubric weights; falls back to
    plain mean when all weights are zero."""
    if not vs:
        return float("nan")
    n = min(len(vs), len(weights))
    w_sum = sum(weights[:n])
    if w_sum <= 0:
        return agg_mean(vs[:n])
    return sum(vs[i] * weights[i] for i in range(n)) / w_sum


# ---------------------------------------------------------------------------
# Decision + metrics
# ---------------------------------------------------------------------------

def _decision_from_delta(pa: float, pb: float, tau: float) -> str:
    """A / B / same from (pa, pb) under tolerance tau.

    Note: ``<=`` (NOT ``<``) so that ``pa == pb`` is judged as ``same`` even
    at tau=0. The historic strict-less-than form silently routed exact
    ties into 'B' (since ``0 > 0`` is False) and inflated acc_no_tie for
    tie-prone aggregations.
    """
    delta = pa - pb
    if abs(delta) <= tau:
        return "same"
    return "A" if delta > 0 else "B"


def _macro_acc(confusion: dict[str, Counter]) -> float:
    """Macro recall over A/B/same (only classes with samples)."""
    recalls: list[float] = []
    for c in ("A", "B", "same"):
        cnt = confusion.get(c) or Counter()
        total = sum(cnt.values())
        if total > 0:
            recalls.append(cnt.get(c, 0) / total)
    return sum(recalls) / len(recalls) if recalls else 0.0


def _summary_row(
    tau: float,
    n_total: int, n_missing: int, n_no_tie: int,
    n_pa_eq_pb: int, correct: int, correct_no_tie: int,
    confusion: dict[str, Counter],
) -> dict[str, Any]:
    return {
        "tau":        tau,
        "n_total":    n_total,
        "n_missing":  n_missing,
        "n_no_tie":   n_no_tie,
        "tie_rate":   round(n_pa_eq_pb / n_total, 4) if n_total else 0.0,
        "acc":        round(correct / n_total, 4) if n_total else 0.0,
        "acc_no_tie": round(correct_no_tie / n_no_tie, 4) if n_no_tie else 0.0,
        "macro_acc":  round(_macro_acc(confusion), 4),
        "confusion":  {gt: dict(cnt) for gt, cnt in confusion.items()},
    }


def _eval_scalar(
    per_pair: list[dict[str, Any]],
    score_fn: Callable,
    use_weights: bool,
    tau: float,
) -> dict[str, Any]:
    n_total = n_missing = correct = correct_no_tie = n_no_tie = n_pa_eq_pb = 0
    confusion: dict[str, Counter] = {"A": Counter(), "B": Counter(), "same": Counter()}
    for rec in per_pair:
        gt = rec.get("gt")
        if gt not in ("A", "B", "same"):
            n_missing += 1
            continue
        ext = _extract_per_pair(rec)
        if ext is None:
            n_missing += 1
            continue
        if use_weights:
            pa = score_fn(ext["scores_a"], ext["weights"])
            pb = score_fn(ext["scores_b"], ext["weights"])
        else:
            pa = score_fn(ext["scores_a"])
            pb = score_fn(ext["scores_b"])
        if pa != pa or pb != pb:
            n_missing += 1
            continue
        n_total += 1
        if pa == pb:
            n_pa_eq_pb += 1
        decision = _decision_from_delta(pa, pb, tau)
        confusion[gt][decision] += 1
        if decision == gt:
            correct += 1
        if gt in ("A", "B"):
            n_no_tie += 1
            if decision == gt:
                correct_no_tie += 1
    return _summary_row(
        tau, n_total, n_missing, n_no_tie,
        n_pa_eq_pb, correct, correct_no_tie, confusion,
    )


def _eval_listwise_direct(per_pair: list[dict[str, Any]]) -> dict[str, Any]:
    """LLM's own decision baseline (no aggregation, no tau)."""
    n_total = correct = correct_no_tie = n_no_tie = 0
    confusion: dict[str, Counter] = {"A": Counter(), "B": Counter(), "same": Counter()}
    for rec in per_pair:
        gt = rec.get("gt")
        decision = rec.get("decision")
        if gt not in ("A", "B", "same") or decision not in ("A", "B", "same"):
            continue
        n_total += 1
        confusion[gt][decision] += 1
        if decision == gt:
            correct += 1
        if gt in ("A", "B"):
            n_no_tie += 1
            if decision == gt:
                correct_no_tie += 1
    return {
        "n_total":    n_total,
        "n_no_tie":   n_no_tie,
        "acc":        round(correct / n_total, 4) if n_total else 0.0,
        "acc_no_tie": round(correct_no_tie / n_no_tie, 4) if n_no_tie else 0.0,
        "macro_acc":  round(_macro_acc(confusion), 4),
        "confusion":  {gt: dict(cnt) for gt, cnt in confusion.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--listwise_pred", required=True,
                   help="path to pairwise_listwise_perrubric/{axis}.json")
    p.add_argument("--axis", required=True,
                   choices=("VQ", "MQ", "TA", "Overall"))
    p.add_argument("--tau_grid",
                   default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5",
                   help="comma-separated tau values for scalar strategies")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    per_pair = _load_listwise(args.listwise_pred)
    print(f"[agg-lw] loaded {len(per_pair)} pairs from {args.listwise_pred}")

    # preflight: usable rows + rank distribution (sanity check that the
    # listwise driver actually produced rubric_rank fields)
    n_with = 0
    K_seen: list[int] = []
    rank_dist: Counter = Counter()
    for rec in per_pair:
        ext = _extract_per_pair(rec)
        if ext is not None:
            n_with += 1
            K_seen.append(len(ext["scores_a"]))
            for r in ext["ranks"]:
                rank_dist[r] += 1
    K = max(K_seen) if K_seen else 0
    print(f"[agg-lw] {n_with}/{len(per_pair)} rows carry usable per_rubric, max K={K}")
    print(f"[agg-lw] rubric_rank distribution: {dict(rank_dist)}")

    tau_grid = [float(x) for x in args.tau_grid.split(",") if x.strip()]

    # ---- run strategies ---------------------------------------------------
    scalar_specs: list[tuple[str, Callable, bool]] = [
        ("mean",          agg_mean,          False),
        ("weighted_mean", agg_weighted_mean, True),
        ("lp4",           agg_lp4,           False),
    ]
    sweeps: dict[str, list[dict[str, Any]]] = {}
    by_strategy: dict[str, dict[str, Any]] = {}
    for name, fn, use_w in scalar_specs:
        rows = [_eval_scalar(per_pair, fn, use_w, tau) for tau in tau_grid]
        sweeps[name] = rows
        # best by (acc_no_tie, acc); tie-broken to prefer smaller tau
        by_strategy[name] = max(rows, key=lambda r: (r["acc_no_tie"], r["acc"], -r["tau"]))

    listwise_direct = _eval_listwise_direct(per_pair)

    # ---- write ------------------------------------------------------------
    out = {
        "axis":            args.axis,
        "listwise_pred":   args.listwise_pred,
        "n_pairs":         len(per_pair),
        "n_with_rubric":   n_with,
        "max_K":           K,
        "rank_dist":       {f"{a}-{b}": v for (a, b), v in rank_dist.items()},
        "listwise_direct": listwise_direct,
        "by_strategy":     by_strategy,
        "sweeps":          sweeps,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    # ---- stdout summary ---------------------------------------------------
    print(f"\n[agg-lw] axis={args.axis}")
    print(f"  {'method':<18} {'acc':>6}  {'acc_no_tie':>10}  {'macro':>6}  {'tie%':>5}  config")
    ld = listwise_direct
    print(f"  {'listwise_direct':<18} {ld['acc']:>6.4f}  {ld['acc_no_tie']:>10.4f}  "
          f"{ld['macro_acc']:>6.4f}  {'  -  ':>5}  (LLM)")
    for name in ["mean", "weighted_mean", "lp4"]:
        b = by_strategy[name]
        cfg = f"tau={b['tau']}"
        tie_str = f"{b['tie_rate']*100:>4.1f}%"
        print(f"  {name:<18} {b['acc']:>6.4f}  {b['acc_no_tie']:>10.4f}  "
              f"{b['macro_acc']:>6.4f}  {tie_str:>5}  {cfg}")
    print(f"[agg-lw] -> {out_path}")


if __name__ == "__main__":
    main()
