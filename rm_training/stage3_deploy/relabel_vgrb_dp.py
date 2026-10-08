"""rm_training.stage3_deploy.relabel_vgrb_dp
===========================================
Rescore an existing DP-eval shard directory under a *different* VGRB
axis label (VQ / MQ / TA / Overall), without re-running the model.

Motivation
----------
``eval_on_vgrb_dp.py`` stores each rank's ``s_a`` / ``s_b`` alongside a
``gt`` value baked in at scoring time — that ``gt`` comes from
``pairwise_view_full/<axis>.json`` (one axis per view). If you already
scored the Overall view but want VQ (or MQ / TA) accuracy on the *same*
25.2K scalar scores, you don't need to hit the GPU again: swap the ``gt``
against the four-axis labels embedded in
``vgrb_pairwise_<N>.json``'s ``labels`` field, re-decide with the same
``_score_diff_to_decision`` (optionally sweeping tau), and recompute
acc / acc_no_tie / confusion.

Usage
-----
::

    # Re-score the Overall shard under the VQ label at the recorded tau.
    python rm_training/stage3_deploy/relabel_vgrb_dp.py \\
        --rank_dir  runs/.../student_rm_dp/Overall_dp/ \\
        --new_axis  VQ \\
        --pair_json /efs/.../videogen_rewardbench/vgrb_pairwise_25233.json \\
        --output    runs/.../student_rm_dp/Overall_scores_under_VQ_label.json

    # Same thing, but sweep tau to find the pick that maximises acc_no_tie.
    python rm_training/stage3_deploy/relabel_vgrb_dp.py \\
        --rank_dir  runs/.../student_rm_dp/Overall_dp/ \\
        --new_axis  VQ \\
        --pair_json /efs/.../videogen_rewardbench/vgrb_pairwise_25233.json \\
        --output    runs/.../student_rm_dp/Overall_scores_under_VQ_label_tausweep.json \\
        --tau_grid  0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5

The output JSON matches ``merge_vgrb_dp.py``'s shape when a single tau
is chosen (``summary`` + ``per_pair``). With ``--tau_grid`` it emits a
list of per-tau summaries under ``per_tau`` (plus a ``per_pair`` block
computed at the ORIGINAL tau, kept for provenance).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rm_training.stage3_deploy.eval_on_vgrb import _score_diff_to_decision


VGRB_AXES = ("VQ", "MQ", "TA", "Overall")


def _load_pair_labels(pair_json: Path) -> dict[str, dict[str, str]]:
    """Return {pair_id: {axis: gt_label}}.

    Reads from the fetcher output (``vgrb_pairwise_<N>.json``) which
    carries the 4-axis labels in ``labels`` per record.
    """
    if pair_json.is_dir():
        # Common footgun: user passed a directory. Give a concrete hint
        # pointing at the fetcher artefact.
        candidates = sorted(pair_json.glob("vgrb_pairwise_*.json"))
        hint = f" (found under it: {candidates[-1]})" if candidates else ""
        raise SystemExit(
            f"[relabel] --pair_json must be a FILE, got directory "
            f"{pair_json}{hint}"
        )
    if not pair_json.exists():
        raise SystemExit(f"[relabel] --pair_json not found: {pair_json}")
    data = json.loads(pair_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"[relabel] unexpected shape in {pair_json}")
    out: dict[str, dict[str, str]] = {}
    for rec in data:
        pid = rec.get("id")
        labels = rec.get("labels") or {}
        if not pid or not isinstance(labels, dict):
            continue
        out[pid] = {a: labels.get(a) for a in VGRB_AXES if labels.get(a) in {"A", "B", "same"}}
    if not out:
        raise SystemExit(f"[relabel] no usable pair labels loaded from {pair_json}")
    return out


def _score_row(row: dict[str, Any],
               new_gt: str,
               tau: float) -> dict[str, Any]:
    s_a = row.get("s_a")
    s_b = row.get("s_b")
    decision = _score_diff_to_decision(s_a, s_b, tau)
    correct = decision == new_gt
    return {
        "id":       row.get("id"),
        "orig_idx": row.get("orig_idx"),
        "s_a":      s_a,
        "s_b":      s_b,
        "decision": decision,
        "gt":       new_gt,
        "correct":  correct,
    }


def _summarise(per_pair: list[dict[str, Any]]) -> dict[str, Any]:
    confusion: dict[str, Counter] = {"A": Counter(), "B": Counter(), "same": Counter()}
    n_correct = 0
    n_no_tie = 0
    n_no_tie_correct = 0
    for row in per_pair:
        decision = row.get("decision", "same")
        gt = row.get("gt", "same")
        correct = bool(row.get("correct"))
        if correct:
            n_correct += 1
        if gt in {"A", "B"}:
            n_no_tie += 1
            if correct:
                n_no_tie_correct += 1
        confusion[gt][decision] += 1

    total = len(per_pair)
    return {
        "n_total":     total,
        "n_no_tie":    n_no_tie,
        "acc":         n_correct / total if total else 0.0,
        "acc_no_tie":  n_no_tie_correct / n_no_tie if n_no_tie else 0.0,
        "confusion":   {k: dict(v) for k, v in confusion.items()},
    }


def relabel(args: argparse.Namespace) -> None:
    rank_dir = Path(args.rank_dir)
    rank_files = sorted(rank_dir.glob("rank_*.json"))
    if not rank_files:
        raise SystemExit(f"[relabel] no rank_*.json under {rank_dir}")

    pair_labels = _load_pair_labels(Path(args.pair_json))
    axis = args.new_axis
    if axis not in VGRB_AXES:
        raise SystemExit(f"[relabel] --new_axis must be one of {VGRB_AXES}; got {axis!r}")

    print(f"[relabel] rank files: {len(rank_files)}   pair labels: {len(pair_labels)}",
          file=sys.stderr)

    # Gather all rows (deduplicate by id; DP stride sharding already
    # guarantees uniqueness but the merger checks it too).
    rows_by_id: dict[str, dict[str, Any]] = {}
    orig_axis: str | None = None
    orig_tau: float | None = None
    student_ckpt = dataset_orig = None
    for rf in rank_files:
        d = json.loads(rf.read_text(encoding="utf-8"))
        if orig_axis is None:
            orig_axis = d.get("axis")
            orig_tau = d.get("tau")
            student_ckpt = d.get("student_ckpt")
            dataset_orig = d.get("dataset")
        for row in d.get("per_pair", []):
            pid = row.get("id")
            if pid and pid not in rows_by_id:
                rows_by_id[pid] = row

    missing_label = 0
    filtered_rows: list[dict[str, Any]] = []
    for pid, row in rows_by_id.items():
        pair = pair_labels.get(pid)
        if not pair or pair.get(axis) not in {"A", "B", "same"}:
            missing_label += 1
            continue
        filtered_rows.append((row, pair[axis]))  # type: ignore[arg-type]

    print(f"[relabel] dropped {missing_label} rows without {axis} labels; "
          f"kept {len(filtered_rows)}", file=sys.stderr)

    # Sweep tau if requested; else single-tau at args.tau (defaults to the
    # original shard's tau).
    if args.tau_grid:
        taus = [float(x) for x in args.tau_grid.split(",") if x.strip()]
    else:
        taus = [args.tau if args.tau is not None else (orig_tau or 0.0)]

    per_tau_summaries: list[dict[str, Any]] = []
    for tau in taus:
        per_pair = [_score_row(row, new_gt, tau) for row, new_gt in filtered_rows]
        # Deterministic ordering to make diffs readable.
        per_pair.sort(key=lambda r: (r.get("orig_idx") if r.get("orig_idx") is not None else 1 << 30,
                                     r.get("id", "")))
        summary = _summarise(per_pair)
        summary["tau"] = tau
        per_tau_summaries.append(summary)
        print(f"[relabel] tau={tau:.3f}  acc={summary['acc']:.4f}  "
              f"acc_no_tie={summary['acc_no_tie']:.4f}  "
              f"n={summary['n_total']}", file=sys.stderr)

    # For the JSON payload, expose per_pair AT THE FIRST TAU so users
    # can inspect individual rows without ambiguity. Multi-tau consumers
    # should read ``per_tau``.
    first_tau = taus[0]
    per_pair_out = [_score_row(row, new_gt, first_tau) for row, new_gt in filtered_rows]
    per_pair_out.sort(key=lambda r: (r.get("orig_idx") if r.get("orig_idx") is not None else 1 << 30,
                                     r.get("id", "")))

    head = {
        "method":              "RewardVerse student RM (relabel of pre-computed DP shard)",
        "orig_axis":           orig_axis,
        "orig_tau":            orig_tau,
        "new_axis":            axis,
        "student_ckpt":        student_ckpt,
        "orig_dataset":        dataset_orig,
        "pair_labels_source":  str(Path(args.pair_json).resolve()),
        "n_rank_files":        len(rank_files),
        "n_rows_ingested":     len(rows_by_id),
        "n_rows_kept":         len(filtered_rows),
        "n_rows_missing_label": missing_label,
    }

    out: dict[str, Any] = {
        "summary":  {**head, **per_tau_summaries[0]} if len(taus) == 1 else head,
        "per_tau":  per_tau_summaries,
        "per_pair": per_pair_out,   # rows scored at taus[0]
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[relabel] wrote -> {out_path}", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank_dir", required=True,
                   help="Directory of rank_*.json files from eval_on_vgrb_dp.py.")
    p.add_argument("--new_axis", required=True, choices=VGRB_AXES,
                   help="Axis whose gt label to use for relabelling.")
    p.add_argument("--pair_json", required=True,
                   help="vgrb_pairwise_<N>.json (fetcher output) carrying "
                        "per-pair 'labels' with all four axes.")
    p.add_argument("--output", required=True,
                   help="Path to write the relabelled summary JSON.")
    p.add_argument("--tau", type=float, default=None,
                   help="Single decision threshold (default: original shard's tau).")
    p.add_argument("--tau_grid", default=None,
                   help="Comma-separated tau sweep. If set, --tau is ignored.")
    args = p.parse_args()
    relabel(args)


if __name__ == "__main__":
    main()
