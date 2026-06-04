"""summarize_external_vgrb.py — turn pairwise eval JSONs into a Markdown table.

Reads:
  <root>/{rewardverse,videoscore_v11,videoscore2}/pairwise/<vgrb_axis>.json
each with top-level ``summary: {acc, acc_no_tie, n_total, n_no_tie, ...}``.

Emits one Markdown file with two tables:
  (1) main:    pair acc per (method, VGRB axis)
  (2) no-tie:  acc on the subset where GT in {A,B} only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_METHODS = [
    ("rewardverse",     "RewardVerse (rubric, pointwise+τ)"),
    ("rewardverse_pw",  "RewardVerse (rubric, pairwise)"),
    ("videoscore_v11",  "VideoScore-v1.1"),
    ("videoscore2",     "VideoScore-2"),
]
_AXES = ["VQ", "MQ", "TA", "Overall"]


def _load_summary(path: Path) -> dict | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return d.get("summary")


def _fmt(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return str(v)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True,
                   help="runs/external_benchmarks/videogen_rewardbench")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    root = Path(args.root)

    grid: dict[str, dict[str, dict]] = {ax: {} for ax in _AXES}
    tau_seen: float | None = None
    for method_key, _ in _METHODS:
        for ax in _AXES:
            s = _load_summary(root / method_key / "pairwise" / f"{ax}.json")
            if s is not None:
                grid[ax][method_key] = s
                if tau_seen is None and "tau" in s:
                    tau_seen = s["tau"]

    lines: list[str] = []
    lines.append("# VideoGen-RewardBench (200 pairs) — external benchmark results\n")
    lines.append(
        f"Pairwise accuracy via delta-threshold rule "
        f"(tau={tau_seen if tau_seen is not None else '?'}). "
        "Pointwise predictions per video are produced once per method "
        "across RewardVerse axes (visual_quality / action / character) "
        "and Overall is the elementwise mean.\n"
    )

    # (1) Main acc table.
    lines.append("## Pair accuracy (3-way: A / B / same)\n")
    headers = ["VGRB axis"] + [m[1] for m in _METHODS] + ["n"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for ax in _AXES:
        row = [ax]
        n_seen = None
        for mk, _ in _METHODS:
            v = grid[ax].get(mk)
            row.append(_fmt(v["acc"]) if v else "—")
            if v and v.get("n_total"):
                n_seen = v["n_total"]
        row.append(str(n_seen) if n_seen else "—")
        lines.append("| " + " | ".join(row) + " |")

    # (2) No-tie acc table.
    lines.append("")
    lines.append("## Pair accuracy on no-tie subset (GT in {A, B})\n")
    headers = ["VGRB axis"] + [m[1] for m in _METHODS] + ["n_no_tie"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for ax in _AXES:
        row = [ax]
        n_seen = None
        for mk, _ in _METHODS:
            v = grid[ax].get(mk)
            row.append(_fmt(v["acc_no_tie"]) if v else "—")
            if v and v.get("n_no_tie"):
                n_seen = v["n_no_tie"]
        row.append(str(n_seen) if n_seen else "—")
        lines.append("| " + " | ".join(row) + " |")

    # Mean across axes.
    lines.append("")
    lines.append("## Mean across 4 axes\n")
    lines.append("| Method | acc | acc_no_tie |")
    lines.append("|---|---|---|")
    for mk, name in _METHODS:
        accs = [grid[ax][mk]["acc"] for ax in _AXES
                if mk in grid[ax] and grid[ax][mk].get("acc") is not None]
        nots = [grid[ax][mk]["acc_no_tie"] for ax in _AXES
                if mk in grid[ax] and grid[ax][mk].get("acc_no_tie") is not None]
        m_a = sum(accs) / len(accs) if accs else None
        m_n = sum(nots) / len(nots) if nots else None
        lines.append(f"| {name} | {_fmt(m_a)} | {_fmt(m_n)} |")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summary] -> {out_path}")


if __name__ == "__main__":
    main()
