"""Per-dimension radar chart: overlaid per-arm 16-axis shape.

Only RewardVerse produces a full radar — scalar-RM baselines only expose 1
axis, so we compute per-dim quality via the *mean score across all RMs*.
Each dim axis is min-max normalized across arms so shapes are comparable
regardless of raw score scale.

Outputs:
  radar_all_arms.pdf/.png — one figure, all arms overlaid.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARM_COLORS = {
    "base": "#666666",
    "arm_A_base": "#666666",
    "arm_B_videoscore": "#d1495b",
    "arm_C_v3zeroshot": "#edae49",
    "arm_D_rewardverse": "#00798c",
    "arm_E_videoalign": "#2e4057",
}

ARM_LABELS = {
    "base": "Base (no RL)",
    "arm_A_base": "Base (no RL)",
    "arm_B_videoscore": "+ VideoScore-v2",
    "arm_C_v3zeroshot": "+ V3-zeroshot",
    "arm_D_rewardverse": "+ RewardVerse (ours)",
    "arm_E_videoalign": "+ VideoAlign",
}


def _load_scores(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _arm_dim_mean(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Return {arm: {dim: mean-across-RMs-of-per-video-mean-score}}."""
    # First mean over videos within (arm, rm, dim), then mean across RMs.
    per_arm_rm_dim: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        per_arm_rm_dim[(r["arm"], r["rm_name"], r["dimension"])].append(float(r["score"]))

    per_arm_dim_rm: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (arm, rm, dim), vs in per_arm_rm_dim.items():
        per_arm_dim_rm[(arm, dim)].append(float(np.mean(vs)))

    result: dict[str, dict[str, float]] = defaultdict(dict)
    for (arm, dim), rm_means in per_arm_dim_rm.items():
        result[arm][dim] = float(np.mean(rm_means))
    return result


def _normalize_per_dim(arm_dim_mean: dict[str, dict[str, float]]) -> tuple[list[str], dict[str, list[float]]]:
    dims = sorted({d for arm in arm_dim_mean for d in arm_dim_mean[arm]})
    arms = sorted(arm_dim_mean.keys())

    # Collect raw per-dim vectors.
    raw: dict[str, list[float]] = {}
    for arm in arms:
        raw[arm] = [arm_dim_mean[arm].get(d, np.nan) for d in dims]

    # Min-max within each dim.
    dim_min = np.array([np.nanmin([raw[a][j] for a in arms]) for j in range(len(dims))])
    dim_max = np.array([np.nanmax([raw[a][j] for a in arms]) for j in range(len(dims))])
    span = np.where((dim_max - dim_min) > 1e-9, dim_max - dim_min, 1.0)

    norm: dict[str, list[float]] = {}
    for arm in arms:
        v = np.array(raw[arm], dtype=float)
        n = (v - dim_min) / span
        n = np.where(np.isnan(n), 0.0, n)
        norm[arm] = n.tolist()
    return dims, norm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    rows = _load_scores(Path(args.scores))
    if not rows:
        raise SystemExit(f"[radar] no rows in {args.scores}")

    per_arm_dim = _arm_dim_mean(rows)
    dims, norm = _normalize_per_dim(per_arm_dim)
    if not dims:
        raise SystemExit("[radar] no dimensions found in scores")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    for arm in sorted(norm.keys()):
        vals = norm[arm] + [norm[arm][0]]
        color = ARM_COLORS.get(arm, None)
        label = ARM_LABELS.get(arm, arm)
        ax.plot(angles, vals, color=color, linewidth=2.0, label=label)
        ax.fill(angles, vals, color=color, alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""])
    ax.grid(alpha=0.4)
    ax.set_title("Per-dimension quality (min-max within dim, mean over RMs)",
                 fontsize=11, pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), fontsize=9, frameon=False)

    plt.tight_layout()
    out_pdf = out_dir / "radar_all_arms.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)
    print(f"[radar] wrote {out_pdf} for arms={list(norm.keys())}, dims={len(dims)}")


if __name__ == "__main__":
    main()
