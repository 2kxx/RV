"""Cross-RM heatmap: for each arm × RM, plot mean score.

Aggregation order (matters for reward-hacking claim):
  1. Group scores by (arm, rm_name, dimension); take the mean per dim.
  2. Group those per-dim means by (arm, rm_name); take the mean across dims.
This prevents dims with more prompts from dominating the cell value.

Two panels:
  * Absolute mean score.
  * Column-normalized (z-score within each RM column) — makes the
    "policy trained on RM_X only wins under RM_X" story visually obvious.

Also emits ``hacking_delta.json`` with per-arm:
    Δ_hacking = mean(score under own RM) − mean(score under other RMs)
Arms trained on a hackable scalar RM have Δ > 0; RewardVerse should be ≈ 0.
The "own RM" mapping infers from arm_name conventions:
    arm_B_videoscore -> videoscore_v2
    arm_C_v3zeroshot -> v3_zeroshot
    arm_D_rewardverse -> rewardverse
    arm_E_videoalign -> videoalign
    (base / unknown -> no delta computed)
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARM_TO_OWN_RM: dict[str, str] = {
    "arm_B_videoscore": "videoscore_v2",
    "arm_C_v3zeroshot": "v3_zeroshot",
    "arm_D_rewardverse": "rewardverse",
    "arm_E_videoalign": "videoalign",
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


def _cell_means(rows: list[dict]) -> tuple[list[str], list[str], np.ndarray]:
    """Return (arm_order, rm_order, matrix[arm, rm]) where each cell is the
    mean-of-per-dim-means score."""
    per_dim: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        per_dim[(r["arm"], r["rm_name"], r["dimension"])].append(float(r["score"]))

    per_dim_mean: dict[tuple[str, str, str], float] = {
        k: float(np.mean(v)) for k, v in per_dim.items()
    }

    arms = sorted({k[0] for k in per_dim_mean})
    rms = sorted({k[1] for k in per_dim_mean})
    matrix = np.full((len(arms), len(rms)), np.nan, dtype=float)
    for i, arm in enumerate(arms):
        for j, rm in enumerate(rms):
            dim_means = [
                v for (a, m, _), v in per_dim_mean.items()
                if a == arm and m == rm
            ]
            if dim_means:
                matrix[i, j] = float(np.mean(dim_means))
    return arms, rms, matrix


def _hacking_deltas(arms: list[str], rms: list[str], matrix: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for i, arm in enumerate(arms):
        own_rm = ARM_TO_OWN_RM.get(arm)
        if own_rm is None or own_rm not in rms:
            continue
        j_own = rms.index(own_rm)
        row = matrix[i]
        if np.isnan(row[j_own]):
            continue
        others = [row[j] for j in range(len(rms)) if j != j_own and not np.isnan(row[j])]
        if not others:
            continue
        out[arm] = float(row[j_own] - float(np.mean(others)))
    return out


def _column_zscore(matrix: np.ndarray) -> np.ndarray:
    """Z-score within each RM column, ignoring NaNs. Cells with < 2 valid arms
    or zero std stay NaN."""
    z = np.full_like(matrix, np.nan, dtype=float)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            continue
        mu = np.mean(col[valid])
        sd = np.std(col[valid])
        if sd < 1e-12:
            continue
        z[valid, j] = (col[valid] - mu) / sd
    return z


def _plot(
    ax: plt.Axes,
    matrix: np.ndarray,
    arm_labels: list[str],
    rm_labels: list[str],
    title: str,
    cmap: str,
    annot_fmt: str,
    diverging: bool,
) -> None:
    if diverging:
        vabs = np.nanmax(np.abs(matrix)) if matrix.size and np.any(~np.isnan(matrix)) else 1.0
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-vabs, vmax=vabs)
    else:
        im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(rm_labels)))
    ax.set_xticklabels(rm_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(arm_labels)))
    ax.set_yticklabels(arm_labels)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                txt = "—"
            else:
                txt = format(v, annot_fmt)
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if not diverging else ("white" if abs(v) > 0.6 * (np.nanmax(np.abs(matrix)) or 1) else "black"))
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    scores_path = Path(args.scores)
    rows = _load_scores(scores_path)
    if not rows:
        raise SystemExit(f"[cross_rm] no rows in {scores_path}")

    arms, rms, matrix = _cell_means(rows)
    z_matrix = _column_zscore(matrix)
    deltas = _hacking_deltas(arms, rms, matrix)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_labels = [ARM_LABELS.get(a, a) for a in arms]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, max(3, 0.7 * len(arms) + 2)))
    _plot(axes[0], matrix, arm_labels, rms,
          title="Cross-RM: mean score", cmap="viridis",
          annot_fmt=".2f", diverging=False)
    _plot(axes[1], z_matrix, arm_labels, rms,
          title="Cross-RM: column-z (reward hacking signature)", cmap="RdBu_r",
          annot_fmt="+.2f", diverging=True)
    plt.tight_layout()
    out_pdf = out_dir / "cross_rm_matrix.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)

    delta_path = out_dir / "hacking_delta.json"
    delta_path.write_text(json.dumps(deltas, indent=2), encoding="utf-8")
    print(f"[cross_rm] wrote {out_pdf}")
    print(f"[cross_rm] arms={arms}")
    print(f"[cross_rm] rms={rms}")
    print(f"[cross_rm] deltas={deltas}")


if __name__ == "__main__":
    main()
