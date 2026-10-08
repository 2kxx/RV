"""Aggregate per-arm training jsonl into the reward-hacking diagnostic plot.

Reads ``logs/downstream/<arm>.jsonl`` for each arm and produces:
  * ``runs/downstream/fig_downstream_hacking.pdf`` — 2-panel figure
    matching the paper's ``fig:downstream_hacking``.
  * ``runs/downstream/hacking_delta.json`` — per-arm scalar Δ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARM_COLORS = {
    "arm_A_base": "#666666",
    "arm_B_videoscore": "#d1495b",
    "arm_C_v3zeroshot": "#edae49",
    "arm_D_rewardverse": "#00798c",
    "arm_E_videoalign": "#2e4057",
}

ARM_LABELS = {
    "arm_A_base": "Base (no RL)",
    "arm_B_videoscore": "+ VideoScore-v2",
    "arm_C_v3zeroshot": "+ V3-zeroshot",
    "arm_D_rewardverse": "+ RewardVerse (ours)",
    "arm_E_videoalign": "+ VideoAlign",
}


def _load_arm(log_dir: Path, arm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = log_dir / f"{arm}.jsonl"
    steps: list[int] = []
    rewards: list[float] = []
    eval_steps: list[int] = []
    eval_scores: list[float] = []
    if not path.exists():
        return np.array([]), np.array([]), np.array([]), np.array([])
    for line in path.open("r", encoding="utf-8"):
        rec = json.loads(line)
        if rec.get("event") == "step" and "mean_reward" in rec:
            steps.append(rec["step"])
            rewards.append(rec["mean_reward"])
        elif rec.get("event") == "eval_fixed" and "mean_reward" in rec:
            eval_steps.append(rec["step"])
            eval_scores.append(rec["mean_reward"])
    return (
        np.array(steps),
        np.array(rewards),
        np.array(eval_steps),
        np.array(eval_scores),
    )


def _normalize(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def hacking_delta(rm_rewards: np.ndarray, indep: np.ndarray) -> float:
    if rm_rewards.size == 0 or indep.size == 0:
        return float("nan")
    r_norm = _normalize(rm_rewards)
    q_norm = _normalize(indep)
    if q_norm.size < rm_rewards.size:
        # align lengths by nearest step: pad by repeating last
        q_norm = np.interp(
            np.arange(rm_rewards.size),
            np.linspace(0, rm_rewards.size - 1, q_norm.size),
            q_norm,
        )
    q0 = q_norm[0] if q_norm.size else 0.0
    delta = float(np.max(r_norm - (q_norm - q0)))
    return delta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="logs/downstream")
    ap.add_argument("--out_pdf", default="runs/downstream/fig_downstream_hacking.pdf")
    ap.add_argument("--out_json", default="runs/downstream/hacking_delta.json")
    ap.add_argument("--arms", nargs="+",
                    default=["arm_B_videoscore", "arm_C_v3zeroshot", "arm_D_rewardverse"])
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    deltas: dict[str, float] = {}

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for arm in args.arms:
        steps, rewards, eval_steps, eval_scores = _load_arm(log_dir, arm)
        color = ARM_COLORS.get(arm, "#000000")
        label = ARM_LABELS.get(arm, arm)
        if steps.size:
            axL.plot(steps, rewards, color=color, label=label, linewidth=1.8)
        if eval_steps.size:
            axR.plot(eval_steps, eval_scores, color=color, label=label, linewidth=1.8)
        deltas[arm] = hacking_delta(rewards, eval_scores)

    axL.set_title("RM reward (train rollouts)")
    axL.set_xlabel("GRPO step")
    axL.set_ylabel("mean RM reward")
    axL.legend(loc="best", fontsize=8, frameon=False)
    axL.grid(alpha=0.3)

    axR.set_title("Independent quality (fixed 32-prompt VBench-subset)")
    axR.set_xlabel("GRPO step")
    axR.set_ylabel("mean subset score")
    axR.grid(alpha=0.3)

    plt.tight_layout()
    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), bbox_inches="tight", dpi=180)
    plt.close(fig)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(deltas, indent=2), encoding="utf-8")
    print(f"[hacking] wrote {out_pdf}")
    print(f"[hacking] deltas: {deltas}")


if __name__ == "__main__":
    main()
