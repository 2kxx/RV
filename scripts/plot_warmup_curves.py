#!/usr/bin/env python3
"""Reconstruct a training curve plot from logs/train_warmup.log.

The 2026-06-21 warm-up run logged every step to logs/train_warmup.log but
wandb wasn't authenticated, so no online dashboard was created. This script
parses the [step N] breadcrumb lines and writes a multi-panel PNG/PDF that
mirrors the panels we'd normally watch on wandb.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STEP_RE = re.compile(
    r"^\[step\s+(?P<step>\d+)\]\s+"
    r"dim=(?P<dim>\S+)\s+"
    r"reward\s+μ=(?P<reward>[-\d.eE+]+)\s+"
    r"r_pref=(?P<rpref>[-\d.eE+]+)\s+"
    r"r_align=(?P<ralign>[-\d.eE+]+)\s+"
    r"r_fmt=(?P<rfmt>[-\d.eE+]+)\s+"
    r"valid=(?P<valid_n>\d+)/(?P<valid_d>\d+)\s+"
    r"λ=(?P<lam>[-\d.eE+]+)\s+"
    r"β=(?P<beta>[-\d.eE+]+)\s+"
    r"loss=(?P<loss>[-\d.eE+]+)\s+"
    r"kl=(?P<kl>[-\d.eE+]+)\s+"
    r"kl_s=(?P<kls>[-\d.eE+]+)\s+"
    r"bt=(?P<bt>[-\d.eE+]+)\s+"
    r"pw=(?P<pw>[-\d.eE+]+)"
)


def parse_log(path: Path):
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = STEP_RE.match(line)
            if not m:
                continue
            d = m.groupdict()
            rows.append(
                {
                    "step": int(d["step"]),
                    "dim": d["dim"],
                    "reward": float(d["reward"]),
                    "r_pref": float(d["rpref"]),
                    "r_align": float(d["ralign"]),
                    "r_fmt": float(d["rfmt"]),
                    "valid_ratio": int(d["valid_n"]) / int(d["valid_d"]),
                    "lambda": float(d["lam"]),
                    "beta": float(d["beta"]),
                    "loss": float(d["loss"]),
                    "kl": float(d["kl"]),
                    "kl_s": float(d["kls"]),
                    "bt": float(d["bt"]),
                    "pw": float(d["pw"]),
                }
            )
    rows.sort(key=lambda r: r["step"])
    return rows


def smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
    if len(x) < 2 or k <= 1:
        return x
    k = min(k, len(x))
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log",
        type=Path,
        default=Path("logs/train_warmup.log"),
        help="path to train_warmup.log",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("runs/grpo_warmup"),
        help="where to write the figures",
    )
    ap.add_argument("--smooth", type=int, default=5, help="moving avg window")
    args = ap.parse_args()

    log_path = args.log.resolve()
    rows = parse_log(log_path)
    if not rows:
        raise SystemExit(f"no [step N] lines parsed from {log_path}")

    steps = np.array([r["step"] for r in rows])
    reward = np.array([r["reward"] for r in rows])
    r_pref = np.array([r["r_pref"] for r in rows])
    r_align = np.array([r["r_align"] for r in rows])
    r_fmt = np.array([r["r_fmt"] for r in rows])
    valid = np.array([r["valid_ratio"] for r in rows])
    lam = np.array([r["lambda"] for r in rows])
    beta = np.array([r["beta"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    kl_s = np.array([r["kl_s"] for r in rows])
    pw = np.array([r["pw"] for r in rows])
    bt = np.array([r["bt"] for r in rows])
    dims = [r["dim"] for r in rows]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), sharex=True)

    ax = axes[0, 0]
    ax.plot(steps, reward, color="#bbb", linewidth=0.9, label="raw")
    ax.plot(steps, smooth(reward, args.smooth), color="C0", linewidth=2.0, label=f"MA({args.smooth})")
    ax.set_ylabel("reward μ")
    ax.set_title("reward (per-step mean over 8 rollouts)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(steps, r_pref, label="r_pref", color="C0", linewidth=1.2)
    ax.plot(steps, r_align, label="r_align", color="C2", linewidth=1.2)
    ax.plot(steps, r_fmt, label="r_fmt", color="C3", linewidth=1.2)
    ax.set_ylabel("component reward")
    ax.set_title("reward components")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.1)

    ax = axes[1, 0]
    ax.plot(steps, loss, color="#bbb", linewidth=0.8)
    ax.plot(steps, smooth(loss, args.smooth), color="C4", linewidth=2.0, label=f"MA({args.smooth})")
    ax.set_ylabel("loss")
    ax.set_title("training loss (raw + smoothed)")
    ax.axhline(0.0, color="k", linewidth=0.5, alpha=0.4)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(steps, pw, color="C5", label="pw L1", linewidth=1.4)
    if np.any(bt > 0):
        ax.plot(steps, bt, color="C8", label="bt aux", linewidth=1.0)
    ax.set_ylabel("aux loss")
    ax.set_title("score-gap supervision (pw) — bt is off (lambda_bt=0)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2, 0]
    ax.plot(steps, kl_s, color="#aaa", linewidth=0.8)
    ax.plot(steps, smooth(kl_s, args.smooth), color="C1", linewidth=2.0, label=f"MA({args.smooth})")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylabel("kl_s (score-side, symlog)")
    ax.set_title("score-side KL drift")
    ax.set_xlabel("step")
    ax.grid(alpha=0.3, which="both")

    ax = axes[2, 1]
    ax.plot(steps, valid, color="C6", linewidth=1.4, label="valid_ratio")
    ax.plot(steps, lam, color="C7", linewidth=1.2, label="λ_t")
    ax.plot(steps, beta, color="C9", linewidth=1.2, label="β_format")
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("ratio / schedule")
    ax.set_title("valid_ratio + λ / β schedules")
    ax.set_xlabel("step")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"GRPO phase-A warm-up — score head only — N={len(rows)} steps   ({log_path.name})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_png = args.out_dir / "fig_warmup_curves.png"
    out_pdf = args.out_dir / "fig_warmup_curves.pdf"
    fig.savefig(out_png, dpi=140)
    fig.savefig(out_pdf)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    uniq = sorted(set(dims))
    cmap = plt.get_cmap("tab20", len(uniq))
    for i, d in enumerate(uniq):
        mask = np.array([x == d for x in dims])
        ax.scatter(steps[mask], reward[mask], s=22, color=cmap(i), label=d, alpha=0.85)
    ax.plot(steps, smooth(reward, args.smooth), color="k", linewidth=1.5, alpha=0.6, label="MA(5)")
    ax.set_xlabel("step")
    ax.set_ylabel("reward μ")
    ax.set_title("reward vs step, colored by dimension")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, ncol=1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out_dir / "fig_warmup_reward_by_dim.png", dpi=140)
    plt.close(fig)

    print(f"[plot] parsed {len(rows)} step rows  steps {steps.min()}..{steps.max()}")
    print(f"[plot] reward μ end-mean(last 10) = {reward[-10:].mean():.3f}   start-mean(first 10) = {reward[:10].mean():.3f}")
    print(f"[plot] kl_s end-mean(last 10) = {kl_s[-10:].mean():.3f}   max = {kl_s.max():.3f}")
    print(f"[plot] wrote {out_png}")
    print(f"[plot] wrote {out_pdf}")
    print(f"[plot] wrote {args.out_dir / 'fig_warmup_reward_by_dim.png'}")


if __name__ == "__main__":
    main()
