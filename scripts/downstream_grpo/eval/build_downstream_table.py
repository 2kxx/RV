"""Emit the §5.4 main-table LaTeX fragment from per-arm eval JSONs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROWS = [
    ("arm_A_base",         r"\texttt{Wan-2.1-1.3B} (Base)"),
    ("arm_B_videoscore",   r"\quad + GRPO w/ VideoScore-v2 (pointwise, unconstrained)"),
    ("arm_C_v3zeroshot",   r"\quad + GRPO w/ V3-zeroshot (rubric + soft-logits, no RGPO)"),
    ("arm_D_rewardverse",  r"\quad \textbf{+ GRPO w/ RewardVerse (ours, RGPO-trained)}"),
]


def _get(d: dict, *keys, default="\\textendash") -> str:
    for k in keys:
        if d is None or k not in d:
            return default
        d = d[k]
    if isinstance(d, float):
        return f"{d:.3f}"
    return str(d)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_table(runs_root: Path) -> str:
    hacking = _load_json(runs_root / "hacking_delta.json") or {}
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Reward Model Policy}",
        r"    & \textbf{VBench-Long}~$\uparrow$",
        r"    & \textbf{VBench-Text}~$\uparrow$",
        r"    & \textbf{Gemini Win-Rate}~$\uparrow$",
        r"    & \textbf{Reward-Hacking $\Delta$}~$\downarrow$ \\",
        r"\midrule",
    ]
    for arm, label in ROWS:
        vb = _load_json(runs_root / f"{arm}_vbench.json")
        gj = _load_json(runs_root / f"{arm}_gemini.json") if arm != "arm_A_base" else None
        vb_long = _get(vb, "long_macro")
        vb_text = _get(vb, "text_macro")
        gemini = "--" if arm == "arm_A_base" else _get(gj, "win_rate_vs_base")
        delta = "--" if arm == "arm_A_base" else _get(hacking, arm)
        lines.append(f"{label} & {vb_long} & {vb_text} & {gemini} & {delta} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default="runs/downstream")
    ap.add_argument("--out", default="runs/downstream/tab_downstream.tex")
    args = ap.parse_args()

    tex = build_table(Path(args.runs_root))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex, encoding="utf-8")
    print(f"[table] wrote {out}")


if __name__ == "__main__":
    main()
