"""Generate held-out eval videos from a checkpoint (or base model).

Emits per-arm mp4s plus an append-only manifest.jsonl that downstream
scoring scripts (`score_matrix.py`) consume.

Usage:
    # base model, one video per prompt
    python -m scripts.downstream_grpo.eval.generate_eval_videos \\
        --arm_name base \\
        --splits scripts/downstream_grpo/configs/prompts/evalverse_eval.jsonl \\
        --out    runs/downstream/eval_videos

    # a trained LoRA arm, N samples per prompt for Best-of-N eval
    python -m scripts.downstream_grpo.eval.generate_eval_videos \\
        --arm_name arm_D_rewardverse \\
        --ckpt     ckpts/downstream/arm_D_rewardverse/checkpoint-final \\
        --splits   scripts/downstream_grpo/configs/prompts/evalverse_eval.jsonl \\
        --out      runs/downstream/eval_videos \\
        --n_per_prompt 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from torchvision.io import write_video

from scripts.downstream_grpo.src.prompt_pool import read_split
from scripts.downstream_grpo.src.sampler_wan import WanGRPOSampler
from scripts.downstream_grpo.src.train_grpo_wan import load_yaml


REPO_ROOT = Path(__file__).resolve().parents[3]


def _existing_video_ids(arm_dir: Path, prompt_id: str, n_per_prompt: int) -> list[Path]:
    """Return list of expected video paths that already exist for this prompt."""
    return [
        arm_dir / f"{prompt_id}_n{i}.mp4"
        for i in range(n_per_prompt)
        if (arm_dir / f"{prompt_id}_n{i}.mp4").exists()
    ]


def _append_manifest(manifest_path: Path, rows: list[dict]) -> None:
    with manifest_path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _load_manifest_pairs(manifest_path: Path, arm_name: str) -> set[tuple[str, int]]:
    """Return (prompt_id, sample_idx) pairs already logged for this arm."""
    if not manifest_path.exists():
        return set()
    pairs: set[tuple[str, int]] = set()
    for line in manifest_path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("arm") == arm_name:
            pairs.add((rec["prompt_id"], int(rec["sample_idx"])))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="scripts/downstream_grpo/configs/wan_grpo_default.yaml")
    ap.add_argument("--ckpt", default="", help="Path to LoRA adapter dir (empty = base model)")
    ap.add_argument("--splits", required=True, help="Prompt jsonl to sample from")
    ap.add_argument("--out", required=True, help="Output root; videos go to <out>/<arm_name>/")
    ap.add_argument("--arm_name", default="base", help="Arm label used in manifest rows and directory")
    ap.add_argument("--n_per_prompt", type=int, default=1)
    ap.add_argument("--num_inference_steps", type=int, default=None,
                    help="Override yaml default. Wan-2.1 uses 30 by default; drop to 20 for wall-clock.")
    args = ap.parse_args()

    cfg = load_yaml(REPO_ROOT / args.config)
    n_inf = args.num_inference_steps if args.num_inference_steps is not None else cfg["num_inference_steps"]

    device = torch.device("cuda")
    sampler = WanGRPOSampler(cfg["wan_model_id"], device=device)
    if args.ckpt:
        sampler.transformer = PeftModel.from_pretrained(sampler.transformer, args.ckpt).to(device)

    out_root = Path(args.out)
    arm_dir = out_root / args.arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"

    prompts = read_split(args.splits)
    already_logged = _load_manifest_pairs(manifest_path, args.arm_name)

    print(
        f"[gen] arm={args.arm_name} ckpt={args.ckpt or '(base)'} "
        f"n_per_prompt={args.n_per_prompt} n_inference_steps={n_inf} "
        f"prompts={len(prompts)} out={arm_dir}"
    )

    for i, p in enumerate(prompts):
        pending_indices = [
            j for j in range(args.n_per_prompt)
            if not (arm_dir / f"{p.prompt_id}_n{j}.mp4").exists()
            or (p.prompt_id, j) not in already_logged
        ]
        if not pending_indices:
            continue

        # Determinism: same seed base per (arm, prompt_id); G in the sampler
        # walks the noise Generator forward so sample_idx 0..N-1 are distinct.
        seed_base = hash((args.arm_name, p.prompt_id)) & 0xFFFFFF
        out = sampler.sample_group(
            p.query,
            G=args.n_per_prompt,
            num_frames=cfg["num_frames"],
            height=cfg["video_height"],
            width=cfg["video_width"],
            num_inference_steps=n_inf,
            guidance_scale=cfg["guidance_scale"],
            scheduler_shift=cfg["scheduler_shift"],
            eta=0.0,               # deterministic sampling; no PPO log-probs needed
            seed_base=seed_base,
        )

        new_rows: list[dict] = []
        for j in range(args.n_per_prompt):
            video_path = arm_dir / f"{p.prompt_id}_n{j}.mp4"
            if not video_path.exists():
                write_video(str(video_path), out.videos[j], fps=cfg["fps"])
            if (p.prompt_id, j) not in already_logged:
                new_rows.append({
                    "arm": args.arm_name,
                    "prompt_id": p.prompt_id,
                    "dimension": p.dimension,
                    "prompt": p.query,
                    "sample_idx": j,
                    "video_path": str(video_path.resolve()),
                })
                already_logged.add((p.prompt_id, j))
        if new_rows:
            _append_manifest(manifest_path, new_rows)

        if (i + 1) % 10 == 0 or i == len(prompts) - 1:
            print(f"[gen] {i+1}/{len(prompts)} done")

    print(f"[gen] arm={args.arm_name} wrote videos under {arm_dir}, manifest={manifest_path}")


if __name__ == "__main__":
    main()
