"""VBench-Long and VBench-Text scorer over an arm's generated eval videos.

Emits per-dim + macro scores as JSON. Uses the official VBench package
(``pip install vbench``) if available; falls back to a lightweight
subset scorer that uses CLIP for text-alignment and a simple frame-
consistency proxy for VBench-Long (documented as an approximation in
the paper's supplementary if the full VBench suite is unavailable).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


VBENCH_LONG_DIMS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]

VBENCH_TEXT_DIMS = ["object_class", "multiple_objects", "human_action", "color", "spatial_relationship"]


def score_with_official_vbench(video_dir: Path, prompt_map: dict) -> dict:
    """Attempt to run the official vbench package."""
    from vbench import VBench  # type: ignore

    vb = VBench(device="cuda", cache_dir=str(video_dir / ".vbench_cache"))
    all_scores: dict = {"long": {}, "text": {}}

    for dim in VBENCH_LONG_DIMS:
        s = vb.evaluate(
            videos_path=str(video_dir),
            dimension=dim,
            prompt_list=prompt_map,
        )
        all_scores["long"][dim] = float(s)
    for dim in VBENCH_TEXT_DIMS:
        s = vb.evaluate(
            videos_path=str(video_dir),
            dimension=dim,
            prompt_list=prompt_map,
        )
        all_scores["text"][dim] = float(s)

    all_scores["long_macro"] = sum(all_scores["long"].values()) / len(all_scores["long"])
    all_scores["text_macro"] = sum(all_scores["text"].values()) / len(all_scores["text"])
    return all_scores


def score_with_fallback(video_dir: Path, prompt_map: dict) -> dict:
    """CLIP-based text alignment + simple frame-diff consistency proxy.

    Documented in the paper's supp as an approximation when VBench is
    not installed. Reported alongside the note that final numbers should
    come from the official suite.
    """
    import torch
    import numpy as np
    from PIL import Image
    from torchvision.io import read_video
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

    def clip_align(video_path: Path, text: str) -> float:
        vid, _, _ = read_video(str(video_path), pts_unit="sec")
        idx = torch.linspace(0, vid.shape[0] - 1, 4).round().long()
        frames = [Image.fromarray(f.numpy()) for f in vid[idx]]
        inputs = proc(text=[text], images=frames, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
        # image-text similarity averaged across frames
        return float(out.logits_per_image.mean().item())

    def consistency(video_path: Path) -> float:
        vid, _, _ = read_video(str(video_path), pts_unit="sec")
        if vid.shape[0] < 2:
            return 1.0
        diffs = (vid[1:].float() - vid[:-1].float()).abs().mean(dim=(1, 2, 3))
        # normalize: lower diff -> more consistent
        return float(1.0 / (1.0 + diffs.mean().item() / 255.0))

    align_scores, cons_scores = [], []
    for prompt_id, text in prompt_map.items():
        vp = video_dir / f"{prompt_id}.mp4"
        if not vp.exists():
            continue
        align_scores.append(clip_align(vp, text))
        cons_scores.append(consistency(vp))

    return {
        "long": {"consistency_proxy": float(sum(cons_scores) / max(1, len(cons_scores)))},
        "text": {"clip_align": float(sum(align_scores) / max(1, len(align_scores)))},
        "long_macro": float(sum(cons_scores) / max(1, len(cons_scores))),
        "text_macro": float(sum(align_scores) / max(1, len(align_scores))),
        "note": "CLIP+diff proxy fallback (official VBench not installed).",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="Directory containing <prompt_id>.mp4")
    ap.add_argument("--splits", required=True, help="Path to evalverse_eval.jsonl")
    ap.add_argument("--out", required=True, help="Path to JSON output")
    args = ap.parse_args()

    from scripts.downstream_grpo.src.prompt_pool import read_split

    prompts = read_split(args.splits)
    prompt_map = {p.prompt_id: p.query for p in prompts}
    video_dir = Path(args.videos)

    try:
        scores = score_with_official_vbench(video_dir, prompt_map)
    except ImportError:
        print("[vbench] official package not installed — using CLIP+consistency fallback")
        scores = score_with_fallback(video_dir, prompt_map)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[vbench] wrote {out}: long_macro={scores['long_macro']:.4f} text_macro={scores['text_macro']:.4f}")


if __name__ == "__main__":
    main()
