"""VideoScore-v2 adapter — an unconstrained pointwise MLP scorer.

Uses TIGER-Lab/VideoScore-v1.1 (VideoScore-v2 not yet public as of
2026-07). Returns the weighted average of its 5 heads (visual quality,
temporal consistency, dynamic degree, text-to-video alignment, factual
consistency) as a scalar. This is intentionally the "no rubric, no
learned calibration" arm and is the paper's designated reward-hacking
demonstrator.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torchvision.io import read_video

from scripts.downstream_grpo.src.reward_adapters import RewardRequest


HF_ID = "TIGER-Lab/VideoScore-v1.1"

HEAD_WEIGHTS = torch.tensor([0.30, 0.25, 0.15, 0.20, 0.10], dtype=torch.float32)


class VideoScoreV2Adapter:
    name = "videoscore_v2"

    def __init__(
        self,
        *,
        model_id: str = HF_ID,
        device: str = "cuda",
        num_frames: int = 8,
    ) -> None:
        from transformers import AutoProcessor, Idefics2ForConditionalGeneration  # noqa: F401

        self.device = torch.device(device)
        self.num_frames = num_frames
        try:
            from mantis.models.idefics2 import Idefics2ForSequenceClassification
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "VideoScore requires `pip install mantis-vl` — see the "
                "README under scripts/downstream_grpo/."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = (
            Idefics2ForSequenceClassification.from_pretrained(
                model_id, torch_dtype=torch.bfloat16
            )
            .to(self.device)
            .eval()
        )

    def _load_video(self, path: str) -> list:
        # returns evenly-sampled list of PIL images matching processor expectations
        from PIL import Image

        vid, _, _ = read_video(path, pts_unit="sec")   # [T, H, W, C] uint8
        T = vid.shape[0]
        if T == 0:
            raise ValueError(f"empty video at {path}")
        idx = torch.linspace(0, T - 1, self.num_frames).round().long()
        frames = vid[idx].numpy()
        return [Image.fromarray(f) for f in frames]

    def _score_one(self, prompt: str, video_path: str) -> float:
        frames = self._load_video(video_path)
        eval_prompt = (
            "Suppose you are an expert in judging generated videos. "
            f"Text prompt: {prompt}\n"
            "Rate the video on visual quality, temporal consistency, "
            "dynamic degree, text-video alignment, and factual consistency."
        )
        inputs = self.processor(
            text=eval_prompt,
            images=frames,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(**inputs).logits.squeeze(0).float().cpu()
        weighted = float((logits * HEAD_WEIGHTS[: logits.numel()]).sum().item())
        return weighted

    def score_batch(self, batch: Sequence[RewardRequest]) -> torch.Tensor:
        out = []
        for req in batch:
            try:
                out.append(self._score_one(req.prompt, req.video_path))
            except Exception as exc:  # noqa: BLE001
                print(f"[videoscore_v2] failed on {req.video_path}: {exc}")
                out.append(3.0)
        return torch.tensor(out, dtype=torch.float32)

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()
