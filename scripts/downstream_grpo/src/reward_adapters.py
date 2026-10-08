"""Unified reward-adapter interface + registry for downstream GRPO.

Each adapter accepts a list of (prompt, dimension, mp4_path) tuples
and returns a [N] float tensor of scalar rewards, higher = better.

Implementations live under ``reward_models/``. This module wires them
into a name -> class registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import torch


@dataclass(frozen=True)
class RewardRequest:
    prompt: str
    dimension: str
    video_path: str


class RewardAdapter(Protocol):
    name: str

    def score_batch(self, batch: Sequence[RewardRequest]) -> torch.Tensor: ...

    def close(self) -> None: ...


# lazy registry (avoid importing every model on every launch)
_REGISTRY: dict[str, str] = {
    "rewardverse": "scripts.downstream_grpo.src.reward_models.rewardverse_adapter:RewardVerseAdapter",
    "v3_zeroshot": "scripts.downstream_grpo.src.reward_models.v3_zeroshot_adapter:V3ZeroshotAdapter",
    "videoscore_v2": "scripts.downstream_grpo.src.reward_models.videoscore_v2_adapter:VideoScoreV2Adapter",
    "videoalign": "scripts.downstream_grpo.src.reward_models.videoalign_adapter:VideoAlignAdapter",
}


def build_adapter(name: str, **kwargs) -> RewardAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown reward adapter '{name}'. Registered: {list(_REGISTRY)}")
    import importlib

    mod_path, cls_name = _REGISTRY[name].split(":")
    module = importlib.import_module(mod_path)
    cls = getattr(module, cls_name)
    return cls(**kwargs)


def available_adapters() -> list[str]:
    return sorted(_REGISTRY)
