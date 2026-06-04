# Copyright (c) [2025] [FastVideo Team]
# SPDX-License-Identifier: Apache License 2.0
"""
Minimal T2V RPO training hook for Auto-Rubric video rewards.

This module documents the integration surface for text-to-video RL. Wire your T2V
sampler to produce two mp4 paths per prompt, then call Judger.get_reward_single.

Example pair item:
    {
        "query": "<text prompt>",
        "response": ["./videos/rank_a.mp4", "./videos/rank_b.mp4"],
    }
"""

from __future__ import annotations

import asyncio
from typing import Any

import torch

from judger import Judger


def get_arr_reward_sync(arr_judge: Judger, qa_data: dict[str, Any], device: torch.device) -> torch.Tensor:
    """Blocking wrapper used inside training loops."""
    return asyncio.run(arr_judge.get_reward_single(qa_data, device))


def build_pair_item(query: str, video_paths: list[str]) -> dict[str, Any]:
    if len(video_paths) != 2:
        raise ValueError("T2V RPO requires exactly two candidate videos per prompt.")
    return {"query": query, "response": video_paths}


def pairwise_rewards_from_videos(
    arr_judge: Judger,
    query: str,
    video_a: str,
    video_b: str,
    device: torch.device,
) -> torch.Tensor:
    """Score a single prompt's two generated videos with ARR pairwise rewards."""
    return get_arr_reward_sync(arr_judge, build_pair_item(query, [video_a, video_b]), device)


# Integration checklist for a full trainer:
# 1. Sample two videos per prompt from your T2V pipeline (same noise seed group optional).
# 2. Build pair dicts and call pairwise_rewards_from_videos / get_arr_reward_sync.
# 3. Use returned tensor [1.0, -0.1] (or swapped) as GRPO advantages.
# 4. Configure judger with rubric_pipeline/config/qwen3vl_8B_instruct_t2v.yaml
#    and a pre-generated rubrics_file for stable training.
