"""Downstream GRPO trainer for Wan-2.1-1.3B with a swappable reward.

Design mirrors ``fastvideo/train_rpo_flux.py``:
  * sample G rollouts per prompt (SDE),
  * compute rewards via a socket-based reward server,
  * standardize advantages within each prompt-group,
  * PPO clipped update over a random subset of denoising steps,
  * KL against the frozen no-LoRA transformer,
  * DeepSpeed ZeRO-2 across 7 ranks; rank 7 GPU hosts the reward server.

Launched by ``launch/run_arm_*.sh`` on the 8xH100 node.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist
import yaml
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model
from torchvision.io import write_video

from scripts.downstream_grpo.src.logging_utils import RunLogger
from scripts.downstream_grpo.src.prompt_pool import (
    Prompt,
    PromptIterator,
    build_and_dump_splits,
    read_split,
)
from scripts.downstream_grpo.src.reward_client import RewardClient
from scripts.downstream_grpo.src.sampler_wan import WanGRPOSampler


REPO_ROOT = Path(__file__).resolve().parents[3]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class TrainCfg:
    arm: str
    cfg: dict
    ckpt_dir: Path
    log_dir: Path
    sample_dir: Path
    max_steps: int
    save_every: int
    eval_every: int
    log_every: int


def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# LoRA / KL helpers
# --------------------------------------------------------------------------- #


def apply_lora(transformer, cfg: dict):
    target_modules = [m.strip() for m in cfg["lora_target_modules"].split(",") if m.strip()]
    lora = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        init_lora_weights=True,
        bias="none",
        target_modules=target_modules,
    )
    transformer.requires_grad_(False)
    transformer = get_peft_model(transformer, lora)
    # Even with LoRA, backward through the 30-layer frozen base saves every
    # layer's activations by default. Wan-2.1-1.3B forward for a 480x832x33
    # video costs ~2 GB activations *per layer* — 60GB+ for a small micro-batch.
    # Gradient checkpointing recomputes activations in the backward pass and
    # brings peak activation memory down ~5-10x for one extra forward.
    base = transformer.base_model.model if hasattr(transformer, "base_model") else transformer
    if hasattr(base, "enable_gradient_checkpointing"):
        base.enable_gradient_checkpointing()
    elif hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()
    if int(os.environ.get("RANK", 0)) == 0:
        transformer.print_trainable_parameters()
    return transformer


class disable_lora:
    """Context manager: temporarily disable PEFT adapters for ref-model logπ.

    Handles both the plain PEFT-wrapped module and the DeepSpeedEngine that wraps it.
    Without this, under DeepSpeed the ``disable_adapter`` attribute is not
    reachable and the KL term degenerates to zero (ref == policy).
    """

    def __init__(self, transformer):
        self.transformer = transformer

    def __enter__(self):
        inner = self.transformer
        # DeepSpeedEngine exposes .module to reach the wrapped PeftModel.
        if hasattr(inner, "module") and hasattr(inner.module, "disable_adapter"):
            inner = inner.module
        if hasattr(inner, "disable_adapter"):
            self._ctx = inner.disable_adapter()
            self._ctx.__enter__()
        else:
            self._ctx = None
        return self

    def __exit__(self, *exc):
        if self._ctx is not None:
            self._ctx.__exit__(*exc)


# --------------------------------------------------------------------------- #
# Rollout + reward gather
# --------------------------------------------------------------------------- #


def save_videos_to_mp4(
    videos: torch.Tensor,       # [G, F, H, W, 3] uint8
    prompt_id: str,
    step: int,
    out_dir: Path,
    fps: int,
    rank: int,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for g in range(videos.shape[0]):
        p = out_dir / f"step{step:05d}_rank{rank}_{prompt_id}_g{g}.mp4"
        write_video(str(p), videos[g], fps=fps)
        paths.append(str(p))
    return paths


def run_rollout_and_score(
    *,
    prompts: Sequence[Prompt],
    sampler: WanGRPOSampler,
    reward_client: RewardClient | None,
    cfg: dict,
    step: int,
    sample_dir: Path,
    rank: int,
    seed_offset: int,
) -> tuple[dict, list[float]]:
    """Sample G videos per prompt, gather rewards.

    Returns:
      rollouts: dict with stacked trajectory / log_probs / conditioning tensors
                over B*G along dim 0
      rewards:  list of B*G floats (same order as rollouts entries)
    """
    G = cfg["num_generations"]
    all_traj: list[torch.Tensor] = []
    all_logp: list[torch.Tensor] = []
    all_prompt_emb: list[torch.Tensor] = []
    all_neg_emb: list[torch.Tensor] = []
    all_video_paths: list[str] = []
    all_prompt_texts: list[str] = []
    all_dim_tags: list[str] = []
    all_sigmas: torch.Tensor | None = None

    for b_idx, p in enumerate(prompts):
        out = sampler.sample_group(
            p.query,
            G=G,
            num_frames=cfg["num_frames"],
            height=cfg["video_height"],
            width=cfg["video_width"],
            num_inference_steps=cfg["num_inference_steps"],
            guidance_scale=cfg["guidance_scale"],
            scheduler_shift=cfg["scheduler_shift"],
            eta=cfg["eta"],
            seed_base=(cfg["seed"] * 1000003 + step * 97 + b_idx * 7 + seed_offset),
        )
        video_paths = save_videos_to_mp4(
            out.videos, p.prompt_id, step, sample_dir, cfg["fps"], rank
        )
        all_traj.append(out.trajectory)
        all_logp.append(out.log_probs)
        all_video_paths.extend(video_paths)
        all_prompt_texts.extend([p.query] * G)
        all_dim_tags.extend([p.dimension] * G)
        if all_sigmas is None:
            all_sigmas = out.sigmas

        # cache encodings for score_transition() during PPO
        emb = sampler.encode_prompts([p.query] * G)
        neg = sampler.encode_prompts([""] * G)
        all_prompt_emb.append(emb.cpu())
        all_neg_emb.append(neg.cpu())

    trajectory = torch.cat(all_traj, dim=0)           # [B*G, S+1, C, T', H', W'] (CPU)
    log_probs = torch.cat(all_logp, dim=0)            # [B*G, S] (CPU)
    prompt_emb = torch.cat(all_prompt_emb, dim=0)     # [B*G, L, D] (CPU)
    neg_emb = torch.cat(all_neg_emb, dim=0)

    # Rewards --- only rank 0 talks to the reward server, then broadcasts.
    if rank == 0 and reward_client is not None:
        rewards_t = reward_client.score([
            {"prompt": pt, "dimension": dt, "video_path": vp}
            for pt, dt, vp in zip(all_prompt_texts, all_dim_tags, all_video_paths)
        ])
    else:
        rewards_t = torch.zeros(len(all_video_paths), dtype=torch.float32)

    if dist.is_initialized():
        # trajectory is on CPU now; use the caller's cuda device for the collective.
        rewards_t = rewards_t.to(torch.device(f"cuda:{torch.cuda.current_device()}"))
        dist.broadcast(rewards_t, src=0)

    rewards = rewards_t.detach().cpu().tolist()

    return (
        {
            "trajectory": trajectory,
            "log_probs": log_probs,
            "prompt_emb": prompt_emb,
            "neg_emb": neg_emb,
            "sigmas": all_sigmas,
            "video_paths": all_video_paths,
            "prompt_texts": all_prompt_texts,
            "dimensions": all_dim_tags,
        },
        rewards,
    )


def compute_group_advantage(
    rewards: Sequence[float], *, G: int, adv_clip_max: float
) -> torch.Tensor:
    """Group-standardize rewards within each B-block."""
    r = torch.tensor(rewards, dtype=torch.float32).view(-1, G)
    mean = r.mean(dim=1, keepdim=True)
    std = r.std(dim=1, keepdim=True) + 1e-8
    adv = ((r - mean) / std).clamp(-adv_clip_max, adv_clip_max)
    return adv.reshape(-1)                            # [B*G]


# --------------------------------------------------------------------------- #
# PPO update over denoising steps
# --------------------------------------------------------------------------- #


def _is_deepspeed_engine(obj) -> bool:
    """True when ``sampler.transformer`` has been wrapped by deepspeed.initialize."""
    return obj.__class__.__module__.startswith("deepspeed")


def _save_lora_ckpt(transformer, out_dir: Path) -> None:
    """Save PEFT LoRA adapters, unwrapping the DeepSpeedEngine if present."""
    inner = transformer.module if _is_deepspeed_engine(transformer) else transformer
    if not hasattr(inner, "save_pretrained"):
        raise RuntimeError(
            f"Cannot save LoRA: {type(inner).__name__} has no save_pretrained method"
        )
    inner.save_pretrained(out_dir)


def ppo_update(
    *,
    rollout: dict,
    advantages: torch.Tensor,       # [B*G]
    old_log_probs: torch.Tensor,    # [B*G, S]
    sampler: WanGRPOSampler,
    optimizer,
    cfg: dict,
    device: torch.device,
    step: int,
    logger: RunLogger,
    rank: int,
) -> dict:
    trajectory = rollout["trajectory"]                # CPU [B*G, S+1, ...]
    # prompt_emb / neg_emb stay on CPU; per-micro-batch slice moves to GPU below.
    prompt_emb_cpu = rollout["prompt_emb"]
    neg_emb_cpu = rollout["neg_emb"]
    sigmas = rollout["sigmas"].to(device)

    engine = sampler.transformer if _is_deepspeed_engine(sampler.transformer) else None

    BG, S_plus_1 = trajectory.shape[0], trajectory.shape[1]
    S = S_plus_1 - 1

    train_S = max(1, int(S * cfg["timestep_fraction"]))
    # sample distinct denoising indices to train on
    idxs = torch.randperm(S)[:train_S].tolist()

    # PPO micro-batch: 32-sample per-step forwards through Wan-2.1-1.3B save
    # ~2 GB activations per layer × 30 layers even with LoRA (frozen base still
    # needs activations for backprop). Split into small micro-batches to cap
    # activation memory; grads accumulate before optimizer.step().
    ppo_micro_batch = int(cfg.get("ppo_micro_batch", 4))
    ppo_micro_batch = max(1, min(ppo_micro_batch, BG))
    n_micro = math.ceil(BG / ppo_micro_batch)

    total_loss = 0.0
    total_kl = 0.0
    total_ratio = 0.0

    for i in idxs:
        z_t_full = trajectory[:, i]                       # CPU [B*G, C, T', H', W']
        z_next_full = trajectory[:, i + 1]

        for mb_start in range(0, BG, ppo_micro_batch):
            mb_end = min(mb_start + ppo_micro_batch, BG)
            sl = slice(mb_start, mb_end)

            z_t = z_t_full[sl].to(device, non_blocking=True)
            z_next = z_next_full[sl].to(device, non_blocking=True)
            p_emb = prompt_emb_cpu[sl].to(device, non_blocking=True)
            n_emb = neg_emb_cpu[sl].to(device, non_blocking=True)

            new_lp = sampler.score_transition(
                latents_t=z_t,
                latents_next=z_next,
                prompt_emb=p_emb,
                neg_emb=n_emb,
                sigmas=sigmas,
                index=i,
                guidance_scale=cfg["guidance_scale"],
                eta=cfg["eta"],
                train=True,
            )

            if cfg["kl_beta"] > 0:
                with torch.no_grad(), disable_lora(sampler.transformer):
                    ref_lp = sampler.score_transition(
                        latents_t=z_t,
                        latents_next=z_next,
                        prompt_emb=p_emb,
                        neg_emb=n_emb,
                        sigmas=sigmas,
                        index=i,
                        guidance_scale=cfg["guidance_scale"],
                        eta=cfg["eta"],
                        train=False,
                    )
                kl = new_lp - ref_lp
            else:
                kl = torch.zeros_like(new_lp)

            adv = advantages[sl].to(device)
            ratio = torch.exp(new_lp - old_log_probs[sl, i].to(device))
            unclipped = -adv * ratio
            clipped = -adv * torch.clamp(ratio, 1.0 - cfg["clip_range"], 1.0 + cfg["clip_range"])
            pg_loss = torch.maximum(unclipped, clipped).mean()
            kl_loss = cfg["kl_beta"] * kl.mean()
            # normalize by (denoise steps we train on) × (micro-batches per step)
            loss = (pg_loss + kl_loss) / (train_S * n_micro)

            if engine is not None:
                engine.backward(loss)
            else:
                loss.backward()

            total_loss += float(loss.detach().item())
            total_kl += float(kl.detach().mean().item())
            total_ratio += float(ratio.detach().mean().item())

            # Free per-micro-batch tensors before the next micro-batch's forward
            # allocates its own activation stack.
            del z_t, z_next, new_lp, ratio, pg_loss, kl_loss, loss, kl
        del z_t_full, z_next_full

    if engine is not None:
        # DS handles clipping via config; still expose an approx grad_norm for logs.
        engine.step()
        grad_norm = torch.tensor(float(getattr(engine, "get_global_grad_norm", lambda: 0.0)() or 0.0))
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in sampler.transformer.parameters() if p.requires_grad],
            cfg["max_grad_norm"],
        )
        optimizer.step()
        optimizer.zero_grad()

    stats = {
        "loss": total_loss,
        "kl": total_kl / max(1, train_S * n_micro),
        "ratio": total_ratio / max(1, train_S * n_micro),
        "grad_norm": float(grad_norm),
    }
    if rank == 0:
        logger.log("ppo", step=step, **stats)
    return stats


# --------------------------------------------------------------------------- #
# In-training eval
# --------------------------------------------------------------------------- #


@torch.no_grad()
def eval_fixed_batch(
    *,
    prompts: Sequence[Prompt],
    sampler: WanGRPOSampler,
    reward_client: RewardClient | None,
    cfg: dict,
    step: int,
    sample_dir: Path,
    logger: RunLogger,
    rank: int,
) -> None:
    """Sample one video per prompt on the fixed-32 batch, gather stats."""
    if rank != 0:
        return
    eval_dir = sample_dir / f"eval_step{step:05d}"
    rewards: list[float] = []
    for p in prompts:
        out = sampler.sample_group(
            p.query,
            G=1,
            num_frames=cfg["num_frames"],
            height=cfg["video_height"],
            width=cfg["video_width"],
            num_inference_steps=cfg["num_inference_steps"],
            guidance_scale=cfg["guidance_scale"],
            scheduler_shift=cfg["scheduler_shift"],
            eta=0.0,                    # deterministic eval
            seed_base=hash(p.prompt_id) & 0xFFFFFF,
        )
        video_path = save_videos_to_mp4(out.videos, p.prompt_id, step, eval_dir, cfg["fps"], rank)[0]
        if reward_client is not None:
            r = reward_client.score([
                {"prompt": p.query, "dimension": p.dimension, "video_path": video_path}
            ])
            rewards.append(float(r.item()))
    logger.log(
        "eval_fixed",
        step=step,
        mean_reward=(sum(rewards) / max(1, len(rewards))) if rewards else 0.0,
        n=len(rewards),
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="e.g. arm_D_rewardverse")
    ap.add_argument("--config", default="scripts/downstream_grpo/configs/wan_grpo_default.yaml")
    ap.add_argument("--reward_socket", required=True)
    ap.add_argument("--wandb_project", default="")
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--eval_every", type=int, default=None)
    ap.add_argument("--save_every", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=None)
    args = ap.parse_args()

    cfg = load_yaml(REPO_ROOT / args.config)
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.eval_every is not None:
        cfg["eval_every"] = args.eval_every
    if args.save_every is not None:
        cfg["save_every"] = args.save_every
    if args.log_every is not None:
        cfg["log_every"] = args.log_every

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group("nccl")

    set_seed(cfg["seed"] + rank)

    ckpt_dir = REPO_ROOT / cfg["ckpt_root"] / args.arm
    log_dir = REPO_ROOT / cfg["log_root"]
    sample_dir = REPO_ROOT / cfg["sample_root"] / args.arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    logger = RunLogger(
        log_dir,
        args.arm,
        wandb_project=args.wandb_project or None,
        wandb_run_name=args.arm,
        rank=rank,
    )
    logger.log("config", step=0, **cfg)

    # Prompt splits (idempotent)
    splits_dir = REPO_ROOT / "scripts/downstream_grpo/configs/prompts"
    if rank == 0 and not (splits_dir / "evalverse_train.jsonl").exists():
        build_and_dump_splits(cfg, splits_dir)
    if dist.is_initialized():
        dist.barrier()

    train_prompts = read_split(splits_dir / "evalverse_train.jsonl")
    fixed32 = read_split(splits_dir / "evalverse_fixed32.jsonl")

    prompt_iter = PromptIterator(train_prompts, seed=cfg["seed"] + rank)

    # Reward client (rank-0 only actually talks; others get zeros pre-broadcast)
    reward_client = RewardClient(args.reward_socket) if rank == 0 else None

    # Sampler / model
    sampler = WanGRPOSampler(cfg["wan_model_id"], device=device)
    sampler.transformer = apply_lora(sampler.transformer, cfg)

    # Optimizer (LoRA params only)
    trainable = [p for p in sampler.transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg["learning_rate"],
        betas=(cfg["adam_beta1"], cfg["adam_beta2"]),
        weight_decay=cfg["adam_weight_decay"],
        eps=cfg["adam_eps"],
    )

    # DeepSpeed init (ZeRO-2 over LoRA params)
    try:
        import deepspeed  # type: ignore

        ds_cfg = json.loads((REPO_ROOT / "scripts/downstream_grpo/configs/ds_zero2.json").read_text())
        ds_cfg["train_batch_size"] = cfg["prompts_per_step"] * cfg["num_generations"] * max(1, world_size)
        ds_cfg["gradient_accumulation_steps"] = cfg["gradient_accumulation_steps"]
        transformer, optimizer, _, _ = deepspeed.initialize(
            model=sampler.transformer,
            optimizer=optimizer,
            config=ds_cfg,
        )
        sampler.transformer = transformer
    except ImportError:
        if rank == 0:
            print("[trainer] deepspeed not installed; running single-GPU.")

    if rank == 0:
        print(f"[trainer] arm={args.arm} steps={cfg['max_steps']} G={cfg['num_generations']}")

    for step in range(1, cfg["max_steps"] + 1):
        t0 = time.time()
        prompts = prompt_iter.next_batch(cfg["prompts_per_step"])

        rollout, rewards = run_rollout_and_score(
            prompts=prompts,
            sampler=sampler,
            reward_client=reward_client,
            cfg=cfg,
            step=step,
            sample_dir=sample_dir,
            rank=rank,
            seed_offset=rank,
        )
        advantages = compute_group_advantage(
            rewards, G=cfg["num_generations"], adv_clip_max=cfg["adv_clip_max"]
        )
        stats = ppo_update(
            rollout=rollout,
            advantages=advantages,
            old_log_probs=rollout["log_probs"],
            sampler=sampler,
            optimizer=optimizer,
            cfg=cfg,
            device=device,
            step=step,
            logger=logger,
            rank=rank,
        )

        if rank == 0:
            logger.log(
                "step",
                step=step,
                arm=args.arm,
                mean_reward=float(sum(rewards) / len(rewards)),
                reward_std=float(torch.tensor(rewards).std().item()),
                seconds=time.time() - t0,
                **stats,
            )

        if step % cfg["eval_every"] == 0 and rank == 0:
            eval_fixed_batch(
                prompts=fixed32,
                sampler=sampler,
                reward_client=reward_client,
                cfg=cfg,
                step=step,
                sample_dir=sample_dir,
                logger=logger,
                rank=rank,
            )

        if step % cfg["save_every"] == 0 and rank == 0:
            ckpt = ckpt_dir / f"checkpoint-{step:05d}"
            ckpt.mkdir(parents=True, exist_ok=True)
            _save_lora_ckpt(sampler.transformer, ckpt)
            print(f"[trainer] saved {ckpt}")

    if rank == 0:
        final = ckpt_dir / "checkpoint-final"
        final.mkdir(parents=True, exist_ok=True)
        _save_lora_ckpt(sampler.transformer, final)
        print(f"[trainer] saved {final}")
    logger.close()


if __name__ == "__main__":
    main()
