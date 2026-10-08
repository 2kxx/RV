"""Wan-2.1-1.3B rollout sampler for GRPO.

Wan uses `UniPCMultistepScheduler` in diffusers. For GRPO we need
per-step Gaussian log-probabilities of the sampled x_{t-1} given the
predicted mean at step t. We implement this by adding SDE noise on
top of the deterministic UniPC step (analogous to how FLUX RPO
adds Euler-SDE noise in `fastvideo/train_rpo_flux.py::flux_step`).

Two rollout modes:
  * ``mode="sample"``  — no reference latents; draw new noise per step,
                          return the trajectory + log-probs.
  * ``mode="score"``   — given a reference trajectory, compute the
                          new-policy log-prob of the SAME transitions
                          (for the PPO update).

We keep VAE + text-encoder frozen and *outside* FSDP/DeepSpeed to avoid
memory blowup during rollout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from diffusers import (
    AutoencoderKLWan,
    UniPCMultistepScheduler,
    WanPipeline,
    WanTransformer3DModel,
)
from transformers import AutoTokenizer, UMT5EncoderModel


@dataclass
class WanSampleOutput:
    videos: torch.Tensor              # [G, T, H, W, 3] uint8
    trajectory: torch.Tensor          # [G, num_steps + 1, C, T', H', W'] fp32
    log_probs: torch.Tensor           # [G, num_steps]  fp32   (sum-log-p over pixels)
    sigmas: torch.Tensor              # [num_steps + 1]        fp32


def _shift_sigmas(sigmas: torch.Tensor, shift: float) -> torch.Tensor:
    return (shift * sigmas) / (1.0 + (shift - 1.0) * sigmas)


def _step_with_logp(
    model_pred: torch.Tensor,      # [G, C, T', H', W']  (predicted velocity)
    latents: torch.Tensor,         # [G, C, T', H', W']
    sigmas: torch.Tensor,          # [num_steps + 1]
    index: int,
    *,
    eta: float,
    prev_sample: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One UniPC-style rectified-flow SDE step with Gaussian log-prob.

    Rectified-flow parameterisation: v = ε - x0 (Wan is trained this way).
    Deterministic Euler mean:   x_{t-1} = x_t + (σ_{t-1} - σ_t) · v_θ(x_t, t).
    We add SDE noise scaled by ``eta`` so the transition kernel becomes
    Gaussian with known variance, enabling per-step log-prob.
    """
    sigma_t = sigmas[index]
    sigma_next = sigmas[index + 1]
    dsigma = sigma_next - sigma_t              # negative (sigmas decrease)

    mean = latents + dsigma * model_pred

    x0_hat = latents - sigma_t * model_pred

    delta = sigma_t - sigma_next               # positive
    std = eta * math.sqrt(max(float(delta), 1e-12))

    if std > 0.0:
        score = -(latents - x0_hat * (1.0 - sigma_t)) / (sigma_t ** 2 + 1e-12)
        log_correction = -0.5 * (eta ** 2) * score
        mean = mean + log_correction * dsigma

    if prev_sample is None:
        noise = torch.randn_like(mean)
        prev = mean + std * noise
    else:
        prev = prev_sample

    if std > 0.0:
        var = std ** 2
        diff = prev.to(torch.float32) - mean.to(torch.float32)
        log_prob = (
            -0.5 * (diff ** 2) / var
            - math.log(std)
            - 0.5 * math.log(2.0 * math.pi)
        )
    else:
        log_prob = torch.zeros_like(prev, dtype=torch.float32)
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return prev, x0_hat, log_prob


class WanGRPOSampler:
    """Thin wrapper around diffusers' WanPipeline exposing per-step logπ."""

    def __init__(
        self,
        model_id: str,
        device: torch.device,
        *,
        dtype: torch.dtype = torch.bfloat16,
        vae_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = device
        self.dtype = dtype

        # Load submodules independently so we can control precisions.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=dtype
        ).to(device).eval()
        self.vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=vae_dtype
        ).to(device).eval()
        # G=8 * 480x832x33 fp32 activations OOM an 80GB H100 that also holds
        # the transformer + LoRA + text encoder. Slicing decodes one video at
        # a time; tiling caps the per-slice spatial footprint.
        self.vae.enable_slicing()
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()
        self.transformer = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=dtype
        ).to(device)
        self._transformer_in_channels = self.transformer.config.in_channels
        self.scheduler = UniPCMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        for p in self.vae.parameters():
            p.requires_grad_(False)

    # -------- text --------
    @torch.no_grad()
    def encode_prompts(self, prompts: Sequence[str]) -> torch.Tensor:
        toks = self.tokenizer(
            list(prompts),
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        out = self.text_encoder(
            input_ids=toks.input_ids,
            attention_mask=toks.attention_mask,
        )
        return out.last_hidden_state.to(self.dtype)   # [B, L, D]

    # -------- latent init --------
    def _latent_shape(self, num_frames: int, h: int, w: int) -> tuple[int, int, int, int]:
        # Wan VAE downsample factors: spatial 8, temporal 4.
        C = self._transformer_in_channels
        return (C, num_frames // 4 + 1, h // 8, w // 8)

    # -------- rollout --------
    def sample_group(
        self,
        prompt: str,
        *,
        G: int,
        num_frames: int,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        scheduler_shift: float,
        eta: float,
        seed_base: int,
    ) -> WanSampleOutput:
        """Sample G videos from one prompt, returning trajectory + log-probs."""
        prompt_emb = self.encode_prompts([prompt] * G)      # [G, L, D]
        neg_emb = self.encode_prompts([""] * G)

        C, T_lat, H_lat, W_lat = self._latent_shape(num_frames, height, width)
        gen = torch.Generator(device=self.device).manual_seed(seed_base)
        latents = torch.randn(
            (G, C, T_lat, H_lat, W_lat),
            generator=gen,
            device=self.device,
            dtype=torch.float32,
        )

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        sigmas_raw = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=self.device)
        sigmas = _shift_sigmas(sigmas_raw, scheduler_shift)

        traj = [latents.clone()]
        log_probs: list[torch.Tensor] = []
        self.transformer.eval()
        for i in range(num_inference_steps):
            sigma = sigmas[i]
            timestep = (sigma * 1000.0).long().expand(G)
            with torch.no_grad(), torch.autocast("cuda", dtype=self.dtype):
                pred_cond = self.transformer(
                    hidden_states=latents.to(self.dtype),
                    timestep=timestep.to(self.dtype),
                    encoder_hidden_states=prompt_emb,
                    return_dict=False,
                )[0]
                pred_uncond = self.transformer(
                    hidden_states=latents.to(self.dtype),
                    timestep=timestep.to(self.dtype),
                    encoder_hidden_states=neg_emb,
                    return_dict=False,
                )[0]
            model_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            latents_next, _, lp = _step_with_logp(
                model_pred.to(torch.float32),
                latents,
                sigmas,
                i,
                eta=eta,
                prev_sample=None,
            )
            latents = latents_next
            traj.append(latents.clone())
            log_probs.append(lp)

        trajectory = torch.stack(traj, dim=1)               # [G, S+1, C, T', H', W']
        log_probs_t = torch.stack(log_probs, dim=1)         # [G, S]

        # Decode final latents.  Wan VAE normalizes with per-channel
        # latents_mean / latents_std, not a scalar scaling_factor.
        with torch.no_grad():
            vae_cfg = self.vae.config
            z_dim = vae_cfg.z_dim
            latents_mean = (
                torch.tensor(vae_cfg.latents_mean)
                .view(1, z_dim, 1, 1, 1)
                .to(latents.device, torch.float32)
            )
            latents_std = (
                torch.tensor(vae_cfg.latents_std)
                .view(1, z_dim, 1, 1, 1)
                .to(latents.device, torch.float32)
            )
            latents_dec = (latents.to(torch.float32) * latents_std + latents_mean).to(self.vae.dtype)
            frames = self.vae.decode(latents_dec, return_dict=False)[0]      # [G, C, F, H, W]
            del latents_dec
        videos = frames.clamp(-1, 1).add(1).mul(127.5).round().to(torch.uint8)
        videos = videos.permute(0, 2, 3, 4, 1).contiguous()                  # [G, F, H, W, 3]
        videos_cpu = videos.cpu()
        del frames, videos
        # Trajectory holds fp32 latents for every denoise step; keeping it on
        # GPU pins ~few GB per prompt across the outer B loop. Move to CPU;
        # trainer restages step-by-step in ppo_update via ``.to(device)``.
        trajectory_cpu = trajectory.detach().cpu()
        log_probs_cpu = log_probs_t.detach().cpu()
        sigmas_cpu = sigmas.detach().cpu()
        del trajectory, log_probs_t
        torch.cuda.empty_cache()

        return WanSampleOutput(
            videos=videos_cpu,
            trajectory=trajectory_cpu,
            log_probs=log_probs_cpu,
            sigmas=sigmas_cpu,
        )

    # -------- log-prob under a (possibly LoRA'd) policy --------
    def score_transition(
        self,
        *,
        latents_t: torch.Tensor,          # [G, C, T', H', W']
        latents_next: torch.Tensor,       # [G, C, T', H', W']
        prompt_emb: torch.Tensor,         # [G, L, D]
        neg_emb: torch.Tensor,            # [G, L, D]
        sigmas: torch.Tensor,
        index: int,
        guidance_scale: float,
        eta: float,
        train: bool,
    ) -> torch.Tensor:
        """Return log π_θ(latents_next | latents_t) at step ``index``.

        Enables PPO ratio computation. Caller is responsible for
        toggling ``transformer.train()`` / gradient enablement.
        """
        if train:
            self.transformer.train()
        else:
            self.transformer.eval()
        sigma = sigmas[index]
        timestep = (sigma * 1000.0).long().expand(latents_t.shape[0])
        with torch.autocast("cuda", dtype=self.dtype):
            pred_cond = self.transformer(
                hidden_states=latents_t.to(self.dtype),
                timestep=timestep.to(self.dtype),
                encoder_hidden_states=prompt_emb,
                return_dict=False,
            )[0]
            pred_uncond = self.transformer(
                hidden_states=latents_t.to(self.dtype),
                timestep=timestep.to(self.dtype),
                encoder_hidden_states=neg_emb,
                return_dict=False,
            )[0]
        model_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
        _, _, log_prob = _step_with_logp(
            model_pred.to(torch.float32),
            latents_t.to(torch.float32),
            sigmas,
            index,
            eta=eta,
            prev_sample=latents_next.to(torch.float32),
        )
        return log_prob
