"""VideoAlign / VideoReward adapter (KwaiVGI/VideoReward). Optional arm E.

Wraps the multi-head VideoReward model released with the paper
"Improving Video Generation with Human Feedback" (Liu et al., 2025).

* Upstream code: https://github.com/KlingAIResearch/VideoAlign (Apache-2.0)
* Checkpoint: https://huggingface.co/KwaiVGI/VideoReward

The HF checkpoint is **not** a standard HF snapshot: it ships a bespoke
``model_config.json`` and a raw PEFT-wrapped ``checkpoint-*/model.pth``.
Loading it requires reconstructing the same nn.Module (Qwen2-VL-2B +
special reward tokens + LoRA + regression head) that produced the
checkpoint. We vendor a minimal port of ``Qwen2VLRewardModelBT`` and the
loader logic from upstream so this adapter is self-contained.

The adapter returns the **VQ** (visual quality) scalar. It optionally
z-score normalizes with the ``inference_config`` block from the
checkpoint's ``model_config.json`` so scores land on the same scale as
the upstream README/report.

Expected layout of ``model_id`` (default: env ``VIDEOREWARD_LOCAL_DIR``
or ``/group/40063/jacuzzitang/models/VideoReward``)::

    <snapshot>/
        model_config.json
        checkpoint-<step>/
            model.pth
            tokenizer/...
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from scripts.downstream_grpo.src.reward_adapters import RewardRequest


DEFAULT_LOCAL_DIR = "/group/40063/jacuzzitang/models/VideoReward"
BASE_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
SPECIAL_TOKENS = ["<|VQ_reward|>", "<|MQ_reward|>", "<|TA_reward|>"]


DETAILED_PROMPT_WITH_SPECIAL_TOKEN = """
You are tasked with evaluating a generated video based on three distinct criteria: Visual Quality, Motion Quality, and Text Alignment. Please provide a rating from 0 to 10 for each of the three categories, with 0 being the worst and 10 being the best. Each evaluation should be independent of the others.

**Visual Quality:**
Evaluate the overall visual quality of the video, with a focus on static factors. The following sub-dimensions should be considered:
- **Reasonableness:** The video should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the video. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the video, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The video should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible.

Please provide the ratings of Visual Quality: <|VQ_reward|>
END

**Motion Quality:**
Assess the dynamic aspects of the video, with a focus on dynamic factors. Consider the following sub-dimensions:
- **Stability:** Evaluate the continuity and stability between frames. There should be no sudden, unnatural jumps, and the video should maintain stable attributes (e.g., no fluctuating colors, textures, or missing body parts).
- **Naturalness:** The movement should align with physical laws and be realistic. For example, clothing should flow naturally with motion, and facial expressions should change appropriately (e.g., blinking, mouth movements).
- **Aesthetic Quality:** The movement should be smooth and fluid. The transitions between different motions or camera angles should be seamless, and the overall dynamic feel should be visually pleasing.
- **Fusion:** Ensure that elements in motion (e.g., edges of the subject, hair, clothing) blend naturally with the background, without obvious artifacts or the feeling of cut-and-paste effects.
- **Clarity of Motion:** The video should be clear and smooth in motion. Pay attention to any areas where the video might have blurry or unsteady sections that hinder visual continuity.
- **Amplitude:** If the video is largely static or has little movement, assign a low score for motion quality.

Please provide the ratings of Motion Quality: <|MQ_reward|>
END

**Text Alignment:**
Assess how well the video matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the video (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Motion Relevance:** Evaluate if the dynamic actions (e.g., gestures, posture, facial expressions like talking or blinking) align with the described prompt. The motion should match the prompt in terms of type, scale, and direction.
- **Environment Relevance:** Assess whether the background and scene fit the prompt. This includes checking if real-world locations or scenes are accurately represented, though some stylistic adaptation is acceptable.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the video adheres to this style.
- **Camera Movement Relevance:** Check if the camera movements (e.g., following the subject, focus shifts) are consistent with the expected behavior from the prompt.

Textual prompt - {text_prompt}
Please provide the ratings of Text Alignment: <|TA_reward|>
END
"""


def _build_prompt(text_prompt: str) -> str:
    return DETAILED_PROMPT_WITH_SPECIAL_TOKEN.format(text_prompt=text_prompt)


def _resolve_snapshot_dir(model_id: str) -> str:
    if os.path.isdir(model_id):
        return model_id
    env_dir = os.environ.get("VIDEOREWARD_LOCAL_DIR", "").strip()
    if env_dir and Path(env_dir).is_dir():
        return env_dir
    if Path(DEFAULT_LOCAL_DIR).is_dir():
        return DEFAULT_LOCAL_DIR
    raise FileNotFoundError(
        f"Cannot locate a VideoReward snapshot. Tried model_id={model_id!r}, "
        f"$VIDEOREWARD_LOCAL_DIR={env_dir!r}, default={DEFAULT_LOCAL_DIR!r}. "
        "Download with `huggingface-cli download KwaiVGI/VideoReward "
        "--local-dir <path>` and re-run."
    )


def _pick_checkpoint(snapshot_dir: str, step: int = -1) -> str:
    candidates = sorted(
        glob.glob(os.path.join(snapshot_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint-* dir under {snapshot_dir!r} — expected e.g. checkpoint-11352/"
        )
    if step is None or step < 0:
        return candidates[0]
    match = os.path.join(snapshot_dir, f"checkpoint-{step}")
    return match if match in candidates else candidates[0]


def _build_qwen2vl_reward_model_class():
    """Return the ``Qwen2VLRewardModelBT`` class (ported from upstream).

    Deferred so we don't pull transformers at module import time.
    """
    from typing import List, Optional
    from transformers import Qwen2VLForConditionalGeneration

    class Qwen2VLRewardModelBT(Qwen2VLForConditionalGeneration):
        """Port of upstream ``trainer.Qwen2VLRewardModelBT`` (KlingAIResearch/VideoAlign, Apache-2.0)."""

        def __init__(self, config, output_dim: int = 1, reward_token: str = "special",
                     special_token_ids: Optional[List[int]] = None):
            super().__init__(config)
            self.output_dim = output_dim
            self.rm_head = nn.Linear(config.hidden_size, output_dim, bias=False)
            self.reward_token = reward_token
            self.special_token_ids = special_token_ids
            if self.special_token_ids is not None:
                self.reward_token = "special"

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            pixel_values=None,
            pixel_values_videos=None,
            image_grid_thw=None,
            video_grid_thw=None,
            rope_deltas=None,
            cache_position=None,
        ):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # Transformers >=4.57 places both towers inside Qwen2VLModel:
            # self.model.visual and self.model.language_model. Let that model
            # perform multimodal embedding injection and M-RoPE construction.
            modern_layout = (
                hasattr(self.model, "visual")
                and hasattr(self.model, "language_model")
            )
            if modern_layout:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    rope_deltas=rope_deltas,
                    cache_position=cache_position,
                )
                hidden_states = outputs[0]
            else:
                # Legacy Transformers layout used by the released checkpoint.
                if inputs_embeds is None:
                    inputs_embeds = self.model.embed_tokens(input_ids)
                    if pixel_values is not None:
                        pixel_values = pixel_values.type(self.visual.get_dtype())
                        image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                        image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
                    if pixel_values_videos is not None:
                        pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                        video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                        video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(inputs_embeds.device)

                outputs = self.model(
                    input_ids=None,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
                hidden_states = outputs[0]  # [B, L, D]
            logits = self.rm_head(hidden_states)  # [B, L, N]

            batch_size = (input_ids if input_ids is not None else inputs_embeds).shape[0]

            if self.reward_token == "special":
                special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
                for tid in self.special_token_ids:
                    special_token_mask = special_token_mask | (input_ids == tid)
                pooled = logits[special_token_mask, ...]
                pooled = pooled.view(batch_size, 3, -1)  # [B, 3, N]
                if self.output_dim == 3:
                    pooled = pooled.diagonal(dim1=1, dim2=2)
                pooled = pooled.view(batch_size, -1)
                return {"logits": pooled}

            # last-token fallback (unused for the released checkpoint)
            if self.config.pad_token_id is None:
                sequence_lengths = -1
            else:
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            pooled = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
            return {"logits": pooled}

    return Qwen2VLRewardModelBT


def _unwrap_checkpoint_state_dict(payload) -> dict[str, torch.Tensor]:
    """Return a tensor state_dict from common checkpoint container layouts."""
    if not isinstance(payload, dict):
        raise TypeError(
            f"VideoReward checkpoint must be a dict, got {type(payload).__name__}."
        )

    def _is_state_dict(obj) -> bool:
        return (
            isinstance(obj, dict)
            and bool(obj)
            and all(isinstance(k, str) for k in obj)
            and all(torch.is_tensor(v) for v in obj.values())
        )

    if _is_state_dict(payload):
        return payload

    for key in ("state_dict", "model_state_dict", "model", "module"):
        candidate = payload.get(key)
        if _is_state_dict(candidate):
            return candidate

    preview = list(payload)[:8]
    raise TypeError(
        "Could not find a tensor state_dict in VideoReward checkpoint; "
        f"top-level keys: {preview}."
    )


def _align_checkpoint_state_dict_to_runtime(
    state_dict: dict[str, torch.Tensor],
    runtime_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], int]:
    """Align a pre-4.57 Qwen2-VL PEFT checkpoint to the runtime module layout.

    Transformers 4.57 moved the text backbone below ``model.language_model``
    and the visual tower below ``model.visual``. PEFT adds the stable
    ``base_model.model`` wrapper outside those paths, so released VideoReward
    keys need these deterministic conversions:

      base_model.model.model.*  -> base_model.model.model.language_model.*
      base_model.model.visual.* -> base_model.model.model.visual.*

    Keys that already match the runtime are kept unchanged. Every remapped key
    must exist in the runtime state_dict and have the same shape.
    """
    runtime_keys = set(runtime_state)
    aligned: dict[str, torch.Tensor] = {}
    remapped = 0

    old_text_prefix = "base_model.model.model."
    new_text_prefix = "base_model.model.model.language_model."
    old_visual_prefix = "base_model.model.visual."
    new_visual_prefix = "base_model.model.model.visual."

    for old_key, tensor in state_dict.items():
        candidates = [old_key]

        if old_key.startswith(old_visual_prefix):
            candidates.append(new_visual_prefix + old_key[len(old_visual_prefix):])

        if (
            old_key.startswith(old_text_prefix)
            and not old_key.startswith(new_text_prefix)
            and not old_key.startswith(new_visual_prefix)
        ):
            candidates.append(new_text_prefix + old_key[len(old_text_prefix):])

        matched_key = None
        shape_candidates = []
        for candidate in candidates:
            if candidate not in runtime_keys:
                continue
            shape_candidates.append(candidate)
            if tuple(runtime_state[candidate].shape) == tuple(tensor.shape):
                matched_key = candidate
                break

        if matched_key is None:
            raise RuntimeError(
                "Cannot align VideoReward checkpoint key to runtime: "
                f"checkpoint key={old_key!r}, shape={tuple(tensor.shape)}, "
                f"candidate runtime keys={shape_candidates or candidates}."
            )

        if matched_key in aligned:
            raise RuntimeError(
                f"Multiple checkpoint keys map to runtime key {matched_key!r}."
            )

        aligned[matched_key] = tensor
        remapped += int(matched_key != old_key)

    aligned_keys = set(aligned)
    missing = sorted(runtime_keys - aligned_keys)
    unexpected = sorted(aligned_keys - runtime_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Aligned VideoReward checkpoint is still incomplete: "
            f"missing={len(missing)} first={missing[:5]}, "
            f"unexpected={len(unexpected)} first={unexpected[:5]}."
        )

    return aligned, remapped


class VideoAlignAdapter:
    """RewardVerse adapter for KwaiVGI/VideoReward. Emits the VQ scalar."""

    name = "videoalign"

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_LOCAL_DIR,
        device: str = "cuda",
        step: int = -1,
        use_norm: bool = True,
        which_head: str = "VQ",
    ) -> None:
        from transformers import AutoProcessor

        assert which_head in {"VQ", "MQ", "TA", "Overall"}, which_head
        self.which_head = which_head
        self.device = torch.device(device)
        self.use_norm = use_norm

        snapshot_dir = _resolve_snapshot_dir(model_id)
        with open(os.path.join(snapshot_dir, "model_config.json")) as f:
            cfg = json.load(f)
        self._data_cfg = cfg.get("data_config", {})
        self._inference_cfg = cfg.get("inference_config") or {}
        model_cfg = cfg.get("model_config", {}) or {}
        peft_cfg = cfg.get("peft_lora_config", {}) or {}

        base_id = model_cfg.get("model_name_or_path") or BASE_MODEL_ID
        output_dim = int(model_cfg.get("output_dim", 1))

        # 1) Processor from the base Qwen2-VL, then add the 3 special reward tokens.
        self.processor = AutoProcessor.from_pretrained(
            base_id, padding_side="right", use_fast=False
        )
        self.processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS}
        )
        self._special_ids = self.processor.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

        # 2) Rebuild exactly the module used by upstream training.
        Qwen2VLRewardModelBT = _build_qwen2vl_reward_model_class()
        model = Qwen2VLRewardModelBT.from_pretrained(
            base_id,
            output_dim=output_dim,
            reward_token=model_cfg.get("reward_token", "special"),
            special_token_ids=self._special_ids,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model.resize_token_embeddings(len(self.processor.tokenizer))

        # 3) The released model.pth is a full PeftModel state_dict. Recreate the
        # same LoRA wrapper first, then load the checkpoint without rewriting keys.
        lora_enabled = bool(peft_cfg.get("lora_enable", False))
        if lora_enabled:
            import peft
            import transformers
            from peft import LoraConfig, get_peft_model

            target_modules = self._collect_lora_targets(model, peft_cfg)
            if not target_modules:
                raise RuntimeError(
                    "VideoReward LoRA is enabled, but no target Linear/Embedding "
                    "modules were selected. Check lora_namespan_exclude and "
                    "num_lora_modules in model_config.json."
                )

            modules_to_save = peft_cfg.get("lora_modules_to_save")
            if isinstance(modules_to_save, str):
                modules_to_save = [modules_to_save]

            lora_config = LoraConfig(
                target_modules=target_modules,
                r=int(peft_cfg.get("lora_r", 16)),
                lora_alpha=int(peft_cfg.get("lora_alpha", 32)),
                lora_dropout=float(peft_cfg.get("lora_dropout", 0.0)),
                task_type=str(peft_cfg.get("lora_task_type", "CAUSAL_LM")),
                use_rslora=bool(peft_cfg.get("use_rslora", False)),
                bias="none",
                modules_to_save=modules_to_save,
            )
            model = get_peft_model(model, lora_config)
            print(
                "[videoalign] loader=upstream-peft-v3-compat "
                f"transformers={transformers.__version__} "
                f"peft={peft.__version__} "
                f"targets={len(target_modules)}"
            )
        else:
            print("[videoalign] loader=plain-model-v1 LoRA disabled")

        ckpt_dir = _pick_checkpoint(snapshot_dir, step=step)
        state_path = os.path.join(ckpt_dir, "model.pth")
        if not os.path.isfile(state_path):
            raise FileNotFoundError(f"missing checkpoint payload: {state_path}")

        payload = torch.load(state_path, map_location="cpu", weights_only=False)
        state_dict = _unwrap_checkpoint_state_dict(payload)

        runtime_state = model.state_dict()
        state_dict, remapped = _align_checkpoint_state_dict_to_runtime(
            state_dict,
            runtime_state,
        )
        print(
            "[videoalign] checkpoint layout aligned to runtime: "
            f"remapped={remapped}/{len(state_dict)} tensors"
        )

        model.load_state_dict(state_dict, strict=True)
        del runtime_state
        del state_dict
        del payload

        print(
            "[videoalign] loaded upstream PEFT checkpoint strictly: "
            f"{state_path}"
        )

        # 4) Move to device, freeze, done.
        model = model.to(self.device).to(torch.bfloat16).eval()
        model.requires_grad_(False)
        model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        self.model = model

    @staticmethod
    def _collect_lora_targets(base_model, peft_cfg) -> list[str]:
        """Reproduce ``find_target_linear_names`` from upstream train_reward.py."""
        exclude = peft_cfg.get("lora_namespan_exclude") or []
        if isinstance(exclude, str):
            exclude = [exclude]
        names: list[str] = []
        for name, mod in base_model.named_modules():
            if any(x in name for x in exclude):
                continue
            if isinstance(mod, (nn.Linear, nn.Embedding)):
                names.append(name)
        num = int(peft_cfg.get("num_lora_modules", -1))
        if num > 0:
            names = names[-num:]
        return names

    def _prepare_batch(self, prompts: Sequence[str], video_paths: Sequence[str]):
        from qwen_vl_utils import process_vision_info

        max_pixels = int(self._data_cfg.get("max_frame_pixels", 200704))
        fps = self._data_cfg.get("fps", 2.0)
        num_frames = self._data_cfg.get("num_frames")
        sample_type = self._data_cfg.get("sample_type", "uniform")

        chat_data = []
        for prompt, vp in zip(prompts, video_paths):
            video_field = {
                "type": "video",
                "video": f"file://{vp}",
                "max_pixels": max_pixels,
                "sample_type": sample_type,
            }
            if num_frames is None:
                video_field["fps"] = fps
            else:
                video_field["nframes"] = int(num_frames)
            chat_data.append(
                [
                    {
                        "role": "user",
                        "content": [
                            video_field,
                            {"type": "text", "text": _build_prompt(prompt)},
                        ],
                    }
                ]
            )

        image_inputs, video_inputs = process_vision_info(chat_data)
        batch = self.processor(
            text=self.processor.apply_chat_template(
                chat_data, tokenize=False, add_generation_prompt=True
            ),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}

    def _norm(self, vq: float, mq: float, ta: float) -> tuple[float, float, float]:
        if not self.use_norm or not self._inference_cfg:
            return vq, mq, ta
        c = self._inference_cfg
        vq = (vq - c["VQ_mean"]) / c["VQ_std"]
        mq = (mq - c["MQ_mean"]) / c["MQ_std"]
        ta = (ta - c["TA_mean"]) / c["TA_std"]
        return vq, mq, ta

    @torch.no_grad()
    def score_batch(self, batch: Sequence[RewardRequest]) -> torch.Tensor:
        out: list[float] = []
        for req in batch:
            try:
                inputs = self._prepare_batch([req.prompt], [req.video_path])
                logits = self.model(return_dict=True, **inputs)["logits"]  # [1, 3]
                vq, mq, ta = (float(x) for x in logits.squeeze(0).float().cpu().tolist())
                vq, mq, ta = self._norm(vq, mq, ta)
                if self.which_head == "VQ":
                    out.append(vq)
                elif self.which_head == "MQ":
                    out.append(mq)
                elif self.which_head == "TA":
                    out.append(ta)
                else:  # Overall
                    out.append(vq + mq + ta)
            except Exception as exc:  # noqa: BLE001
                print(f"[videoalign] failed on {req.video_path}: {exc}")
                out.append(0.0)
        return torch.tensor(out, dtype=torch.float32)

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()
