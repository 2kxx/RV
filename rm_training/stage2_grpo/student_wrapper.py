"""rm_training.stage2_grpo.student_wrapper
========================================
Thin wrapper over Qwen2.5-VL-7B-Instruct (or any compatible video-LLM)
exposing the operations Stage ② needs:

    1. ``generate_rubric(query, caption) -> str``
    2. ``generate_rubric_with_meta(query, caption) -> RubricRollout``
       Same as #1 but also returns ``prompt_ids`` and ``sampled_ids`` so
       the trainer can re-run a teacher-forcing forward to obtain
       grad-on log-probabilities. Required for the GRPO policy update.
    3. ``score_video(query, caption, video_path, rubric_text) -> str``
       Greedy by default during training (so reward variance comes from
       rubric sampling, not score sampling).
    4. ``compute_log_probs(prompt_ids, sampled_ids, *, ref=False)``
       Teacher-forcing forward. ``ref=True`` disables the LoRA adapter
       (one model serves as both policy and frozen reference) — used by
       GRPO's KL term. With ``ref=False`` and ``with torch.enable_grad()``
       this is the path the policy gradient flows through.

LoRA injection
--------------
When ``cfg.lora_config_path`` is set, ``_ensure_model`` wraps the base
model with PEFT LoRA adapters as configured in
``configs/lora_qwen25vl.yaml``. The adapter is named so it can be
toggled off via ``with model.disable_adapter():`` for the reference
log-prob computation.

Dummy mode
----------
Set env ``RM_DUMMY=1`` (or ``StudentConfig(dummy=True)``) and the
wrapper returns deterministic mock outputs WITHOUT touching torch /
transformers. Trainable-parameter / log-prob calls are no-ops in dummy
mode (they return Python floats / empty tensors), so the trainer's
optimiser path is exercised on CPU but no gradient actually flows.
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

__all__ = ["StudentConfig", "StudentWrapper", "RubricRollout", "DimensionContext"]


class _NullCtx:
    """Trivial context manager used as a fall-through for ``with`` blocks."""
    def __enter__(self): return None
    def __exit__(self, *exc): return False


@dataclass
class DimensionContext:
    """Per-dimension prompt scaffolding consumed by RUBRIC/SCORE templates.

    All fields are plain strings so the wrapper doesn't need to import
    ``stage1_teacher.task_descriptions`` (avoids cross-module dependency
    at import time). The trainer is responsible for materialising one
    ``DimensionContext`` per dim from its loaded ``TaskSpec`` map.
    """
    dimension:        str = "video_quality"
    evalverse_path:   str = ""        # e.g. "Acting > Action"
    task_description: str = ""        # short version
    sub_aspects:      list[str] = field(default_factory=list)

    def sub_aspects_block(self) -> str:
        """Bulleted list, or a one-line stand-in when sub_aspects is empty."""
        if not self.sub_aspects:
            return ("- (free-form: cover the most discriminative facets of "
                    "this dimension; do not invent a fixed taxonomy)")
        return "\n".join(f"- {s}" for s in self.sub_aspects)

    @classmethod
    def generic(cls, dimension: str = "video_quality") -> "DimensionContext":
        """Fallback when the trainer can't find a TaskSpec for ``dimension``.

        Renders a minimal scaffold: just a generic "judge this dimension"
        sentence. Lets the wrapper run end-to-end on dummy / smoke data
        without forcing every caller to load yamls.
        """
        return cls(
            dimension=dimension or "video_quality",
            evalverse_path="",
            task_description=(
                f"Evaluate the **{dimension or 'video_quality'}** dimension of "
                f"the generated video. Be concrete and observable."
            ),
            sub_aspects=[],
        )


# ---------------------------------------------------------------------------
# Default prompts (kept at module scope for easy override / inspection).
# ---------------------------------------------------------------------------
RUBRIC_PROMPT_TMPL = """You are a strict video reward judge for the **{dimension}** dimension ({evalverse_path}).

## Task
{task_description}

## Sub-aspects (your 5 rubrics MUST collectively cover these)
{sub_aspects_block}

## Rubric requirements
- Observable from a single video clip alone (no pair comparison)
- Discriminative: the rubric must distinguish good vs. bad executions of THIS dimension
- Each grounded in one or more sub-aspects above
- Concrete (no "looks better" / "feels right" — name what to look for)
- Plain natural language; no benchmark-specific bracket prefixes
- Stay tied to **{dimension}**; do NOT drift into neighbouring dimensions

## Caption (the prompt that drove video generation)
{caption}

## Output
STRICT JSON with this schema (no markdown fences, no commentary):

{{
  "rubrics": [
    {{"path": "Theme: <short theme>", "weight": <float in (0,1)>, "tips": ["<tip1>", "<tip2>"]}}
  ]
}}

Exactly 5 rubrics. Weights must sum to 1. Each rubric must carry at least 2 non-empty tips.
"""

# Plain-text score scaffold (matches stage0 D5 + soft_logits aggregation).
# Format is deliberately deterministic — every "theme_k:" is followed by
# exactly one digit token, which lets the soft-logits path read per-position
# logits over {1..5} ids without re-tokenisation surprises.
SCORE_PROMPT_TMPL = """You are a strict video reward judge for the **{dimension}** dimension ({evalverse_path}).

## Task
{task_description}

Score the SINGLE video on each rubric on a 1-5 integer scale (1=very poor,
2=poor, 3=fair, 4=good, 5=excellent).

Respond with EXACTLY this one-line format, replacing each <X> with a single
digit 1-5. No commas, no extra words, no explanations:

{format_line}

## Caption (the prompt that drove video generation)
{caption}

## Rubric (DO NOT change paths or weights)
{rubric_text}
"""


# ---------------------------------------------------------------------------
# Config + main wrapper.
# ---------------------------------------------------------------------------
@dataclass
class StudentConfig:
    model_path:        str | None = None       # e.g. /group/.../Qwen2.5-VL-7B-Instruct
    device_map:        str = "auto"
    torch_dtype:       str = "bfloat16"
    max_new_tokens:    int = 1024
    temperature:       float = 0.7              # GRPO sampling temperature
    top_p:             float = 0.95
    num_video_frames:  int = 16
    fps:               float = 2.0
    rubric_prompt_tmpl: str = RUBRIC_PROMPT_TMPL
    score_prompt_tmpl:  str = SCORE_PROMPT_TMPL
    # ── LoRA. None → no LoRA (full fine-tune; not what we want for 7B
    # video models). When set, ``_ensure_model`` wraps the base model
    # via peft.get_peft_model() with the yaml's settings.
    lora_config_path:  str | None = None
    # If True, score_video is forced to greedy (temperature=0) during
    # training so the reward variance comes purely from the rubric
    # sampling — keeps GRPO advantage well-defined.
    score_greedy:      bool = True
    # Score-extraction method (matches stage0 D5 ablation findings):
    #   "soft_logits"   — read per-position logits at each "theme_k:"
    #                     anchor, take softmax over {1..5} digit ids, and
    #                     report E[k]=Σ k·p_k. Continuous floats in [1,5];
    #                     no parse_fail. Default.
    #   "integer_text"  — let the model emit text "theme_1: 3 theme_2: 4
    #                     ...", parse with regex. Coarser but no logits
    #                     bookkeeping needed.
    # Falls back to integer_text at runtime if the tokenizer can't resolve
    # distinct token ids for "1".."5".
    score_method:      str = "soft_logits"
    # Max new tokens for the score forward — short scaffold needs much less
    # than the rubric-gen forward. ~64 fits 5 themes comfortably.
    score_max_new_tokens: int = 64
    # Dummy mode flag; overridden by env RM_DUMMY=1.
    dummy:             bool = False
    seed:              int = 0


@dataclass
class RubricRollout:
    """One rubric-gen rollout, packed for the policy-gradient update.

    ``prompt_ids`` and ``sampled_ids`` are 1-D LongTensors on the model's
    device; the trainer concatenates them and passes them back through
    ``compute_log_probs`` to obtain grad-on log-probabilities. ``text``
    is the decoded rubric (caller parses via ``rubric_parser``).
    """
    text:         str
    prompt_ids:   "object"     # torch.Tensor at runtime; lazy type
    sampled_ids:  "object"     # torch.Tensor at runtime
    # Optional: cached log-prob from sampling time (mostly for debug,
    # NOT used by the policy gradient — backprop needs a fresh forward).
    sampling_logprob_sum: float | None = None


class StudentWrapper:
    """Lazy-loaded Qwen2.5-VL-Instruct interface."""

    def __init__(self, cfg: StudentConfig):
        self.cfg = cfg
        self._dummy = bool(cfg.dummy) or os.environ.get("RM_DUMMY", "0") == "1"
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._rng = random.Random(cfg.seed)

        if self._dummy:
            print("[student] running in DUMMY mode — no torch/transformers loaded.",
                  file=sys.stderr)
            return

        if not cfg.model_path:
            raise ValueError("StudentConfig.model_path is required when RM_DUMMY is not set.")

    # ------------------------------------------------------------------
    # Lazy real-model loader.
    # ------------------------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None or self._dummy:
            return
        # Imports kept inside the function so dummy mode never pays the
        # transformers import cost.
        import torch
        from transformers import (
            AutoProcessor,
            AutoTokenizer,
            Qwen2_5_VLForConditionalGeneration,
        )

        torch_dtype = getattr(torch, self.cfg.torch_dtype, torch.bfloat16)
        print(f"[student] loading {self.cfg.model_path} dtype={self.cfg.torch_dtype}",
              file=sys.stderr)
        self._processor = AutoProcessor.from_pretrained(self.cfg.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_path)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.cfg.model_path,
            torch_dtype=torch_dtype,
            device_map=self.cfg.device_map,
        )

        # ── Optional LoRA injection.
        if self.cfg.lora_config_path:
            self._inject_lora(self.cfg.lora_config_path)

        # ── Soft-logits score-token-id resolution (mirrors stage0 D5).
        # Builds the digit-id groups + colon anchors once at load time.
        self._init_score_token_ids()

        # We default to eval mode; the trainer will flip ``.train()`` on
        # the LoRA params right before calling compute_log_probs.
        self._model.eval()

    def _init_score_token_ids(self) -> None:
        """Resolve (id of "1"), ..., (id of "5") + colon-anchor ids.

        Soft-logits scoring needs to:
          (a) detect WHICH generated token is the score digit for theme_k
              — done by requiring the previous token end with ':';
          (b) read logits OVER the {1..5} ids at that step and softmax
              them into E[k].

        BPE tokenisers occasionally encode " 1" / "1" / ":1" as different
        ids; we collect ALL such variants so probability mass on any of
        them aggregates onto integer value k correctly.

        Sets:
          self._score_token_ids        : [id('1'), ..., id('5')]
          self._score_token_ids_space  : [id(' 1'), ..., id(' 5')]
          self._score_digit_id_groups  : list[list[int]] — all variants per digit
          self._score_id_to_value      : dict[int, int]
          self._score_colon_ids        : set[int]
          self._soft_logits_ok         : bool
        """
        tok = self._tokenizer or getattr(self._processor, "tokenizer", None)
        self._score_token_ids        = None
        self._score_token_ids_space  = None
        self._score_digit_id_groups  = []
        self._score_id_to_value      = {}
        self._score_colon_ids        = set()
        self._soft_logits_ok         = False
        if tok is None:
            print("[student] no tokenizer found; soft_logits disabled.",
                  file=sys.stderr)
            return

        def _try_get_ids(strategy: str) -> list[int] | None:
            ids: list[int] = []
            try:
                for k in range(1, 6):
                    if strategy == "A":
                        tid = tok.convert_tokens_to_ids(str(k))
                        if tid == tok.unk_token_id:
                            return None
                    elif strategy == "B":
                        enc = tok.encode("score: " + str(k), add_special_tokens=False)
                        tid = enc[-1]
                    else:
                        enc = tok.encode(str(k), add_special_tokens=False)
                        tid = enc[-1]
                    ids.append(tid)
            except Exception:
                return None
            return ids if len(set(ids)) == 5 else None

        ids_bare: list[int] | None = None
        for strat in ("A", "B", "C"):
            ids_bare = _try_get_ids(strat)
            if ids_bare is not None:
                break
        if ids_bare is None:
            print("[student] could not resolve distinct ids for '1'..'5'; "
                  "score_method will fall back to integer_text.",
                  file=sys.stderr)
            return

        ids_space: list[int] = []
        for k in range(1, 6):
            try:
                ids_space.append(tok.encode(" " + str(k), add_special_tokens=False)[-1])
            except Exception:
                ids_space.append(ids_bare[k - 1])

        # All last-token variants for each digit value.
        digit_groups: list[list[int]] = [[] for _ in range(5)]
        for k_idx in range(5):
            for pref in ("", " ", ": ", ":", "\n", " : "):
                try:
                    enc = tok.encode(pref + str(k_idx + 1), add_special_tokens=False)
                except Exception:
                    continue
                if enc:
                    digit_groups[k_idx].append(enc[-1])
            # Dedup per group, preserve order.
            seen: set[int] = set()
            uniq: list[int] = []
            for tid in digit_groups[k_idx]:
                if tid not in seen:
                    seen.add(tid); uniq.append(tid)
            digit_groups[k_idx] = uniq

        id_to_value: dict[int, int] = {}
        for k_idx, grp in enumerate(digit_groups):
            for tid in grp:
                id_to_value.setdefault(tid, k_idx + 1)
        # Drop dupes (id appearing in both digit-3 and digit-4 groups, etc.).
        clean_groups: list[list[int]] = [[] for _ in range(5)]
        for k_idx, grp in enumerate(digit_groups):
            for tid in grp:
                if id_to_value.get(tid) == k_idx + 1:
                    clean_groups[k_idx].append(tid)

        # Colon-anchor scan (cap to vocab to keep startup cheap).
        colon_ids: set[int] = set()
        try:
            vocab_size = tok.vocab_size
        except Exception:
            vocab_size = 0
        if vocab_size and vocab_size < 200_000:
            for tid in range(vocab_size):
                try:
                    s = tok.decode([tid])
                except Exception:
                    continue
                if s and s.rstrip(" ").endswith(":"):
                    colon_ids.add(tid)

        self._score_token_ids        = ids_bare
        self._score_token_ids_space  = ids_space
        self._score_digit_id_groups  = clean_groups
        self._score_id_to_value      = id_to_value
        self._score_colon_ids        = colon_ids
        self._soft_logits_ok         = True
        print(f"[student] soft_logits ready  digit_ids={ids_bare}  "
              f"group_sizes={[len(g) for g in clean_groups]}  "
              f"colon_ids={len(colon_ids)}", file=sys.stderr)

    def _inject_lora(self, yaml_path: str) -> None:
        """Wrap ``self._model`` with PEFT LoRA adapters from a yaml file."""
        try:
            import yaml                                          # type: ignore
        except ImportError as e:                                 # pragma: no cover
            raise RuntimeError(
                "PyYAML required for LoRA injection — pip install pyyaml"
            ) from e
        try:
            from peft import LoraConfig, get_peft_model          # type: ignore
        except ImportError as e:                                 # pragma: no cover
            raise RuntimeError(
                "peft required for LoRA injection — pip install peft"
            ) from e

        with open(yaml_path, "r", encoding="utf-8") as f:
            spec = yaml.safe_load(f) or {}
        target = list(spec.get("target_modules") or [])
        if spec.get("include_mlp_proj"):
            target += list(spec.get("extra_target_modules_mlp") or [])

        lcfg = LoraConfig(
            r              = int(spec.get("r", 16)),
            lora_alpha     = int(spec.get("lora_alpha", 32)),
            lora_dropout   = float(spec.get("lora_dropout", 0.05)),
            bias           = str(spec.get("bias", "none")),
            task_type      = str(spec.get("task_type", "CAUSAL_LM")),
            target_modules = target or ["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        adapter_name = str(spec.get("adapter_name", "default"))
        self._model = get_peft_model(self._model, lcfg, adapter_name=adapter_name)
        # Trainable params summary.
        try:
            self._model.print_trainable_parameters()
        except Exception:
            pass
        print(f"[student] LoRA injected (adapter={adapter_name}, "
              f"r={lcfg.r}, target={lcfg.target_modules})",
              file=sys.stderr)

    # ------------------------------------------------------------------
    # Trainable-parameter accessors (used by train_grpo to build optimiser).
    # ------------------------------------------------------------------
    def trainable_parameters(self):
        """Iterator over parameters with ``requires_grad=True``.

        In dummy mode returns an empty list — the trainer's optimiser
        creation will detect the empty iterable and skip backward.
        """
        if self._dummy or self._model is None:
            return iter([])
        return (p for p in self._model.parameters() if p.requires_grad)

    @property
    def model(self):
        """Underlying nn.Module (PEFT-wrapped if LoRA is on). None in dummy mode."""
        return self._model

    @property
    def device(self):
        if self._dummy or self._model is None:
            return None
        return next(self._model.parameters()).device

    # ------------------------------------------------------------------
    # Public op #1: rubric-only generation (no video).
    # ------------------------------------------------------------------
    def generate_rubric(self,
                        caption:    str | None = None,
                        *,
                        dim_ctx:    DimensionContext | None = None,
                        # Legacy aliases for older callers — caption used to
                        # be called ``query``. Kept until the last call site
                        # migrates.
                        query:      str | None = None) -> str:
        """Sample a rubric for one (caption, dimension).

        ``dim_ctx`` provides the per-dim task description + sub-aspects;
        when None, falls back to ``DimensionContext.generic()``.
        ``caption`` is the prompt that drove video generation (the
        v_arr_mvp pair file's ``query`` field — see naming clarification
        in the project README).
        """
        ctx    = dim_ctx or DimensionContext.generic()
        cap    = caption if caption is not None else (query or "")
        prompt = self._format_rubric_prompt(ctx, cap)
        if self._dummy:
            return self._dummy_rubric(cap or ctx.dimension)

        self._ensure_model()
        return self._generate_text_only(prompt)

    def generate_rubric_with_meta(self,
                                  caption:    str | None = None,
                                  *,
                                  dim_ctx:    DimensionContext | None = None,
                                  query:      str | None = None
                                  ) -> "RubricRollout":
        """Same contract as ``generate_rubric`` but returns prompt/sampled ids."""
        ctx    = dim_ctx or DimensionContext.generic()
        cap    = caption if caption is not None else (query or "")
        prompt = self._format_rubric_prompt(ctx, cap)
        if self._dummy:
            return RubricRollout(
                text=self._dummy_rubric(cap or ctx.dimension),
                prompt_ids=None,
                sampled_ids=None,
            )

        self._ensure_model()
        return self._generate_text_only_with_meta(prompt)

    def generate_rubrics_batch(self, captions: Iterable[str],
                               *, dim_ctxs: Iterable[DimensionContext] | None = None
                               ) -> list[str]:
        """Convenience: same as ``generate_rubric`` over a list of captions."""
        captions = list(captions)
        if dim_ctxs is None:
            dim_ctxs = [DimensionContext.generic()] * len(captions)
        else:
            dim_ctxs = list(dim_ctxs)
        return [self.generate_rubric(c, dim_ctx=ctx)
                for c, ctx in zip(captions, dim_ctxs)]

    def is_dummy(self) -> bool:
        return self._dummy

    # ------------------------------------------------------------------
    # Prompt formatters (shared by sampling, with-meta, and dummy paths).
    # ------------------------------------------------------------------
    def _format_rubric_prompt(self, ctx: DimensionContext, caption: str) -> str:
        return self.cfg.rubric_prompt_tmpl.format(
            dimension         = ctx.dimension or "video_quality",
            evalverse_path    = ctx.evalverse_path or ctx.dimension or "video_quality",
            task_description  = (ctx.task_description or "").strip(),
            sub_aspects_block = ctx.sub_aspects_block(),
            caption           = caption or "",
        )

    def _format_score_prompt(self, ctx: DimensionContext, caption: str,
                             rubric_text: str, format_line: str) -> str:
        return self.cfg.score_prompt_tmpl.format(
            dimension         = ctx.dimension or "video_quality",
            evalverse_path    = ctx.evalverse_path or ctx.dimension or "video_quality",
            task_description  = (ctx.task_description or "").strip(),
            caption           = caption or "",
            rubric_text       = rubric_text or "",
            format_line       = format_line,
        )

    # ------------------------------------------------------------------
    # Public op #2: score one video given a rubric.
    # ------------------------------------------------------------------
    def score_video(self,
                    caption:      str | None,
                    video_path:   str,
                    rubric_text:  str,
                    *,
                    dim_ctx:      DimensionContext | None = None,
                    rubric_paths: list[str] | None = None,
                    # Legacy positional alias: caption used to be passed
                    # as the first arg called ``query``. Old call sites
                    # should migrate to keyword form.
                    query:        str | None = None,
                    ) -> tuple[str, list[float | None]]:
        """Score one video on each rubric path, returning ``(text, scores)``.

        ``scores`` is a list of length ``len(rubric_paths)`` (or 5 if
        ``rubric_paths`` is None) of either floats in [1.0, 5.0] (soft_logits
        mode) or ints in {1..5} (integer_text mode). ``None`` at any slot
        means that theme's score could not be recovered.

        ``text`` is the model's raw decoded output, kept for diagnostics
        and r_format scoring (matched against the plain-text scaffold).

        The plain-text scaffold the model is asked to emit is::

            theme_1: <X> theme_2: <X> theme_3: <X> theme_4: <X> theme_5: <X>

        which is the same shape stage0 D5 validated as the most stable
        format for 7B Qwen2.5-VL. soft_logits then reads logits at each
        ``<X>`` position over the 5 digit-id variants and reports
        E[k]=Σ k·softmax(logits)_k as a continuous score in [1.0, 5.0].
        """
        if caption is None:
            caption = query or ""
        ctx = dim_ctx or DimensionContext.generic()
        n   = len(rubric_paths) if rubric_paths else 5
        format_line = " ".join(f"theme_{k}: <X>" for k in range(1, n + 1))
        prompt = self._format_score_prompt(ctx, caption, rubric_text or "", format_line)
        if self._dummy:
            return self._dummy_score(rubric_text, video_path, n)

        self._ensure_model()
        method = self.cfg.score_method
        if method == "soft_logits" and not getattr(self, "_soft_logits_ok", False):
            method = "integer_text"
        if method == "soft_logits":
            return self._score_video_softlogits(
                prompt, video_path, n,
                force_greedy=self.cfg.score_greedy,
            )
        # integer_text fallback / opt-in.
        text = self._generate_with_video(
            prompt, video_path,
            force_greedy=self.cfg.score_greedy,
            max_new_tokens=self.cfg.score_max_new_tokens,
        )
        from .rubric_parser import parse_score_scaffold
        return text, parse_score_scaffold(text, n)

    # ------------------------------------------------------------------
    # Real model: text-only forward.
    # ------------------------------------------------------------------
    def _generate_text_only(self, prompt: str) -> str:
        import torch
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        chat = self._processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._processor(text=[chat], return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
            )
        gen = out_ids[0, inputs["input_ids"].shape[1]:]
        return self._processor.tokenizer.decode(gen, skip_special_tokens=True)

    def _generate_text_only_with_meta(self, prompt: str) -> "RubricRollout":
        """Same as ``_generate_text_only`` but also returns prompt + sampled ids."""
        import torch
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        chat = self._processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._processor(text=[chat], return_tensors="pt").to(self._model.device)
        prompt_ids = inputs["input_ids"][0]                # (P,)
        with torch.no_grad():
            out_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                do_sample=True,
                temperature=self.cfg.temperature,
                top_p=self.cfg.top_p,
            )
        sampled_ids = out_ids[0, prompt_ids.shape[0]:]     # (T_new,)
        text = self._processor.tokenizer.decode(sampled_ids, skip_special_tokens=True)
        return RubricRollout(
            text=text,
            prompt_ids=prompt_ids.detach(),
            sampled_ids=sampled_ids.detach(),
        )

    # ------------------------------------------------------------------
    # Teacher-forcing log-prob forward (used by GRPO policy gradient + KL).
    # ------------------------------------------------------------------
    def compute_log_probs(self, prompt_ids, sampled_ids, *, ref: bool = False):
        """Return per-token log-prob of ``sampled_ids`` given ``prompt_ids``.

        Returns
        -------
        torch.Tensor of shape ``(T,)`` where ``T = len(sampled_ids)``.
        With ``ref=False`` this carries grad iff the model has trainable
        params (LoRA on); with ``ref=True`` we ``with disable_adapter()``
        the LoRA so the same forward gives base-model log-probs (used
        for the KL term), wrapped in ``torch.no_grad``.

        Implementation notes
        --------------------
        * Concatenate prompt + sampled into one sequence; logits at
          position ``i`` predict token ``i+1``. We slice to align with
          the ``T`` sampled tokens only.
        * Compute log-softmax in float32 for numerical stability even
          when the model runs in bf16.
        * ``ref=True`` requires the model to be PEFT-wrapped; if it's a
          plain HF model (no LoRA), there's no "policy vs ref" distinction
          and we just run the same forward under no_grad.
        """
        import torch
        if self._dummy or self._model is None:
            return torch.zeros(0)

        full = torch.cat([prompt_ids, sampled_ids], dim=0).unsqueeze(0)
        full = full.to(self._model.device)
        T = sampled_ids.shape[0]
        # Decide adapter context.
        if ref and hasattr(self._model, "disable_adapter"):
            ctx = self._model.disable_adapter()
            grad_ctx = torch.no_grad()
        elif ref:
            # Plain HF model: ref == policy; just no_grad.
            ctx = _NullCtx()
            grad_ctx = torch.no_grad()
        else:
            ctx = _NullCtx()
            grad_ctx = _NullCtx()      # caller controls grad via outer enable_grad / no_grad

        with ctx, grad_ctx:
            out = self._model(input_ids=full)
            logits = out.logits[0]                              # (P+T, V)
            # Predicting token at position i+1 from logits at position i.
            shift = logits[-(T + 1):-1, :]                      # (T, V)
            shift = shift.float()
            log_probs = torch.log_softmax(shift, dim=-1)
            tgt = sampled_ids.to(self._model.device).long()
            tok_logp = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # (T,)
        return tok_logp

    # ------------------------------------------------------------------
    # Real model: forward with one video as multimodal input.
    # ------------------------------------------------------------------
    def _generate_with_video(self, prompt: str, video_path: str,
                             *, force_greedy: bool = False,
                             max_new_tokens: int | None = None) -> str:
        import torch
        # Qwen2.5-VL expects the chat template to declare a "video" content item.
        msgs = [{
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path),
                 "fps": self.cfg.fps, "max_pixels": 360 * 420},
                {"type": "text",  "text": prompt},
            ],
        }]
        chat = self._processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        # qwen-vl-utils handles the actual frame extraction.
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
            image_inputs, video_inputs = process_vision_info(msgs)
        except ImportError as e:                                  # pragma: no cover
            raise RuntimeError(
                "qwen_vl_utils not installed — pip install qwen-vl-utils"
            ) from e
        inputs = self._processor(
            text=[chat],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)
        budget = max_new_tokens if max_new_tokens is not None else self.cfg.max_new_tokens
        with torch.no_grad():
            if force_greedy:
                out_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=budget,
                    do_sample=False,
                )
            else:
                out_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=budget,
                    do_sample=True,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                )
        gen = out_ids[0, inputs["input_ids"].shape[1]:]
        return self._processor.tokenizer.decode(gen, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Soft-logits score path (mirror of stage0 D5 score_video_multi_theme_softlogits).
    # ------------------------------------------------------------------
    def _score_video_softlogits(self, prompt: str, video_path: str,
                                n_themes: int, *,
                                force_greedy: bool = True
                                ) -> tuple[str, list[float | None]]:
        """One forward pass, ``output_scores=True``, soft-logits over {1..5}.

        Algorithm (cf. stage0 ``score_video_multi_theme_softlogits``):
          1. Generate up to score_max_new_tokens with the plain-text
             scaffold prompt, capturing per-step logits.
          2. Walk the generated ids; every digit token whose previous
             token decodes to ":<...>" (colon-anchor) is a hit.
          3. At each hit position, take the logits over the union of all
             {1..5} digit-id variants, softmax → expected value E[k].
          4. Return the first ``n_themes`` hits in [1.0, 5.0] (None
             padded for missing positions).
        """
        import torch
        from qwen_vl_utils import process_vision_info  # type: ignore

        msgs = [{
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path),
                 "fps": self.cfg.fps, "max_pixels": 360 * 420},
                {"type": "text",  "text": prompt},
            ],
        }]
        chat = self._processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(msgs)
        inputs = self._processor(
            text=[chat],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)

        budget = self.cfg.score_max_new_tokens
        with torch.no_grad():
            gen_out = self._model.generate(
                **inputs,
                max_new_tokens=budget,
                do_sample=not force_greedy,
                temperature=(self.cfg.temperature if not force_greedy else 1.0),
                top_p=(self.cfg.top_p if not force_greedy else 1.0),
                return_dict_in_generate=True,
                output_scores=True,
            )
        trimmed = gen_out.sequences[:, inputs["input_ids"].shape[1]:]
        gen_ids = trimmed[0].tolist()
        step_scores = gen_out.scores or ()
        text = self._processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Build flat-id tensors once.
        groups = self._score_digit_id_groups
        flat_ids:    list[int] = []
        flat_values: list[int] = []
        for k_idx, grp in enumerate(groups):
            for tid in grp:
                flat_ids.append(tid)
                flat_values.append(k_idx + 1)
        score_ids_set = set(self._score_id_to_value.keys())
        colon_ids     = self._score_colon_ids
        tok           = self._processor.tokenizer

        def _ends_with_colon(tid: int) -> bool:
            if tid in colon_ids:
                return True
            try:
                s = tok.decode([tid])
            except Exception:
                return False
            return bool(s) and s.rstrip(" ").endswith(":")

        # Walk for colon-anchored digit hits.
        T = len(gen_ids)
        hits: list[int] = []
        for pos in range(T):
            tid = gen_ids[pos]
            if tid not in score_ids_set:
                continue
            anchor_ok = False
            for back in (1, 2):
                j = pos - back
                if j < 0:
                    break
                prev_tid = gen_ids[j]
                if _ends_with_colon(prev_tid):
                    anchor_ok = True
                    break
                if back == 1:
                    try:
                        if tok.decode([prev_tid]).strip() == "":
                            continue
                    except Exception:
                        pass
                break
            if not anchor_ok:
                continue
            hits.append(pos)
            if len(hits) >= n_themes:
                break

        if not flat_ids or not step_scores:
            return text, [None] * n_themes

        device = step_scores[0].device
        flat_ids_t    = torch.tensor(flat_ids,    device=device, dtype=torch.long)
        flat_values_t = torch.tensor(flat_values, device=device, dtype=torch.float32)

        scores: list[float | None] = []
        for k in range(n_themes):
            if k >= len(hits):
                scores.append(None); continue
            pos = hits[k]
            if pos >= len(step_scores):
                scores.append(None); continue
            logits = step_scores[pos][0]                       # [V]
            sel    = logits.index_select(0, flat_ids_t).float()
            probs  = torch.softmax(sel, dim=-1)
            ev     = float((probs * flat_values_t).sum().item())
            ev = max(1.0, min(5.0, ev))
            scores.append(ev)
        return text, scores

    # ------------------------------------------------------------------
    # Dummy emissions — deterministic enough for tests, varied enough
    # to exercise the parser/reward path.
    # ------------------------------------------------------------------
    _DUMMY_PATHS = [
        "Theme: Visual Quality and Temporal Consistency",
        "Theme: Character Fidelity and Appearance",
        "Theme: Environment, Lighting, and Cinematic Style",
        "Theme: Action Plausibility and Character Dynamics",
        "Theme: Camera Motion, Framing, and Narrative Flow",
    ]

    def _dummy_rubric(self, query: str) -> str:
        # Stable per-query rubric: hash query into a small permutation
        # so identical queries → identical rubrics, but different
        # queries → slightly different ordering / weights.
        rng = random.Random(abs(hash(query)) % (2**31))
        paths = list(self._DUMMY_PATHS)
        rng.shuffle(paths)
        # Random-but-summing-to-1 weights.
        ws = [rng.uniform(0.5, 1.5) for _ in paths]
        s  = sum(ws)
        ws = [round(w / s, 4) for w in ws]
        # Fix rounding drift on the last weight.
        ws[-1] = round(1.0 - sum(ws[:-1]), 4)
        rubrics = [
            {"path": p, "weight": w, "tips": ["dummy tip"]}
            for p, w in zip(paths, ws)
        ]
        return json.dumps({"rubrics": rubrics}, ensure_ascii=False)

    def _dummy_score(self, rubric_text: str, video_path: str,
                     n_themes: int = 5) -> tuple[str, list[float | None]]:
        """Plain-text scaffold + soft-logits-shaped scores for dummy mode.

        Mirrors the SCORE_PROMPT_TMPL output format, so r_format and the
        scaffold parser exercise the same code as real mode. The values
        are deterministic per (video_path) with a small per-theme jitter
        so GRPO advantage stays non-degenerate within the G group.
        """
        rng = random.Random(abs(hash(video_path)) % (2**31))
        # win/lose bias: keeps preference signal in the smoke run.
        bias = 0.0
        vp = (video_path or "").lower()
        if "win" in vp or vp.endswith("_a.mp4"):
            bias = 1.0
        elif "lose" in vp or vp.endswith("_b.mp4"):
            bias = -0.6
        scores: list[float] = []
        for _ in range(n_themes):
            v = 3.0 + bias + rng.uniform(-0.4, 0.4)
            v = max(1.0, min(5.0, round(v, 3)))
            scores.append(v)
        # Plain-text scaffold matching SCORE_PROMPT_TMPL format_line.
        # Round to nearest int when text-emitting (real model emits a digit;
        # the soft-logits floats live OUTSIDE the text path).
        text = " ".join(
            f"theme_{k+1}: {int(round(scores[k]))}"
            for k in range(n_themes)
        )
        return text, scores
