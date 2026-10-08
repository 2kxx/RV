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

import contextlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

__all__ = ["StudentConfig", "StudentWrapper", "RubricRollout", "DimensionContext",
           "BadVideoError"]


class BadVideoError(RuntimeError):
    """Raised when ``process_vision_info`` cannot decode a video.

    Caveat — SIGSEGV is uncatchable
    -------------------------------
    decord / ffmpeg can SEGFAULT inside a frame decode (we've seen this
    on a handful of newly-extended pacing/rhythm/action/expression
    pairs). A native segfault kills the worker process before any Python
    handler runs — try/except cannot intercept it. The ONLY way to
    avoid those is to scan offline and remove the offending video pairs
    from the dataset (see ``scripts/scan_bad_videos.py``).

    What this exception DOES catch
    ------------------------------
    The cases where ffmpeg/decord raise a Python exception instead of
    crashing — corrupt header, truncated file, codec it doesn't support,
    ``OSError`` from a partial cache. The trainer's step loop catches
    this and skips the affected rollout / sample so one bad video can't
    break a whole step's batch.
    """



class _NullCtx:
    """Trivial context manager used as a fall-through for ``with`` blocks."""
    def __enter__(self): return None
    def __exit__(self, *exc): return False


def _under_accelerate_launch() -> bool:
    """Detect whether the process was started by ``accelerate launch``.

    Used to suppress ``device_map="auto"`` in `from_pretrained` — under
    accelerate the launcher is responsible for placement, and passing
    ``device_map`` confuses the DeepSpeed wrapping.
    """
    return any(k in os.environ for k in
               ("ACCELERATE_USE_DEEPSPEED", "LOCAL_RANK",
                "TORCHELASTIC_RUN_ID"))


def _param_full_numel(p) -> int:
    """Return the *unpartitioned* element count of an nn.Parameter.

    Under DeepSpeed ZeRO-3 with ``zero3_init_flag: true``, freshly-loaded
    params are placeholder tensors with ``p.numel() == 0`` -- the real
    storage lives in a sharded buffer. The true full-tensor size is
    exposed via the ``ds_numel`` attribute that ZeRO-3 stamps on the
    parameter.

    Use this helper anywhere we want to print a "real" parameter count
    without first having to gather the params, so the trainer can log
    sane numbers in both ZeRO-3 and single-process modes.
    """
    n = getattr(p, "ds_numel", None)
    if isinstance(n, int) and n > 0:
        return n
    return p.numel()


def _normalize_mm_kwargs(mm_kwargs, *, target_device, batch_dim_add: bool = True) -> dict:
    """Build a dict of multimodal kwargs to pass into a Qwen2.5-VL forward.

    Used by the grad-on re-forward path (compute_log_probs and
    compute_digit_logits) so the vision tower is invoked instead of the
    language-only fallback. See Patch 6 (2026-06-16) for the full
    rationale: without these tensors the re-forward computes the
    LANGUAGE-ONLY marginal of π_θ rather than the multimodal π_θ that
    produced the sample, biasing the GRPO ratio and making BT-aux
    vision-blind.

    Parameters
    ----------
    mm_kwargs : dict | None
        Per-row mm tensors saved on RubricRollout. Expected keys:
        - "pixel_values_videos" : Tensor of shape (block_len, ...) for
          ONE video (no batch dim).
        - "video_grid_thw"      : Tensor of shape (1, 3) (already has
          batch dim).
        - "second_per_grid_ts"  : list[float] | None.
        Image-only mm rollouts could pass "pixel_values" + "image_grid_thw"
        instead; this function passes those through verbatim.
    target_device : torch.device
        Device of the receiving model. Tensors are moved here.
    batch_dim_add : bool
        If True (default), unsqueeze a leading batch dim of 1 on
        ``pixel_values_videos`` since a row=1 forward expects (B=1, ...).
        Qwen2.5-VL's processor outputs pixel_values_videos with batch
        already CONCATENATED (not stacked), so saving per-row gives a
        (block_len, ...) tensor; the model's forward expects this shape
        directly without an extra leading batch dim. Empirically the
        forward accepts the concatenated shape — we set this False by
        default after testing on the actual processor output. Kept as a
        flag for future-proofing if HF changes the convention.
    """
    if not mm_kwargs:
        return {}
    out: dict = {}
    pvv = mm_kwargs.get("pixel_values_videos")
    if pvv is not None:
        # Move to device. Don't unsqueeze a batch dim — Qwen2.5-VL
        # expects pixel_values_videos as a flat concatenation along
        # the first dim, indexed by video_grid_thw.
        out["pixel_values_videos"] = pvv.to(target_device)
    vgt = mm_kwargs.get("video_grid_thw")
    if vgt is not None:
        # Already has shape (1, 3) per row.
        out["video_grid_thw"] = vgt.to(target_device)
    spgt = mm_kwargs.get("second_per_grid_ts")
    if spgt is not None:
        # Forward expects a list[float] (one per video) or a 1-D Tensor.
        out["second_per_grid_ts"] = spgt
    # Image-only mm (rare in our pipeline but supported for safety).
    pvi = mm_kwargs.get("pixel_values")
    if pvi is not None:
        out["pixel_values"] = pvi.to(target_device)
    igt = mm_kwargs.get("image_grid_thw")
    if igt is not None:
        out["image_grid_thw"] = igt.to(target_device)
    return out


def _is_zero3_wrapped(model) -> bool:
    """Return True iff ``model`` has DeepSpeed ZeRO-3 partitioned parameters.

    ZeRO-3 stamps every nn.Parameter with a ``ds_id`` attribute and a
    ``ds_status`` enum (NOT_AVAILABLE / AVAILABLE / INFLIGHT). Checking
    one parameter is enough — partitioning is all-or-nothing per param
    group, and the model wrapper itself (DeepSpeedEngine) doesn't have a
    cleanly importable type to isinstance-check across versions.
    """
    if model is None:
        return False
    # Unwrap a possible DeepSpeedEngine: ``engine.module`` is the user's
    # original nn.Module. params are stamped on that side.
    inner = getattr(model, "module", model)
    try:
        for p in inner.parameters():
            # Only look at the first param — ZeRO-3 stamps all or none.
            return hasattr(p, "ds_id")
    except Exception:
        return False
    return False


def _summon_full_params_ctx(model, *, writeback: bool = False):
    """Context manager that gathers ZeRO-3 sharded params **and disables
    DeepSpeed's per-submodule forward hooks** for the duration.

    Wraps ``deepspeed.zero.GatheredParameters`` for the no-grad sampling
    path (writeback=False). For backward (writeback=True) we fall through
    to a no-op and let DeepSpeedEngine's own per-layer pre/post-forward
    hooks gather/scatter, which is the supported way to drive backward
    under ZeRO-3.

    No-op when not ZeRO-3 wrapped (single-process / dummy / ZeRO-2).

    Why we need this
    ----------------
    Two embedding-layer bugs hit Qwen2.5-VL forward in transformers 4.5x
    when params are sharded:

    1. ``model.generate()`` autoregressive sampling fails with
       ``RuntimeError: 'weight' must be 2-D`` from the embedding layer —
       sharded backends store params as 1-D flat views and the inner
       generate loop bypasses the wrapper's pre-forward hook that would
       un-flatten them.
    2. The teacher-forcing forward inside ``compute_log_probs`` (policy
       and ref both) hits the same bug.

    The standard fix in PPO/GRPO/RLHF is to issue an explicit gather
    around the whole forward, so every layer sees the original
    (un-sharded, 2-D) weight tensor.

    Why we ALSO disable hooks
    -------------------------
    Once we have the full params resident on every rank, DeepSpeed's
    ``_pre_forward_module_hook`` (registered by ``parameter_offload``)
    will STILL fire on every submodule and call
    ``param_coordinator.fetch_sub_module(...) → all_gather``. With the
    params already AVAILABLE on every rank that's pure waste — but worse
    than that, on long ``generate()`` rollouts those redundant fetches
    allocate transient partition buffers that fragment the caching
    allocator and eventually OOM (we hit this around step 13 with
    ``RM_SCORE_MB=8``: peak ~64 GB used, 1 GB partition buffer alloc
    failed inside ``fetch_sub_module``).

    The fix is what ``accelerate.utils.unwrap_model_for_generation``
    does for ZeRO-3 RLHF: gather + disable_hooks for the generate scope,
    then restore on exit. This is the same pattern TRL / OpenRLHF / verl
    all use. Verified against accelerate 1.13 + deepspeed 0.19.

    DP safety
    ---------
    Under DeepSpeed+DP each rank processes a different instance, but
    ``GatheredParameters`` is a NCCL collective. Won't this desync?
    No — NCCL matches collectives by call ORDER on each rank, not by
    wall-clock time. Every rank executes the same code path
    (rollout → policy_update), so collectives line up; fast ranks just
    block at the next one waiting for slow ranks. The whole
    ``model.generate`` autoregressive loop runs WITHIN one gather scope
    on each rank without issuing further collectives, so inter-rank
    generate-time variance does NOT cause issues. Disabling hooks does
    not change this — the gather is still issued in lock-step on entry
    and the partition is still issued in lock-step on exit.
    """
    if not _is_zero3_wrapped(model):
        return _NullCtx()
    if writeback:
        # Backward path is engine-managed under ZeRO-3.
        return _NullCtx()
    try:
        import deepspeed                                        # noqa: F401
        from deepspeed.runtime.zero.partition_parameters import GatheredParameters
    except ImportError:                                         # pragma: no cover
        return _NullCtx()
    inner = getattr(model, "module", model)
    params = [p for p in inner.parameters() if hasattr(p, "ds_id")]
    if not params:
        return _NullCtx()

    @contextlib.contextmanager
    def _gather_and_disable_hooks():
        # 1) Gather every shard onto every rank (modifier_rank=None →
        #    read-only, no broadcast on exit).
        gather = GatheredParameters(params, modifier_rank=None, fwd_module=inner)
        # 2) Locate the parameter_offload object that owns the per-
        #    submodule forward hooks. The attribute path has churned
        #    across DS versions; we probe the known names and degrade
        #    gracefully if none match (then we get the old "double
        #    gather" behaviour but still correct results).
        offload, restore = _resolve_ds_hook_handle(model)

        gather.__enter__()
        try:
            if offload is not None:
                try:
                    offload.disable_hooks()
                except Exception as e:                      # pragma: no cover
                    print(f"[student] disable_hooks failed ({e}); "
                          f"falling back to double-gather path.",
                          file=sys.stderr)
                    offload = None
            try:
                yield
            finally:
                if offload is not None and restore:
                    try:
                        offload.enable_hooks()
                    except Exception as e:                  # pragma: no cover
                        print(f"[student] enable_hooks (restore) "
                              f"failed: {e}", file=sys.stderr)
        finally:
            gather.__exit__(None, None, None)

    return _gather_and_disable_hooks()


def _resolve_ds_hook_handle(model) -> tuple[object | None, bool]:
    """Find the DeepSpeed object exposing ``disable_hooks/enable_hooks``.

    Returns ``(handle, was_enabled)``. ``handle`` is None if we can't
    locate it on this DeepSpeed version — caller falls back to the
    pure-gather path.

    Across DS 0.13 → 0.19 the hook owner has lived in different places:

      * ``engine.optimizer.parameter_offload``  (ZeRO-3 with offloading)
      * ``engine.parameter_offload``            (some intermediate DS)
      * direct attribute on the engine itself
      * methods grafted onto the user's nn.Module (older DS)

    We probe in that order and return the first that has both methods.
    ``was_enabled`` is best-effort: if we can read the engine's
    ``_zero3_hooks_enabled`` flag we use it, otherwise assume True (the
    default) so we try to restore on exit.
    """
    # ``model`` here is whatever the trainer passed in — typically a
    # DeepSpeedEngine (has .module pointing at the user nn.Module).
    candidates = []
    opt = getattr(model, "optimizer", None)
    if opt is not None:
        candidates.append(getattr(opt, "parameter_offload", None))
    candidates.append(getattr(model, "parameter_offload", None))
    candidates.append(model)
    inner = getattr(model, "module", None)
    if inner is not None:
        candidates.append(inner)

    for c in candidates:
        if c is None:
            continue
        if callable(getattr(c, "disable_hooks", None)) and \
           callable(getattr(c, "enable_hooks",  None)):
            was = bool(getattr(model, "_zero3_hooks_enabled", True))
            return c, was
    return None, False


class _PauseGradCheckpointing:
    """Context manager: temporarily disable gradient_checkpointing during ``model.generate()``.

    Why
    ---
    Under ZeRO-3 + ``gradient_checkpointing_enable``,
    the autoregressive ``generate()`` loop on Qwen2.5-VL (transformers 4.5x)
    can produce NaN/Inf logits, which then crash ``torch.multinomial`` with::

        RuntimeError: probability tensor contains either `inf`, `nan`
        or element < 0

    Root cause: grad-ckpt installs forward hooks that re-run a block's
    forward inside ``no_grad`` and, on certain transformers/torch combos,
    those hooks interact badly with the freshly-gathered ZeRO-3 params --
    the second forward sees a stale view of the weights and emits
    garbage logits.

    Generation is ``no_grad`` anyway, so disabling grad-ckpt for the
    duration of ``generate()`` costs nothing on memory (no activations to
    save) and removes the buggy interaction. We restore grad-ckpt on
    exit so the subsequent ``compute_log_probs`` backward path keeps
    its activation memory savings.

    Idempotent: if grad-ckpt isn't enabled, this is a no-op.
    """

    def __init__(self, model, *, use_reentrant: bool = False):
        self._model = model
        self._use_reentrant = use_reentrant
        self._was_enabled = False

    def __enter__(self):
        m = self._model
        if m is None:
            return self
        # ``is_gradient_checkpointing`` is the canonical HF flag; some
        # PEFT-wrapped paths only expose ``gradient_checkpointing`` on
        # the inner module. Check both.
        was = bool(getattr(m, "is_gradient_checkpointing", False))
        if not was:
            inner = getattr(m, "base_model", None)
            if inner is not None:
                was = bool(getattr(inner, "is_gradient_checkpointing", False))
        self._was_enabled = was
        if was:
            try:
                m.gradient_checkpointing_disable()
            except Exception as e:                            # pragma: no cover
                print(f"[student] gradient_checkpointing_disable failed: {e}",
                      file=sys.stderr)
                self._was_enabled = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._was_enabled or self._model is None:
            return False
        m = self._model
        try:
            m.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": bool(self._use_reentrant),
                },
            )
        except TypeError:
            try:
                m.gradient_checkpointing_enable()
            except Exception as e:                            # pragma: no cover
                print(f"[student] gradient_checkpointing_enable (restore) failed: {e}",
                      file=sys.stderr)
        except Exception as e:                                # pragma: no cover
            print(f"[student] gradient_checkpointing_enable (restore) failed: {e}",
                  file=sys.stderr)
        return False


def _pause_grad_ckpt(model, *, use_reentrant: bool = False):
    """Public alias for ``_PauseGradCheckpointing``."""
    return _PauseGradCheckpointing(model, use_reentrant=use_reentrant)


@dataclass
class DimensionContext:
    """Per-dimension prompt scaffolding consumed by RUBRIC/SCORE templates.

    All fields are plain strings so the wrapper doesn't need to import
    ``stage1_teacher.task_descriptions`` (avoids cross-module dependency
    at import time). The trainer is responsible for materialising one
    ``DimensionContext`` per dim from its loaded ``TaskSpec`` map.

    Two flavors of task description (2026-06-13 split)
    --------------------------------------------------
    * ``task_description``        — long, used by the RUBRIC-gen prompt.
      Contains the EvalVerse hierarchy header, sub-aspect IDs, anti-drift
      instructions ("ground itself in one or more"), Audio block (where
      applicable), etc. The student NEEDS this context to write 5 rubrics
      that cover the right facets without bleeding into neighbouring dims.

    * ``task_description_score``  — short, used by the SCORE prompt.
      One-line "Judge each rubric on the {dim} dimension". The score
      forward only needs to know which dimension the video is being
      judged on; the rubric_text already carries the per-rubric criterion
      and tips. Stuffing the long rubric-gen task_description into the
      score prompt was empirically harmful: prompt-diff (probe v8 vs
      train) showed train-time score saw a ~14-line task_description
      block including instructions like "should ground itself in one or
      more of these" and EvalVerse IDs ("Visual_Concept_Design-Character-
      Costume_Rationality"); these are rubric-gen-targeted instructions
      that polluted the autoregressive prior at score time.
    """
    dimension:        str = "video_quality"
    evalverse_path:   str = ""        # e.g. "Acting > Action"
    task_description: str = ""        # long; consumed by RUBRIC_PROMPT_TMPL
    task_description_score: str = ""  # short; consumed by SCORE_PROMPT_TMPL
    sub_aspects:      list[str] = field(default_factory=list)

    def sub_aspects_block(self) -> str:
        """Bulleted list, or a one-line stand-in when sub_aspects is empty."""
        if not self.sub_aspects:
            return ("- (free-form: cover the most discriminative facets of "
                    "this dimension; do not invent a fixed taxonomy)")
        return "\n".join(f"- {s}" for s in self.sub_aspects)

    def score_task_description(self) -> str:
        """Resolve the short, score-side task description.

        Falls back to a generic single-line sentence if
        ``task_description_score`` was not set (e.g. older callers, or
        ``DimensionContext.generic`` from before this split). This keeps
        score prompts lean regardless of how the ctx was constructed.
        """
        if self.task_description_score:
            return self.task_description_score.strip()
        return (
            f"Judge the generated video on the **{self.dimension or 'video_quality'}** "
            f"dimension. Be concrete and observable."
        )

    @classmethod
    def generic(cls, dimension: str = "video_quality") -> "DimensionContext":
        """Fallback when the trainer can't find a TaskSpec for ``dimension``.

        Renders a minimal scaffold: just a generic "judge this dimension"
        sentence. Lets the wrapper run end-to-end on dummy / smoke data
        without forcing every caller to load yamls.
        """
        dim = dimension or "video_quality"
        return cls(
            dimension=dim,
            evalverse_path="",
            task_description=(
                f"Evaluate the **{dim}** dimension of "
                f"the generated video. Be concrete and observable."
            ),
            task_description_score=(
                f"Judge the generated video on the **{dim}** dimension. "
                f"Be concrete and observable."
            ),
            sub_aspects=[],
        )


# ---------------------------------------------------------------------------
# Default prompts (kept at module scope for easy override / inspection).
# ---------------------------------------------------------------------------
RUBRIC_PROMPT_TMPL = """You are a cinematic video evaluation expert. Generate **exactly 5** rubrics for scoring videos on the **{dimension}** dimension ({evalverse_path}).

## Task
{task_description}

## Reference: EvalVerse sub-aspect taxonomy (BACKGROUND ONLY)
This dimension is broken into the following internal sub-aspect IDs.
Use them as a check-list to make sure your 5 rubrics collectively cover
the relevant facets, but **do NOT copy these IDs into your output**:
{sub_aspects_block}

## Rubric format
Each rubric has three fields:
- **theme**: a concise human-readable name for the criterion (e.g.
  "Sharpness and Temporal Stability"). NEVER use EvalVerse-style IDs
  like "Aesthetics-Lighting-Lighting_Logic".
- **weight**: a float in (0, 1). The 5 weights MUST sum to 1.0.
- **tips**: ≥ 2 specific, observable guidance bullets (e.g. "Penalize
  flicker on faces during turning", not "looks better").

## Requirements
- Exactly 5 rubrics; the 5 themes MUST be distinct (no near-duplicates).
- Each rubric grounds in the **{dimension}** dimension and stays there;
  do NOT drift into neighbouring dimensions.
- Tips are concrete and observable from a single video clip alone.
- Tips name failure modes the model can detect ("watermark visible",
  "limb morphs during throw") rather than vague preferences.
- Output rubric text only; do not score the current video.

## Caption (the prompt that drove video generation)
{caption}

## Output (STRICT JSON, no markdown fences, no commentary)
{{
  "rubrics": [
    {{"theme": "<concise human-readable theme>", "weight": <float in (0,1)>, "tips": ["<tip1>", "<tip2>"]}},
    {{"theme": "...", "weight": ..., "tips": [...]}},
    {{"theme": "...", "weight": ..., "tips": [...]}},
    {{"theme": "...", "weight": ..., "tips": [...]}},
    {{"theme": "...", "weight": ..., "tips": [...]}}
  ]
}}

### Example (visual_quality)
{{
  "rubrics": [
    {{"theme": "Sharpness and Structural Integrity", "weight": 0.25,
      "tips": ["Check edge clarity on faces and props.",
               "Penalize blur, color bleeding, or pixelation."]}},
    {{"theme": "Temporal Stability of Subjects", "weight": 0.20,
      "tips": ["Track main subjects across frames; identity must not morph.",
               "Background landmarks should not duplicate or vanish."]}},
    {{"theme": "Anatomy and Object Coherence under Motion", "weight": 0.20,
      "tips": ["Limbs and fingers should not melt or warp during action.",
               "Held props must keep rigid geometry."]}},
    {{"theme": "Visual Effects Rendering Quality", "weight": 0.15,
      "tips": ["Glows, particles, weather should integrate cleanly.",
               "No bleed onto unintended subjects."]}},
    {{"theme": "Overall Frame Cleanness", "weight": 0.20,
      "tips": ["No watermarks, captions, or letterbox artifacts.",
               "Aspect ratio and crop respect the requested framing."]}}
  ]
}}
"""

# Plain-text score scaffold (matches stage0 D5 + soft_logits aggregation).
# Format is deliberately deterministic — every "theme_k:" is followed by
# exactly one digit token, which lets the soft-logits path read per-position
# logits over {1..5} ids without re-tokenisation surprises.
#
# Prompt-boundary anchoring (2026-06-12 fix)
# ------------------------------------------
# Earlier versions ended the prompt with the rubric blob:
#
#     ## Rubric (DO NOT change themes or weights)
#     {rubric_text}
#
# Under SAMPLING (do_sample=True, score_greedy=False) the model frequently
# treated the trailing rubric_text as "an in-progress JSON document" and
# auto-regressed into:
#   * echoing rubric tips ("theme interior."] ⏎ "Verify for any...")
#   * 1-token EOS ("theme")
#   * truncated scaffolds ("theme_1: 3 theme_2: 5 theme_3")
# all of which evade the colon-anchored digit walker (see
# ``_score_video_softlogits_batch``) and report no_anchor.
#
# probe_v8 variant=A (8-card ZeRO-3, sampling): 33% finite, 28/64 no_anchor
# probe_v8 variant=K (gold rubric instead): 100% finite
# probe_v4 (single-card, GREEDY): 100% finite — greedy hides the
#   ambiguity because EOS is never argmax after "theme_k:".
#
# Fix: after the rubric blob, repeat the scaffold marker + a fresh
# "Output:" header so the LAST tokens the model sees are scaffold-shaped,
# not rubric-shaped. The autoregressive prior is then anchored on
# "theme_1: <digit>" rather than on "tips": [...]". Costs ~30 prompt
# tokens; works under both greedy and sampling.
SCORE_PROMPT_TMPL = """You are a strict video reward judge for the **{dimension}** dimension ({evalverse_path}).

## Task
{task_description}

Score the SINGLE video on each rubric on a 1-5 integer scale (1=very poor,
2=poor, 3=fair, 4=good, 5=excellent).

Respond with EXACTLY this one-line format. Each "theme_k:" prefix MUST
appear in your output verbatim, followed by a single space and a single
digit 1-5. Do NOT abbreviate "theme_k" away (e.g. "4:3:4:4:4" is WRONG —
the "theme_k" tokens are required). No commas, no extra words, no
explanations, no markdown, no leading prose:

{format_line}

### Example (5 themes)
theme_1: 4 theme_2: 3 theme_3: 5 theme_4: 4 theme_5: 4

## Caption (the prompt that drove video generation)
{caption}

## Rubric (reference only — DO NOT echo, do NOT continue this JSON)
{rubric_text}

## Output (one line, exactly the format below; nothing else)
{format_line}
Answer:"""


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
    # ── LoRA. None → no LoRA (full fine-tune; needs ZeRO-3 to fit
    # 7B params + AdamW state on 8×80GB cards). When set, ``_ensure_model``
    # wraps the base model via peft.get_peft_model() with the yaml's
    # settings.
    lora_config_path:  str | None = None
    # ── Attention impl. "sdpa" (PyTorch 2.6 builtin flash kernel — no
    # extra deps) is the default. "flash_attention_2" needs the
    # ``flash-attn`` pip package (pip install flash-attn --no-build-isolation)
    # but is ~10-20% faster on H100. "eager" disables fused kernels (slow,
    # mostly for debugging numeric mismatches).
    attn_implementation: str = "sdpa"
    # ── Reference policy for KL. Two paths:
    #
    #   (a) LoRA mode + ref_model_path is None:
    #       compute_log_probs(ref=True) calls model.disable_adapter() so
    #       the same weights serve as both policy (LoRA on) and reference
    #       (LoRA off / base). Free.
    #
    #   (b) Full-param mode (lora_config_path is None) + ref_model_path
    #       set: load a SECOND frozen Qwen2.5-VL from the same weights
    #       directory. Adds ~14 GB per card (full-replicated on every rank). Pass
    #       ref_model_path=cfg.model_path for the most common case (KL
    #       against the base instruct model).
    #
    #   (c) Full-param mode without ref_model_path: the trainer should
    #       set beta_kl=0 — no second model is loaded and KL is silently
    #       skipped. (compute_log_probs(ref=True) returns the same
    #       no-grad forward as the policy in this case, so KL≈0 by
    #       construction.)
    ref_model_path:    str | None = None
    # ── ZeRO-3 partitioning of the ref model (added 2026-06-16, Patch 7).
    #
    # Default (False): ref is loaded full-replicated on every rank, ~14GB
    # / card. Old default for memory simplicity; matches the comment block
    # above option (b).
    #
    # When True: ref participates in DeepSpeed ZeRO-3 partitioning at
    # construction time, ~1.75 GB / card on 8 GPUs. The forward path
    # uses ``_summon_full_params_ctx(ref)`` to gather params on demand,
    # mirroring exactly what the policy forward does. This is the
    # KEY FIX for the "kl_s ≈ 4.6 nats at step 0 when student==ref"
    # bug: the asymmetry came from policy going through ZeRO-3 hooks
    # (DeepSpeedEngine pre/post-forward gather/scatter, fused kernel
    # selection on sharded weights, ...) while ref took the unsharded
    # plain-HF path. With ref also ZeRO-3-partitioned, both forwards go
    # through the same code path → KL_s ≈ 0 by construction at step 0.
    #
    # Memory budget on 8×H100 80GB (full-param 7B):
    #     policy sharded weight     :  1.75 GB / card
    #     policy optimizer (fp32×2) :  3.50 GB / card
    #     policy gradient (bf16)    :  1.75 GB / card
    #     ref sharded weight        :  1.75 GB / card  (was 14 GB!)
    #     activation peak           : 10-20 GB / card
    #     ───────────────────────────────────────────
    #     total                     : ~18-30 GB / card  (was ~30-44 GB)
    # i.e. SAVE ~12 GB / card while fixing kl_s. Pareto improvement.
    ref_zero3:         bool = False
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
    # ── Gradient checkpointing. Trades ~30% wall-time for ~30-50% activation
    # memory. When ``use_gradient_checkpointing`` is True, ``_ensure_model``
    # calls ``model.gradient_checkpointing_enable`` and
    # ``model.enable_input_require_grads`` (the latter is REQUIRED with
    # LoRA — without it the input embedding's grad is None and the LoRA
    # update silently no-ops). ``use_reentrant=False`` is recommended for
    # LoRA + multimodal forwards; reentrant=True works but is 2-3x slower
    # on backward and incompatible with custom autograd.Function nodes.
    use_gradient_checkpointing: bool = True
    grad_checkpoint_use_reentrant: bool = False
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
    # ── BT-aux (added 2026-06-16): bookkeeping for the supervised BT loss.
    # ``model.generate`` runs under ``torch.no_grad`` (line ~2264), so the
    # logits emitted during sampling are detached from autograd and cannot
    # be used for backprop. Instead we save the TOKEN OFFSETS at which the
    # colon-anchor walk recovered a digit hit; the trainer then re-runs a
    # grad-on forward over the same (prompt_ids, sampled_ids) window.
    digit_token_offsets: "object | None" = None    # list[int] or None
    digit_mask:          "object | None" = None    # torch.BoolTensor or None
    # ── Multi-modal tensors (added 2026-06-16, Patch 6a). Score forward
    # is multimodal — its prompt contains video placeholder tokens that
    # are replaced by Qwen2.5-VL's vision tower output during sampling.
    # If we re-run compute_log_probs / compute_digit_logits without
    # ``pixel_values_videos`` and ``video_grid_thw``, the vision tower
    # is NOT invoked, the placeholder tokens fall back to plain LM
    # embeddings, and the resulting logp is the LANGUAGE-ONLY marginal
    # of π_θ — not the multimodal π_θ that produced the sample. That
    # makes both (a) the GRPO score-PG ratio biased and (b) the BT-aux
    # supervision use vision-blind logits. We save here the per-row
    # slice of mm tensors so any grad-on re-forward can see the same
    # frames the sampler did. Per-row means batch dim is squeezed out;
    # the trainer adds it back on a single-row forward.
    #   pixel_values_videos : (T_frames * Hp * Wp, C)  or whatever shape
    #                         the processor returns for ONE video.
    #   video_grid_thw      : (1, 3) [t, h, w]  per-row.
    #   second_per_grid_ts  : list[float] | None  positional time stamps
    #                         (Qwen2.5-VL temporal RoPE). Optional —
    #                         present in newer processor versions.
    pixel_values_videos: "object | None" = None    # torch.Tensor or None
    video_grid_thw:      "object | None" = None    # torch.Tensor or None
    second_per_grid_ts:  "object | None" = None    # list[float] or None


class StudentWrapper:
    """Lazy-loaded Qwen2.5-VL-Instruct interface."""

    def __init__(self, cfg: StudentConfig):
        self.cfg = cfg
        self._dummy = bool(cfg.dummy) or os.environ.get("RM_DUMMY", "0") == "1"
        self._model = None
        self._ref_model = None        # set by _load_ref_model in full-param mode
        self._is_full_param = False   # set by _ensure_model
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
        print(f"[student] loading {self.cfg.model_path} dtype={self.cfg.torch_dtype} "
              f"attn={self.cfg.attn_implementation}", file=sys.stderr)
        self._processor = AutoProcessor.from_pretrained(self.cfg.model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_path)
        # ``device_map`` is intentionally NOT passed when running under
        # accelerate / DeepSpeed: the launcher is responsible for placement.
        # We only honour cfg.device_map in single-process fall-back mode
        # (e.g. local smoke tests without accelerate).
        load_kwargs = dict(
            torch_dtype=torch_dtype,
            attn_implementation=self.cfg.attn_implementation,
        )
        if not _under_accelerate_launch():
            load_kwargs["device_map"] = self.cfg.device_map
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.cfg.model_path,
            **load_kwargs,
        )

        # ── Optional LoRA injection. When lora_config_path is None we
        # leave the model fully trainable — every param requires_grad
        # by default after from_pretrained. The trainer then builds an
        # AdamW over the full param set; ZeRO-3 handles sharding. Note
        # this is INCOMPATIBLE with the disable_adapter() KL-ref path —
        # full-param mode REQUIRES a separate frozen reference model
        # (see _load_ref_model below).
        if self.cfg.lora_config_path:
            self._inject_lora(self.cfg.lora_config_path)
            self._is_full_param = False
        else:
            self._is_full_param = True
            # Use ``ds_numel`` when present so ZeRO-3 placeholder params
            # (numel==0) report the true unpartitioned size.
            n_params = sum(_param_full_numel(p) for p in self._model.parameters())
            print(f"[student] FULL-PARAM mode (no LoRA). All "
                  f"{n_params:,} "
                  f"params trainable.", file=sys.stderr)

        # ── Optional gradient checkpointing.
        # IMPORTANT: ``enable_input_require_grads`` MUST be called BEFORE
        # ``gradient_checkpointing_enable`` when LoRA is present —
        # otherwise the input embedding sits in no_grad and the LoRA
        # adapter's grad is silently None. Order matters.
        if self.cfg.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()

        # ── Optional reference policy for KL (full-param mode only).
        if self._is_full_param and self.cfg.ref_model_path:
            self._load_ref_model(torch_dtype)

        # ── Soft-logits score-token-id resolution (mirrors stage0 D5).
        # Builds the digit-id groups + colon anchors once at load time.
        self._init_score_token_ids()

        # We default to eval mode; the trainer will flip ``.train()`` on
        # the LoRA params right before calling compute_log_probs.
        self._model.eval()

    def _load_ref_model(self, torch_dtype) -> None:
        """Load a SECOND frozen 7B as the KL reference policy.

        Required for full-param training because the LoRA `disable_adapter()`
        trick is unavailable. The ref is loaded with `requires_grad=False`
        on every param so backward never flows through it.

        Sharding policy
        ---------------
        We deliberately keep the ref **full-replicated on every rank**
        (~14 GB / card for 7B bf16), NOT sharded:

        * Under DeepSpeed ZeRO-3, ``zero3_init_flag: true`` causes HF's
          ``from_pretrained`` to partition every param at construction
          time. The ref then has ``ds_id``-stamped params and every
          ref forward needs a ``GatheredParameters`` collective —
          which: (a) defeats the "no extra collectives on ref forward"
          assumption the trainer relies on, (b) costs ~6× wall-clock vs.
          the replicated path because ZeRO-3 has to gather every layer
          for every ref call. We force-disable ZeRO-3 init for the ref
          load by clearing transformers' internal weakref pointer.

        Memory budget: 8×H100 80GB, with ZeRO-3-sharded policy ~30 GB
        and full-replicated ref ~14 GB still leaves ~30 GB headroom for
        activations + KV cache. Comfortable.

        Why a weakref dance instead of ``deepspeed.zero.Init(enabled=False)``
        ---------------------------------------------------------------------
        HF transformers does NOT consult ``deepspeed.zero.Init``'s status to
        decide whether to partition. It checks
        ``transformers.integrations.deepspeed._hf_deepspeed_config_weak_ref``
        (or its modern alias). When accelerate creates the DeepSpeedPlugin
        it sets that weakref globally, so every subsequent
        ``Qwen.from_pretrained`` call goes through the ZeRO-3-init path
        regardless of any ``deepspeed.zero.Init(enabled=False)`` context.
        The reliable knock-out is to clear that weakref (or call HF's own
        ``unset_hf_deepspeed_config``) for the duration of the ref load,
        then restore it. This is the trl / DeepSpeed-Chat pattern.
        """
        from transformers import Qwen2_5_VLForConditionalGeneration as Qwen
        print(f"[student] loading frozen REF policy from {self.cfg.ref_model_path}",
              file=sys.stderr)
        ref_kwargs = dict(
            torch_dtype=torch_dtype,
            attn_implementation=self.cfg.attn_implementation,
        )
        if not _under_accelerate_launch():
            ref_kwargs["device_map"] = self.cfg.device_map

        # ── Patch 7 (2026-06-16): when ref_zero3 is True we LET ZeRO-3
        # init partition the ref's params. This is the key fix for the
        # kl_s ≈ 4.6 nats noise floor at step 0 when student == ref:
        # policy forwards go through DeepSpeed's per-layer hooks
        # (gather → forward → re-shard) while the previously
        # non-partitioned ref took the plain-HF code path, producing
        # bf16 numerical asymmetries the Schulman k3 estimator could
        # not distinguish from real KL. With both partitioned, the
        # forward routes match — both sides go through summon_full_params
        # gather scopes — and KL_s ≈ 0 by construction.
        #
        # When ref_zero3 is False we keep the legacy disable-dance,
        # both for backwards compat and as an escape hatch if the
        # ZeRO-3 ref forward path has a bug we haven't found yet.
        use_zero3_ref = bool(getattr(self.cfg, "ref_zero3", False))

        # ── Force-disable ZeRO-3 init for this from_pretrained call.
        #
        # transformers.integrations.deepspeed exposes the global weakref
        # under one of a few names depending on the version. We try them
        # in order and silently no-op if the module isn't importable
        # (single-process runs).
        #
        # Skipped entirely when ref_zero3=True (Patch 7).
        restore_callbacks: list = []
        if not use_zero3_ref:
            try:
                from transformers.integrations import deepspeed as _hf_ds_int

                # Modern HF: explicit setter/unsetter API.
                unset_fn = getattr(_hf_ds_int, "unset_hf_deepspeed_config", None)
                set_fn   = getattr(_hf_ds_int, "set_hf_deepspeed_config", None)
                # The actual weakref attribute. Names seen in the wild:
                #   _hf_deepspeed_config_weak_ref  (transformers >= 4.38)
                #   hf_deepspeed_config_weak_ref   (older)
                wr_attrname = None
                for name in ("_hf_deepspeed_config_weak_ref",
                             "hf_deepspeed_config_weak_ref"):
                    if hasattr(_hf_ds_int, name):
                        wr_attrname = name
                        break

                saved_wr = None
                saved_obj = None  # for the set_fn-based restore path

                if wr_attrname is not None:
                    saved_wr = getattr(_hf_ds_int, wr_attrname)
                    # Resolve the strong reference for restore (if any).
                    if saved_wr is not None and callable(saved_wr):
                        try:
                            saved_obj = saved_wr()
                        except Exception:
                            saved_obj = None
                    # Clear it: HF will now treat ZeRO-3 as inactive.
                    setattr(_hf_ds_int, wr_attrname, None)

                if unset_fn is not None:
                    # API-level unset is a no-op when nothing is registered;
                    # call it after the manual clear so both states agree.
                    try:
                        unset_fn()
                    except Exception:
                        pass

                def _restore():
                    # Prefer the API-level setter if we still have the strong
                    # ref to the original config object.
                    if saved_obj is not None and set_fn is not None:
                        try:
                            set_fn(saved_obj)
                            return
                        except Exception:
                            pass
                    # Fallback: direct attribute restore.
                    if wr_attrname is not None and saved_wr is not None:
                        setattr(_hf_ds_int, wr_attrname, saved_wr)

                restore_callbacks.append(_restore)
            except ImportError:
                # transformers integrations.deepspeed not present — nothing
                # to disable. Happens on single-process runs.
                pass
            except Exception as e:                                 # pragma: no cover
                print(f"[student] HF deepspeed-zero3 disable for ref load skipped: {e}",
                      file=sys.stderr)

        # ── Belt-and-suspenders: also wrap in deepspeed.zero.Init(enabled=False)
        # so any downstream code path that does check the Init context
        # also sees it disabled. No-op when DS isn't installed.
        # Skipped entirely when ref_zero3=True (we WANT zero3 init to fire).
        import contextlib
        ds_disable_init_ctx = contextlib.nullcontext()
        if not use_zero3_ref:
            try:
                from deepspeed.zero import Init as _DSZeroInit
                ds_disable_init_ctx = _DSZeroInit(enabled=False)
            except ImportError:
                pass
            except Exception:
                ds_disable_init_ctx = contextlib.nullcontext()

        try:
            with ds_disable_init_ctx:
                self._ref_model = Qwen.from_pretrained(self.cfg.ref_model_path, **ref_kwargs)
        finally:
            for cb in restore_callbacks:
                try:
                    cb()
                except Exception as e:                         # pragma: no cover
                    print(f"[student] HF deepspeed-zero3 restore failed: {e}",
                          file=sys.stderr)

        for p in self._ref_model.parameters():
            p.requires_grad = False
        self._ref_model.eval()
        # Sanity: confirm partitioning state matches the flag (Patch 7).
        first_param = next(self._ref_model.parameters(), None)
        is_partitioned = (
            first_param is not None and hasattr(first_param, "ds_id")
        )
        if use_zero3_ref and not is_partitioned:
            print("[student] WARNING: ref_zero3=True but ref params have NO "
                  "ds_id stamps — ZeRO-3 init did not fire. Check that "
                  "DeepSpeed is configured (HF deepspeed plugin must be "
                  "active when ref load runs).", file=sys.stderr)
        elif (not use_zero3_ref) and is_partitioned:
            print("[student] WARNING: ref model has ZeRO-3 ds_id stamps — "
                  "the disable dance did not take effect. Ref forward "
                  "will be slow and may desync the trainer.",
                  file=sys.stderr)
        elif use_zero3_ref:
            print("[student] ref policy is ZeRO-3 partitioned (Patch 7) — "
                  "ref forwards will go through summon_full_params just "
                  "like the policy, so kl_s reflects real divergence.",
                  file=sys.stderr)
        # ZeRO-3 / partitioned params have ``numel() == 0`` placeholder
        # storage; the real size is exposed via ``ds_numel``. Use that
        # so the print is meaningful in both modes.
        n_params = sum(_param_full_numel(p) for p in self._ref_model.parameters())
        print(f"[student] ref policy ready ({n_params:,} frozen params).",
              file=sys.stderr)

    def _enable_gradient_checkpointing(self) -> None:
        """Turn on activation checkpointing on the underlying HF model.

        Two prerequisites for LoRA + gradient_checkpointing to actually
        train:

        1. The base model's input embedding must allow grad to flow
           through it. HF's ``enable_input_require_grads`` registers a
           forward hook that sets ``requires_grad=True`` on the embedded
           tokens. Without this hook the LoRA adapter's grad is None.

        2. ``use_reentrant=False`` is the recommended path post-PyTorch
           1.11. Reentrant=True works but: (a) it's ~2-3x slower on
           backward; (b) it's incompatible with non-tensor inputs that
           multimodal forwards sometimes pass through.

        Caveats handled here:
          * If the model is PEFT-wrapped, ``model`` is a ``PeftModel``
            and we have to enable on its ``base_model`` for some HF
            versions — we try both code paths to be safe.
        """
        m = self._model
        # Some HF versions raise on already-enabled checkpointing — a
        # broad except keeps idempotency.
        try:
            if hasattr(m, "enable_input_require_grads"):
                m.enable_input_require_grads()
            else:
                # PeftModel forwards the call through; handle plain HF
                # models that don't expose it (rare, very old versions).
                if hasattr(m, "get_input_embeddings"):
                    emb = m.get_input_embeddings()
                    if emb is not None:
                        def _make_inputs_require_grad(_module, _input, output):
                            output.requires_grad_(True)
                        emb.register_forward_hook(_make_inputs_require_grad)
        except Exception as e:
            print(f"[student] enable_input_require_grads failed: {e}",
                  file=sys.stderr)

        try:
            m.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": bool(self.cfg.grad_checkpoint_use_reentrant),
                },
            )
        except TypeError:
            # Older HF / PEFT versions don't accept the kwargs dict.
            try:
                m.gradient_checkpointing_enable()
            except Exception as e:
                print(f"[student] gradient_checkpointing_enable failed: {e}",
                      file=sys.stderr)
                return
        # NOTE: we DON'T flip use_cache here. The sampling forward
        # (model.generate) wants use_cache=True for KV caching speed,
        # while only the backward forward (compute_log_probs) needs
        # use_cache=False to avoid the HF gradient-checkpointing warning.
        # ``compute_log_probs`` toggles use_cache locally per call.
        print(f"[student] gradient checkpointing ON  "
              f"(use_reentrant={self.cfg.grad_checkpoint_use_reentrant})",
              file=sys.stderr)

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

    def generate_rubrics_with_meta_batch(
        self,
        caption:    str | None = None,
        *,
        num_return_sequences: int,
        dim_ctx:    DimensionContext | None = None,
        query:      str | None = None,
    ) -> list["RubricRollout"]:
        """Sample ``num_return_sequences`` rubrics for ONE (caption, dim) in
        a single ``model.generate`` call.

        Why
        ---
        The naive G-rollout loop in ``step_one_instance`` calls
        ``generate_rubric_with_meta`` G times back-to-back with the SAME
        prompt — every call repays the full prefill cost (~700 input
        tokens) and the full ZeRO-3 gather + grad-ckpt-disable + summon
        ceremony. ``num_return_sequences=G`` keeps the prefill at 1× and
        only batches the suffix decode.

        Returns one ``RubricRollout`` per sample. ``prompt_ids`` is the
        same tensor shared across all G entries (the trainer's
        ``compute_log_probs`` doesn't share state across rollouts so this
        is fine — they each get their own ``sampled_ids``).
        """
        ctx    = dim_ctx or DimensionContext.generic()
        cap    = caption if caption is not None else (query or "")
        prompt = self._format_rubric_prompt(ctx, cap)
        if num_return_sequences < 1:
            raise ValueError("num_return_sequences must be >= 1")
        if self._dummy:
            # Dummy mode: deterministic-but-jittered rubrics so the
            # downstream parser/reward path stays exercised. Vary by
            # appending the rollout index so identical captions produce
            # G different rubrics within a group.
            return [
                RubricRollout(
                    text=self._dummy_rubric(f"{cap or ctx.dimension}::{i}"),
                    prompt_ids=None,
                    sampled_ids=None,
                )
                for i in range(num_return_sequences)
            ]

        self._ensure_model()
        return self._generate_text_only_with_meta_batch(
            prompt, num_return_sequences=num_return_sequences,
        )

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
        # SCORE-side uses the SHORT task description (one-liner). The long
        # rubric-gen-targeted task_description (sub-aspect IDs, "ground
        # itself" instructions, Audio block) hurt score under student
        # rubrics — prompt-diff showed it polluted the autoregressive
        # prior. ``ctx.score_task_description()`` falls back to a generic
        # one-liner if the trainer didn't populate the short flavour.
        return self.cfg.score_prompt_tmpl.format(
            dimension         = ctx.dimension or "video_quality",
            evalverse_path    = ctx.evalverse_path or ctx.dimension or "video_quality",
            task_description  = ctx.score_task_description(),
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
                    dim_ctx:       DimensionContext | None = None,
                    rubric_themes: list[str] | None = None,
                    # Legacy positional alias: caption used to be passed
                    # as the first arg called ``query``. Old call sites
                    # should migrate to keyword form.
                    query:        str | None = None,
                    # Deprecated alias kept for old probe scripts; prefer
                    # ``rubric_themes``. Will be removed in a later release.
                    rubric_paths: list[str] | None = None,
                    ) -> tuple[str, list[float | None]]:
        """Score one video on each rubric theme, returning ``(text, scores)``.

        ``scores`` is a list of length ``len(rubric_themes)`` (or 5 if
        ``rubric_themes`` is None) of either floats in [1.0, 5.0] (soft_logits
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
        # Accept the deprecated ``rubric_paths`` kwarg.
        if rubric_themes is None and rubric_paths is not None:
            rubric_themes = rubric_paths
        n   = len(rubric_themes) if rubric_themes else 5
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

    def score_videos_batch(
        self,
        caption:      str | None,
        video_path:   str,
        rubric_texts: list[str],
        *,
        dim_ctx:           DimensionContext | None = None,
        rubric_themes_per: list[list[str] | None] | None = None,
        query:        str | None = None,
        return_meta:  bool = False,
        # ── Deprecated alias kept for old probe scripts. New callers use
        # ``rubric_themes_per``. If both are supplied, ``rubric_themes_per``
        # wins; the old name will be removed in a later release.
        rubric_paths_per: list[list[str] | None] | None = None,
    ) -> list[tuple[str, list[float | None]]] | list[tuple[str, list[float | None], "RubricRollout | None"]]:
        """Score ONE video against ``len(rubric_texts)`` different rubrics
        in a single batched forward.

        Returns a list of ``(text, scores)`` tuples, length =
        ``len(rubric_texts)``. With ``return_meta=True`` AND
        ``cfg.score_greedy=False`` (i.e. sampling is on), each tuple gets
        a third element: a ``RubricRollout``-shaped struct carrying the
        prompt/sampled token ids so the trainer can compute log-probs
        for the score-side policy gradient. Greedy mode + return_meta
        downgrades to a None third element (no sampling → no PG).

        See ``_score_video_softlogits_batch`` for the rationale and
        caveats. Falls back to per-row ``score_video`` calls in
        integer_text mode (no soft-logits → no clear win from batching
        on the parse side, and the integer parser is sequential anyway).
        ``return_meta`` is silently ignored on the integer_text path —
        that path doesn't expose sampled_ids and the trainer should not
        be on it when ``score_pg`` is active.
        """
        if caption is None:
            caption = query or ""
        ctx = dim_ctx or DimensionContext.generic()
        # Accept the deprecated ``rubric_paths_per`` keyword from old probe
        # scripts. ``rubric_themes_per`` takes precedence when both are set.
        if rubric_themes_per is None and rubric_paths_per is not None:
            rubric_themes_per = rubric_paths_per
        if rubric_themes_per is None:
            rubric_themes_per = [None] * len(rubric_texts)
        if len(rubric_themes_per) != len(rubric_texts):
            raise ValueError("rubric_themes_per must have the same length as rubric_texts")

        # Build per-row prompts (same scaffold, different rubric body).
        prompts: list[str] = []
        n_themes_per: list[int] = []
        for rubric_text, rubric_themes in zip(rubric_texts, rubric_themes_per):
            n = len(rubric_themes) if rubric_themes else 5
            n_themes_per.append(n)
            format_line = " ".join(f"theme_{k}: <X>" for k in range(1, n + 1))
            prompts.append(
                self._format_score_prompt(ctx, caption, rubric_text or "", format_line)
            )

        if self._dummy:
            base = [
                self._dummy_score(rt, video_path, n)
                for rt, n in zip(rubric_texts, n_themes_per)
            ]
            if return_meta:
                return [(t, s, None) for (t, s) in base]
            return base

        self._ensure_model()
        method = self.cfg.score_method
        if method == "soft_logits" and not getattr(self, "_soft_logits_ok", False):
            method = "integer_text"
        if method == "soft_logits":
            return self._score_video_softlogits_batch(
                prompts, video_path, n_themes_per,
                force_greedy=self.cfg.score_greedy,
                return_meta=return_meta,
            )
        # integer_text fallback: no logit aggregation needed, so the
        # batched-prefill win is smaller. Keep the per-row loop here for
        # clarity; the caller gets the same return shape either way.
        from .rubric_parser import parse_score_scaffold
        out: list[tuple[str, list[float | None]] | tuple[str, list[float | None], None]] = []
        for prompt, n in zip(prompts, n_themes_per):
            text = self._generate_with_video(
                prompt, video_path,
                force_greedy=self.cfg.score_greedy,
                max_new_tokens=self.cfg.score_max_new_tokens,
            )
            if return_meta:
                out.append((text, parse_score_scaffold(text, n), None))
            else:
                out.append((text, parse_score_scaffold(text, n)))
        return out

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
        # ZeRO-3: gather sharded params for the duration of generate() so
        # every layer sees a 2-D (un-partitioned) weight tensor. The
        # autoregressive sampling loop otherwise hits the "weight must be
        # 2-D" embedding bug on the first step.
        #
        # Additionally pause grad-checkpointing for the duration of
        # generate(): the grad-ckpt forward hooks interact badly with
        # freshly-gathered ZeRO-3 params on transformers 4.5x and produce
        # NaN/Inf logits → ``torch.multinomial`` then raises
        # "probability tensor contains either `inf`, `nan` or element < 0".
        # Generation is no_grad so disabling grad-ckpt costs no memory.
        # transformers ≥4.4x rejects do_sample=True with temperature=0.0;
        # at deploy time we want true greedy, so flip do_sample off and
        # drop sampling kwargs when the configured temperature is 0.
        gen_kwargs: dict = {"max_new_tokens": self.cfg.max_new_tokens}
        if self.cfg.temperature and self.cfg.temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.cfg.temperature
            gen_kwargs["top_p"] = self.cfg.top_p
        else:
            gen_kwargs["do_sample"] = False
        with _pause_grad_ckpt(
                self._model,
                use_reentrant=self.cfg.grad_checkpoint_use_reentrant,
             ), _summon_full_params_ctx(self._model), torch.no_grad():
            out_ids = self._model.generate(**inputs, **gen_kwargs)
        gen = out_ids[0, inputs["input_ids"].shape[1]:]
        return self._processor.tokenizer.decode(gen, skip_special_tokens=True)

    def _generate_text_only_with_meta(self, prompt: str) -> "RubricRollout":
        """Same as ``_generate_text_only`` but also returns prompt + sampled ids.

        Thin wrapper over the batched path with G=1 — kept for back-compat
        callers (e.g. ablation paths that don't use group sampling).
        """
        return self._generate_text_only_with_meta_batch(prompt, num_return_sequences=1)[0]

    def _generate_text_only_with_meta_batch(
        self,
        prompt: str,
        *,
        num_return_sequences: int,
    ) -> list["RubricRollout"]:
        """Sample ``num_return_sequences`` rubrics from ONE prompt in a single
        ``model.generate`` call.

        Why this exists
        ---------------
        The naive G-rollout loop calls generate G times back-to-back with
        identical prompt + identical KV-cache prefill — i.e. the prompt's
        ~700 input tokens get re-encoded G times for no reason. Using
        ``num_return_sequences=G`` makes generate run prefill ONCE and only
        the suffix decode is G-replicated. With G=8 this cuts the rubric
        phase wall-time roughly to 1/G + small batched-decode overhead;
        on H100 we see ~5-7× speedup.

        Returns one RubricRollout per sample, each with its own
        ``sampled_ids`` (variable length — the generate call still EOS's
        each beam independently).

        Memory note — micro-batching
        ----------------------------
        The KV cache for ``num_return_sequences=G`` lives at
        ``(G, P + max_new)``. With G=8, P~700, max_new=1024, 7B in bf16
        and ZeRO-3 currently in the GATHERED state (the outer
        ``_summon_full_params_ctx`` holds the full 7B replicated on each
        rank during this call), the activation + KV memory adds ~6-10 GB
        on top of the gathered weights. We split into micro-batches of
        size ``RM_RUBRIC_MB`` (default 4) so the worst-case is half of
        that. Set ``RM_RUBRIC_MB=8`` to skip the split if you have
        headroom.
        """
        import torch
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        chat = self._processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._processor(text=[chat], return_tensors="pt").to(self._model.device)
        prompt_ids_1d = inputs["input_ids"][0]                # (P,)
        P = prompt_ids_1d.shape[0]

        try:
            mb_default = int(os.environ.get("RM_RUBRIC_MB", "4"))
        except ValueError:
            mb_default = 4
        G_total = int(num_return_sequences)
        mb = max(1, min(G_total, mb_default))

        pad_id = (
            getattr(self._processor.tokenizer, "pad_token_id", None)
            or getattr(self._processor.tokenizer, "eos_token_id", None)
        )

        rollouts: list[RubricRollout] = []
        remaining = G_total
        # Same ZeRO-3 gather + grad-ckpt safety dance as the single-shot
        # path. We open ONE summon for the whole micro-batch loop so the
        # 7B isn't gathered + scattered repeatedly per chunk — that would
        # negate most of the win.
        with _pause_grad_ckpt(
                self._model,
                use_reentrant=self.cfg.grad_checkpoint_use_reentrant,
             ), _summon_full_params_ctx(self._model), torch.no_grad():
            while remaining > 0:
                k = min(mb, remaining)
                out_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.cfg.max_new_tokens,
                    do_sample=True,
                    temperature=self.cfg.temperature,
                    top_p=self.cfg.top_p,
                    num_return_sequences=int(k),
                )
                # ``out_ids`` shape: (k, P+T_pad). The prompt prefix is
                # identical across rows so we can slice once. Trailing PAD
                # tokens (after EOS) are stripped per-row so the trainer's
                # compute_log_probs sees only real sampled tokens.
                for i in range(out_ids.shape[0]):
                    seq = out_ids[i]
                    sampled = seq[P:]
                    if pad_id is not None:
                        nz = (sampled != pad_id).nonzero(as_tuple=False)
                        if nz.numel() > 0:
                            last = int(nz[-1].item()) + 1
                            sampled = sampled[:last]
                        else:
                            sampled = sampled[:0]
                    text = self._processor.tokenizer.decode(
                        sampled, skip_special_tokens=True,
                    )
                    rollouts.append(RubricRollout(
                        text=text,
                        prompt_ids=prompt_ids_1d.detach().clone(),
                        sampled_ids=sampled.detach().clone(),
                    ))
                remaining -= k
        return rollouts

    # ------------------------------------------------------------------
    # Teacher-forcing log-prob forward (used by GRPO policy gradient + KL).
    # ------------------------------------------------------------------
    def compute_log_probs(self, prompt_ids, sampled_ids, *, ref: bool = False,
                          _skip_summon: bool = False,
                          detach_prompt: bool = False,
                          mm_kwargs: dict | None = None):
        """Return per-token log-prob of ``sampled_ids`` given ``prompt_ids``.

        Returns
        -------
        torch.Tensor of shape ``(T,)`` where ``T = len(sampled_ids)``.
        With ``ref=False`` this carries grad iff the model has trainable
        params (LoRA on); with ``ref=True`` we ``with disable_adapter()``
        the LoRA so the same forward gives base-model log-probs (used
        for the KL term), wrapped in ``torch.no_grad``.

        ``_skip_summon`` (internal): when True, the caller has already
        opened a ``summon_full_params`` scope around the entire G-rollout
        loop and we MUST NOT open another one — re-entering
        ``summon_full_params(writeback=True)`` within one optimiser step
        corrupts the gathered parameters (rollouts after the first see
        garbage logits). Used by ``policy_update`` to keep the per-rollout
        backward / activation-release pattern (memory-bounded by ONE
        rollout) while still doing only one gather + one scatter per step.

        ``detach_prompt`` (Trick A — EasyR1 / VeRL / OpenRLHF style):
            Standard PG forward is ``forward(prompt + sampled)`` end-to-end
            with grad through the entire activation graph. For a *score*
            forward the prompt contains thousands of video placeholder
            tokens (~6000), so the activation graph dominates memory and
            backward OOMs.

            With ``detach_prompt=True`` we split the forward in two:

              1. prompt forward of length ``P-1`` runs under ``no_grad``
                 with ``use_cache=True`` to obtain ``past_key_values``.
                 No autograd graph for the long video prefix.
              2. sampled forward of length ``T+1`` (= prompt's last token
                 prepended to sampled_ids) runs grad-on against the cached
                 K/V. Every logp[t] still receives gradient — including
                 ``logp[0]``: its hidden state at the prepended prompt[-1]
                 position is computed via attention over the (detached)
                 K/V cache + position-P query, so the query weights still
                 receive gradient.

            This is mathematically the same simplification VeRL / TRL /
            OpenRLHF / EasyR1 / DeepSpeed-Chat all make (prompt is a
            condition, not a prediction target — ``log π_θ(generated |
            prompt)`` doesn't need ∇_θ through the prompt forward to be
            well-defined PG).

            Caveats:
              * Requires ``use_cache=True`` on the prompt forward, which
                conflicts with ``gradient_checkpointing`` only while
                ``self.training`` is True. We temporarily flip the model
                to ``eval()`` for the prompt forward (Qwen2.5-VL has no
                BatchNorm/Dropout that change behaviour, so eval ≡ train
                for the forward outputs), then restore. The sampled
                forward stays in eval too — there's no value in re-enabling
                grad-ckpt for a 32-token forward.
              * Not used for ``ref=True``. The ref forward is already
                under ``no_grad`` end-to-end so it pays no activation cost
                from grad — splitting the forward there only adds a kv
                cache allocation for nothing.
              * Falls back to the legacy single-forward path when
                ``ref=True``, ``T == 0``, or ``P < 2`` (no prompt to cache).

        Implementation notes (legacy path)
        ----------------------------------
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

        T = sampled_ids.shape[0]
        P = prompt_ids.shape[0]

        # detach_prompt only meaningful with a non-empty sampled tail and
        # a prompt long enough to split. Historically we restricted this
        # to the policy forward (``not ref``) to avoid an extra KV cache
        # allocation on a ``no_grad`` ref pass. That restriction was
        # REMOVED 2026-06-16: when score-PG is enabled, the policy went
        # through Trick A's two-stage forward while the ref took the
        # legacy single-forward, and the resulting numerical asymmetry
        # produced kl_s = 4–22 nats at step 0 (when student == ref) — pure
        # path noise that the Schulman k3 estimator could not distinguish
        # from a real KL. Letting ref also take the detach path makes
        # both forwards numerically identical at step 0, restoring KL ≈ 0.
        # The extra KV cache allocation is small relative to the policy's
        # activation graph that we already pay for.
        use_detach = bool(detach_prompt) and T > 0 and P >= 2

        # NOTE: build the input WITHOUT moving it to a device yet — the
        # right device depends on whether we're forwarding through the
        # policy (DeepSpeed-wrapped, lives on cuda:<local_rank>) or the
        # frozen ref (un-wrapped, also on cuda:<local_rank> after the
        # trainer .to() it). We compute target_model first, then move.
        full = torch.cat([prompt_ids, sampled_ids], dim=0).unsqueeze(0)
        # Decide which model + grad mode to use for this forward.
        #
        #   ref=True paths (3 cases, in priority order):
        #     1. self._ref_model is set    → frozen second 7B (full-param mode)
        #     2. model has disable_adapter → LoRA mode, drop the adapter
        #     3. plain HF model            → ref == policy, just no_grad
        #          (this gives KL ≈ 0; only useful when beta_kl=0 anyway)
        if ref and self._ref_model is not None:
            target_model = self._ref_model
            ctx = _NullCtx()
            grad_ctx = torch.no_grad()
        elif ref and hasattr(self._model, "disable_adapter"):
            target_model = self._model
            ctx = self._model.disable_adapter()
            grad_ctx = torch.no_grad()
        elif ref:
            target_model = self._model
            ctx = _NullCtx()
            grad_ctx = torch.no_grad()
        else:
            target_model = self._model
            ctx = _NullCtx()
            grad_ctx = _NullCtx()      # caller controls grad via outer enable_grad / no_grad

        # Now move ``full`` to whichever device the chosen target lives
        # on. ``target_model.device`` is reliable for ZeRO-3-wrapped
        # models (DeepSpeedEngine.device follows the active GPU) and for
        # the plain ref model (set explicitly by the trainer).
        full = full.to(target_model.device)

        # ZeRO-3 gather scope. Two cases:
        #   * ref=True forwards are no_grad — gather with writeback=False
        #     gives a read-only un-partitioned view for the duration of
        #     the forward and avoids the "weight must be 2-D" embedding
        #     bug.
        #   * ref=False forwards need backward to flow through to the
        #     sharded params. We do NOT issue a manual gather; instead
        #     we let DeepSpeedEngine's per-layer pre/post-forward hooks
        #     gather/scatter automatically, which is the supported way
        #     to drive backward under ZeRO-3. ``writeback=True`` here
        #     therefore resolves to a no-op (see _summon_full_params_ctx).
        #
        # When ``_skip_summon`` is set, the caller has already opened
        # the gather scope around a multi-rollout loop and we must NOT
        # open another one (re-entering corrupts the gathered params).
        #
        # ── Patch 8 (2026-06-17, restored): ref forward path selection.
        #
        # Three possible ref states (detected at call time, no flag needed):
        #
        # 1. Ref is wrapped in a DeepSpeedEngine (post `accelerator.
        #    prepare_model(ref, evaluation_mode=True)`). Detected by
        #    presence of `module` attribute or
        #    `set_gradient_accumulation_boundary` method. The engine's
        #    per-layer pre/post-forward hooks gather and scatter
        #    automatically, exactly mirroring the policy path. We use
        #    ``_NullCtx()`` and let those hooks fire.
        #
        # 2. Ref params are ZeRO-3 partitioned (have ``ds_id``) but the
        #    module was NEVER prepared by accelerate / deepspeed —
        #    no hooks installed. Forward would crash with "weight
        #    must be 2-D" on the embedding layer. We need manual
        #    ``_summon_full_params_ctx(writeback=False)`` to gather
        #    params for the duration of the forward. This is the
        #    legacy Patch 7 path.
        #
        # 3. Ref is full-replicated (no ds_id, plain HF module). No
        #    gather needed — ``_NullCtx()``.
        #
        # The caller's summon scope (when _skip_summon=False) covers the
        # POLICY model only; ref handling is independent regardless.
        if _skip_summon and ref and self._ref_model is not None:
            sample = next(self._ref_model.parameters(), None)
            ref_is_partitioned = (
                sample is not None and hasattr(sample, "ds_id")
            )
            ref_has_engine = (
                hasattr(self._ref_model, "module")
                or hasattr(self._ref_model, "set_gradient_accumulation_boundary")
            )
            if ref_has_engine:
                # Case 1: DSEngine-wrapped — hooks fire automatically.
                summon_ctx = _NullCtx()
            elif ref_is_partitioned:
                # Case 2: partitioned but no hooks — manual gather required.
                summon_ctx = _summon_full_params_ctx(
                    self._ref_model, writeback=False,
                )
            else:
                # Case 3: full-replicated — no gather needed.
                summon_ctx = _NullCtx()
        elif _skip_summon:
            summon_ctx = _NullCtx()
        else:
            summon_ctx = _summon_full_params_ctx(
                target_model, writeback=(not ref),
            )

        with ctx, grad_ctx, summon_ctx:
            # use_cache MUST be False when gradient_checkpointing is ON
            # — otherwise HF emits a warning and the checkpointing
            # silently no-ops. Save / restore so generate() still gets
            # KV caching on its no_grad sampling path.
            #
            # IMPORTANT: when target_model is a DeepSpeedEngine (the
            # accelerate-prepared policy), ``engine.config`` returns the
            # DeepSpeed runtime JSON (a plain dict) -- it shadows the HF
            # ``PretrainedConfig``. Setting ``.use_cache`` on a dict
            # would AttributeError. Unwrap to the inner HF module first
            # via ``engine.module``; for the un-wrapped ref model this
            # is a no-op.
            inner_for_cfg = getattr(target_model, "module", target_model)
            cfg_obj = getattr(inner_for_cfg, "config", None)
            if not hasattr(cfg_obj, "use_cache"):
                # Some HF configs expose use_cache only via __dict__;
                # fall back to None so we skip the toggle silently.
                cfg_obj = None
            prev_cache = None
            toggled_cache = False
            if cfg_obj is not None and self.cfg.use_gradient_checkpointing:
                prev_cache = getattr(cfg_obj, "use_cache", None)
                cfg_obj.use_cache = False
                toggled_cache = True
            try:
                if use_detach:
                    tok_logp = self._compute_log_probs_detach_prompt(
                        target_model=target_model,
                        prompt_ids=prompt_ids,
                        sampled_ids=sampled_ids,
                        T=T, P=P,
                        no_grad_tail=bool(ref),
                        mm_kwargs=mm_kwargs,
                    )
                else:
                    # Qwen2.5-VL is multimodal: when called with text-only
                    # input_ids and no attention_mask, transformers 4.53's
                    # ``_preprocess_mask_arguments`` tries to expand a 3D
                    # multimodal RoPE position_ids tensor to 2D and crashes
                    # ("expand([3,1,T], size=[1,-1])"). Passing an explicit
                    # attention_mask short-circuits that path: the model
                    # treats this as a regular causal LM forward.
                    #
                    # ZeRO-3 note: ref=True forwards run inside the read-only
                    # gather above; ref=False forwards rely on DeepSpeedEngine's
                    # pre/post-forward hooks to gather and on backward to
                    # reduce-scatter grads. The trainer wraps the policy
                    # compute_log_probs in ``with torch.enable_grad()`` and
                    # the engine's ``backward()`` handles the rest.
                    attn_mask = torch.ones_like(full)
                    # Pass use_cache=False as a forward kwarg (NOT only via
                    # config) so HF's gradient-checkpointing path doesn't
                    # spam the "use_cache=True is incompatible with gradient
                    # checkpointing" warning every step. The kwarg overrides
                    # whatever config.use_cache was, regardless of which
                    # underlying config object the model actually consults
                    # (DeepSpeedEngine.config shadows the HF one — see the
                    # cfg_obj unwrap above).
                    fwd_kwargs = {"input_ids": full, "attention_mask": attn_mask}
                    if self.cfg.use_gradient_checkpointing:
                        fwd_kwargs["use_cache"] = False
                    # ── Patch 6b (2026-06-16): inject mm tensors. Without
                    # these the Qwen2.5-VL forward sees the video_pad
                    # placeholder tokens but skips the vision tower, so
                    # the resulting logp is the language-only marginal
                    # rather than the multimodal π_θ that produced the
                    # sample. With score forwards this is essential —
                    # score-PG ratios become unbiased only when the
                    # logp is from the same conditional distribution
                    # as the sampling step. The mm_kwargs dict is built
                    # by the trainer from RubricRollout's per-row mm
                    # fields (pixel_values_videos, video_grid_thw,
                    # second_per_grid_ts).
                    fwd_kwargs.update(_normalize_mm_kwargs(
                        mm_kwargs, target_device=target_model.device,
                    ))
                    out = target_model(**fwd_kwargs)
                    logits = out.logits[0]                              # (P+T, V)
                    # Predicting token at position i+1 from logits at position i.
                    shift = logits[-(T + 1):-1, :]                      # (T, V)
                    shift = shift.float()
                    log_probs = torch.log_softmax(shift, dim=-1)
                    tgt = sampled_ids.to(target_model.device).long()
                    tok_logp = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # (T,)
            finally:
                if toggled_cache and cfg_obj is not None:
                    # Restore even when prev_cache was None — None is a
                    # legitimate "use HF default" sentinel and writing
                    # False back would change behaviour for the next
                    # generate() call.
                    cfg_obj.use_cache = prev_cache
        return tok_logp

    # ------------------------------------------------------------------
    # BT-aux helper (added 2026-06-16): grad-on digit logits at offsets.
    # ------------------------------------------------------------------
    def compute_digit_logits(
        self,
        prompt_ids,
        sampled_ids,
        digit_token_offsets: list[int],
        *,
        _skip_summon: bool = False,
        detach_prompt: bool = True,
        mm_kwargs: dict | None = None,
    ):
        """Re-run a grad-on forward and return per-position 5-digit logits.

        Why this exists
        ---------------
        ``model.generate`` runs under ``torch.no_grad`` so the digit logits
        captured during the score forward are detached from autograd. To
        compute a supervised BT loss on ``S⁺ − S⁻`` we need a fresh forward
        whose logits flow back to the policy params. This mirrors the trick
        already used by score-PG (``compute_log_probs(detach_prompt=True)``)
        — we cache the long video-token prefix under ``no_grad`` and only
        backprop through the short sampled tail.

        Returns
        -------
        ``(K_valid, 5)`` float tensor where ``K_valid == len(digit_token_offsets)``,
        rows are the 5 likert-digit logits at each offset, REQUIRES_GRAD on
        the policy path. The trainer then applies ``softmax`` and the
        digit-value vector ``[1,2,3,4,5]`` to obtain expected scores.

        Notes
        -----
        * ``digit_token_offsets`` are offsets into ``sampled_ids`` (0-based).
          Empty input → empty tensor (caller should mask out).
        * ``detach_prompt=True`` by default since BT-aux is only meaningful
          on the policy (not on a frozen ref) and the prompt is the same
          ~6000-token video prefix that score-PG already chose to detach.
        * ``_skip_summon`` semantics match ``compute_log_probs``: when True,
          the caller has already opened the ZeRO-3 gather scope.
        """
        import torch
        if self._dummy or self._model is None or not digit_token_offsets:
            return torch.zeros(0, 5)

        # Resolve digit token ids (5-vector). ``_score_digit_id_groups`` may
        # have multiple variants per digit; we use the FIRST id of each
        # group (matches the ``ids_bare`` printed at startup).
        groups = self._score_digit_id_groups
        if not groups or len(groups) != 5:
            return torch.zeros(0, 5)
        digit_ids = [grp[0] for grp in groups]    # length 5
        device    = self._model.device
        digit_ids_t = torch.tensor(digit_ids, device=device, dtype=torch.long)

        T = sampled_ids.shape[0]
        P = prompt_ids.shape[0]
        use_detach = bool(detach_prompt) and T > 0 and P >= 2

        # Filter offsets to the [0, T) window — defensive; the score path
        # already enforced this when packing the meta.
        offsets = [int(o) for o in digit_token_offsets if 0 <= int(o) < T]
        if not offsets:
            return torch.zeros(0, 5, device=device)

        target_model = self._model
        if _skip_summon:
            summon_ctx = _NullCtx()
        else:
            summon_ctx = _summon_full_params_ctx(target_model, writeback=True)

        # use_cache toggle (same dance as compute_log_probs).
        inner_for_cfg = getattr(target_model, "module", target_model)
        cfg_obj = getattr(inner_for_cfg, "config", None)
        if not hasattr(cfg_obj, "use_cache"):
            cfg_obj = None
        prev_cache = None
        toggled_cache = False
        if cfg_obj is not None and self.cfg.use_gradient_checkpointing:
            prev_cache = getattr(cfg_obj, "use_cache", None)
            cfg_obj.use_cache = False
            toggled_cache = True

        try:
            with summon_ctx:
                if use_detach:
                    # Mirror _compute_log_probs_detach_prompt's tail forward
                    # but keep the FULL logits tensor instead of gathering on
                    # sampled_ids — we need the 5-digit slice instead.
                    prompt_ids_d  = prompt_ids.to(device)
                    sampled_ids_d = sampled_ids.to(device).long()
                    # ── Patch 6c (2026-06-16): inject mm tensors into the
                    # PROMPT prefix forward so the vision tower runs and the
                    # cached past_kv encodes visual context. Without this
                    # the BT-aux supervision would be vision-blind.
                    prefix_mm = _normalize_mm_kwargs(
                        mm_kwargs, target_device=device,
                    )
                    was_training  = bool(getattr(target_model, "training", False))
                    if was_training:
                        target_model.eval()
                    try:
                        prompt_prefix = prompt_ids_d[:P - 1].unsqueeze(0)
                        attn_prefix   = torch.ones_like(prompt_prefix)
                        with torch.no_grad():
                            prefix_out = target_model(
                                input_ids=prompt_prefix,
                                attention_mask=attn_prefix,
                                use_cache=True,
                                **prefix_mm,
                            )
                        past_kv = getattr(prefix_out, "past_key_values", None)
                        if past_kv is None:
                            # Fallback: full grad-on forward.
                            full_local = torch.cat(
                                [prompt_ids_d, sampled_ids_d], dim=0
                            ).unsqueeze(0)
                            attn_full  = torch.ones_like(full_local)
                            with torch.enable_grad():
                                out = target_model(
                                    input_ids=full_local,
                                    attention_mask=attn_full,
                                    use_cache=False,
                                    **prefix_mm,
                                )
                            logits_full = out.logits[0]                # (P+T, V)
                            # logits at position i predict token i+1, so
                            # logits at sampled token offset ``o`` (predicting
                            # sampled_ids[o]) sit at absolute index P+o-1.
                            # We want the logits AT the digit's own position
                            # (i.e. those used during sampling to PRODUCE that
                            # digit) — that is logits[P+o-1].
                            sel_rows = logits_full[
                                torch.tensor([P + o - 1 for o in offsets],
                                             device=device, dtype=torch.long)
                            ]                                           # (K, V)
                        else:
                            tail_ids = torch.cat(
                                [prompt_ids_d[P - 1:P], sampled_ids_d],
                                dim=0,
                            ).unsqueeze(0)                              # (1, T+1)
                            attn_full = torch.ones(
                                (1, (P - 1) + (T + 1)),
                                dtype=prompt_prefix.dtype, device=device,
                            )
                            with torch.enable_grad():
                                tail_out = target_model(
                                    input_ids=tail_ids,
                                    attention_mask=attn_full,
                                    past_key_values=past_kv,
                                    use_cache=False,
                                )
                            logits_tail = tail_out.logits[0]            # (T+1, V)
                            # tail position 0 corresponds to absolute position
                            # P-1 (the prompt's last token), and tail[t]
                            # predicts sampled_ids[t] (for t in 0..T-1). So
                            # logits AT the digit (i.e. predicting it) live
                            # at tail index ``o`` for offset ``o``.
                            sel_rows = logits_tail[
                                torch.tensor(offsets, device=device,
                                             dtype=torch.long)
                            ]                                           # (K, V)
                    finally:
                        if was_training:
                            target_model.train()
                else:
                    # Legacy single forward, grad-on.
                    full = torch.cat(
                        [prompt_ids.to(device),
                         sampled_ids.to(device).long()], dim=0
                    ).unsqueeze(0)
                    attn = torch.ones_like(full)
                    fwd_kwargs = {"input_ids": full, "attention_mask": attn}
                    if self.cfg.use_gradient_checkpointing:
                        fwd_kwargs["use_cache"] = False
                    fwd_kwargs.update(_normalize_mm_kwargs(
                        mm_kwargs, target_device=device,
                    ))
                    with torch.enable_grad():
                        out = target_model(**fwd_kwargs)
                    logits_full = out.logits[0]                         # (P+T, V)
                    sel_rows = logits_full[
                        torch.tensor([P + o - 1 for o in offsets],
                                     device=device, dtype=torch.long)
                    ]                                                   # (K, V)

                # Project onto the 5 likert digits.
                digit_logits = sel_rows.index_select(-1, digit_ids_t).float()
                return digit_logits                                     # (K, 5)
        finally:
            if toggled_cache and cfg_obj is not None:
                cfg_obj.use_cache = prev_cache

    # ------------------------------------------------------------------
    # Trick A helper: prompt-detach + sampled-only backward.
    # ------------------------------------------------------------------
    def _compute_log_probs_detach_prompt(
        self,
        *,
        target_model,
        prompt_ids,
        sampled_ids,
        T: int,
        P: int,
        no_grad_tail: bool = False,
        mm_kwargs: dict | None = None,
    ):
        """Two-stage forward: no_grad prompt prefill → grad-on (or grad-off) sampled tail.

        Splits ``prompt_ids`` (length ``P``) into a ``P-1`` prefix consumed
        under ``no_grad`` to build the K/V cache, then forwards a single
        ``T+1`` token tail (``[prompt[-1]] + sampled``) against that cache.
        Returns the same ``(T,)`` tok_logp tensor as the legacy
        single-forward path.

        ``no_grad_tail`` (added 2026-06-16)
        -----------------------------------
        When True, the tail forward also runs under ``no_grad``. This is the
        correct mode for the REF policy: the ref is frozen, so building an
        autograd graph on the tail would be wasted memory. Used by
        ``compute_log_probs(ref=True, detach_prompt=True)`` to make the ref
        forward numerically identical to the policy forward (same eval
        mode, same KV-cached tail, same attention mask shape) so KL_s ≈ 0
        at step 0 when student == ref. Without this flag the tail would
        run under ``enable_grad`` and Qwen2.5-VL's ``no_grad`` context
        from the caller would be silently overridden.

        Memory win
        ----------
        The legacy path keeps the full (P+T) activation graph alive
        because backward needs to reach every layer's hidden states for
        the prompt tokens too — even though the prompt is a *condition*
        not a target, autograd doesn't know that and saves activations
        for every position in the sequence. With Trick A the activation
        graph covers only the ``T+1`` tail, while the K/V cache (no
        autograd) carries the prompt context.

        For score forwards in this codebase, ``P`` ≈ 6000 (video tokens)
        and ``T`` ≈ 32 (score scaffold). The activation graph shrinks
        roughly ``(P+T)/(T+1) ≈ 180×``. Real-world memory savings are
        smaller (ZeRO-3 grad bucket and KV cache itself don't shrink),
        but the activation peak which has been the OOM trigger does.

        Mathematical equivalence to the legacy path
        -------------------------------------------
        At sampled position ``t``, the legacy path computes:
            logp[t] = log P( sampled[t] | prompt + sampled[:t] )
                    = log_softmax( logits[ P + t - 1 ] )[ sampled[t] ]
                          # logits[P+t-1] sits at input position P+t-1
                          # (last is sampled[t-1]), predicting token P+t.

        With Trick A:
            tail = [ prompt[-1], sampled[0], ..., sampled[T-1] ]    # length T+1
            past_kv = no_grad forward over prompt[:-1]              # cached
            logits_tail = forward(tail, past_kv)                    # (T+1, V)
            logits_tail[t] sits at position (P-1)+t in the absolute index,
            so it predicts the token after position (P-1)+t — which is
            prompt[0+1]=prompt[0]? No: the prompt's last token IS at
            absolute position P-1 (0-indexed), so logits_tail[0] predicts
            absolute position P, i.e. sampled[0]. Mapping:
                logits_tail[ 0 ]  predicts sampled[0]
                logits_tail[ t ]  predicts sampled[t]    (for t in 0..T-1)
                logits_tail[ T ]  predicts the next-after-sampled token
                                  (unused; we drop this slice).
            So tok_logp[t] = log_softmax( logits_tail[t] )[ sampled[t] ]
                            for t in 0..T-1.

        Numerically identical to the legacy ``logits[-(T+1):-1, :]`` slice
        (within float reordering tolerance).

        Eval-mode toggle
        ----------------
        HF's transformer layers force ``use_cache=False`` whenever
        ``self.training and self.gradient_checkpointing`` are both True.
        We need the cache, so we flip the model to ``eval()`` for the
        prompt forward (and keep it in eval for the sampled forward, no
        reason to re-enable grad-ckpt on a 32-token tail). Qwen2.5-VL has
        no BatchNorm or train-only Dropout that change forward outputs,
        so eval ≡ train numerically; only the grad-ckpt accounting
        changes.
        """
        import torch

        device = target_model.device
        prompt_ids_d = prompt_ids.to(device)
        sampled_ids_d = sampled_ids.to(device).long()

        # Save and clear the train-mode flag for the inner module so the
        # transformer's per-layer ``use_cache=False`` override does NOT
        # fire. ``module.training`` is a recursive flag; we flip via
        # ``model.eval()`` and restore via ``train(prev_train)`` after.
        # For DeepSpeedEngine wrappers, ``.eval()`` propagates through
        # to the inner module (engine forwards train/eval calls).
        was_training = bool(getattr(target_model, "training", False))
        try:
            if was_training:
                target_model.eval()

            # ── Stage 1: prompt[:-1] forward, no_grad, kv-cache on.
            prompt_prefix = prompt_ids_d[:P - 1].unsqueeze(0)          # (1, P-1)
            attn_prefix   = torch.ones_like(prompt_prefix)
            # ── Patch 6b (2026-06-16): inject mm tensors into the PROMPT
            # prefix forward. Video placeholder tokens live in the prompt;
            # without pixel_values_videos the vision tower is skipped and
            # the prefix's hidden states (cached as past_key_values for
            # the tail) reflect language-only context. The tail forward
            # below contains only the score scaffold + sampled digits —
            # no video tokens — so mm doesn't need to be re-injected
            # there; the past_kv cache already encodes the visual
            # context.
            prefix_mm = _normalize_mm_kwargs(
                mm_kwargs, target_device=device,
            )
            with torch.no_grad():
                prefix_out = target_model(
                    input_ids       = prompt_prefix,
                    attention_mask  = attn_prefix,
                    use_cache       = True,
                    **prefix_mm,
                )
            past_kv = getattr(prefix_out, "past_key_values", None)
            if past_kv is None:
                # Engine refused to return KV cache (e.g. some grad-ckpt
                # paths in older HF). Fall back to the legacy single
                # forward; caller pays the activation cost but doesn't
                # crash. This branch is also exercised by ``ref=True``
                # callers — though those skip Trick A upstream.
                full_local  = torch.cat(
                    [prompt_ids_d, sampled_ids_d], dim=0).unsqueeze(0)
                attn_full   = torch.ones_like(full_local)
                fallback_kw = {"input_ids": full_local,
                               "attention_mask": attn_full,
                               "use_cache": False}
                fallback_kw.update(prefix_mm)
                # Mirror the tail-grad mode for the fallback so ref vs
                # policy stay numerically aligned. (HF model forward is
                # eager either way; the no_grad context just avoids
                # building a useless autograd graph for the ref.)
                fallback_grad_ctx = (
                    torch.no_grad() if no_grad_tail else torch.enable_grad()
                )
                with fallback_grad_ctx:
                    out = target_model(**fallback_kw)
                logits = out.logits[0]                                 # (P+T, V)
                shift  = logits[-(T + 1):-1, :].float()
                log_probs = torch.log_softmax(shift, dim=-1)
                return log_probs.gather(
                    -1, sampled_ids_d.unsqueeze(-1)).squeeze(-1)

            # ── Stage 2: tail forward = [prompt[-1]] ++ sampled.
            # Grad mode depends on caller: policy → enable_grad (build the
            # autograd graph the trainer will backprop through); ref →
            # no_grad (frozen weights, no backprop wanted, save activation
            # memory). This is the key correctness fix from Patch 5b
            # (2026-06-16): when ref takes the SAME forward route as
            # policy, kl_s ≈ 0 at step 0 — restoring the textbook PPO/GRPO
            # invariant.
            tail_ids = torch.cat(
                [prompt_ids_d[P - 1:P], sampled_ids_d], dim=0
            ).unsqueeze(0)                                             # (1, T+1)
            attn_full = torch.ones(
                (1, (P - 1) + (T + 1)),
                dtype=prompt_prefix.dtype, device=device,
            )
            tail_grad_ctx = (
                torch.no_grad() if no_grad_tail else torch.enable_grad()
            )
            with tail_grad_ctx:
                tail_out = target_model(
                    input_ids       = tail_ids,
                    attention_mask  = attn_full,
                    past_key_values = past_kv,
                    use_cache       = False,
                )
            logits_tail = tail_out.logits[0]                           # (T+1, V)
            # Drop the last position: it predicts the token after
            # sampled[T-1], which we don't have a target for.
            shift = logits_tail[:-1, :].float()                        # (T, V)
            log_probs = torch.log_softmax(shift, dim=-1)
            return log_probs.gather(
                -1, sampled_ids_d.unsqueeze(-1)).squeeze(-1)
        finally:
            if was_training:
                target_model.train()

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
        except ImportError as e:                                  # pragma: no cover
            raise RuntimeError(
                "qwen_vl_utils not installed — pip install qwen-vl-utils"
            ) from e
        try:
            image_inputs, video_inputs = process_vision_info(msgs)
        except Exception as e:
            # ffmpeg/decord raised a Python exception (truncated file,
            # corrupt header, unsupported codec, …). Bubble up as a
            # typed error so the trainer can skip this sample. SIGSEGVs
            # are NOT caught here — see ``scripts/scan_bad_videos.py``.
            raise BadVideoError(
                f"process_vision_info failed for {video_path}: "
                f"{type(e).__name__}: {e}"
            ) from e
        inputs = self._processor(
            text=[chat],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)
        budget = max_new_tokens if max_new_tokens is not None else self.cfg.max_new_tokens
        # Score forward is no_grad (it reads logits, not gradients), so
        # we can safely gather ZeRO-3-sharded params for the duration.
        # We also pause gradient_checkpointing — see the comment block in
        # ``_generate_text_only`` for the NaN/Inf logits interaction.
        with _pause_grad_ckpt(
                self._model,
                use_reentrant=self.cfg.grad_checkpoint_use_reentrant,
             ), _summon_full_params_ctx(self._model), torch.no_grad():
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
        """Single-prompt softlogits score. Thin wrapper over the batched path."""
        out = self._score_video_softlogits_batch(
            [prompt], video_path, [n_themes], force_greedy=force_greedy,
        )
        return out[0]

    def _score_video_softlogits_batch(
        self,
        prompts:    list[str],
        video_path: str,
        n_themes_per: list[int],
        *,
        force_greedy: bool = True,
        return_meta:  bool = False,
    ) -> list[tuple[str, list[float | None]]] | list[tuple[str, list[float | None], "RubricRollout | None"]]:
        """Score ONE video against G different prompts in batched forwards.

        Why
        ---
        ``step_one_instance`` was calling ``score_video`` G×2 times — once
        per (rubric, side). Each call re-paid:
          * the video-frame decode (~16 frames) + processor work,
          * the prefill on a long multimodal prompt (~700 text tokens
            plus the visual tokens),
          * the ZeRO-3 gather + grad-ckpt-pause ceremony.
        The video itself is identical across the G rubrics for a given
        side, and the visual-token prefix in the prompt is identical too.
        Batching the G rubrics for one side into ONE generate gets us:
          * 1 video decode (we share the frames across the batch dim),
          * 1 prefill (KV-cache batched B=mb),
          * 1 gather/scatter cycle.

        Memory caveat — micro-batching
        ------------------------------
        A naive B=G batched generate holds B copies of the KV cache for
        the FULL multimodal prefix (visual tokens dominate; ~1.5K-2K
        tokens at fps=2.0 + 360x420). With G=8 and 7B params on H100
        80GB this OOMs after the policy backward has already eaten most
        of the activation headroom. We chunk into micro-batches of size
        ``score_micro_batch`` (default 4, override via env
        ``RM_SCORE_MB``). Each micro-batch is one batched generate; we
        concatenate the per-row results.

        Padding caveat — decoder-only requires LEFT-pad
        -----------------------------------------------
        Decoder-only models (Qwen2.5-VL is one) MUST be left-padded for
        ``generate`` — right-pad makes the model start decoding from PAD
        tokens on shorter rows, which scrambles the output. We flip the
        tokenizer's ``padding_side`` to "left" around the processor
        call and restore it on exit.

        ``return_meta`` (default False)
        -------------------------------
        When True AND ``not force_greedy`` (i.e. we're sampling, not
        greedy decoding), each row gets a third tuple element: a
        ``RubricRollout``-shaped struct carrying
        ``(prompt_ids, sampled_ids)`` so the trainer can teacher-force a
        gradient-on log-prob over the sampled scaffold tokens for the
        score-side PG update. With ``force_greedy=True`` the meta is
        ``None`` (no exploration → no PG signal to compute). Greedy
        scoring is left untouched in the legacy two-element shape.

        We capture ``prompt_ids`` AS THE UN-PADDED PREFIX (post-multimodal
        chat-template tokenisation, before generation) so the downstream
        ``compute_log_probs`` doesn't accidentally score the left-pad
        tokens. ``sampled_ids`` is the row's generated suffix with EOS
        and trailing pad/eos tokens stripped.
        """
        import torch
        from qwen_vl_utils import process_vision_info  # type: ignore

        if len(prompts) != len(n_themes_per):
            raise ValueError("prompts and n_themes_per must have the same length")
        B_total = len(prompts)
        if B_total == 0:
            return []
        # ``return_meta=True`` + greedy is a contradiction (no gradients to
        # collect from a deterministic forward) — silently downgrade.
        meta_active = bool(return_meta) and not bool(force_greedy)

        # Decode the video ONCE; replicate the (image, video) tensors per
        # micro-batch row so the processor's batched path treats every row
        # as looking at the same clip.
        ref_msgs = [{
            "role": "user",
            "content": [
                {"type": "video", "video": str(video_path),
                 "fps": self.cfg.fps, "max_pixels": 360 * 420},
                {"type": "text",  "text": prompts[0]},
            ],
        }]
        try:
            image_inputs_one, video_inputs_one = process_vision_info(ref_msgs)
        except Exception as e:
            # See BadVideoError docstring at top of module for the
            # SIGSEGV-vs-Python-exception distinction.
            raise BadVideoError(
                f"process_vision_info failed for {video_path}: "
                f"{type(e).__name__}: {e}"
            ) from e

        # Pre-build all B chat strings (different prompt text, identical
        # video prefix). We split into micro-batches AFTER this so all
        # rows share the same vision inputs.
        chat_texts: list[str] = []
        for p in prompts:
            msgs = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path),
                     "fps": self.cfg.fps, "max_pixels": 360 * 420},
                    {"type": "text",  "text": p},
                ],
            }]
            chat_texts.append(self._processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            ))

        # ── Prompt-dump diagnostic. Set RM_SCORE_DUMP_PROMPT=<path> in the
        # env to write the FIRST chat_text emitted by this wrapper to disk
        # and exit early on next call (idempotent: subsequent calls see
        # the file exists and are no-ops). Use to capture the actual
        # train-time score prompt — task_description, rubric_text, Output
        # primer, all of it — and diff it against the probe v8 prompt to
        # find prompt-shape divergences. The chat_text already contains
        # the full multimodal-formatted prompt minus the visual tokens
        # themselves (those come in via processor() at micro-batch time);
        # for prompt-shape diffing the text part is what matters.
        _dump_path = os.environ.get("RM_SCORE_DUMP_PROMPT", "")
        if _dump_path and chat_texts:
            try:
                from pathlib import Path as _P
                _dp = _P(_dump_path)
                if not _dp.exists():
                    _dp.parent.mkdir(parents=True, exist_ok=True)
                    _dp.write_text(chat_texts[0], encoding="utf-8")
                    print(f"[score_dump] wrote first chat_text "
                          f"(len={len(chat_texts[0])}) to {_dump_path}",
                          flush=True)
            except Exception as _e:
                print(f"[score_dump] WARN failed to write {_dump_path}: {_e}",
                      flush=True)

        # Resolve micro-batch size. Default 4; overridable via env.
        try:
            mb_default = int(os.environ.get("RM_SCORE_MB", "4"))
        except ValueError:
            mb_default = 4
        mb = max(1, min(B_total, mb_default))

        # Build flat-id tensors once (shared across micro-batches).
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

        results: list[tuple[str, list[float | None]]] = []

        # Save and force the tokenizer's padding side to "left" for the
        # whole batched run; restore on exit. The processor's tokenizer
        # is the one used for text padding inside the multimodal call.
        prev_pad_side = getattr(tok, "padding_side", "right")
        tok.padding_side = "left"
        # Some processors keep their own tokenizer reference — flip both
        # if they happen to differ.
        proc_tok = getattr(self._processor, "tokenizer", None)
        prev_proc_pad_side = None
        if proc_tok is not None and proc_tok is not tok:
            prev_proc_pad_side = getattr(proc_tok, "padding_side", "right")
            proc_tok.padding_side = "left"

        try:
            for start in range(0, B_total, mb):
                end = min(start + mb, B_total)
                B   = end - start
                chunk_texts = chat_texts[start:end]

                # Replicate vision inputs to match this micro-batch size.
                if video_inputs_one:
                    videos_b = [video_inputs_one[0]] * B if len(video_inputs_one) == 1 \
                               else (list(video_inputs_one) * B)
                else:
                    videos_b = None
                images_b = None
                if image_inputs_one:
                    images_b = [image_inputs_one[0]] * B if len(image_inputs_one) == 1 \
                               else (list(image_inputs_one) * B)

                inputs = self._processor(
                    text=chunk_texts,
                    images=images_b,
                    videos=videos_b,
                    padding=True,
                    return_tensors="pt",
                ).to(self._model.device)

                budget = self.cfg.score_max_new_tokens
                with _pause_grad_ckpt(
                        self._model,
                        use_reentrant=self.cfg.grad_checkpoint_use_reentrant,
                     ), _summon_full_params_ctx(self._model), torch.no_grad():
                    gen_out = self._model.generate(
                        **inputs,
                        max_new_tokens=budget,
                        do_sample=not force_greedy,
                        temperature=(self.cfg.temperature if not force_greedy else 1.0),
                        top_p=(self.cfg.top_p if not force_greedy else 1.0),
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                # With left-padding, every row's prompt-end aligns to the
                # same column = input_ids.shape[1]. Slicing past it gives
                # only the generated suffix.
                prompt_len  = inputs["input_ids"].shape[1]
                trimmed_all = gen_out.sequences[:, prompt_len:]
                step_scores = gen_out.scores or ()

                if flat_ids and step_scores:
                    device = step_scores[0].device
                    flat_ids_t    = torch.tensor(flat_ids,    device=device, dtype=torch.long)
                    flat_values_t = torch.tensor(flat_values, device=device, dtype=torch.float32)
                else:
                    flat_ids_t    = None
                    flat_values_t = None

                # Per-row prompt prefix (un-padded). Under left-pad the
                # padding_mask of the row tells us where the real tokens
                # start in the (B, P_padded) tensor; we also need it later
                # to slice prompt_ids out of the full sequence so
                # ``compute_log_probs`` doesn't try to score the pad slots.
                #
                # Strategy: per-row, find the first non-pad column in
                # ``inputs.input_ids[b]``; the prompt is that slice up to
                # ``prompt_len`` (which is the right edge for every row
                # since we're left-padding).
                pad_id_for_prompt = tok.pad_token_id
                if pad_id_for_prompt is None:
                    pad_id_for_prompt = tok.eos_token_id  # safety fallback

                for b in range(B):
                    n_themes = int(n_themes_per[start + b])
                    gen_ids  = trimmed_all[b].tolist()
                    text     = self._processor.tokenizer.decode(
                        gen_ids, skip_special_tokens=True,
                    ).strip()

                    # ── Score-side PG meta capture (only when sampling).
                    # Strip trailing pad/eos from ``gen_ids`` so backprop
                    # through compute_log_probs scores actual generated
                    # tokens, not the pad-fill that ``generate`` adds when
                    # one row in the micro-batch hits EOS earlier than
                    # the budget. Build a fresh torch.Tensor (1-D, long)
                    # for both prompt and sampled.
                    score_meta = None
                    if meta_active:
                        # Sampled: drop trailing pad/eos.
                        sampled_trimmed = list(gen_ids)
                        while (sampled_trimmed and
                               (sampled_trimmed[-1] == pad_id_for_prompt
                                or sampled_trimmed[-1] == tok.eos_token_id)):
                            sampled_trimmed.pop()
                        if sampled_trimmed:
                            # Prompt: per-row strip of the left-pad prefix.
                            prompt_row = inputs["input_ids"][b]   # (P_padded,)
                            mask = prompt_row != pad_id_for_prompt
                            # Find first non-pad position; fall back to 0
                            # if the row is entirely pads (shouldn't happen).
                            nonzero = mask.nonzero(as_tuple=False)
                            first = int(nonzero[0].item()) if nonzero.numel() else 0
                            prompt_unpadded = prompt_row[first:].detach().clone()
                            sampled_t = torch.tensor(
                                sampled_trimmed,
                                device=prompt_unpadded.device,
                                dtype=prompt_unpadded.dtype,
                            )
                            # ── Patch 6a (2026-06-16): save per-row mm tensors.
                            # The grad-on re-forward (compute_log_probs /
                            # compute_digit_logits) needs these to invoke
                            # the vision tower; without them the re-forward
                            # computes language-only logp and the GRPO
                            # ratio + BT-aux become vision-blind.
                            #
                            # Slicing: we share one video across all B
                            # rows in this micro-batch, but the processor
                            # has already replicated it B times in
                            # ``pixel_values_videos`` (sequential blocks
                            # along dim 0). For row b, the block size is
                            # the row's vision-token count, derivable
                            # from video_grid_thw[b]. Slice cleanly so a
                            # row=1 re-forward sees exactly ONE video.
                            mm_pv = None
                            mm_grid = None
                            mm_seconds = None
                            pvv = inputs.get("pixel_values_videos", None)
                            vgt = inputs.get("video_grid_thw", None)
                            spgt = inputs.get("second_per_grid_ts", None)
                            if pvv is not None and vgt is not None:
                                # video_grid_thw shape: (B, 3).
                                # Per-row block length on dim 0 of
                                # pixel_values_videos = t * h * w (in
                                # patch units, matching the processor's
                                # chunking convention).
                                vgt_b = vgt[b]                       # (3,)
                                t_, h_, w_ = (int(vgt_b[0].item()),
                                              int(vgt_b[1].item()),
                                              int(vgt_b[2].item()))
                                block_len = t_ * h_ * w_
                                # Compute this row's block start by summing
                                # block_len of all rows before b.
                                block_start = 0
                                for bb in range(b):
                                    vgt_bb = vgt[bb]
                                    block_start += (
                                        int(vgt_bb[0].item())
                                        * int(vgt_bb[1].item())
                                        * int(vgt_bb[2].item())
                                    )
                                mm_pv = pvv[block_start:block_start + block_len].detach().clone()
                                mm_grid = vgt_b.unsqueeze(0).detach().clone()  # (1, 3)
                                if spgt is not None:
                                    # second_per_grid_ts is a list-like;
                                    # one entry per row.
                                    try:
                                        mm_seconds = [float(spgt[b])]
                                    except Exception:
                                        mm_seconds = None
                            score_meta = RubricRollout(
                                text=text,
                                prompt_ids=prompt_unpadded,
                                sampled_ids=sampled_t,
                                pixel_values_videos=mm_pv,
                                video_grid_thw=mm_grid,
                                second_per_grid_ts=mm_seconds,
                            )

                    if flat_ids_t is None:
                        if return_meta:
                            results.append((text, [None] * n_themes, score_meta))
                        else:
                            results.append((text, [None] * n_themes))
                        continue

                    # Walk for colon-anchored digit hits (per-row).
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

                    scores: list[float | None] = []
                    # ── BT-aux capture (added 2026-06-16): record the
                    # SAMPLED-IDS OFFSET of every successful colon-anchored
                    # digit hit, plus a per-theme bool mask. The trainer will
                    # re-run a grad-on forward over (prompt_ids, sampled_ids)
                    # and slice 5-digit logits at these offsets; we cannot
                    # use the no_grad logits captured here because
                    # model.generate ran under torch.no_grad.
                    bt_offsets: list[int] = []
                    bt_mask: list[bool] = []
                    # ``hits`` are positions in gen_ids; the meta path strips
                    # trailing pad/eos to form ``sampled_trimmed`` — the
                    # offset into sampled_trimmed equals the position in
                    # gen_ids as long as that position is < len(sampled_trimmed).
                    sampled_trim_len = (
                        len(score_meta.sampled_ids)
                        if (score_meta is not None
                            and score_meta.sampled_ids is not None)
                        else 0
                    )
                    # ── DIAG (RM_SCORE_DIAG=1): when no anchor was found OR
                    # logits are NaN, dump per-row state so we can tell on
                    # the 8-GPU box whether the failure is "model produced
                    # no scaffold" or "logits came back NaN/Inf". One-line
                    # per-failure dump → grep-able in train_grpo log.
                    diag_on = os.environ.get("RM_SCORE_DIAG", "0") == "1"
                    diag_logits_nan_count = 0
                    for k in range(n_themes):
                        if k >= len(hits):
                            scores.append(None)
                            bt_mask.append(False)
                            continue
                        pos = hits[k]
                        if pos >= len(step_scores):
                            scores.append(None)
                            bt_mask.append(False)
                            continue
                        # step_scores[pos] is shape (B, V) in batched mode.
                        logits = step_scores[pos][b]
                        sel    = logits.index_select(0, flat_ids_t).float()
                        if diag_on:
                            try:
                                if not bool(torch.isfinite(sel).all().item()):
                                    diag_logits_nan_count += 1
                            except Exception:
                                pass
                        probs  = torch.softmax(sel, dim=-1)
                        ev     = float((probs * flat_values_t).sum().item())
                        # ── BUG-FIX (2026-06-11): math.{min,max}(NaN, x)
                        # silently returns x, so NaN logits used to be
                        # clipped to 5.0 and treated as a valid finite
                        # score. That contaminated r_pref and inflated
                        # ``valid_count``. Detect NaN explicitly and
                        # report None so format_score / valid_filter
                        # correctly drop the rollout.
                        if not math.isfinite(ev):
                            scores.append(None)
                            bt_mask.append(False)
                        else:
                            ev = max(1.0, min(5.0, ev))
                            scores.append(ev)
                            # BT-aux: record the offset only if it falls
                            # inside the sampled tail kept on the meta. A
                            # digit at pos beyond sampled_trim_len would
                            # mean the trainer's re-forward couldn't reach
                            # that slot — drop it from the BT loss.
                            if pos < sampled_trim_len:
                                bt_offsets.append(int(pos))
                                bt_mask.append(True)
                            else:
                                bt_mask.append(False)
                    # ── FALLBACK (2026-06-11): when soft_logits could not
                    # recover ≥1 score (anchor walk missed a theme_k:
                    # boundary, OR logits came back NaN), retry with the
                    # plain-text positional digit scrape — handles
                    # abbreviations like "4:3:4:4:4" that 7B Qwen
                    # sometimes emits in place of the explicit scaffold.
                    if sum(1 for s in scores
                           if s is not None and math.isfinite(s)) < n_themes:
                        from .rubric_parser import parse_score_scaffold
                        fallback = parse_score_scaffold(text, n_themes)
                        for k in range(n_themes):
                            sk = scores[k]
                            if (sk is None or not math.isfinite(sk)) and \
                               fallback[k] is not None:
                                scores[k] = float(fallback[k])
                    if diag_on:
                        finite_count = sum(1 for s in scores
                                           if s is not None and isinstance(s, float)
                                           and math.isfinite(s))
                        if finite_count < n_themes:
                            try:
                                rank = int(os.environ.get("LOCAL_RANK", "0"))
                            except Exception:
                                rank = 0
                            print(
                                f"[score_diag rank={rank}] hits={len(hits)}/{n_themes} "
                                f"finite={finite_count}/{n_themes} "
                                f"logits_nan_anchors={diag_logits_nan_count} "
                                f"text_len={len(text)} "
                                f"text_repr={text[:120]!r}",
                                file=sys.stderr, flush=True,
                            )
                    if return_meta:
                        # ── BT-aux: attach offsets + mask to score_meta. The
                        # trainer reads these inside _compute_grpo_loss to
                        # slice digit logits from a fresh grad-on forward
                        # over (prompt_ids, sampled_ids).
                        if score_meta is not None and bt_offsets:
                            score_meta.digit_token_offsets = list(bt_offsets)
                            score_meta.digit_mask = torch.tensor(
                                bt_mask,
                                dtype=torch.bool,
                                device=score_meta.sampled_ids.device,
                            )
                        results.append((text, scores, score_meta))
                    else:
                        results.append((text, scores))
        finally:
            tok.padding_side = prev_pad_side
            if proc_tok is not None and proc_tok is not tok and prev_proc_pad_side is not None:
                proc_tok.padding_side = prev_proc_pad_side

        return results

    # ------------------------------------------------------------------
    # Dummy emissions — deterministic enough for tests, varied enough
    # to exercise the parser/reward path.
    # ------------------------------------------------------------------
    _DUMMY_THEMES = [
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
        themes = list(self._DUMMY_THEMES)
        rng.shuffle(themes)
        # Random-but-summing-to-1 weights.
        ws = [rng.uniform(0.5, 1.5) for _ in themes]
        s  = sum(ws)
        ws = [round(w / s, 4) for w in ws]
        # Fix rounding drift on the last weight.
        ws[-1] = round(1.0 - sum(ws[:-1]), 4)
        rubrics = [
            {"theme": t, "weight": w, "tips": ["dummy tip"]}
            for t, w in zip(themes, ws)
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
