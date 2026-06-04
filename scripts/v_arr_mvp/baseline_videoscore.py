"""RewardVerse — VideoScore-v1.1 baseline runner (TIGER-Lab/VideoScore-v1.1).

This is a thin wrapper around the official VideoScore-v1.1 inference flow
(``Mantis-Idefics2-8B`` fine-tune that emits five real-valued sub-scores
in [1.0, 4.0] for a generated video given its text prompt). We use it as a
"strong open-source VLM reward model" baseline against the RewardVerse
rubric pipeline.

VideoScore-v1.1 outputs five sub-scores per (prompt, video):

    1. visual quality
    2. temporal consistency
    3. dynamic degree
    4. text-to-video alignment
    5. factual consistency

We map them to the 18-dim ARR L2 / 7-dim ARR L1 axes for downstream
aggregation:

    visual quality       -> aesthetics  (and visual_quality, lighting,
                                          materiality, chromaticity in L2)
    temporal consistency -> acting:consistency / multi_shot:logic / pacing
    dynamic degree       -> acting:action / pacing / cinematography
    t2v alignment        -> visual_concept_design (character, scene),
                            cinematography:composition
    factual consistency  -> visual_concept_design + acting (composite)

Downstream comparison tables handle that mapping at column-merge time;
this script just emits the raw 5-dim and an overall (mean of the 5
sub-scores).

Inference parameters mirror the v1.1 README: 48 uniformly sampled frames,
float16 on CUDA, autoregressive greedy decoding.

Usage::

    python scripts/v_arr_mvp/baseline_videoscore.py \\
        --dataset data/v_arr_mvp/visual_quality_mvp/shared_test_pointwise.json \\
        --output runs/baselines/videoscore/shared_test.json \\
        --model_path TIGER-Lab/VideoScore-v1.1 \\
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_assets import (  # type: ignore  # noqa: E402
    ensure_model_local,
    resolve_local_model_dir,
)


# Default HF repo id; the actual on-disk path comes from local_assets.
_VS1_REPO_ID = "TIGER-Lab/VideoScore-v1.1"


# ---------------------------------------------------------------------------
# Pointwise metric helpers (independent copy so this file can run on a GPU
# node without pulling the full ARR judger graph).
# ---------------------------------------------------------------------------
def _ranks_with_ties(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _compute_metrics(preds: list[float], labels: list[float],
                     *, mae_mode: str = "raw"):
    """Pointwise PLCC / SRCC / MAE.

    Args:
        preds, labels: equal-length sequences of floats.
        mae_mode:
            ``"raw"`` — MAE is computed on the raw values as given.
            ``"zscore"`` — both ``preds`` and ``labels`` are z-score
                normalized first, then MAE is computed on the normalized
                series. Use this when ``preds`` and ``labels`` live on
                different scales (e.g. raw VideoScore logits centered
                near 0 vs. ARR labels in [1, 5]); PLCC/SRCC are
                scale-invariant so they're unaffected, but MAE on raw
                logits is meaningless across baselines.
            ``"affine"`` — fit ``a * pred + b`` minimizing MSE to labels
                (a single global linear calibration), then MAE on the
                calibrated preds. This is the cleanest comparison when
                you want to know "how well does the predictor's
                relative ordering correspond to label scale" without
                being penalized by a constant scale offset.
    """
    if not preds or not labels or len(preds) != len(labels):
        return None, None, None
    n = len(preds)
    plcc = None
    if n >= 2:
        mp = sum(preds) / n
        ml = sum(labels) / n
        num = sum((p - mp) * (l - ml) for p, l in zip(preds, labels))
        dp = sum((p - mp) ** 2 for p in preds) ** 0.5
        dl = sum((l - ml) ** 2 for l in labels) ** 0.5
        if dp > 1e-12 and dl > 1e-12:
            plcc = num / (dp * dl)
    srcc = None
    if n >= 2:
        rp = _ranks_with_ties(preds)
        rl = _ranks_with_ties(labels)
        mp = sum(rp) / n
        ml = sum(rl) / n
        num = sum((a - mp) * (b - ml) for a, b in zip(rp, rl))
        dp = sum((a - mp) ** 2 for a in rp) ** 0.5
        dl = sum((b - ml) ** 2 for b in rl) ** 0.5
        if dp > 1e-12 and dl > 1e-12:
            srcc = num / (dp * dl)

    if mae_mode == "raw":
        mae = sum(abs(p - l) for p, l in zip(preds, labels)) / n
    elif mae_mode == "zscore":
        def _z(xs):
            mu = sum(xs) / len(xs)
            var = sum((x - mu) ** 2 for x in xs) / len(xs)
            sd = var ** 0.5
            if sd < 1e-12:
                return [0.0] * len(xs)
            return [(x - mu) / sd for x in xs]
        zp = _z(preds); zl = _z(labels)
        mae = sum(abs(a - b) for a, b in zip(zp, zl)) / n
    elif mae_mode == "affine":
        # Closed-form least-squares: a, b = argmin sum (a*p + b - l)^2
        mp = sum(preds) / n
        ml = sum(labels) / n
        num = sum((p - mp) * (l - ml) for p, l in zip(preds, labels))
        den = sum((p - mp) ** 2 for p in preds)
        if den < 1e-12:
            mae = sum(abs(p - l) for p, l in zip(preds, labels)) / n
        else:
            a = num / den
            b = ml - a * mp
            mae = sum(abs(a * p + b - l) for p, l in zip(preds, labels)) / n
    else:
        raise ValueError(f"unknown mae_mode={mae_mode!r}")
    return plcc, srcc, mae


# ---------------------------------------------------------------------------
# Video acquisition: VideoScore needs frames extracted locally.
# ---------------------------------------------------------------------------
def _download_to(url: str, dst: Path, *, timeout: int = 120) -> Path:
    """Download an HTTP(S) URL to ``dst`` if it's not already on disk.

    No-op if ``url`` is already a local file path.
    """
    if not url.startswith(("http://", "https://")):
        return Path(url)
    if dst.is_file() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "varr-baseline/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return dst


def _sample_frames(video_path: Path, num_frames: int = 48):
    """Uniformly sample ``num_frames`` PIL frames from ``video_path``.

    Mirrors the official VideoScore-v1.1 README which uses 48 frames at
    inference time for the v1.1 checkpoint ("support for 48 frames in
    inference now"). Falls back to whatever the clip provides if it has
    fewer than ``num_frames`` decoded frames.

    Tries decoding backends in this order:

      1. ``decord``        — fastest, but PyPI wheels are flaky on
                             newer glibc / Py3.10 setups.
      2. ``torchvision``   — usually present in any DL env (env-3.10.6
                             ships torchvision 0.20.1+cu121).
      3. ``opencv-python`` — last-resort, frame-by-frame decode.
      4. ``imageio``       — only as a final fallback.

    The first backend that imports successfully AND can read at least
    one frame from the file is used; we never silently swallow a real
    decode error from a chosen backend.
    """
    from PIL import Image  # Pillow is always required.

    p = str(video_path)

    def _pick(idxs_total: int, n: int):
        """Uniform sampling indices into a clip of length ``idxs_total``."""
        if idxs_total <= 0:
            return []
        if idxs_total <= n:
            return list(range(idxs_total))
        step = (idxs_total - 1) / max(n - 1, 1)
        return [min(idxs_total - 1, int(round(i * step))) for i in range(n)]

    # ---- backend 1: decord ----------------------------------------------
    try:
        import decord  # type: ignore
        decord.bridge.set_bridge("native")
        reader = decord.VideoReader(p)
        total = len(reader)
        if total > 0:
            idxs = _pick(total, num_frames)
            return [Image.fromarray(reader[i].asnumpy()) for i in idxs]
    except ImportError:
        pass  # try next backend
    except Exception:
        pass  # corrupt or unsupported codec — let next backend try

    # ---- backend 2: torchvision -----------------------------------------
    try:
        import torch  # noqa: F401
        from torchvision.io import read_video  # type: ignore
        # `read_video` returns (T,H,W,C) uint8 tensor.
        vframes, _aframes, _info = read_video(p, pts_unit="sec")
        total = int(vframes.shape[0])
        if total > 0:
            idxs = _pick(total, num_frames)
            return [Image.fromarray(vframes[i].numpy()) for i in idxs]
    except ImportError:
        pass
    except Exception:
        pass

    # ---- backend 3: opencv-python ---------------------------------------
    try:
        import cv2  # type: ignore
        cap = cv2.VideoCapture(p)
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total <= 0:
                # fallback: scan
                total = 0
                while cap.grab():
                    total += 1
                cap.release()
                cap = cv2.VideoCapture(p)
            idxs = _pick(total, num_frames)
            wanted = set(idxs)
            order = {idx: pos for pos, idx in enumerate(idxs)}
            frames_by_pos: dict = {}
            i = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if i in wanted:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frames_by_pos[order[i]] = Image.fromarray(frame_rgb)
                    if len(frames_by_pos) == len(idxs):
                        break
                i += 1
            if frames_by_pos:
                return [frames_by_pos[k] for k in sorted(frames_by_pos)]
        finally:
            cap.release()
    except ImportError:
        pass
    except Exception:
        pass

    # ---- backend 4: imageio (final fallback) ----------------------------
    try:
        import imageio.v3 as iio  # type: ignore
        frames_np = list(iio.imiter(p))
        total = len(frames_np)
        if total > 0:
            idxs = _pick(total, num_frames)
            return [Image.fromarray(frames_np[i]) for i in idxs]
    except ImportError:
        pass

    raise ImportError(
        "VideoScore baseline could not decode the video: none of "
        "{decord, torchvision.io, opencv-python, imageio} is available "
        "or able to open the file. Install one of them, e.g.:\n"
        "    pip install decord                   # fastest\n"
        "    pip install opencv-python-headless   # most portable\n"
        f"file: {video_path}"
    )


# ---------------------------------------------------------------------------
# VideoScore-v1.1 prompt template.
#
# IMPORTANT: This prompt is COPIED VERBATIM from the official v1.1 model
# card README. VideoScore-v1.1 is a regression head trained on top of
# Mantis-Idefics2; it reads the last-hidden-state at the end of the prompt
# and projects it to 5 logits in [1.0, 4.0]. Both the textual content AND
# the way <image> tokens are interleaved are part of the training
# distribution -- deviating in either dimension causes the regression
# head to output garbage (often collapsing to clip(<0)=1.0 across all
# 5 dimensions, mistakenly looking like the model itself is broken).
#
# Two specific gotchas, both inherited from the official inference
# example (see VideoScore-v1.1/README.md `### Inference`):
#   1. The prompt is a RAW STRING -- do not wrap it with the processor's
#      chat_template (no `User:` / `Assistant:` framing).
#   2. All <image> tokens are appended at the END of the text prompt,
#      one per sampled frame, separated by spaces. Do NOT interleave or
#      put images before the text -- training-time format places them
#      after the regression query.
# ---------------------------------------------------------------------------
VIDEOSCORE_PROMPT = (
    "Suppose you are an expert in judging and evaluating the quality of "
    "AI-generated videos,\n"
    "please watch the following frames of a given video and see the text "
    "prompt for generating the video,\n"
    "then give scores from 5 different dimensions:\n"
    "(1) visual quality: the quality of the video in terms of clearness, "
    "resolution, brightness, and color\n"
    "(2) temporal consistency, both the consistency of objects or humans "
    "and the smoothness of motion or movements\n"
    "(3) dynamic degree, the degree of dynamic changes\n"
    "(4) text-to-video alignment, the alignment between the text prompt "
    "and the video content\n"
    "(5) factual consistency, the consistency of the video content with "
    "the common-sense and factual knowledge\n"
    "for each dimension, output a float number from 1.0 to 4.0,\n"
    "the higher the number is, the better the video performs in that "
    "sub-score,\n"
    "the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the "
    "video is like a real video)\n"
    "Here is an output example:\n"
    "visual quality: 3.2\n"
    "temporal consistency: 2.7\n"
    "dynamic degree: 4.0\n"
    "text-to-video alignment: 2.3\n"
    "factual consistency: 1.8\n"
    'For this video, the text prompt is "{query}",\n'
    "all the frames of video are as follows:\n"
)

# Order is canonical — never sort by Python dict ordering when emitting
# downstream, the SOTA-table builder expects this list.
VIDEOSCORE_KEYS: tuple[str, ...] = (
    "visual quality",
    "temporal consistency",
    "dynamic degree",
    "text-to-video alignment",
    "factual consistency",
)


# ---------------------------------------------------------------------------
# Coverage matrix: which VideoScore-v1.1 sub-score is the canonical
# "matching" head for each EvalVerse 18-dim L2 axis.
#
# `None` means VideoScore-v1.1 has no semantically aligned sub-score for
# that EvalVerse axis (e.g. anything Sound Design — the model is
# vision-only — or fine-grained character/scene design which collapses
# at the level of "text-to-video alignment"). For those axes we still
# emit the raw 5 sub-scores in the per-sample record, but we report the
# main metric as N/A in the SOTA table builder rather than picking a
# bad proxy.
#
# Cluster groupings (L1 -> L2) follow the project taxonomy:
#     Visual Concept Design = {Character, Scene}
#     Acting                = {Consistency, Action, Expression}
#     Cinematography        = {Composition, Pacing, Lens}
#     Aesthetics            = {Visual Quality, Chromaticity, Lighting,
#                              Materiality}
#     Affectivity           = {Grounding, Progression}
#     Multi-Shot            = {Logic, Rhythm}
#     Sound Design          = {Vocal, Soundscape}
# ---------------------------------------------------------------------------
BASELINE_COVERAGE: dict[str, str | None] = {
    # Visual Concept Design -- VideoScore can't separate character vs
    # scene; the closest signal is global text-to-video alignment.
    "character":            "text-to-video alignment",
    "scene":                "text-to-video alignment",
    # Acting
    "consistency":          "temporal consistency",
    "action":               "dynamic degree",
    "expression":           None,
    # Cinematography
    "composition":          None,
    "pacing":               "dynamic degree",
    "lens":                 None,
    # Aesthetics — all four map to the visual-quality head; that head
    # is the only fine-grained proxy VideoScore-v1.1 emits for these.
    "visual_quality":       "visual quality",
    "chromaticity":         "visual quality",
    "lighting":             "visual quality",
    "materiality":          "visual quality",
    # Affectivity
    "grounding":            "text-to-video alignment",
    "progression":          None,
    # Multi-Shot
    "logic":                "factual consistency",
    "rhythm":               "temporal consistency",
    # Sound Design — VideoScore is vision-only.
    "vocal":                None,
    "soundscape":           None,
}


def _infer_dimension_from_dataset(dataset_path: str) -> str | None:
    """Try to infer the EvalVerse L2 axis name from the dataset path.

    e.g. ``data/v_arr_mvp/visual_quality_mvp/shared_test_pointwise.json``
    -> ``"visual_quality"``. Returns ``None`` if no match. The lookup is
    purely on path components — ``BASELINE_COVERAGE`` keys are matched
    case-insensitively, and the suffix ``_mvp`` (which is the v-arr MVP
    naming convention) is stripped before lookup.
    """
    parts = [p.lower() for p in Path(dataset_path).parts]
    for part in parts:
        stem = part[:-4] if part.endswith("_mvp") else part
        if stem in BASELINE_COVERAGE:
            return stem
    return None


# ---------------------------------------------------------------------------
# Model wrapper.
#
# Encapsulated as a class so the dry-run harness can monkey-patch it with
# a ``MockModel`` to exercise the rest of the pipeline without GPU.
# ---------------------------------------------------------------------------
class VideoScoreModel:
    """Lazy-loaded VideoScore-v1.1 wrapper.

    The actual transformers/Mantis import happens inside ``load()`` so
    importing this module in a CPU-only / dry-run environment doesn't
    explode.
    """

    def __init__(self, model_path: str, device: str = "cuda:0",
                 dtype: str = "fp16"):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.processor = None
        self.model = None

    def load(self):
        import torch  # noqa
        from transformers import AutoProcessor  # noqa

        # ------------------------------------------------------------------
        # timm <-> transformers compat shim.
        #
        # Some envs ship timm 0.6.x while transformers 4.48+ expects
        # `ImageNetInfo` / `infer_imagenet_subset` to exist in `timm.data`
        # (used by transformers' lazy-loaded `timm_wrapper` even when no
        # timm-backed model is being instantiated). Without this stub the
        # very `AutoProcessor.from_pretrained` call below crashes with
        # `cannot import name 'ImageNetInfo' from 'timm.data'`.
        # The shim is a no-op when timm is absent or already up-to-date.
        # ------------------------------------------------------------------
        try:
            import timm.data as _timm_data  # type: ignore
            if not hasattr(_timm_data, "ImageNetInfo"):
                class _StubImageNetInfo:  # noqa: D401
                    def __init__(self, *a, **kw): pass
                _timm_data.ImageNetInfo = _StubImageNetInfo  # type: ignore
            if not hasattr(_timm_data, "infer_imagenet_subset"):
                _timm_data.infer_imagenet_subset = lambda *a, **kw: None  # type: ignore
        except ImportError:
            pass  # timm not installed — fine, transformers will skip the wrapper.

        # ------------------------------------------------------------------
        # Official VideoScore-v1.1 inference path requires Mantis-VL's
        # `Idefics2ForSequenceClassification` (it has the 5-way regression
        # head fine-tuned for VideoScore; `AutoModel` would only load the
        # Idefics2 backbone and drop that head -> outputs become garbage).
        # ------------------------------------------------------------------
        try:
            from mantis.models.idefics2 import (  # type: ignore  # noqa
                Idefics2ForSequenceClassification as _Cls,
            )
        except Exception as e:
            raise ImportError(
                "VideoScore-v1.1 requires the `mantis-vl` package "
                "(provides Idefics2ForSequenceClassification with the "
                "5-way regression head). Install it INTO THE SAME conda "
                "env that has a working torch + visible GPUs, e.g.:\n"
                "    conda activate env-3.10.6   # torch 2.5.1+cu121, 2 GPUs\n"
                "    pip install mantis-vl\n"
                "Note: do NOT use the `rewardverse` env — its torch is "
                "broken (cu12.4 cusparse vs nvJitLink mismatch)."
            ) from e

        # ------------------------------------------------------------------
        # mantis-vl <-> transformers 4.56 compat shim.
        #
        # transformers >= 4.50 calls `self.tie_weights(recompute_mapping=False)`
        # from `init_weights()`, but mantis's override of `tie_weights` was
        # written for older transformers and has signature `tie_weights(self)`
        # — no `**kwargs`. So `from_pretrained` blows up with:
        #     TypeError: Idefics2ForSequenceClassification.tie_weights() got
        #                an unexpected keyword argument 'recompute_mapping'
        #
        # We swap it for a wrapper that silently absorbs unknown kwargs and
        # then delegates to the original (kwarg-less) implementation. This
        # is safe: `tie_weights` only ties input/output embedding weights
        # and never affects the regression head we actually care about.
        # ------------------------------------------------------------------
        _orig_tie = _Cls.tie_weights
        try:
            import inspect as _inspect
            _sig = _inspect.signature(_orig_tie)
            _accepts_kwargs = any(
                p.kind == _inspect.Parameter.VAR_KEYWORD
                for p in _sig.parameters.values()
            )
        except (TypeError, ValueError):
            _accepts_kwargs = False
        if not _accepts_kwargs:
            def _tie_weights_compat(self, *args, **kwargs):  # noqa: D401
                return _orig_tie(self)
            _Cls.tie_weights = _tie_weights_compat  # type: ignore[assignment]

        # ------------------------------------------------------------------
        # mantis-vl <-> transformers >= 4.50 compat shim (attn impl kwarg).
        #
        # transformers 4.50 deprecated the `use_flash_attention_2` kwarg
        # to `_autoset_attn_implementation` and replaced it with the
        # broader `attn_implementation` selector. mantis idefics2 still
        # forwards `use_flash_attention_2=use_flash_attention_2` verbatim,
        # producing on transformers 4.53.x:
        #     TypeError: PreTrainedModel._autoset_attn_implementation()
        #                got an unexpected keyword argument
        #                'use_flash_attention_2'
        #
        # We wrap the mantis override so that it (a) translates the legacy
        # bool into `attn_implementation="flash_attention_2"` when True,
        # otherwise drops it (letting transformers pick the default eager
        # / sdpa path). This is what mantis itself would do under modern
        # transformers; we just shim it in until the package catches up.
        # No-op if a future mantis release fixes the call site upstream.
        # ------------------------------------------------------------------
        _orig_autoset = _Cls._autoset_attn_implementation
        if getattr(_orig_autoset, "_arr_patched", False) is False:
            def _autoset_attn_compat(  # type: ignore[override]
                cls,
                config,
                use_flash_attention_2: bool = False,
                torch_dtype=None,
                device_map=None,
                check_device_map: bool = True,
                **kwargs,
            ):
                # Translate legacy bool -> modern selector. mantis only
                # accesses `config.vision_config._attn_implementation`
                # after the parent call returns, so we just need the
                # parent to set `config._attn_implementation` correctly.
                if use_flash_attention_2 and "attn_implementation" not in kwargs:
                    kwargs["attn_implementation"] = "flash_attention_2"
                # Call PreTrainedModel's classmethod directly to bypass
                # mantis's signature-incompatible override (we're already
                # inside it conceptually).
                from transformers import PreTrainedModel as _PTM
                config = _PTM._autoset_attn_implementation.__func__(
                    cls,
                    config=config,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                    check_device_map=check_device_map,
                    **kwargs,
                )
                # Mirror mantis's post-step: propagate to vision sub-config.
                if hasattr(config, "vision_config"):
                    config.vision_config._attn_implementation = (
                        config._attn_implementation
                    )
                return config
            _autoset_attn_compat._arr_patched = True  # type: ignore[attr-defined]
            _Cls._autoset_attn_implementation = classmethod(
                _autoset_attn_compat
            )  # type: ignore[assignment]

        # ------------------------------------------------------------------
        # mantis-vl <-> transformers 4.56 compat shim (KV cache).
        #
        # transformers 4.56 dropped the `DynamicCache.from_legacy_cache`
        # classmethod. mantis idefics2 still calls it inside its forward:
        #     past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        # which now blows up with:
        #     AttributeError: type object 'DynamicCache' has no
        #                     attribute 'from_legacy_cache'
        #
        # The new `DynamicCache.__init__` already accepts the legacy
        # tuple-of-tuples format directly via its `ddp_cache_data` arg,
        # so we just rewire the missing classmethod to call the ctor.
        # No-op if a future transformers version restores the method.
        # ------------------------------------------------------------------
        try:
            from transformers.cache_utils import DynamicCache as _DC  # type: ignore
            if not hasattr(_DC, "from_legacy_cache"):
                @classmethod
                def _from_legacy_cache(cls, past_key_values=None):
                    if past_key_values is None:
                        return cls()
                    return cls(past_key_values)
                _DC.from_legacy_cache = _from_legacy_cache  # type: ignore[attr-defined]
        except Exception:
            # If the import path changes, we'll just rely on the
            # use_cache=False forward path below.
            pass

        torch_dtype = (
            torch.float16 if self.dtype == "fp16"
            else torch.bfloat16 if self.dtype == "bf16"
            else torch.float32
        )

        # ------------------------------------------------------------------
        # Resolve / fall back the requested device BEFORE moving the model
        # onto it, so users on CPU-only nodes get a clear warning instead
        # of an opaque CUDA init crash.
        # ------------------------------------------------------------------
        if self.device.startswith("cuda"):
            cuda_ok = torch.cuda.is_available()
            dev_id  = int(self.device.split(":")[1]) if ":" in self.device else 0
            if (not cuda_ok) or dev_id >= torch.cuda.device_count():
                print(f"[VideoScore] WARNING: requested {self.device} but it is not "
                      f"available (cuda={cuda_ok}, n_devices={torch.cuda.device_count()}); "
                      f"falling back to CPU.", flush=True)
                self.device = "cpu"

        # transformers 4.56 renamed the kwarg `torch_dtype` -> `dtype`.
        # The old name still works but emits a deprecation warning every
        # call; pass under both names so we work on 4.4x and 4.5x+.
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, dtype=torch_dtype,
            )
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(
                self.model_path, torch_dtype=torch_dtype,
            )
        try:
            self.model = _Cls.from_pretrained(
                self.model_path, dtype=torch_dtype,
            ).eval().to(self.device)
        except TypeError:
            self.model = _Cls.from_pretrained(
                self.model_path, torch_dtype=torch_dtype,
            ).eval().to(self.device)

        # ------------------------------------------------------------------
        # mantis-vl <-> transformers 4.56 compat shim (pad_token_id).
        #
        # mantis Idefics2ForSequenceClassification.forward references
        # `self.config.pad_token_id` to compute pooled logits indices.
        # On transformers 4.56, when the checkpoint's top-level
        # config.json has `"pad_token_id": null` (which is exactly the
        # case for VideoScore-v1.1), `Idefics2Config` no longer exposes
        # that attribute at all, and forward blows up with:
        #     AttributeError: 'Idefics2Config' object has no
        #                     attribute 'pad_token_id'
        #
        # Fix: copy a real pad id onto the top-level config from the
        # text sub-config (Mistral text backbone has pad_token_id=0),
        # falling back to the tokenizer's pad_token_id, then to 0.
        # For batch_size=1 inference (our case) the choice doesn't
        # change pooled-logits semantics: mantis just argmax's the
        # first occurrence of pad_token_id; if there is none in the
        # row it wraps via modulo to position -1 (i.e. last token),
        # which is exactly what we want for a regression head.
        # ------------------------------------------------------------------
        try:
            cfg = self.model.config
            pad_id = getattr(cfg, "pad_token_id", None)
            if pad_id is None:
                tcfg = getattr(cfg, "text_config", None)
                if tcfg is not None:
                    pad_id = getattr(tcfg, "pad_token_id", None)
            if pad_id is None and self.processor is not None:
                tok = getattr(self.processor, "tokenizer", None)
                if tok is not None:
                    pad_id = getattr(tok, "pad_token_id", None)
            if pad_id is None:
                pad_id = 0
            # Set on both the top-level config (what mantis reads) and
            # the text sub-config (what the language model itself reads).
            cfg.pad_token_id = pad_id
            tcfg = getattr(cfg, "text_config", None)
            if tcfg is not None and getattr(tcfg, "pad_token_id", None) is None:
                tcfg.pad_token_id = pad_id
        except Exception:
            # Worst case the mantis forward will raise the original
            # AttributeError per-sample, which we already surface in
            # predictions[*].error with full context.
            pass

    @staticmethod
    def _build_prompt(query: str, num_frames: int) -> str:
        """Build the exact training-time prompt for VideoScore-v1.1.

        Mirrors the official README inference example verbatim:
          1. Format the regression query with the user's text prompt.
          2. Count any literal '<image>' tokens already present (zero in
             our template, but kept for parity with the upstream code).
          3. Append '<image> ' once per sampled frame at the END of the
             prompt -- this is the placement the regression head was
             trained on; do NOT prepend or interleave.
        """
        eval_prompt = VIDEOSCORE_PROMPT.format(query=query)
        num_image_token = eval_prompt.count("<image>")
        if num_image_token < num_frames:
            eval_prompt += "<image> " * (num_frames - num_image_token)
        return eval_prompt

    def score_video(self, query: str, frames: list,
                    *, max_new_tokens: int = 200) -> list[float]:
        """Return 5 raw sub-scores in the canonical VIDEOSCORE_KEYS order.

        VideoScore-v1.1 is a sequence-classification head in the v1.1
        release (the regression head for 5 sub-scores), so we read
        ``logits`` directly. Each sub-score lives in [1.0, 4.0].

        We deliberately bypass ``processor.apply_chat_template`` here:
        the v1.1 regression head was trained on a RAW prompt string with
        <image> tokens appended at the end (see VIDEOSCORE_PROMPT), and
        wrapping it in the Idefics2 'User:/Assistant:' chat frame shifts
        the last-hidden-state position the regression head reads from,
        which silently produces garbage logits.
        """
        if self.model is None:
            raise RuntimeError("VideoScoreModel not loaded; call load() first.")
        import torch
        prompt = self._build_prompt(query, len(frames))
        inputs = self.processor(
            text=prompt, images=frames, return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            # use_cache=False short-circuits the KV-cache code path inside
            # mantis idefics2 (which still calls the removed classmethod
            # `DynamicCache.from_legacy_cache` on transformers >= 4.56).
            # We're doing a single forward for the 5-way regression head,
            # so KV cache has no effect on the result.
            out = self.model(**inputs, use_cache=False)
        # ``logits`` shape (1, 5) for the 5 sub-scores in the v1.1 head.
        logits = out.logits.squeeze(0).float().cpu().tolist()
        if len(logits) != len(VIDEOSCORE_KEYS):
            raise RuntimeError(
                f"unexpected VideoScore head width: got {len(logits)} "
                f"logits, expected {len(VIDEOSCORE_KEYS)} (=len(VIDEOSCORE_KEYS))"
            )
        return logits


# ---------------------------------------------------------------------------
# Per-sample scoring.
# ---------------------------------------------------------------------------
def _video_url_of(item: dict) -> str:
    resp = item.get("response")
    if isinstance(resp, str):
        return resp
    if isinstance(resp, list) and resp:
        return str(resp[0])
    raise ValueError(f"item {item.get('id')!r} has no response URL")


def _url_to_cache_name(url: str) -> str:
    """Deterministic, cross-process-stable cache filename for a video URL.

    ** Why not ``abs(hash(url))``? ** Python's built-in ``hash()`` is
    salted per interpreter process when PYTHONHASHSEED is unset (the
    default since 3.3), so the same URL maps to a *different* filename
    every time you launch the script. That silently invalidates the
    cache directory across runs. SHA-256 is deterministic across
    processes and across Python versions, so v1.1 and v2 runners share
    the same cache automatically.
    """
    import hashlib
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32] + ".mp4"

def _score_one(model: VideoScoreModel, item: dict, video_dir: Path,
               num_frames: int) -> dict:
    sample_id = item.get("id")
    query = item.get("query") or item.get("prompt") or ""
    url = _video_url_of(item)
    # Local cache path; VideoScore-v1.1 needs frames on disk for decord.
    cache_path = video_dir / _url_to_cache_name(url)
    if not cache_path.is_file():
        cache_path = _download_to(url, cache_path)
    # Sanity-check the file is non-empty -- a 0-byte cache from a
    # previously-aborted download would crash decord with SIGFPE.
    if cache_path.stat().st_size == 0:
        cache_path.unlink()
        cache_path = _download_to(url, cache_path)
    frames = _sample_frames(cache_path, num_frames=num_frames)
    raw_scores = model.score_video(query, frames)
    sub_scores = dict(zip(VIDEOSCORE_KEYS, raw_scores))
    overall = sum(raw_scores) / len(raw_scores)
    # Project [1, 4] -> [1, 5] linearly so the value is directly comparable
    # with our T1/T2 label_score axis. ARR labels live in [1.0, 5.0]; the
    # VideoScore head's natural support is [1.0, 4.0]; an affine map
    # (1, 4) -> (1, 5) is the cleanest comparison.
    pred_5 = 1.0 + (overall - 1.0) * (4.0 / 3.0)
    pred_5 = round(max(1.0, min(5.0, pred_5)), 3)
    return {
        "id":              sample_id,
        "query":           query,
        "label_score":     item.get("label_score"),
        "raw_sub_scores":  sub_scores,
        "raw_overall_4":   round(overall, 4),
        "pred_score":      pred_5,
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model_path", default=None,
                   help="local model dir or HF repo id. "
                        f"Default: ../../models/VideoScore-v1.1/ (auto-"
                        f"downloaded from {_VS1_REPO_ID} on first use).")
    p.add_argument("--no_download", action="store_true",
                   help="if set, refuse to auto-download missing weights; "
                        "the local model dir must already be populated.")
    p.add_argument("--device", default=os.environ.get("ARR_VIDEOSCORE_DEVICE",
                                                       "cuda:0"))
    # Default bf16 to match the official VideoScore-v1.1 README, which loads
    # both processor and model with torch_dtype=torch.bfloat16. fp16 still
    # works numerically on the Idefics2 backbone but we want strict parity.
    p.add_argument("--dtype", default="bf16", choices=("fp16", "bf16", "fp32"))
    p.add_argument("--num_frames", type=int, default=48)
    p.add_argument("--video_dir", default="runs/baselines/_video_cache",
                   help="local mp4 cache directory; URLs are downloaded once.")
    p.add_argument("--limit", type=int, default=None,
                   help="if set, only score the first N samples (smoke test).")
    p.add_argument("--target_subscore",
                   default=None,
                   choices=(None, "auto",
                            "visual quality", "temporal consistency",
                            "dynamic degree", "text-to-video alignment",
                            "factual consistency",
                            "raw_overall_4", "raw_overall_5"),
                   help="Which raw VideoScore sub-score to use as the main "
                        "baseline output for PLCC/SRCC against GT label_score. "
                        "`auto` (recommended) infers the canonical sub-score "
                        "from BASELINE_COVERAGE based on the dataset path "
                        "(e.g. visual_quality_mvp -> 'visual quality'). "
                        "`raw_overall_4` averages the first 4 sub-scores "
                        "(drops factual consistency); `raw_overall_5` "
                        "averages all 5. If unset, behaves like `auto`.")
    p.add_argument("--dimension", default=None,
                   help="EvalVerse L2 axis name (e.g. 'visual_quality') "
                        "to record in the output JSON; if omitted, inferred "
                        "from --dataset path.")
    args = p.parse_args()

    samples = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        samples = [samples]
    if args.limit:
        samples = samples[:args.limit]

    video_dir = Path(args.video_dir).expanduser()
    video_dir.mkdir(parents=True, exist_ok=True)

    # Resolve --model_path. None / repo-id / non-existent path all map to
    # the canonical ../../models/VideoScore-v1.1/ slot, with a one-shot
    # snapshot_download on first use.
    if args.model_path:
        if Path(args.model_path).is_dir():
            model_path = args.model_path
        else:
            model_path = str(ensure_model_local(
                args.model_path, allow_download=not args.no_download,
            ))
    else:
        model_path = str(ensure_model_local(
            _VS1_REPO_ID, allow_download=not args.no_download,
        ))
    print(f"[VideoScore] model_path resolved to {model_path}", flush=True)

    model = VideoScoreModel(
        model_path=model_path,
        device=args.device,
        dtype=args.dtype,
    )
    print(f"[VideoScore] loading {model_path} on {args.device} ({args.dtype}) ...",
          flush=True)
    t0 = time.time()
    model.load()
    print(f"[VideoScore] loaded in {time.time()-t0:.1f}s", flush=True)

    results: list[dict] = []
    failures = 0
    for i, item in enumerate(samples, 1):
        try:
            r = _score_one(model, item, video_dir, args.num_frames)
            results.append(r)
        except Exception as e:
            failures += 1
            results.append({"id": item.get("id"), "error": str(e)})
        if i % 5 == 0 or i == len(samples):
            print(f"[VideoScore] {i}/{len(samples)} done ({failures} failed)",
                  flush=True)

    # ------------------------------------------------------------------
    # Pointwise metrics — three-track report:
    #
    #   (a) `metrics`           : main report. Uses the raw continuous
    #       output of the canonical sub-score for the EvalVerse axis
    #       being evaluated (resolved via --target_subscore / --dimension
    #       / dataset-path inference / BASELINE_COVERAGE). PLCC and
    #       SRCC are scale-invariant; MAE is reported in z-score space
    #       so it's comparable across different baselines whose raw
    #       outputs live on different scales.
    #
    #   (b) `metrics_per_subscore`: for each of the 5 raw sub-scores +
    #       raw_overall_4 + raw_overall_5, full PLCC/SRCC/MAE(zscore).
    #       Lets you diagnose which VideoScore head best aligns with
    #       this dataset's GT axis without rerunning.
    #
    #   (c) `metrics_legacy_pred5`: the *old* metric — uses the
    #       1-to-5-clipped `pred_score`. Kept only for backward compat
    #       with prior runs; in practice it severely under-reports the
    #       true model capability because ~70% of preds get clipped to
    #       the 1.0 floor (raw logits are centered near 0, not 1-4).
    # ------------------------------------------------------------------
    # Resolve which sub-score to feed into the *main* metrics block.
    dim = args.dimension or _infer_dimension_from_dataset(args.dataset)
    target = args.target_subscore
    if target in (None, "auto"):
        # Use BASELINE_COVERAGE if the dimension is recognised; else
        # fall back to raw_overall_4 (drop factual_consistency, which
        # was empirically *negatively* correlated on visual_quality_mvp
        # — see prior diagnostic run).
        target = BASELINE_COVERAGE.get(dim or "", None) or "raw_overall_4"
    print(f"[VideoScore] target_subscore={target!r} (dimension={dim!r})",
          flush=True)

    def _series(name: str, items: list[dict]):
        """Yield aligned (pred, label) pairs for sub-score ``name``."""
        ps, ls = [], []
        for r in items:
            if "error" in r:
                continue
            l = r.get("label_score")
            if l is None:
                continue
            if name == "raw_overall_4":
                # mean of first 4 sub-scores (drops factual consistency).
                sub = r.get("raw_sub_scores") or {}
                vals = [sub.get(k) for k in VIDEOSCORE_KEYS[:4]]
                if any(v is None for v in vals):
                    continue
                p = sum(vals) / len(vals)
            elif name == "raw_overall_5":
                sub = r.get("raw_sub_scores") or {}
                vals = [sub.get(k) for k in VIDEOSCORE_KEYS]
                if any(v is None for v in vals):
                    continue
                p = sum(vals) / len(vals)
            elif name == "pred_score_legacy_clip5":
                p = r.get("pred_score")
            else:
                sub = r.get("raw_sub_scores") or {}
                p = sub.get(name)
            if p is None:
                continue
            try:
                ps.append(float(p))
                ls.append(float(l))
            except (TypeError, ValueError):
                continue
        return ps, ls

    # (a) Main metrics — the chosen target sub-score, raw continuous.
    main_preds, main_labels = _series(target, results)
    main_plcc, main_srcc, main_mae_zs = _compute_metrics(
        main_preds, main_labels, mae_mode="zscore",
    )
    _, _, main_mae_aff = _compute_metrics(
        main_preds, main_labels, mae_mode="affine",
    )

    # (b) Per-sub-score metrics (diagnostics). Always all heads + two
    #     overalls so the SOTA-table builder can pick whichever it
    #     wants without rerunning the model.
    metrics_per_subscore: dict[str, dict] = {}
    for name in (*VIDEOSCORE_KEYS, "raw_overall_4", "raw_overall_5"):
        ps, ls = _series(name, results)
        plcc_n, srcc_n, mae_n = _compute_metrics(ps, ls, mae_mode="zscore")
        metrics_per_subscore[name] = {
            "plcc":    plcc_n,
            "srcc":    srcc_n,
            "mae_zs": mae_n,
            "n":       len(ps),
        }

    # (c) Legacy 1..5-clipped pred metric (don't rely on this one).
    legacy_preds, legacy_labels = _series("pred_score_legacy_clip5", results)
    legacy_plcc, legacy_srcc, legacy_mae = _compute_metrics(
        legacy_preds, legacy_labels, mae_mode="raw",
    )

    out = {
        "method":      "VideoScore-v1.1",
        "model_path":  args.model_path,
        "dataset":     args.dataset,
        "dimension":   dim,
        "target_subscore": target,
        "total":       len(samples),
        "parseable":   len(main_preds),
        "failures":    failures,
        "metrics": {
            "plcc":    main_plcc,
            "srcc":    main_srcc,
            "mae_zs": main_mae_zs,
            "mae_affine": main_mae_aff,
            "n":       len(main_preds),
        },
        "metrics_per_subscore": metrics_per_subscore,
        "metrics_legacy_pred5": {
            "plcc":    legacy_plcc,
            "srcc":    legacy_srcc,
            "mae":     legacy_mae,
            "n":       len(legacy_preds),
        },
        "predictions": results,
    }
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    def _fmt(x):
        return f"{x:.4f}" if isinstance(x, float) else "n/a"
    print(f"[VideoScore] [main:{target!r}] PLCC={_fmt(main_plcc)} "
          f"SRCC={_fmt(main_srcc)} MAE_zs={_fmt(main_mae_zs)} "
          f"MAE_aff={_fmt(main_mae_aff)} "
          f"N={len(main_preds)}/{len(samples)} (failures={failures})",
          flush=True)
    # Per-sub-score one-liner table for quick diagnostic read-out.
    print("[VideoScore] per-sub-score (PLCC / SRCC / MAE_zs / N):", flush=True)
    for name, m in metrics_per_subscore.items():
        print(f"               {name:28s}  "
              f"PLCC={_fmt(m['plcc'])}  SRCC={_fmt(m['srcc'])}  "
              f"MAE_zs={_fmt(m['mae_zs'])}  N={m['n']}",
              flush=True)
    print(f"[VideoScore] [legacy clip5] PLCC={_fmt(legacy_plcc)} "
          f"SRCC={_fmt(legacy_srcc)} MAE={_fmt(legacy_mae)} "
          f"N={len(legacy_preds)} (kept for back-compat only)",
          flush=True)
    print(f"[VideoScore] wrote -> {out_path}")


if __name__ == "__main__":
    main()
