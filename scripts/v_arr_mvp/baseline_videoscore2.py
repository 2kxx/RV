"""RewardVerse — VideoScore2 baseline runner (TIGER-AI-Lab/VideoScore2).

Counterpart to ``baseline_videoscore.py`` for the VideoScore-2 release whose
official inference flow lives at:

    https://github.com/TIGER-AI-Lab/VideoScore2/blob/main/vs2_inference.py

Sandbox-network limitation note (2026-05-21)
--------------------------------------------
This file was authored from a network-restricted environment that could not
fetch ``vs2_inference.py`` directly. The structure mirrors the VideoScore-v1.1
baseline (load → frame-sample → score → 1-5 projection). Three interface
knobs are isolated as module-level constants so a single five-minute patch
aligns this script with the exact VideoScore2 inference contract once the
official script is in hand:

    _VS2_MODEL_ID        — HF repo id for the v2 checkpoint
    _VS2_NUM_FRAMES      — frame budget per clip
    _VS2_KEYS            — canonical sub-score key order
    _VS2_PROMPT_TEMPLATE — chat prompt VideoScore2 expects

Edit the constants if any of them disagree with vs2_inference.py — the rest
of the pipeline (download, decord sampling, 1-4→1-5 projection,
PLCC/SRCC/MAE evaluation) stays correct.

Usage::

    python scripts/v_arr_mvp/baseline_videoscore2.py \\
        --dataset data/v_arr_mvp/visual_quality_mvp/shared_test_pointwise.json \\
        --output runs/baselines/videoscore2/shared_test.json \\
        --model_path TIGER-AI-Lab/VideoScore2 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# Enable PyTorch's expandable-segments allocator BEFORE torch is imported
# anywhere in this process. The default caching allocator pre-reserves
# fixed-size segments per stream, which fragments badly for VideoScore2:
# each sample's prefill needs a single ~3-25 GiB contiguous block for
# ``Q@K^T`` whose exact size depends on the visual-token sequence, and
# the freed-but-not-released blocks from the previous sample force
# subsequent allocs to fail even when 30+ GiB is "free". Expandable
# segments grow on demand and split/merge cleanly between samples.
# Idempotent: respects user-provided value if already set.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:512",
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_assets import (  # type: ignore  # noqa: E402
    ensure_model_local,
    resolve_local_model_dir,
)

# Re-use the v1.1 baseline's centralised pieces so VS2 reports under
# the same three-track metric schema (main / per-subscore / legacy clip5)
# and plugs into the same EvalVerse-axis -> sub-score routing table.
# The v1.1 module has them as plain module-level objects, so a direct
# import is safe and keeps a single source of truth.
from baseline_videoscore import (  # type: ignore  # noqa: E402
    _compute_metrics as _compute_metrics_v1,
    BASELINE_COVERAGE,
    _infer_dimension_from_dataset,
)


# ---------------------------------------------------------------------------
# Pointwise metric helpers (kept independent of judger.py).
# ---------------------------------------------------------------------------
def _ranks_with_ties(xs):
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


def _compute_metrics_legacy(preds, labels):
    """Legacy raw-MAE pointwise metrics. Kept only so the
    ``metrics_legacy_pred5`` block in the output JSON keeps the same
    semantics as before this module's three-track refactor. New code
    should use ``_compute_metrics_v1`` (imported from baseline_videoscore)
    with an explicit ``mae_mode``.
    """
    if not preds or not labels or len(preds) != len(labels):
        return None, None, None
    n = len(preds)
    mae = sum(abs(p - l) for p, l in zip(preds, labels)) / n
    plcc = None
    if n >= 2:
        mp = sum(preds) / n; ml = sum(labels) / n
        num = sum((p - mp) * (l - ml) for p, l in zip(preds, labels))
        dp = sum((p - mp) ** 2 for p in preds) ** 0.5
        dl = sum((l - ml) ** 2 for l in labels) ** 0.5
        if dp > 1e-12 and dl > 1e-12:
            plcc = num / (dp * dl)
    srcc = None
    if n >= 2:
        rp = _ranks_with_ties(preds); rl = _ranks_with_ties(labels)
        mp = sum(rp) / n; ml = sum(rl) / n
        num = sum((a - mp) * (b - ml) for a, b in zip(rp, rl))
        dp = sum((a - mp) ** 2 for a in rp) ** 0.5
        dl = sum((b - ml) ** 2 for b in rl) ** 0.5
        if dp > 1e-12 and dl > 1e-12:
            srcc = num / (dp * dl)
    return plcc, srcc, mae


# ---------------------------------------------------------------------------
# Video acquisition.
# ---------------------------------------------------------------------------
def _download_to(url: str, dst: Path, *, timeout: int = 120) -> Path:
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


def _sample_frames(video_path: Path, num_frames: int):
    try:
        import decord  # type: ignore
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "VideoScore2 baseline requires `decord` and `Pillow`. "
            "Install via `pip install decord Pillow`."
        ) from e
    reader = decord.VideoReader(str(video_path))
    total = len(reader)
    if total == 0:
        raise ValueError(f"video has no frames: {video_path}")
    if total <= num_frames:
        idxs = list(range(total))
    else:
        step = (total - 1) / max(num_frames - 1, 1)
        idxs = [min(total - 1, int(round(i * step))) for i in range(num_frames)]
    return [Image.fromarray(reader[i].asnumpy()) for i in idxs]


# ---------------------------------------------------------------------------
# VideoScore2 contract knobs (per official README, 2026-05).
# ---------------------------------------------------------------------------
# Important: VideoScore2 is NOT a 5-head regression model like v1.1.
# It's a Qwen2.5-VL fine-tune that emits scores GENERATIVELY for THREE
# dimensions, then uses a log-likelihood-based soft-score over the
# integer score tokens 1..5.  See README ### Inference for the exact
# pipeline; everything below mirrors that code path verbatim.
_VS2_MODEL_ID = "TIGER-AI-Lab/VideoScore2"

# Default sampling fps for qwen-vl-utils; the README demo uses 2.0.
_VS2_INFER_FPS = 2.0
# Per-sample max-new-tokens. The canonical answer template is
# ~50 tokens ("visual quality: N; text-to-video alignment: N, ...")
# and any extra CoT preamble fits well below 256. We keep 256 (vs
# README's 1024) to cut KV-cache + ``output_scores=True`` logits-
# stash memory by ~60%; this alone is NOT what fixes the OOM on
# our 95 GiB H800 (root cause is visual tokens, see _VS2_TOTAL_PIXELS).
# Per-sample max-new-tokens. The model emits a CoT scratchpad inside
# ``<think>...</think>`` before producing the final 3-line score block:
#
#     (1) visual quality \u2013 ... : <int>
#     (2) text-to-video alignment \u2013 ... : <int>
#     (3) physical/common-sense consistency \u2013 ... : <int>
#
# Empirically the CoT alone is 600-900 tokens, so 512 truncates ~50% of
# samples mid-third-score (label 14/30 hit on visual_quality_test,
# 2026-05-27). We pin 1024 to match the README demo and give 100-300
# tokens of headroom past the answer block. KV-cache + ``output_scores``
# logits-stash memory remains modest at this length.
_VS2_MAX_NEW_TOKENS = 1024
# Score token range is integer 1..5 (per README's `range(1, 6)`).
_VS2_SCORE_RANGE: tuple[int, ...] = (1, 2, 3, 4, 5)

# IMPORTANT: we intentionally do NOT set ``total_pixels`` /
# ``min_pixels`` on the video content block. Both the official
# README demo (``vs2_inference.py``) and the official batch eval
# (``eval/`` -- ``eval_VideoScore2_float.evaluate_video``) leave
# them at qwen-vl-utils' defaults; capping them aggressively (we
# tried 128*28*28) crushes per-frame resolution to ~89x89 thumbnails
# and was the root cause of the negative-PLCC results in our first
# v2 run. OOMs are now handled at runtime via per-sample subprocess
# isolation in the runner, not by mangling the input.

# Sub-score keys VideoScore2 emits, in canonical order. THREE dims only:
#   visual quality, text-to-video alignment, physical/common-sense consistency.
# These match the README's regex parser exactly.
_VS2_KEYS: tuple[str, ...] = (
    "visual quality",
    "text-to-video alignment",
    "physical/common-sense consistency",
)

# Verbatim VS2 prompt template from the official README; substitute uses
# `string.Template`-style $-placeholders, but we use Python's str.format
# style here for symmetry with the rest of the file (the model never
# sees the placeholder so the surface form is interchangeable).
_VS2_PROMPT_TEMPLATE = (
    "\n    You are an expert for evaluating AI-generated videos from "
    "three dimensions:\n"
    "    (1) visual quality \u2013 clarity, smoothness, artifacts;\n"
    "    (2) text-to-video alignment \u2013 fidelity to the prompt;\n"
    "    (3) physical/common-sense consistency \u2013 naturalness and "
    "physics plausibility.\n"
    "\n"
    "    Video prompt: {query}\n"
    "\n"
    "    Please output in this format:\n"
    "    visual quality: <v_score>; \n"
    "    text-to-video alignment: <t_score>, \n"
    "    physical/common-sense consistency: <p_score>\n    "
)

# Coverage matrix specific to VideoScore2's 3-head taxonomy. EvalVerse
# axes that don't have a defensible v2 head are mapped to None and the
# SOTA table will report "\u2014". Cluster groupings follow the project
# taxonomy (see baseline_videoscore.py BASELINE_COVERAGE for v1.1).
BASELINE_COVERAGE_V2: dict[str, "str | None"] = {
    # Visual Concept Design -- text-to-video alignment is the closest
    # head for character / scene faithfulness.
    "character":            "text-to-video alignment",
    "scene":                "text-to-video alignment",
    # Acting -- consistency / action / expression are temporal phenomena;
    # v2's "physical/common-sense consistency" head best captures whether
    # motion / continuity look plausible.
    "consistency":          "physical/common-sense consistency",
    "action":               "physical/common-sense consistency",
    "expression":           None,  # too fine-grained for any v2 head
    # Cinematography
    "composition":          None,
    "pacing":               "physical/common-sense consistency",
    "lens":                 None,
    # Aesthetics -- all four map to visual quality (clarity / artifacts).
    "visual_quality":       "visual quality",
    "chromaticity":         "visual quality",
    "lighting":             "visual quality",
    "materiality":          "visual quality",
    # Affectivity
    "grounding":            "text-to-video alignment",
    "progression":          None,
    # Multi-Shot -- logic / rhythm are about cross-shot coherence; v2's
    # "physical/common-sense consistency" is the closest fit.
    "logic":                "physical/common-sense consistency",
    "rhythm":               "physical/common-sense consistency",
    # Sound Design -- v2 is vision-only.
    "vocal":                None,
    "soundscape":           None,
}


# ---------------------------------------------------------------------------
# Model wrapper.
# ---------------------------------------------------------------------------
class VideoScore2Model:
    """Wrapper around the VideoScore2 (TIGER-Lab) Qwen2.5-VL fine-tune.

    VideoScore2 emits scores generatively (NOT via a regression head
    like v1.1). The official inference path (per README ### Inference):

      1. Build a chat-style message list with ONE user turn that
         contains a ``video`` content block (path + fps) and a text
         block with the v2 query template.
      2. Apply chat template -> processor; let qwen-vl-utils handle
         video frame sampling internally at fps=2.0.
      3. ``model.generate(...)`` with output_scores=True so we can
         recover per-token logits.
      4. Regex-parse the textual output to find each <hard score>; for
         each, also compute a log-likelihood-based soft score over
         tokens '1'..'5' at the position the hard score was generated.
      5. The soft score is what the README returns as the final per-
         dim score (it modulates the hard integer by the model's
         confidence: ``best_score * max_prob / total_prob``).

    We mirror that pipeline exactly, with two deliberate departures:
      * deterministic decoding (do_sample=False) for reproducibility;
      * a single forward without temperature, since temperature=0.7 in
        the README is for demoing variability, not for evaluation.
    """

    def __init__(self, model_path: str = _VS2_MODEL_ID,
                 device: str = "cuda:0", dtype: str = "bf16",
                 infer_fps: float = _VS2_INFER_FPS,
                 attn_impl: str = "sdpa"):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.infer_fps = infer_fps
        self.attn_impl = attn_impl
        self.model = None
        self.processor = None
        self.tokenizer = None

    def load(self) -> None:
        import torch
        # Per README: AutoModelForVision2Seq + AutoProcessor + AutoTokenizer.
        from transformers import (  # noqa: E402
            AutoModelForVision2Seq,
            AutoProcessor,
            AutoTokenizer,
        )

        torch_dtype = (
            torch.float16 if self.dtype == "fp16"
            else torch.bfloat16 if self.dtype == "bf16"
            else torch.float32
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True,
        )
        self.tokenizer = (
            getattr(self.processor, "tokenizer", None)
            or AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True, use_fast=False,
            )
        )
        # transformers 4.53 still accepts `torch_dtype`; pin it explicitly
        # rather than via `dtype` to keep behaviour stable across patch
        # versions of 4.53.x.
        #
        # ``attn_implementation="sdpa"`` (default) matches what the
        # official README demo and batch eval use implicitly (they don't
        # pass the kwarg, transformers picks sdpa on Qwen2.5-VL). sdpa
        # is both faster and more memory-efficient than eager, and we
        # have not observed the historical SIGFPE on the current
        # torch+cuBLAS combo. Override to "eager" via --attn_impl if a
        # future toolchain regression reintroduces the issue.
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation=self.attn_impl,
        ).eval().to(self.device)

    # ------------------------------------------------------------------
    # Helpers (lifted verbatim from the official README ### Inference).
    # ------------------------------------------------------------------
    @staticmethod
    def _find_score_token_index_by_prompt(
        prompt_text: str,
        tokenizer,
        gen_ids: list[int],
    ) -> int:
        """Locate the index in ``gen_ids`` of the digit immediately following
        ``prompt_text`` in the decoded output. Returns -1 on miss.

        ``prompt_text`` is the *bare* label, e.g. ``"visual quality"`` (no
        trailing colon). The match is tolerant of an optional list-prefix
        like ``"(1) "`` and of any descriptive tail (e.g. ``" \u2013
        clarity, smoothness, artifacts"``) before the colon and digit.
        """
        import re
        gen_str = tokenizer.decode(gen_ids, skip_special_tokens=False)
        # Allow optional "(1) " / leading newline prefix, then the bare
        # label, then any non-newline / non-colon descriptive tail, then
        # ':' and a digit.
        pattern = (
            r"(?:\(\d+\)\s*|\n\s*)?"
            + re.escape(prompt_text)
            + r"[^\n\r:]*?:\s*\d"
        )
        match = re.search(pattern, gen_str, flags=re.IGNORECASE)
        if not match:
            return -1
        # match.end() points just after the digit; the digit is the last
        # character of the match. Build the prefix that ends *at* that
        # digit so we can binary-walk gen_ids to its boundary.
        target_substr = gen_str[: match.end()]
        for i in range(len(gen_ids)):
            partial = tokenizer.decode(gen_ids[: i + 1],
                                       skip_special_tokens=False)
            if partial == target_substr:
                return i
        return -1

    @staticmethod
    def _ll_based_soft_score_normed(
        hard_val,
        token_idx: int,
        scores,
        tokenizer,
    ):
        """Compute the README-style soft score: ``best * max_prob / total_prob``.

        Walks the scores tensor at ``token_idx``, gets log-softmax over the
        score-token ids 1..5, normalises, and returns ``best_score *
        normalized_prob`` rounded to 4 dp. Returns None on failure.
        """
        import numpy as np
        import torch
        if hard_val is None or token_idx < 0:
            return None
        logits = scores[token_idx][0]
        score_probs = []
        for s in _VS2_SCORE_RANGE:
            ids = tokenizer.encode(str(s), add_special_tokens=False)
            if len(ids) == 1:
                tid = ids[0]
                logp = torch.log_softmax(logits, dim=-1)[tid].item()
                prob = float(np.exp(logp))
                score_probs.append((s, prob))
        if not score_probs:
            return None
        scores_list, probs_list = zip(*score_probs)
        total_prob = sum(probs_list)
        max_prob   = max(probs_list)
        best_score = scores_list[probs_list.index(max_prob)]
        normalized_prob = max_prob / total_prob if total_prob > 0 else 0.0
        return round(best_score * normalized_prob, 4)

    def _build_messages(self, query: str, video_path: str):
        # Match the official batch-eval contract verbatim: only
        # {type, video, fps}. qwen-vl-utils' default per-frame pixel
        # budget is what TIGER-Lab uses for VideoScore-Bench-v2 /
        # VideoGenReward-Bench evaluations; capping it ourselves
        # would silently degrade scoring quality (we hit -PLCC the
        # one time we did). OOM control belongs to the runner, not
        # to the input encoder.
        video_block = {
            "type": "video",
            "video": video_path,
            "fps": self.infer_fps,
        }
        return [{
            "role": "user",
            "content": [
                video_block,
                {"type": "text",
                 "text": _VS2_PROMPT_TEMPLATE.format(query=query)},
            ],
        }]

    def score_video(self, query: str, video_path: str,
                    *, max_new_tokens: int = _VS2_MAX_NEW_TOKENS
                    ) -> dict:
        """Return v2's three soft sub-scores plus diagnostic fields.

        Output dict shape::

            {
              "raw_sub_scores": {"visual quality": float|None,
                                 "text-to-video alignment": float|None,
                                 "physical/common-sense consistency": float|None},
              "hard_sub_scores": {<dim>: int|None, ...},  # parsed integers
              "raw_text":        "<full generated text>",
            }
        """
        if self.model is None:
            raise RuntimeError("VideoScore2Model not loaded; call load() first.")
        import re
        import torch
        try:
            from qwen_vl_utils import process_vision_info  # noqa: E402
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "qwen_vl_utils is required for VideoScore2 inference. "
                "Install it with `pip install qwen-vl-utils`."
            ) from e

        messages = self._build_messages(query, video_path)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=self.infer_fps,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            gen_out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                output_scores=True,
                return_dict_in_generate=True,
                do_sample=False,  # deterministic eval
            )
        sequences = gen_out.sequences
        scores    = gen_out.scores
        input_len = inputs["input_ids"].shape[1]
        gen_token_ids = sequences[0, input_len:].tolist()
        output_text = self.processor.batch_decode(
            sequences[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Parse hard integer scores. The official README regex assumes
        # "<label>: <digit>" with the colon directly after the label, but
        # our prompt template instructs the model to emit
        #   (1) visual quality – clarity, smoothness, artifacts: 4
        #   (2) text-to-video alignment – fidelity to the prompt: 4
        #   (3) physical/common-sense consistency – ... : 4
        # i.e. each label is followed by an em-dash + descriptive tail
        # before the colon. We therefore match each label, then any
        # non-colon / non-newline descriptive tail, then ":" and the
        # digit. Also prefer the segment after a closing </think> tag
        # when present, to avoid matching numbers inside the chain-of-
        # thought scratchpad.
        parse_text = output_text
        if "</think>" in parse_text:
            parse_text = parse_text.split("</think>", 1)[1]
        pattern = (
            r"visual\s+quality\b[^\n\r:]*?:\s*(\d+)"
            r".*?text[\s\-]*to[\s\-]*video\s+alignment\b[^\n\r:]*?:\s*(\d+)"
            r".*?physical/common[\s\-]?sense\s+consistency\b[^\n\r:]*?:\s*(\d+)"
        )
        m = re.search(pattern, parse_text, re.DOTALL | re.IGNORECASE)
        v_hard = int(m.group(1)) if m else None
        t_hard = int(m.group(2)) if m else None
        p_hard = int(m.group(3)) if m else None

        # Recover the token-position of each hard score in gen_token_ids,
        # then compute the LL-based soft score.
        # NB: pass the *bare* label (no trailing colon); the helper now
        # tolerates a descriptive tail (" – clarity, ...") before the
        # colon, matching what our prompt template asks the model to
        # emit.
        idx_v = self._find_score_token_index_by_prompt(
            "visual quality", self.tokenizer, gen_token_ids,
        )
        idx_t = self._find_score_token_index_by_prompt(
            "text-to-video alignment", self.tokenizer, gen_token_ids,
        )
        idx_p = self._find_score_token_index_by_prompt(
            "physical/common-sense consistency", self.tokenizer,
            gen_token_ids,
        )
        v_soft = self._ll_based_soft_score_normed(
            v_hard, idx_v, scores, self.tokenizer,
        )
        t_soft = self._ll_based_soft_score_normed(
            t_hard, idx_t, scores, self.tokenizer,
        )
        p_soft = self._ll_based_soft_score_normed(
            p_hard, idx_p, scores, self.tokenizer,
        )

        return {
            "raw_sub_scores": {
                "visual quality":                    v_soft,
                "text-to-video alignment":           t_soft,
                "physical/common-sense consistency": p_soft,
            },
            "hard_sub_scores": {
                "visual quality":                    v_hard,
                "text-to-video alignment":           t_hard,
                "physical/common-sense consistency": p_hard,
            },
            "raw_text": output_text,
        }


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
    cache directory across runs and forces re-download. SHA-256 is
    deterministic across processes and across Python versions, so v1.1
    and v2 runners share the same cache automatically.
    """
    import hashlib
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:32] + ".mp4"

def _score_one(model: VideoScore2Model, item: dict, video_dir: Path,
               num_frames: int) -> dict:
    """Score one sample. ``num_frames`` is unused for v2 (qwen-vl-utils
    samples internally at fps=infer_fps); kept in the signature for
    parity with v1.1."""
    del num_frames  # unused; v2 samples internally
    sample_id = item.get("id")
    query = item.get("query") or item.get("prompt") or ""
    url = _video_url_of(item)
    cache_path = video_dir / _url_to_cache_name(url)
    if not cache_path.is_file():
        cache_path = _download_to(url, cache_path)
    # Sanity-check the file is non-empty -- a 0-byte cache from a
    # previously-aborted download would crash decord with SIGFPE.
    if cache_path.stat().st_size == 0:
        cache_path.unlink()
        cache_path = _download_to(url, cache_path)
    out = model.score_video(query, str(cache_path))
    sub_scores = out["raw_sub_scores"]
    # raw_overall: mean over the 3 soft scores. None values are skipped
    # (a missing dim shouldn't drag the overall down to 0).
    valid = [v for v in sub_scores.values() if v is not None]
    overall = (sum(valid) / len(valid)) if valid else None
    # Legacy clip-to-[1,5] for back-compat metrics block.
    if overall is None:
        pred_5 = None
    else:
        pred_5 = round(max(1.0, min(5.0, float(overall))), 3)
    return {
        "id":              sample_id,
        "query":           query,
        "label_score":     item.get("label_score"),
        "raw_sub_scores":  sub_scores,
        "hard_sub_scores": out.get("hard_sub_scores"),
        "raw_text":        out.get("raw_text"),
        "raw_overall":     round(overall, 4) if overall is not None else None,
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
                   help=f"local model dir or HF repo id. Default: "
                        f"../../models/VideoScore2/ (auto-downloaded from "
                        f"{_VS2_MODEL_ID} on first use).")
    p.add_argument("--no_download", action="store_true",
                   help="if set, refuse to auto-download missing weights.")
    p.add_argument("--device", default=os.environ.get(
        "ARR_VIDEOSCORE2_DEVICE", "cuda:0"))
    p.add_argument("--dtype", default="bf16", choices=("fp16", "bf16", "fp32"),
                   help="compute dtype. Default bf16 (H20 native, matches "
                        "VideoScore2's training dtype). Use fp16 only if you "
                        "need the legacy precision; some H20 + torch 2.5 + "
                        "cuBLAS 12.1 combinations SIGFPE on fp16.")
    p.add_argument("--attn_impl", default="sdpa",
                   choices=("eager", "sdpa", "flash_attention_2"),
                   help="transformers attn_implementation. Default 'sdpa' "
                        "(what the official VideoScore2 README demo and "
                        "batch eval use implicitly). Switch to 'eager' if "
                        "you hit a cuBLAS/sdpa SIGFPE, or to "
                        "'flash_attention_2' once flash-attn is installed "
                        "for ~2x speedup.")
    p.add_argument("--num_frames", type=int, default=None,
                   help="DEPRECATED for v2. v1.1 used a fixed 48-frame budget; "
                        "v2 lets qwen-vl-utils sample internally at "
                        "--infer_fps. Accepted only for CLI compatibility "
                        "with the v1.1 driver scripts; ignored at runtime.")
    p.add_argument("--infer_fps", type=float, default=_VS2_INFER_FPS,
                   help=f"frame-sampling fps passed to qwen-vl-utils "
                        f"(default {_VS2_INFER_FPS}, matches the official "
                        f"VideoScore2 README demo).")
    # --total_pixels / --min_pixels are accepted but IGNORED. They were
    # used in an early version of this script to clamp visual-token
    # length and turned out to crush per-frame resolution to a degree
    # that flipped PLCC negative. Kept here only so existing shell
    # scripts that pass them don't error out.
    p.add_argument("--total_pixels", type=int, default=None,
                   help="DEPRECATED: ignored. Officially-aligned v2 inference "
                        "does NOT cap visual tokens. Kept only for CLI "
                        "back-compat with older runners.")
    p.add_argument("--min_pixels", type=int, default=None,
                   help="DEPRECATED: ignored. See --total_pixels.")
    p.add_argument("--video_dir", default="runs/baselines/_video_cache",
                   help="local mp4 cache directory; URLs are downloaded once.")
    p.add_argument("--limit", type=int, default=None,
                   help="if set, only score the first N samples (smoke test).")
    # --- single-sample subprocess mode (used by the in-process runner to
    #     isolate OOM-prone samples in a fresh CUDA context) ---
    p.add_argument("--single_sample_id", default=None,
                   help="INTERNAL: if set, score ONLY the sample with this "
                        "id from --dataset, write the per-sample dict to "
                        "--single_sample_out as JSON, and exit. Used by the "
                        "main loop to isolate OOM-failed samples in a fresh "
                        "process so allocator fragmentation cannot accumulate "
                        "across samples.")
    p.add_argument("--single_sample_out", default=None,
                   help="INTERNAL: path to write the single-sample result "
                        "dict when --single_sample_id is set.")
    p.add_argument("--target_subscore",
                   default=None,
                   choices=(None, "auto",
                            "visual quality",
                            "text-to-video alignment",
                            "physical/common-sense consistency",
                            "raw_overall_3"),
                   help="Which VS2 soft sub-score to use as the main "
                        "baseline output for PLCC/SRCC against GT label_score. "
                        "`auto` (default) infers from BASELINE_COVERAGE_V2 "
                        "based on the dataset path. v2 only has THREE heads "
                        "(unlike v1.1's five).")
    p.add_argument("--dimension", default=None,
                   help="EvalVerse L2 axis name (e.g. 'visual_quality') to "
                        "record in the output JSON; if omitted, inferred "
                        "from --dataset path.")
    args = p.parse_args()

    samples = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        samples = [samples]
    # Single-sample subprocess mode: filter to the requested id and
    # short-circuit the metric-aggregation tail, since the parent
    # process owns the metrics. We still go through the full load +
    # _score_one path so the result dict is bit-identical to what an
    # in-process call would have produced.
    if args.single_sample_id is not None:
        if not args.single_sample_out:
            raise SystemExit(
                "--single_sample_out is required with --single_sample_id"
            )
        match = [s for s in samples
                 if str(s.get("id")) == str(args.single_sample_id)]
        if not match:
            raise SystemExit(
                f"sample id {args.single_sample_id!r} not found in "
                f"{args.dataset}"
            )
        samples = match[:1]
    elif args.limit:
        samples = samples[:args.limit]

    video_dir = Path(args.video_dir).expanduser()
    video_dir.mkdir(parents=True, exist_ok=True)

    # Resolve --model_path; map repo-id / missing path to canonical
    # ../../models/VideoScore2/ slot.
    if args.model_path:
        if Path(args.model_path).is_dir():
            model_path = args.model_path
        else:
            model_path = str(ensure_model_local(
                args.model_path, allow_download=not args.no_download,
            ))
    else:
        model_path = str(ensure_model_local(
            _VS2_MODEL_ID, allow_download=not args.no_download,
        ))
    print(f"[VideoScore2] model_path resolved to {model_path}", flush=True)

    if args.total_pixels is not None or args.min_pixels is not None:
        print("[VideoScore2] WARN: --total_pixels / --min_pixels are "
              "deprecated and ignored (officially-aligned inference "
              "does not cap visual tokens).", flush=True)
    model = VideoScore2Model(
        model_path=model_path,
        device=args.device,
        dtype=args.dtype,
        infer_fps=args.infer_fps,
        attn_impl=args.attn_impl,
    )
    print(f"[VideoScore2] loading {model_path} on {args.device} "
          f"({args.dtype}) ...", flush=True)
    t0 = time.time()
    model.load()
    print(f"[VideoScore2] loaded in {time.time()-t0:.1f}s", flush=True)

    import torch as _torch_for_cleanup  # local alias to avoid touching imports

    def _gpu_cleanup():
        """Aggressively reclaim CUDA memory between samples.

        v2 calls ``model.generate(..., output_scores=True)`` which keeps
        ``max_new_tokens`` logits tensors plus a long KV cache alive for
        the duration of the call. Once the call returns, Python ref-
        counts drop to zero but PyTorch's caching allocator does NOT
        return the freed blocks to the driver -- it keeps them in its
        pool, fragmented. The next sample (whose KV cache size depends
        on video length and may be larger) then fails to find a single
        contiguous 49 GiB block and OOMs, even though >40 GiB is free
        on paper. ``empty_cache()`` forces the pool to release fully-
        free blocks back to the driver, defragmenting between samples.
        """
        try:
            import gc
            gc.collect()
            if _torch_for_cleanup.cuda.is_available():
                _torch_for_cleanup.cuda.empty_cache()
                _torch_for_cleanup.cuda.ipc_collect()
        except Exception:
            pass  # cleanup must never raise

    results: list[dict] = []
    failures = 0

    def _run_isolated(item: dict):
        """Last-resort fallback: re-run a single OOM-failed sample in a
        fresh Python subprocess so that PyTorch's allocator fragmentation
        from the parent's accumulated runs can't strand a 30+ GiB
        contiguous block. The child process loads the model from scratch
        (~30s on bf16/sdpa) and writes its single-sample result dict to
        a temp file we read back. Returns ``None`` on subprocess failure.
        """
        import subprocess as _sp
        import tempfile as _tf
        sid = item.get("id")
        with _tf.NamedTemporaryFile(
            mode="r", suffix=".json", delete=False,
        ) as tmp:
            tmp_path = tmp.name
        try:
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--dataset", args.dataset,
                "--output", "/dev/null",
                "--device", args.device,
                "--dtype", args.dtype,
                "--attn_impl", args.attn_impl,
                "--infer_fps", str(args.infer_fps),
                "--video_dir", str(video_dir),
                "--model_path", model_path,
                "--no_download",
                "--single_sample_id", str(sid),
                "--single_sample_out", tmp_path,
            ]
            print(f"[VideoScore2] launching isolated subprocess for "
                  f"sample {sid!r}", flush=True)
            t_iso = time.time()
            cp = _sp.run(cmd, capture_output=True, text=True, timeout=600)
            dt_iso = time.time() - t_iso
            if cp.returncode != 0:
                tail = (cp.stderr or cp.stdout or "")[-1500:]
                print(f"[VideoScore2] isolated subprocess for {sid!r} "
                      f"failed (rc={cp.returncode}, {dt_iso:.1f}s); "
                      f"tail:\n{tail}", flush=True)
                return None
            try:
                data = json.loads(Path(tmp_path).read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[VideoScore2] isolated subprocess for {sid!r} "
                      f"returned unreadable output: {e}", flush=True)
                return None
            print(f"[VideoScore2] isolated subprocess for {sid!r} ok "
                  f"({dt_iso:.1f}s)", flush=True)
            return data
        finally:
            try: os.unlink(tmp_path)
            except OSError: pass

    for i, item in enumerate(samples, 1):
        attempt_err = None
        r = None
        for attempt in range(2):  # 1 retry on OOM, then give up
            try:
                r = _score_one(model, item, video_dir, args.num_frames)
                attempt_err = None
                break
            except _torch_for_cleanup.cuda.OutOfMemoryError as e:
                attempt_err = e
                print(f"[VideoScore2] OOM on sample {item.get('id')!r} "
                      f"(attempt {attempt+1}/2); cleaning up and retrying",
                      flush=True)
                _gpu_cleanup()
            except Exception as e:
                attempt_err = e
                break  # non-OOM errors: don't retry
        if r is not None:
            results.append(r)
        elif isinstance(attempt_err, _torch_for_cleanup.cuda.OutOfMemoryError):
            # In-process retries exhausted on OOM; fall back to a fully
            # isolated subprocess for this one sample. This is the only
            # reliable cure for cross-sample allocator fragmentation.
            _gpu_cleanup()
            iso = _run_isolated(item)
            if iso is not None:
                results.append(iso)
            else:
                failures += 1
                results.append({"id": item.get("id"),
                                "error": f"OOM (isolated retry also failed): "
                                         f"{attempt_err}"})
        else:
            failures += 1
            results.append({"id": item.get("id"), "error": str(attempt_err)})
        # Always defragment between samples, success or not. Cheap (~50ms)
        # compared to the 5-15s per-sample inference cost.
        _gpu_cleanup()
        if i % 5 == 0 or i == len(samples):
            print(f"[VideoScore2] {i}/{len(samples)} done "
                  f"({failures} failed)", flush=True)

    # Single-sample subprocess mode: write only the per-sample result
    # dict and exit. The parent process aggregates metrics, so we must
    # NOT compute or write them here (and must not overwrite the
    # parent's --output file -- /dev/null is passed by the parent).
    if args.single_sample_id is not None:
        out_one = results[0] if results else {
            "id": args.single_sample_id,
            "error": "no result produced",
        }
        Path(args.single_sample_out).write_text(
            json.dumps(out_one, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[VideoScore2] single-sample done -> "
              f"{args.single_sample_out}", flush=True)
        return

    # ------------------------------------------------------------------
    # Three-track metrics (mirrors baseline_videoscore.py / v1.1):
    #   (a) `metrics`           : raw continuous output of the canonical
    #       sub-score for the EvalVerse axis being evaluated. PLCC/SRCC
    #       + MAE in z-score and affine-calibrated forms.
    #   (b) `metrics_per_subscore`: 5 raw heads + raw_overall_4 +
    #       raw_overall_5, each with PLCC/SRCC/MAE_zs.
    #   (c) `metrics_legacy_pred5`: legacy 1..5 clipped pred_score (kept
    #       for back-compat; do not use for the SOTA table).
    # ------------------------------------------------------------------
    dim = args.dimension or _infer_dimension_from_dataset(args.dataset)
    target = args.target_subscore
    if target in (None, "auto"):
        # NOTE: v2 has its own coverage table (3 heads, not 5). Don't
        # reuse v1.1's BASELINE_COVERAGE -- it would route to keys that
        # don't exist in v2's output (e.g. "temporal consistency").
        target = BASELINE_COVERAGE_V2.get(dim or "", None) or "raw_overall_3"
    print(f"[VideoScore2] target_subscore={target!r} (dimension={dim!r})",
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
            if name == "raw_overall_3":
                # mean over all 3 soft sub-scores; skip any None entry.
                sub = r.get("raw_sub_scores") or {}
                vals = [sub.get(k) for k in _VS2_KEYS
                        if sub.get(k) is not None]
                if not vals:
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

    # (a) main
    main_preds, main_labels = _series(target, results)
    main_plcc, main_srcc, main_mae_zs = _compute_metrics_v1(
        main_preds, main_labels, mae_mode="zscore",
    )
    _, _, main_mae_aff = _compute_metrics_v1(
        main_preds, main_labels, mae_mode="affine",
    )

    # (b) per-sub-score (3 heads + raw_overall_3)
    metrics_per_subscore: dict[str, dict] = {}
    for name in (*_VS2_KEYS, "raw_overall_3"):
        ps, ls = _series(name, results)
        plcc_n, srcc_n, mae_n = _compute_metrics_v1(ps, ls, mae_mode="zscore")
        metrics_per_subscore[name] = {
            "plcc":   plcc_n,
            "srcc":   srcc_n,
            "mae_zs": mae_n,
            "n":      len(ps),
        }

    # (c) legacy
    legacy_preds, legacy_labels = _series("pred_score_legacy_clip5", results)
    legacy_plcc, legacy_srcc, legacy_mae = _compute_metrics_legacy(
        legacy_preds, legacy_labels,
    )

    out = {
        "method":      "VideoScore2",
        "model_path":  args.model_path,
        "dataset":     args.dataset,
        "dimension":   dim,
        "target_subscore": target,
        "total":       len(samples),
        "parseable":   len(main_preds),
        "failures":    failures,
        "metrics": {
            "plcc":       main_plcc,
            "srcc":       main_srcc,
            "mae_zs":     main_mae_zs,
            "mae_affine": main_mae_aff,
            "n":          len(main_preds),
        },
        "metrics_per_subscore": metrics_per_subscore,
        "metrics_legacy_pred5": {
            "plcc": legacy_plcc,
            "srcc": legacy_srcc,
            "mae":  legacy_mae,
            "n":    len(legacy_preds),
        },
        "predictions": results,
    }
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    def _fmt(x):
        return f"{x:.4f}" if isinstance(x, float) else "n/a"
    print(f"[VideoScore2] [main:{target!r}] PLCC={_fmt(main_plcc)} "
          f"SRCC={_fmt(main_srcc)} MAE_zs={_fmt(main_mae_zs)} "
          f"MAE_aff={_fmt(main_mae_aff)} "
          f"N={len(main_preds)}/{len(samples)} (failures={failures})",
          flush=True)
    print("[VideoScore2] per-sub-score (PLCC / SRCC / MAE_zs / N):",
          flush=True)
    for name, m in metrics_per_subscore.items():
        print(f"               {name:28s}  "
              f"PLCC={_fmt(m['plcc'])}  SRCC={_fmt(m['srcc'])}  "
              f"MAE_zs={_fmt(m['mae_zs'])}  N={m['n']}",
              flush=True)
    print(f"[VideoScore2] [legacy clip5] PLCC={_fmt(legacy_plcc)} "
          f"SRCC={_fmt(legacy_srcc)} MAE={_fmt(legacy_mae)} "
          f"N={len(legacy_preds)} (kept for back-compat only)",
          flush=True)
    print(f"[VideoScore2] wrote -> {out_path}")


if __name__ == "__main__":
    main()