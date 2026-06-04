"""RewardVerse — zero-shot Gemini baseline runner with three explicit modes.

Same model (gemini-3.1-pro), same venus llmproxy, same native-video path
as the rubric judge — but **no rubric**, so the score difference attributable
to the rubric+EvalVerse prior is isolated.

Modes
-----
B1a : visual-quality only (verbatim copy of ``pointwise_zero_shot_judge.py``).
B1b : single second-level dimension (need ``--dim``); 18 separate runs cover
      the full 18-dim space.
B1c : 7 first-level groups in ONE Gemini call.

Usage
-----
::

    python scripts/v_arr_mvp/baseline_zero_shot_18d.py \\
        --mode B1a \\
        --dataset data/v_arr_mvp/visual_quality_mvp/shared_test_pointwise.json \\
        --output runs/baselines/B1a/shared_test.json

    python scripts/v_arr_mvp/baseline_zero_shot_18d.py \\
        --mode B1b --dim character \\
        --dataset data/v_arr_mvp/character_mvp/test_pointwise_character.json \\
        --output runs/baselines/B1b/character.json

    python scripts/v_arr_mvp/baseline_zero_shot_18d.py \\
        --mode B1c \\
        --dataset data/v_arr_mvp/visual_quality_mvp/shared_test_pointwise.json \\
        --output runs/baselines/B1c/shared_test.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Same env defaults as the rubric pointwise judger — keep B1 honest as a
# "same-model different-prompt" comparison rather than accidentally
# introducing a video preprocessing difference.
os.environ.setdefault("ARR_VIDEO_INPUT_STRATEGY", "native_video")
os.environ.setdefault("ARR_VIDEO_BLOCK_TYPE", "gemini_multimodal_url")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dimension_registry import (  # type: ignore  # noqa: E402
    DIMENSIONS_BY_NAME,
    list_dimension_names,
)

from rubric_pipeline.models.openai_chat_model import OpenAIChatModel  # noqa: E402
from rubric_pipeline.utils.media import (  # noqa: E402
    add_videos_to_messages,
    extract_video_paths,
    is_video_task,
)


# ---------------------------------------------------------------------------
# Pointwise metric helpers (independent of judger.py to keep this file
# self-contained for downstream SOTA-table importing).
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


def _compute_metrics(preds: list[float], labels: list[float]):
    if not preds or not labels or len(preds) != len(labels):
        return None, None, None
    n = len(preds)
    mae = sum(abs(p - l) for p, l in zip(preds, labels)) / n
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
    return plcc, srcc, mae


# ---------------------------------------------------------------------------
# Prompt templates.
# ---------------------------------------------------------------------------

# B1a — verbatim copy of the visual-quality prompt used in
# scripts/v_arr_mvp/pointwise_zero_shot_judge.py.
_PROMPT_B1A = """
You are an impartial video-quality judge. You will see exactly ONE generated
video clip for the text prompt below. Score the visual quality on a
continuous 1.0-5.0 scale considering rendering quality, physics plausibility,
and temporal consistency. Do NOT inspect or invent any taxonomy; answer purely
from what you observe.

Scoring rubric (use the WHOLE 1.0-5.0 range, do not bias to 3.0; one decimal
place is fine, e.g. 4.3 / 2.7):
  5.0 = excellent overall visual quality, no obvious artifacts
  4.0 = good, minor issues only
  3.0 = average, noticeable but acceptable issues
  2.0 = poor, multiple obvious issues
  1.0 = very poor, severe artifacts / broken physics / heavy flicker

Prompt: {query}

Return STRICT JSON in this exact shape:
{{"score": <float in [1.0, 5.0]>, "reason": "<one short sentence>"}}
""".strip()


# B1b — single-dimension zero-shot.
_PROMPT_B1B = """
You are an impartial video judge. You will see exactly ONE generated video
clip for the text prompt below. You will score the video on ONE specific
quality dimension named "{dim_aspect} > {dim_main}" (slug: {dim_name}).

Definition of this dimension (use only this; do NOT score other dimensions):
{dim_definition}

{prompt_clause}{audio_clause}

Score on a continuous 1.0-5.0 scale. Use the whole 1.0-5.0 range; one
decimal place is fine (e.g. 4.3 / 2.7):
  5.0 = excellent on this dimension, essentially flawless
  4.0 = good, only minor issues on this dimension
  3.0 = average, noticeable but acceptable issues on this dimension
  2.0 = poor, multiple obvious issues on this dimension
  1.0 = very poor, this dimension is essentially broken

Return STRICT JSON in this exact shape:
{{"score": <float in [1.0, 5.0]>, "reason": "<one short sentence>"}}
""".strip()


# B1c — 7 first-level groups in one call.
_L1_DEFS: dict[str, str] = {
    "visual_concept_design":
        "Are the on-screen characters and the world / scene clearly identifiable, "
        "consistent with what the prompt asks for, and visually coherent?",
    "acting":
        "Are character expressions, actions, and identities consistent and "
        "physically/emotionally plausible across the clip?",
    "cinematography":
        "Are the framing, lens choices, camera moves and shot composition "
        "deliberate, stable, and well-executed?",
    "aesthetics":
        "Are colour, lighting, materials/textures, and overall rendering quality "
        "(physics, temporal consistency, fine detail) at a high standard?",
    "affectivity":
        "Does the clip establish a clear, stable emotional tone and progression "
        "appropriate to the scene?",
    "multi_shot":
        "If the clip has cuts, are the cuts logically connected (continuity, "
        "spatial coherence) and well-paced (rhythm, shot duration)?",
    "sound_design":
        "Is the audio (vocals, music, ambient sound, SFX) present, plausible, "
        "synchronised with the visuals, and consistent with the depicted scene?",
}


_PROMPT_B1C_HEAD = """
You are an impartial video judge. You will see exactly ONE generated video
clip for the text prompt below. Score the video on each of the SEVEN quality
groups described in this list. Treat each group INDEPENDENTLY (a failure on
group X must NOT drag down group Y).

Score each group on a continuous 1.0-5.0 scale (one decimal place is fine):
  5.0 = excellent on this group
  4.0 = good, minor issues only
  3.0 = average
  2.0 = poor, multiple obvious issues
  1.0 = very poor / essentially broken on this group

Use the whole 1.0-5.0 range; do not bias toward 3.0. If a group is not
applicable to the clip (e.g. ``sound_design`` for a clip with no audio
track), set its score to ``null`` and its reason to ``"not applicable"``.

The seven groups (use these slugs verbatim in the JSON):
{group_block}

Prompt: {query}

Return STRICT JSON in this exact shape (do NOT wrap in code fences):
{{
  "visual_concept_design": {{"score": <float|null>, "reason": "<one short sentence>"}},
  "acting":                {{"score": <float|null>, "reason": "..."}},
  "cinematography":        {{"score": <float|null>, "reason": "..."}},
  "aesthetics":            {{"score": <float|null>, "reason": "..."}},
  "affectivity":           {{"score": <float|null>, "reason": "..."}},
  "multi_shot":            {{"score": <float|null>, "reason": "..."}},
  "sound_design":          {{"score": <float|null>, "reason": "..."}}
}}
""".strip()


def _build_b1c_prompt(query: str) -> str:
    block_lines = [
        f"  - {name}: {definition}"
        for name, definition in _L1_DEFS.items()
    ]
    return _PROMPT_B1C_HEAD.format(
        group_block="\n".join(block_lines),
        query=query,
    )


# ---------------------------------------------------------------------------
# Output parsing.
# ---------------------------------------------------------------------------
_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(raw: str) -> dict | None:
    if not raw:
        return None
    m = _JSON_BLOCK_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _coerce_1_5(value: Any) -> float | None:
    """Coerce a model-emitted scalar into a clamped float in [1.0, 5.0].

    Returns None if value is null/non-numeric/way out of range. We tolerate a
    +/-0.5 overshoot (clamp into range) for models that occasionally emit
    out-of-range scalars (e.g. 5.5 or 0.8).
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f < 1.0 - 0.5 or f > 5.0 + 0.5:
        return None
    return round(max(1.0, min(5.0, f)), 3)


# ---------------------------------------------------------------------------
# Helpers shared by all three modes.
# ---------------------------------------------------------------------------
def _extract_paths(item: dict, task_type: str) -> list[str]:
    paths = extract_video_paths(item, task_type) if is_video_task(task_type) else []
    if paths:
        return paths
    resp = item.get("response")
    if isinstance(resp, str):
        return [resp]
    if isinstance(resp, list) and resp:
        return [str(resp[0])]
    return []


def _build_messages(user_text: str, paths: list[str], task_type: str) -> list[dict]:
    messages = [
        {"role": "system",
         "content": "Output strictly valid JSON. No commentary outside JSON."},
        {"role": "user", "content": user_text},
    ]
    return add_videos_to_messages(messages, paths, task_type, has_base=False)


async def _achat_text(model: OpenAIChatModel, messages: list[dict]) -> str:
    try:
        response = await model.achat(messages=messages)
    except Exception as exc:
        return json.dumps({"_error": str(exc)})
    if isinstance(response, dict):
        return response.get("content", "")
    return getattr(response, "content", "") or ""


# ---------------------------------------------------------------------------
# Per-mode async judging.
# ---------------------------------------------------------------------------
async def _judge_b1a_one(model: OpenAIChatModel, item: dict) -> dict:
    sample_id = item.get("id")
    query     = item.get("query") or item.get("prompt") or ""
    task_type = item.get("task_type", "t2v")
    paths     = _extract_paths(item, task_type)
    if len(paths) != 1:
        return {"id": sample_id, "error": f"expected 1 video path, got {len(paths)}"}
    user_text = _PROMPT_B1A.format(query=query)
    messages = _build_messages(user_text, paths, task_type)
    raw = await _achat_text(model, messages)
    parsed = _extract_json(raw) or {}
    return {
        "id":          sample_id,
        "query":       query,
        "label_score": item.get("label_score"),
        "pred_score":  _coerce_1_5(parsed.get("score")),
        "raw":         raw,
    }


async def _judge_b1b_one(model: OpenAIChatModel, item: dict, dim_name: str) -> dict:
    spec = DIMENSIONS_BY_NAME[dim_name]
    sample_id = item.get("id")
    query     = item.get("query") or item.get("prompt") or ""
    task_type = item.get("task_type", "t2v")
    paths     = _extract_paths(item, task_type)
    if len(paths) != 1:
        return {"id": sample_id, "error": f"expected 1 video path, got {len(paths)}"}

    if spec.expose_prompt:
        prompt_clause = f"User instruction prompt for this clip: {query}"
    else:
        prompt_clause = (
            "This dimension is INTRINSIC: judge it from the video alone, "
            "WITHOUT consulting any user prompt. If a prompt string is "
            "supplied, treat it as a placeholder and ignore it."
        )

    audio_clause = ""
    if spec.needs_audio:
        audio_clause = (
            "\n\nThis dimension REQUIRES the audio track. The video clip "
            "carries its native audio; cite audible evidence (presence/"
            "naturalness of dialogue or vocals, lip-sync, sound effects "
            "synced to visual events, music-scene alignment) — not just "
            "visual cues."
        )

    user_text = _PROMPT_B1B.format(
        dim_aspect=spec.aspect,
        dim_main=spec.main,
        dim_name=spec.name,
        dim_definition=spec.description_en,
        prompt_clause=prompt_clause,
        audio_clause=audio_clause,
    )
    messages = _build_messages(user_text, paths, task_type)
    raw = await _achat_text(model, messages)
    parsed = _extract_json(raw) or {}
    return {
        "id":          sample_id,
        "dim":         dim_name,
        "query":       query,
        "label_score": item.get("label_score"),
        "pred_score":  _coerce_1_5(parsed.get("score")),
        "raw":         raw,
    }


async def _judge_b1c_one(model: OpenAIChatModel, item: dict) -> dict:
    sample_id = item.get("id")
    query     = item.get("query") or item.get("prompt") or ""
    task_type = item.get("task_type", "t2v")
    paths     = _extract_paths(item, task_type)
    if len(paths) != 1:
        return {"id": sample_id, "error": f"expected 1 video path, got {len(paths)}"}
    user_text = _build_b1c_prompt(query)
    messages = _build_messages(user_text, paths, task_type)
    raw = await _achat_text(model, messages)
    parsed = _extract_json(raw) or {}
    pred_l1: dict[str, float | None] = {}
    for k in _L1_DEFS:
        block = parsed.get(k)
        if isinstance(block, dict):
            pred_l1[k] = _coerce_1_5(block.get("score"))
        elif isinstance(block, (int, float, str)):
            pred_l1[k] = _coerce_1_5(block)
        else:
            pred_l1[k] = None
    return {
        "id":          sample_id,
        "query":       query,
        "label_score": item.get("label_score"),
        "pred_l1":     pred_l1,
        "raw":         raw,
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
async def main_async(args: argparse.Namespace) -> None:
    samples = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if not isinstance(samples, list):
        samples = [samples]

    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = OpenAIChatModel(
        model=args.model,
        api_key=api_key,
        base_url=args.base_url,
        temperature=0,
    )

    sem = asyncio.Semaphore(args.concurrency)

    async def _runner(item: dict) -> dict:
        async with sem:
            if args.mode == "B1a":
                return await _judge_b1a_one(model, item)
            elif args.mode == "B1b":
                return await _judge_b1b_one(model, item, args.dim)
            elif args.mode == "B1c":
                return await _judge_b1c_one(model, item)
            else:
                raise ValueError(f"unknown mode {args.mode!r}")

    results: list[dict] = await asyncio.gather(*(_runner(s) for s in samples))

    # Metrics.
    if args.mode in {"B1a", "B1b"}:
        preds: list[float] = []
        labels: list[float] = []
        for r in results:
            s = r.get("pred_score")
            l = r.get("label_score")
            if s is None or l is None:
                continue
            try:
                preds.append(float(s))
                labels.append(float(l))
            except (TypeError, ValueError):
                continue
        plcc, srcc, mae = _compute_metrics(preds, labels)
        metrics = {"plcc": plcc, "srcc": srcc, "mae": mae, "n": len(preds)}
    else:  # B1c
        labels_f: list[float | None] = []
        for r in results:
            v = r.get("label_score")
            try:
                labels_f.append(float(v) if v is not None else None)
            except (TypeError, ValueError):
                labels_f.append(None)
        metrics = {}
        for grp in _L1_DEFS:
            preds_l = [r.get("pred_l1", {}).get(grp) for r in results]
            paired = [(p, l) for p, l in zip(preds_l, labels_f)
                      if p is not None and l is not None]
            if len(paired) < 2:
                metrics[grp] = None
                continue
            p_vec = [p for p, _ in paired]
            l_vec = [l for _, l in paired]
            plcc, srcc, mae = _compute_metrics(p_vec, l_vec)
            metrics[grp] = {"plcc": plcc, "srcc": srcc, "mae": mae, "n": len(paired)}

    parseable = sum(
        1 for r in results
        if (args.mode != "B1c" and r.get("pred_score") is not None)
        or (args.mode == "B1c" and any(v is not None for v in r.get("pred_l1", {}).values()))
    )

    out = {
        "mode":        f"{args.mode}:{args.dim}" if args.mode == "B1b" else args.mode,
        "model":       args.model,
        "dataset":     args.dataset,
        "total":       len(samples),
        "parseable":   parseable,
        "metrics":     metrics,
        "predictions": results,
    }
    out_path = Path(args.output).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[B1] mode={out['mode']} parseable={parseable}/{len(samples)}")
    if args.mode in {"B1a", "B1b"}:
        m = metrics
        plcc = f"{m['plcc']:.4f}" if m["plcc"] is not None else "n/a"
        srcc = f"{m['srcc']:.4f}" if m["srcc"] is not None else "n/a"
        mae  = f"{m['mae']:.4f}"  if m["mae"]  is not None else "n/a"
        print(f"     PLCC={plcc} SRCC={srcc} MAE={mae} N={m['n']}")
    else:
        for grp, m in metrics.items():
            if m is None:
                print(f"     L1 {grp:<22} (no parseable rows)")
                continue
            plcc = f"{m['plcc']:.4f}" if m["plcc"] is not None else "n/a"
            srcc = f"{m['srcc']:.4f}" if m["srcc"] is not None else "n/a"
            print(f"     L1 {grp:<22} PLCC={plcc} SRCC={srcc} N={m['n']}")
    print(f"[B1] wrote -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", required=True, choices=("B1a", "B1b", "B1c"),
                   help="B1a=visual-quality only; B1b=single dim (need --dim); "
                        "B1c=7 first-level groups in one call.")
    p.add_argument("--dim", default=None,
                   help="dimension slug (REQUIRED when --mode B1b).")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=os.environ.get("ARR_MODEL", "gemini-3.1-pro"))
    p.add_argument("--base_url", default=os.environ.get(
        "ARR_BASE_URL", "http://v2.open.venus.oa.com/llmproxy"))
    p.add_argument("--concurrency", type=int, default=2)
    args = p.parse_args()
    if args.mode == "B1b":
        if not args.dim:
            p.error("--mode B1b requires --dim <slug>")
        if args.dim not in DIMENSIONS_BY_NAME:
            p.error(
                f"unknown --dim {args.dim!r}; valid: {','.join(list_dimension_names())}"
            )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
