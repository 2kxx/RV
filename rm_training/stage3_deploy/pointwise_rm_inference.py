"""rm_training.stage3_deploy.pointwise_rm_inference
===============================================
Production-time pointwise reward-model inference.

Given **(query, caption, video)** the trained student emits its own
rubric R̂ and per-criterion scores, then aggregates::

    S(y) = Σ w_k · c_k / Σ w_k

This is the same code path used inside Stage ② (one rubric-gen + one
score forward), with two practical differences:
  • Sampling temperature defaults to 0 (greedy) for reproducibility.
  • If the student is a LoRA checkpoint, the wrapper merges the LoRA
    weights into the base model on first call.

CLI
---
::

    python rm_training/stage3_deploy/pointwise_rm_inference.py \\
        --student_ckpt ckpts/student_grpo/checkpoint-final \\
        --query "A red sports car drifts around a mountain bend" \\
        --video data/v_arr_mvp/videos/varr_seed_001_win.mp4

Returns a JSON record on stdout::

    {
      "query": "...",
      "video": "...",
      "rubric": [...5 entries...],
      "scores": [{"rubric_index":1, "score": 2.5}, ...],
      "S": 2.83
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rm_training.stage1_teacher.task_descriptions import (
    TaskSpec,
    load_task_specs,
)
from rm_training.stage2_grpo.rubric_parser   import (
    build_full_from_scores,
    format_score,
    parse_rubric_only,
)
from rm_training.stage2_grpo.student_wrapper import (
    DimensionContext,
    StudentConfig,
    StudentWrapper,
)


@dataclass
class InferenceResult:
    query:   str
    video:   str
    rubric:  list[dict]
    scores:  list[dict]
    s_total: float | None
    raw_rubric_text: str
    raw_score_text:  str

    def to_json(self) -> str:
        return json.dumps({
            "query":  self.query,
            "video":  self.video,
            "rubric": self.rubric,
            "scores": self.scores,
            "S":      self.s_total,
        }, ensure_ascii=False, indent=2)


def score_one(
    student:     StudentWrapper,
    dimension:   str,
    caption:     str,
    video_path:  str,
    *,
    task_specs:  dict[str, TaskSpec] | None = None,
) -> InferenceResult:
    """Single (dimension, caption, video) → S(y).

    Two-forward implementation matching Stage ② (deployment-consistent).

    ``dimension`` selects which TaskSpec to inject as DimensionContext;
    if no spec is registered for it (or task_specs is None), falls back
    to a generic stub. ``caption`` is the prompt that drove the video
    generation (a.k.a. the v_arr_mvp pair file's ``query`` field).
    """
    if task_specs:
        spec = task_specs.get(dimension)
    else:
        spec = None
    if spec is not None:
        dim_ctx = DimensionContext(
            dimension        = spec.dimension,
            evalverse_path   = (
                f"{spec.evalverse_aspect} > {spec.evalverse_main}"
                if spec.evalverse_aspect and spec.evalverse_main
                else spec.dimension
            ),
            task_description = spec.task_description_short,
            sub_aspects      = list(spec.sub_aspects),
        )
    else:
        dim_ctx = DimensionContext.generic(dimension)

    rubric_text   = student.generate_rubric(caption, dim_ctx=dim_ctx)
    rubric_parsed = parse_rubric_only(rubric_text)
    score_text, scores = student.score_video(
        caption, video_path, rubric_text,
        dim_ctx=dim_ctx,
        rubric_paths=[r.path for r in rubric_parsed.rubrics],
    )
    score_parsed = build_full_from_scores(
        rubric_parsed.rubrics, scores, raw_text=score_text,
    )

    s_total = format_score(score_parsed)

    return InferenceResult(
        query=f"[{dimension}] {caption}",
        video=video_path,
        rubric=[{"path": r.path, "weight": r.weight, "tips": r.tips}
                for r in score_parsed.rubrics],
        scores=[{"rubric_index": i + 1, "score": s}
                for i, s in enumerate(score_parsed.scores)],
        s_total=s_total,
        raw_rubric_text=rubric_text,
        raw_score_text=score_text,
    )


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student_ckpt", default=None,
                   help="Trained student checkpoint dir (omit for dummy mode)")
    p.add_argument("--dimension", required=True,
                   help="Dimension slug (e.g. 'action', 'visual_quality'). "
                        "Selects the TaskSpec injected into the prompt.")
    p.add_argument("--caption",   required=True,
                   help="Video generation prompt (the 'query' field in "
                        "v_arr_mvp pair files). NOT the dimension name.")
    p.add_argument("--video",     required=True)
    p.add_argument("--no_task_specs", action="store_true",
                   help="Bypass yaml-loaded task descriptions; use a generic "
                        "scaffold instead. Mostly for ablation.")
    p.add_argument("--temperature",  type=float, default=0.0)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    args = p.parse_args()

    student = StudentWrapper(StudentConfig(
        model_path=args.student_ckpt,
        dummy=(args.student_ckpt is None),
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    ))
    task_specs = None if args.no_task_specs else load_task_specs(ROOT)
    res = score_one(student, args.dimension, args.caption, args.video,
                    task_specs=task_specs)
    print(res.to_json())


if __name__ == "__main__":
    main()
