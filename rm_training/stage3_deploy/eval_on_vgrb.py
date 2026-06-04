"""rm_training.stage3_deploy.eval_on_vgrb
=======================================
Pairwise evaluation on VideoGen-RewardBench using the trained student.

Workflow
--------
For each pair (id, query, [video_a, video_b], gt_label):
    S_a = pointwise_rm_inference.score_one(query, video_a)
    S_b = pointwise_rm_inference.score_one(query, video_b)
    decision = A if S_a > S_b + τ
               B if S_b > S_a + τ
               same otherwise
Then compute acc / acc_no_tie / macro_acc against gt_label and write a
summary JSON in the same shape as the legacy
``runs/external_benchmarks/.../aggregation_*.json`` files.

This is the **M1 milestone entrypoint** (see runs/STATUS_SYNC.md §8.3
M1): once a student RM is good enough, point this script at the full
25.2k VGRB CSV and you've crossed the milestone.

CLI
---
::

    python rm_training/stage3_deploy/eval_on_vgrb.py \\
        --student_ckpt ckpts/student_grpo/checkpoint-final \\
        --vgrb_dataset /group/40063/jacuzzitang/datasets/_v_arr_external_subsets/videogen_rewardbench/pairwise_view/Overall.json \\
        --output       runs/external_benchmarks/videogen_rewardbench/student_rm/Overall.json \\
        --tau 0.0 --concurrency 1 --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rm_training.stage1_teacher.task_descriptions import load_task_specs
from rm_training.stage2_grpo.student_wrapper import StudentConfig, StudentWrapper
from rm_training.stage3_deploy.pointwise_rm_inference import score_one


# ---------------------------------------------------------------------------
# VGRB axis → RewardVerse dimension slug mapping.
#
# VGRB scores three axes in addition to Overall:
#   VQ  — Video Quality       (≈ our visual_quality dim)
#   MQ  — Motion Quality      (≈ our consistency / action mix; default to consistency)
#   TA  — Text Alignment      (≈ our grounding dim)
# Overall blends all three. We pick the closest 1-to-1 mapping; the user
# can override via --dimension on the CLI.
# ---------------------------------------------------------------------------
VGRB_AXIS_TO_DIM = {
    "VQ":      "visual_quality",
    "MQ":      "consistency",
    "TA":      "grounding",
    "Overall": "visual_quality",   # generic fallback
}


# ---------------------------------------------------------------------------
# Helpers (mirror of pairwise_rubric_judge.py decision logic for symmetry).
# ---------------------------------------------------------------------------
def _score_diff_to_decision(s_a: float | None,
                            s_b: float | None,
                            tau: float) -> str:
    if s_a is None or s_b is None:
        return "same"
    delta = s_a - s_b
    if delta >  tau: return "A"
    if delta < -tau: return "B"
    return "same"


def _label_rank_to_gt(rank) -> str:
    if not isinstance(rank, (list, tuple)) or len(rank) != 2:
        return "same"
    a, b = rank[0], rank[1]
    try:
        a_i, b_i = int(a), int(b)
    except (TypeError, ValueError):
        return "same"
    if a_i < b_i: return "A"
    if a_i > b_i: return "B"
    return "same"


# ---------------------------------------------------------------------------
# Main eval loop.
# ---------------------------------------------------------------------------
def evaluate(args: argparse.Namespace) -> None:
    rows = json.loads(Path(args.vgrb_dataset).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"[eval] unexpected dataset shape in {args.vgrb_dataset}")
    if args.limit is not None and args.limit > 0:
        rows = rows[:args.limit]
    print(f"[eval] {len(rows)} pairs to score from {args.vgrb_dataset}", file=sys.stderr)

    student = StudentWrapper(StudentConfig(
        model_path=args.student_ckpt,
        dummy=(args.student_ckpt is None),
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    ))
    task_specs = None if args.no_task_specs else load_task_specs(ROOT)

    # Resolve which dimension to feed score_one. Priority order:
    #   --dimension CLI override > VGRB_AXIS_TO_DIM lookup by file stem >
    #   "visual_quality" default.
    axis = Path(args.vgrb_dataset).stem
    dim = (
        args.dimension or VGRB_AXIS_TO_DIM.get(axis, "visual_quality")
    )
    print(f"[eval] axis={axis}  dimension={dim}", file=sys.stderr)

    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    per_pair: list[dict] = []
    confusion: dict[str, Counter] = {"A": Counter(), "B": Counter(), "same": Counter()}
    n_correct = 0
    n_no_tie  = 0
    n_no_tie_correct = 0

    t0 = time.time()
    for i, r in enumerate(rows):
        pid = r.get("id", f"pair_{i:05d}")
        query = r.get("query") or r.get("prompt") or ""
        responses = r.get("response") or []
        if len(responses) < 2:
            continue
        v_a, v_b = responses[0], responses[1]

        try:
            res_a = score_one(student, dim, query, v_a, task_specs=task_specs)
            res_b = score_one(student, dim, query, v_b, task_specs=task_specs)
        except Exception as e:
            print(f"[eval][{pid}] inference failed: {e}", file=sys.stderr)
            res_a = res_b = None

        s_a = res_a.s_total if res_a else None
        s_b = res_b.s_total if res_b else None
        decision = _score_diff_to_decision(s_a, s_b, args.tau)
        gt = _label_rank_to_gt(r.get("label_rank"))
        correct = (decision == gt)
        if correct: n_correct += 1
        if gt in ("A", "B"):
            n_no_tie += 1
            if correct: n_no_tie_correct += 1
        confusion[gt][decision] += 1

        per_pair.append({
            "id":       pid,
            "s_a":      s_a,
            "s_b":      s_b,
            "decision": decision,
            "gt":       gt,
            "correct":  correct,
        })
        if (i + 1) % max(1, args.log_every) == 0:
            print(f"[eval] {i+1}/{len(rows)}  acc_running={n_correct/(i+1):.3f}  "
                  f"wall={time.time()-t0:.1f}s", file=sys.stderr)

    total = len(per_pair)
    acc        = n_correct / total if total else 0.0
    acc_no_tie = n_no_tie_correct / n_no_tie if n_no_tie else 0.0

    summary = {
        "method":  "RewardVerse student RM (pointwise → tau-decision)",
        "axis":    Path(args.vgrb_dataset).stem,
        "tau":     args.tau,
        "n_total":       total,
        "n_no_tie":      n_no_tie,
        "acc":           acc,
        "acc_no_tie":    acc_no_tie,
        "confusion":     {k: dict(v) for k, v in confusion.items()},
        "wall_s":        round(time.time() - t0, 2),
        "student_ckpt":  args.student_ckpt or "DUMMY",
        "dataset":       str(args.vgrb_dataset),
    }
    out = {
        "summary":  summary,
        "per_pair": per_pair,
    }
    Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    print(f"[eval] wrote -> {args.output}  acc={acc:.3f}  acc_no_tie={acc_no_tie:.3f}",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student_ckpt",  default=None,
                   help="Trained student checkpoint dir (omit -> dummy)")
    p.add_argument("--vgrb_dataset",  required=True,
                   help="JSON list with {id, query, response:[a,b], label_rank} records")
    p.add_argument("--output",        required=True)
    p.add_argument("--tau",           type=float, default=0.0,
                   help="|S_a - S_b| ≤ τ → predict 'same'")
    p.add_argument("--temperature",   type=float, default=0.0)
    p.add_argument("--max_new_tokens", type=int,  default=1024)
    p.add_argument("--limit",         type=int,   default=None)
    p.add_argument("--log_every",     type=int,   default=10)
    p.add_argument("--dimension",     default=None,
                   help="Override the dimension fed into the RM. Default "
                        "auto-derives from the dataset filename stem "
                        "(VQ→visual_quality, MQ→consistency, TA→grounding, "
                        "Overall→visual_quality).")
    p.add_argument("--no_task_specs", action="store_true",
                   help="Bypass yaml-loaded task descriptions; use generic "
                        "scaffold. Mostly for ablation.")
    args = p.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
