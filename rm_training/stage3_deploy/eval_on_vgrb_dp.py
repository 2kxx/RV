"""rm_training.stage3_deploy.eval_on_vgrb_dp
==========================================
Data-parallel shard runner for ``eval_on_vgrb.py``.

Each rank loads the student ckpt independently, processes
``rows[rank::world_size]`` of the VGRB pairwise JSON, and writes a
per-rank result file ``<output_dir>/rank_<rank>.json`` with the same
``per_pair`` shape as the single-process script. Aggregation is done
afterwards by ``merge_vgrb_dp.py``.

Launcher pattern (one process per visible GPU)::

    for r in 0..7:
        CUDA_VISIBLE_DEVICES=$r python eval_on_vgrb_dp.py \\
            --rank $r --world_size 8 \\
            --student_ckpt <ckpt> \\
            --vgrb_dataset .../Overall.json \\
            --output_dir runs/.../Overall_dp/

The intent is to keep the per-rank logic *identical* to the single-
process code path so any divergence between the two is a bug. We do not
share state across ranks; the model load is duplicated per rank because
StudentWrapper isn't designed for cross-process weight sharing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rm_training.stage1_teacher.task_descriptions import load_task_specs
from rm_training.stage2_grpo.student_wrapper import StudentConfig, StudentWrapper
from rm_training.stage3_deploy.eval_on_vgrb import (
    VGRB_AXIS_TO_DIM,
    _label_rank_to_gt,
    _score_diff_to_decision,
)
from rm_training.stage3_deploy.pointwise_rm_inference import (
    _score_with_rubric,
    build_dim_ctx,
)


def evaluate_shard(args: argparse.Namespace) -> None:
    rows = json.loads(Path(args.vgrb_dataset).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"[eval-dp] unexpected dataset shape in {args.vgrb_dataset}")
    if args.limit is not None and args.limit > 0:
        rows = rows[: args.limit]

    # Stride-shard: row i goes to rank (i % world_size). Stride keeps each
    # shard size-balanced even if rows are sorted by something heavy.
    shard = [(i, r) for i, r in enumerate(rows) if i % args.world_size == args.rank]
    print(
        f"[eval-dp][rank={args.rank}/{args.world_size}] "
        f"{len(shard)}/{len(rows)} pairs to score from {args.vgrb_dataset}",
        file=sys.stderr,
    )

    student = StudentWrapper(StudentConfig(
        model_path=args.student_ckpt,
        dummy=(args.student_ckpt is None),
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        attn_implementation=args.attn_implementation,
    ))
    task_specs = None if args.no_task_specs else load_task_specs(ROOT)

    axis = Path(args.vgrb_dataset).stem
    dim = args.dimension or VGRB_AXIS_TO_DIM.get(axis, "visual_quality")
    dim_ctx = build_dim_ctx(dim, task_specs)
    print(f"[eval-dp][rank={args.rank}] axis={axis} dimension={dim}",
          file=sys.stderr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"rank_{args.rank:02d}.json"

    per_pair: list[dict] = []
    t0 = time.time()
    for k, (orig_idx, r) in enumerate(shard):
        pid = r.get("id", f"pair_{orig_idx:05d}")
        query = r.get("query") or r.get("prompt") or ""
        responses = r.get("response") or []
        if len(responses) < 2:
            continue
        v_a, v_b = responses[0], responses[1]

        try:
            # Rubric depends only on (caption, dim_ctx); generate once and
            # share across A/B so the comparison is rubric-consistent and
            # we save one full text-generate per pair.
            rubric_text = student.generate_rubric(query, dim_ctx=dim_ctx)
            res_a = _score_with_rubric(student, dim, query, v_a,
                                       rubric_text=rubric_text, dim_ctx=dim_ctx)
            res_b = _score_with_rubric(student, dim, query, v_b,
                                       rubric_text=rubric_text, dim_ctx=dim_ctx)
        except Exception as e:
            print(f"[eval-dp][rank={args.rank}][{pid}] inference failed: {e}",
                  file=sys.stderr)
            res_a = res_b = None

        s_a = res_a.s_total if res_a else None
        s_b = res_b.s_total if res_b else None
        decision = _score_diff_to_decision(s_a, s_b, args.tau)
        gt = _label_rank_to_gt(r.get("label_rank"))

        per_pair.append({
            "id":        pid,
            "orig_idx":  orig_idx,
            "s_a":       s_a,
            "s_b":       s_b,
            "decision":  decision,
            "gt":        gt,
            "correct":   decision == gt,
        })
        if (k + 1) % max(1, args.log_every) == 0:
            n_done = k + 1
            n_correct = sum(1 for x in per_pair if x["correct"])
            print(
                f"[eval-dp][rank={args.rank}] {n_done}/{len(shard)} "
                f"acc_running={n_correct/n_done:.3f} "
                f"wall={time.time()-t0:.1f}s",
                file=sys.stderr,
            )

    # Per-rank summary kept lightweight; merge_vgrb_dp recomputes totals.
    payload = {
        "rank":          args.rank,
        "world_size":    args.world_size,
        "axis":          axis,
        "dimension":     dim,
        "tau":           args.tau,
        "n_pairs_shard": len(shard),
        "n_total_dataset": len(rows),
        "wall_s":        round(time.time() - t0, 2),
        "student_ckpt":  args.student_ckpt or "DUMMY",
        "dataset":       str(args.vgrb_dataset),
        "per_pair":      per_pair,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[eval-dp][rank={args.rank}] wrote -> {out_path} "
          f"({len(per_pair)} rows, wall={payload['wall_s']:.1f}s)",
          file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rank",          type=int, required=True)
    p.add_argument("--world_size",    type=int, required=True)
    p.add_argument("--student_ckpt",  default=None)
    p.add_argument("--vgrb_dataset",  required=True)
    p.add_argument("--output_dir",    required=True,
                   help="Directory for per-rank rank_<NN>.json files.")
    p.add_argument("--tau",           type=float, default=0.0)
    p.add_argument("--temperature",   type=float, default=0.0)
    p.add_argument("--max_new_tokens", type=int,  default=1024)
    p.add_argument("--limit",         type=int,   default=None)
    p.add_argument("--log_every",     type=int,   default=10)
    p.add_argument("--dimension",     default=None)
    p.add_argument("--no_task_specs", action="store_true")
    # ── attn impl: "sdpa" (default, no extra deps), "flash_attention_2"
    # (~10-20% faster on H100, requires `pip install flash-attn
    # --no-build-isolation`), or "eager" for debug.
    p.add_argument("--attn_implementation", default="sdpa",
                   choices=["sdpa", "flash_attention_2", "eager"])
    args = p.parse_args()
    if args.world_size <= 0 or not (0 <= args.rank < args.world_size):
        raise SystemExit(f"[eval-dp] invalid rank/world_size: "
                         f"{args.rank}/{args.world_size}")
    evaluate_shard(args)


if __name__ == "__main__":
    main()
