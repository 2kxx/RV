"""dump_rubrics_singlecard.py
=============================
Pre-generate G rubrics per case under SINGLE-GPU + NO accelerate (matches
probe_score_v4_e2e setup), and dump them to JSONL keyed by pair_id.

Used by probe_v8_bisect.py variant R, which feeds these rubrics into the
8-card ZeRO-3 score forward to disambiguate:

    * If R ≈ 100% → the rubric STRING content from a 1-card environment
      is fine even when scored under 8-card ZeRO-3. The v8 system is
      not the poison; only the rubrics that 8-card ZeRO-3 generates
      are. Repair → fix rubric_gen-side ZeRO-3 behavior.

    * If R ≈ 33% → student rubric strings are intrinsically poison
      regardless of where they were sampled. Repair → fix score-side
      prompt / decoding.

Usage
-----
On a SINGLE-GPU machine (or just CUDA_VISIBLE_DEVICES=0 + no accelerate):

    python scripts/dump_rubrics_singlecard.py \\
        --captions_from ckpts/student_grpo_8gpu_zero3/train_log.jsonl \\
        --gold_path data/rm_train/gold_rubrics_18d.clean.jsonl \\
        --n_cases 4 --G 8 \\
        --out logs/rubrics_singlecard.jsonl

Then sync logs/rubrics_singlecard.jsonl over to the 8-card box and run:

    bash scripts/probe_v8_bisect.sh R   # set VARIANTS=(R) in the .sh

Output JSONL row schema
-----------------------
    {"pair_id":  "<pid>",
     "dimension": "<dim>",
     "caption":   "<caption>",
     "rubrics":   ["<text 0>", "<text 1>", ...]}   # exactly G entries
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import argparse
import json
import time
import torch

from rm_training.stage2_grpo.student_wrapper import (
    StudentConfig, StudentWrapper, DimensionContext,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/efs/jacuzzitang/models/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--captions_from", default="ckpts/student_grpo_8gpu_zero3/train_log.jsonl")
    p.add_argument("--gold_path", default="data/rm_train/gold_rubrics_18d.clean.jsonl",
                   help="Used only to filter cases to dimensions that have a "
                        "gold rubric (mirrors probe_v8_bisect's case selection).")
    p.add_argument("--n_cases", type=int, default=4,
                   help="How many distinct (pair_id, dim, caption) cases to "
                        "dump. Match probe_v8_bisect's --n_runs.")
    p.add_argument("--G", type=int, default=8,
                   help="Rubrics per case (num_return_sequences). Match "
                        "probe_v8_bisect's --G.")
    p.add_argument("--out", default="logs/rubrics_singlecard.jsonl")
    p.add_argument("--seed_base", type=int, default=0,
                   help="Per-case torch.manual_seed = seed_base + case_idx. "
                        "probe_v8_bisect uses ``case_idx + rank*100`` (rank=0 "
                        "implicit on single GPU); leaving seed_base=0 here "
                        "matches rank-0's seed sequence exactly.")
    args = p.parse_args()

    if not (REPO / args.model).exists() and not Path(args.model).exists():
        # ``args.model`` is usually an absolute path — that's fine; we
        # only check existence so the user gets a clear error early.
        raise SystemExit(f"--model path does not exist: {args.model}")

    # Resolve cases via the same selection rule probe_v8_bisect uses.
    gold_by_dim: dict[str, list[dict]] = {}
    with open(REPO / args.gold_path) as f:
        for ln in f:
            r = json.loads(ln); d = r.get("dimension")
            if d and d not in gold_by_dim and r.get("gold_rubric"):
                gold_by_dim[d] = r["gold_rubric"]

    cases: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    with open(REPO / args.captions_from) as f:
        for ln in f:
            r = json.loads(ln)
            pid = r.get("pair_id_local") or r.get("pair_id")
            d   = r.get("dimension"); c = r.get("caption")
            if d and c and pid and pid not in seen and d in gold_by_dim:
                cases.append((d, c, pid)); seen.add(pid)
            if len(cases) >= args.n_cases:
                break

    print(f"[dump] resolved {len(cases)} cases from {args.captions_from}",
          flush=True)
    if not cases:
        raise SystemExit("[dump] no cases — check --captions_from / --gold_path")

    # Single-GPU student. NO accelerate, NO ZeRO. Mirrors probe_score_v4_e2e
    # at line 56-67: cuda:0, bf16, eval mode, sampling on (default in
    # generate_rubrics_with_meta_batch — do_sample=True).
    cfg = StudentConfig(
        model_path=args.model,
        device_map={"": "cuda:0"},
        torch_dtype="bfloat16",
        score_method="soft_logits",
        score_max_new_tokens=64,
        score_greedy=True,                  # irrelevant here (no scoring)
        attn_implementation="sdpa",
        use_gradient_checkpointing=False,   # speed; no backward needed
        ref_model_path=None,
        temperature=0.7, top_p=0.95,
        max_new_tokens=512,                 # rubric-gen budget
    )
    student = StudentWrapper(cfg)
    student._ensure_model()
    student._model.eval()

    # rubric-gen micro-batch matches v8 (RM_RUBRIC_MB=8). With G=8 this
    # folds the whole sample set into a single num_return_sequences=8 call.
    os.environ["RM_RUBRIC_MB"] = "8"

    out_path = (REPO / args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for case_idx, (dim, caption, pid) in enumerate(cases):
        ctx = DimensionContext.generic(dim)
        torch.manual_seed(args.seed_base + case_idx)
        t0 = time.time()
        metas = student.generate_rubrics_with_meta_batch(
            caption, dim_ctx=ctx, num_return_sequences=args.G,
        )
        rubric_texts = [m.text for m in metas]
        dt = time.time() - t0
        lens = [len(rt or "") for rt in rubric_texts]
        print(f"[dump] case {case_idx+1}/{len(cases)} pid={pid} dim={dim} "
              f"{dt:.1f}s lens={lens}", flush=True)
        rows.append({
            "pair_id":   pid,
            "dimension": dim,
            "caption":   caption,
            "rubrics":   rubric_texts,
        })

    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[dump] wrote {len(rows)} rows → {out_path}", flush=True)


if __name__ == "__main__":
    main()
