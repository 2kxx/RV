"""rm_training.stage2_grpo.eval_ckpt
====================================
Offline pointwise PLCC/SRCC validation against a saved checkpoint.

Why this exists
---------------
The in-train val hook (``--val_every`` in ``train_grpo.py``) pays a heavy
tax we want to drop:

  * Every rank runs the val forwards in lockstep so ZeRO-3 ``all_gather``
    collectives stay in sync (170 forwards × 8 ranks ≈ 10 min/round even
    at ``--val_max_samples_per_dim 10``).
  * The val path enters ``_summon_full_params_ctx`` for every score
    forward — same memory peak as a training step. With the warm-up
    score-PG run already at ~85% of card budget, val rounds nudge the
    allocator into the territory where NCCL's tiny channel-init calloc
    can fail and tip the whole job into a 600 s monitoredBarrier
    deadlock (observed 2026-06-13 at step 60: val finished, the next
    backward hung waiting on ranks 3-6 to reach reduce_scatter).

Running val OUT-OF-PROCESS on saved checkpoints sidesteps both:
  * No ZeRO-3 lockstep — single-rank, full-replica load is enough; we
    don't need to backward through anything.
  * The trainer can keep its allocator state untouched and just save +
    keep training.

Usage
-----
::

    # Single checkpoint (writes <ckpt>/val.json):
    python -m rm_training.stage2_grpo.eval_ckpt \\
        --ckpt_dir   ckpts/student_grpo_8gpu_zero3/checkpoint-30 \\
        --base_model_path /efs/jacuzzitang/models/Qwen2.5-VL-7B-Instruct \\
        --max_samples_per_dim 10

    # Sweep every checkpoint under an output dir (writes per-ckpt val.json
    # plus a top-level val_summary.jsonl):
    python -m rm_training.stage2_grpo.eval_ckpt \\
        --output_dir ckpts/student_grpo_8gpu_zero3 \\
        --base_model_path /efs/jacuzzitang/models/Qwen2.5-VL-7B-Instruct \\
        --max_samples_per_dim 10

    # With student-generated rubrics instead of gold (slower, more
    # representative of deployment):
    python -m rm_training.stage2_grpo.eval_ckpt \\
        --ckpt_dir   ckpts/student_grpo_8gpu_zero3/checkpoint-latest \\
        --base_model_path /efs/jacuzzitang/models/Qwen2.5-VL-7B-Instruct \\
        --no-use_gold_rubric

The CLI mirrors ``_run_val_eval`` in ``train_grpo.py`` so the numbers
emitted here are directly comparable to in-train val numbers from
earlier runs.

Why ``--base_model_path``
-------------------------
Pre-2026-06-14 ``_save_ckpt`` only wrote model weights + tokenizer; it
did NOT save the AutoProcessor (preprocessor_config.json +
image-processor cfg). ``AutoProcessor.from_pretrained(ckpt_dir)`` then
fails with ``"Can't load image processor for <ckpt>"``. The newer
trainer fixes this at save time; for OLD checkpoints we copy the
missing files from ``base_model_path`` into the ckpt dir before loading.
Pass the same Qwen2.5-VL-7B-Instruct path the run trained against.
Newer ckpts that already contain ``preprocessor_config.json`` skip the
copy automatically (idempotent — won't overwrite).

Memory
------
Loads ONE replica of the 7B student in bf16 with ``device_map="cuda:0"``
(~14 GB). No optimiser, no ref, no LoRA second copy. Fits comfortably
on a single H100 alongside whatever else you have running, but DON'T
launch this on the same GPUs that an active training run is using —
the allocator pressure that motivated this script in the first place
applies in reverse.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the trainer's helper machinery — _make_val_score_fn already
# encodes the exact rubric / score path used during training (gold or
# student-generated, with the per-dim DimensionContext + JSON scaffold).
# Reusing it keeps offline numbers numerically identical to in-train
# numbers for the same checkpoint.
from rm_training.stage1_teacher.task_descriptions import load_task_specs
from rm_training.stage2_grpo.data_collator   import GoldRubricDataset
from rm_training.stage2_grpo.eval_pointwise  import (
    prefetch_pointwise_videos,
    run_pointwise_eval,
)
from rm_training.stage2_grpo.student_wrapper import (
    StudentConfig,
    StudentWrapper,
)
# Trainer-internal helpers we need to drive val:
#   _build_gold_by_dim → per-dim gold rubric for use_gold_rubric=True
#   _make_val_score_fn → rubric→score scaffold (matches training contract)
from rm_training.stage2_grpo.train_grpo import (
    _build_gold_by_dim,
    _make_val_score_fn,
)


# ---------------------------------------------------------------------------
# 1. Checkpoint loading.
# ---------------------------------------------------------------------------
# Files the AutoProcessor + tokenizer load path looks for. If any of these
# are missing in ckpt_dir but present in base_model_path, we copy them in
# before instantiating StudentWrapper. (The trainer's _save_ckpt
# pre-2026-06-14 only wrote model + tokenizer files, so the image
# processor metadata was missing for old checkpoints. New ckpts contain
# all of these and the copy is a no-op.)
_PROCESSOR_FILES = (
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    # Tokenizer fallbacks too — newer ckpts already have these but old
    # ckpts that DID save tokenizer should still tolerate a re-copy
    # (shutil.copy2 with dst-exists check is idempotent).
    "chat_template.json",
)


def _backfill_processor_files(ckpt_dir: Path, base_model_path: Path) -> list[str]:
    """Copy missing processor files from base_model_path → ckpt_dir.

    Idempotent: only copies files that exist in base but NOT in ckpt.
    Returns the list of filenames that were actually copied (for logging).
    Empty list on a fully-equipped ckpt.
    """
    copied: list[str] = []
    if not base_model_path.exists():
        return copied
    for name in _PROCESSOR_FILES:
        src = base_model_path / name
        dst = ckpt_dir / name
        if src.exists() and not dst.exists():
            try:
                shutil.copy2(str(src), str(dst))
                copied.append(name)
            except Exception as e:
                print(f"[eval_ckpt] failed to copy {name}: {e}",
                      file=sys.stderr, flush=True)
    return copied


def _load_student_from_ckpt(
    ckpt_dir: Path,
    *,
    base_model_path: str | None,
    attn_implementation: str,
    score_max_new_tokens: int,
    device: str = "cuda:0",
) -> StudentWrapper:
    """Build a StudentWrapper that points at a saved checkpoint.

    The trainer saves with ``save_pretrained`` (full-param mode), so
    ``ckpt_dir`` already contains a complete HF model spec
    (config.json + safetensors shards). We just instantiate
    StudentWrapper with ``model_path = ckpt_dir`` — its internal load
    path goes through ``Qwen2_5_VLForConditionalGeneration.from_pretrained``
    which handles the safetensors sharding transparently.

    If ``base_model_path`` is provided we backfill any missing
    AutoProcessor files into ``ckpt_dir`` first (see
    ``_backfill_processor_files``). This is needed for checkpoints saved
    by the pre-2026-06-14 trainer that omitted the image-processor cfg.
    """
    if not ckpt_dir.exists():
        raise SystemExit(f"[eval_ckpt] ckpt_dir does not exist: {ckpt_dir}")

    # Backfill missing preprocessor files from the base model. Required
    # for old checkpoints; no-op for new ones. Without this AutoProcessor
    # crashes with "Can't load image processor for <ckpt>".
    if base_model_path:
        base = Path(base_model_path).resolve()
        copied = _backfill_processor_files(ckpt_dir, base)
        if copied:
            print(f"[eval_ckpt] backfilled processor files into {ckpt_dir}: "
                  f"{copied}", file=sys.stderr, flush=True)

    cfg = StudentConfig(
        model_path=str(ckpt_dir),
        # bf16 + flash-attn-2 to match training inference path.
        attn_implementation=attn_implementation,
        # Force single-GPU placement at eval time. The default
        # device_map="auto" lets HF shard the 7B across cuda:0 + cuda:1
        # for "balanced" placement, which then breaks score_video's
        # _processor(...).to(self._model.device) call (input tensors land
        # on the first-layer device, but later layers live on cuda:1 →
        # "expected all tensors to be on the same device" at forward).
        # 7B bf16 ≈ 14 GB, fits on one H100; sharding gives nothing.
        # ``device`` is passed in by the worker so multi-GPU eval can spawn
        # one replica per card (each worker pins itself to its own cuda:N).
        device_map=device,
        # Force greedy + 64-token budget for val — matches the override
        # _run_val_eval applies in-train (see the long comment in
        # train_grpo._run_val_eval). Sampling at low budget early-EOSes
        # ~25-30% of rows on student rubrics.
        score_greedy=True,
        score_max_new_tokens=score_max_new_tokens,
        # No grad-ckpt at val time — we run no backward.
        use_gradient_checkpointing=False,
        # No ref model needed (no KL at val).
        ref_model_path=None,
        dummy=False,
    )
    student = StudentWrapper(cfg)
    if student.model is not None:
        student.model.eval()
    return student


# ---------------------------------------------------------------------------
# 2. Run val for a single checkpoint.
# ---------------------------------------------------------------------------
def _run_one(
    ckpt_dir: Path,
    *,
    use_gold_rubric: bool,
    max_samples_per_dim: int | None,
    base_model_path: str | None,
    attn_implementation: str,
    score_max_new_tokens: int,
    gold_rubrics_jsonl: Path,
    write_artifacts: bool = True,
    dimensions: list[str] | None = None,
    device: str = "cuda:0",
    prefetch: bool = True,
    include_records: bool = False,
) -> dict:
    """Load a checkpoint, run pointwise val, write val.json beside it.

    ``dimensions``: optional sub-list of dim slugs to score. When None,
    evaluates all 16 active dims (whole sweep). Multi-GPU eval splits
    the full list across workers via this parameter.

    ``device``: cuda device id to pin this evaluator to (e.g. ``cuda:3``).
    Forwarded to ``StudentConfig.device_map`` so the 7B replica lands
    on exactly one card.

    ``prefetch``: download missing URL-only videos before scoring. Only
    rank-0 in a multi-worker run should set True — others race on the
    same cache dir.
    """
    print(f"[eval_ckpt] loading {ckpt_dir} on {device}...",
          file=sys.stderr, flush=True)
    t0 = time.time()
    student = _load_student_from_ckpt(
        ckpt_dir,
        base_model_path=base_model_path,
        attn_implementation=attn_implementation,
        score_max_new_tokens=score_max_new_tokens,
        device=device,
    )
    load_wall = round(time.time() - t0, 2)
    print(f"[eval_ckpt] loaded in {load_wall}s ({device})",
          file=sys.stderr, flush=True)

    # Mirror the trainer's val setup: load per-dim task specs (the
    # rubric / score prompts depend on these) and per-dim gold rubrics
    # (used when use_gold_rubric=True).
    task_specs = load_task_specs(ROOT)
    ds = GoldRubricDataset.from_jsonl(str(gold_rubrics_jsonl), rebase_root=ROOT)
    gold_by_dim = _build_gold_by_dim(ds)

    # Prefetch any URL-only test videos before the score loop. Skipped in
    # worker processes; rank-0 (or single-worker runs) handle this once
    # before fanning out.
    if prefetch:
        try:
            prefetch_pointwise_videos(ROOT, max_samples_per_dim=max_samples_per_dim)
        except Exception as e:
            print(f"[eval_ckpt] prefetch failed: {e} — continuing; rows whose URL "
                  "didn't materialise will be skip_no_video.",
                  file=sys.stderr, flush=True)

    score_fn = _make_val_score_fn(
        student, task_specs,
        use_gold_rubric=use_gold_rubric,
        gold_by_dim=gold_by_dim,
    )

    val_t0 = time.time()
    report = run_pointwise_eval(
        score_fn=score_fn,
        repo_root=ROOT,
        dimensions=dimensions,
        max_samples_per_dim=max_samples_per_dim,
        progress_every=20,
        prefetch_missing=False,
        include_records=include_records,
    )
    # Replace run_pointwise_eval's flat-pooled overall with a macro-average
    # over per-dim summaries. Each dim's PLCC/SRCC/MAE is computed on its
    # own samples (statistically well-defined), then the 16 dim metrics
    # get arithmetic-averaged. Macro-avg gives each dim equal weight —
    # closer to what we care about ("how well does the RM do on each
    # capability") than flat-pool (which is dominated by dims that happen
    # to have more test rows). Matches the multi-GPU merger so single- and
    # multi-worker numbers are directly comparable.
    heatmap = report.get("heatmap") or {}
    if heatmap:
        n_d = len(heatmap)
        report["overall"] = {
            "plcc":  round(sum(s.get("plcc", 0.0) for s in heatmap.values()) / n_d, 4),
            "srcc":  round(sum(s.get("srcc", 0.0) for s in heatmap.values()) / n_d, 4),
            "mae":   round(sum(s.get("mae",  0.0) for s in heatmap.values()) / n_d, 4),
            "n":     sum(int(s.get("n", 0)) for s in heatmap.values()),
            "n_dims": n_d,
            "macro_average": True,
        }
    wall = round(time.time() - val_t0, 2)
    overall = report.get("overall", {})
    print(
        f"[eval_ckpt] {ckpt_dir.name} ({device}): "
        f"plcc={overall.get('plcc', 0):+.3f}  "
        f"srcc={overall.get('srcc', 0):+.3f}  "
        f"mae={overall.get('mae', 0):.3f}  "
        f"n={overall.get('n', 0)}  "
        f"use_gold={use_gold_rubric}  wall={wall}s",
        file=sys.stderr, flush=True,
    )

    if write_artifacts:
        # Drop val.json beside the ckpt so it travels with it. Same shape
        # as in-train val_step{step}.json, plus a "ckpt_dir" tag.
        out_path = ckpt_dir / "val.json"
        payload = {
            "ckpt_dir":            str(ckpt_dir),
            "use_gold_rubric":     use_gold_rubric,
            "max_samples_per_dim": max_samples_per_dim,
            "wall_s":              wall,
            "load_wall_s":         load_wall,
            "result":              report,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"[eval_ckpt] wrote {out_path}", file=sys.stderr, flush=True)

    return {
        "ckpt": str(ckpt_dir),
        "overall": overall,
        "wall_s": wall,
        "result": report,
    }


# ---------------------------------------------------------------------------
# 3. Walk every checkpoint-N under an output dir.
# ---------------------------------------------------------------------------
def _enumerate_ckpts(output_dir: Path) -> list[Path]:
    """List ``checkpoint-N`` dirs under ``output_dir``, sorted by N.

    Skips ``checkpoint-latest`` (it's a symlink to one of the above).
    """
    if not output_dir.exists():
        raise SystemExit(f"[eval_ckpt] output_dir does not exist: {output_dir}")
    out: list[tuple[int, Path]] = []
    for p in output_dir.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith("checkpoint-"):
            continue
        suffix = p.name[len("checkpoint-"):]
        if not suffix.isdigit():
            continue                       # e.g. "checkpoint-latest"
        out.append((int(suffix), p))
    out.sort(key=lambda x: x[0])
    return [p for _, p in out]


# ---------------------------------------------------------------------------
# 3b. Multi-GPU eval: spawn N workers, each loads its own 7B replica on
#     a distinct cuda:N, evaluates a shard of dims, writes partial JSON.
#     Parent merges partials.
#
# Why this and not DDP / FSDP?
# ---------------------------
# Eval is embarrassingly parallel at the (dim, video) level — no gradient,
# no cross-card synchronisation. Each worker carries a full 7B replica
# (~14 GB bf16, fits on one H100) and works through its own dim shard
# independently. Wall-clock ≈ (single-card time) / num_workers for the
# 16-active-dim workload. Setup overhead per worker = one model load
# (~6 s) + one prefetch warmup (rank-0 only).
# ---------------------------------------------------------------------------
def _split_dims(dims: list[str], num_workers: int) -> list[list[str]]:
    """Round-robin split so wall-clock is balanced even if some dims have
    many more samples / heavier videos than others."""
    shards: list[list[str]] = [[] for _ in range(num_workers)]
    for i, d in enumerate(dims):
        shards[i % num_workers].append(d)
    return [s for s in shards if s]    # drop empty if num_workers > len(dims)


def _worker_entry(
    rank: int,
    world_size: int,
    *,
    ckpt_dir_s: str,
    use_gold_rubric: bool,
    max_samples_per_dim: int | None,
    base_model_path: str | None,
    attn_implementation: str,
    score_max_new_tokens: int,
    gold_rubrics_jsonl_s: str,
    dim_shards: list[list[str]],
    out_partial_dir_s: str,
) -> None:
    """Run in a torch.multiprocessing-spawned child.

    Pins itself to cuda:<rank>, evaluates dim_shards[rank], writes
    partial result to <out_partial_dir>/partial_<rank>.json.
    """
    import torch
    # Pin this child to exactly one GPU. CUDA_VISIBLE_DEVICES isolation
    # would also work but is more fragile under nested spawns; sticking
    # to a single-process cuda:<rank> is sufficient because the model
    # load goes through device_map=cuda:N (see _load_student_from_ckpt).
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    dims = dim_shards[rank]
    if not dims:
        # No work for this rank (more workers than dims). Write empty
        # partial so the merger doesn't block waiting on it.
        Path(out_partial_dir_s, f"partial_{rank}.json").write_text(
            json.dumps({"rank": rank, "device": device, "result": None,
                        "empty": True}),
            encoding="utf-8",
        )
        return

    result = _run_one(
        Path(ckpt_dir_s),
        use_gold_rubric=use_gold_rubric,
        max_samples_per_dim=max_samples_per_dim,
        base_model_path=base_model_path,
        attn_implementation=attn_implementation,
        score_max_new_tokens=score_max_new_tokens,
        gold_rubrics_jsonl=Path(gold_rubrics_jsonl_s),
        write_artifacts=False,             # rank-0 writes the merged val.json
        dimensions=dims,
        device=device,
        prefetch=(rank == 0),              # only rank-0 hits the network/cache
        # Macro-avg merge needs only per-dim heatmap — no raw records to
        # ship back to the parent.
    )
    out_path = Path(out_partial_dir_s, f"partial_{rank}.json")
    out_path.write_text(
        json.dumps({"rank": rank, "device": device, "dims": dims,
                    "result": result.get("result", {}),
                    "wall_s": result.get("wall_s", 0)},
                   ensure_ascii=False),
        encoding="utf-8",
    )


def _merge_partials(partials: list[dict], dims_order: list[str]
                    ) -> dict:
    """Merge per-worker reports into a single report matching
    run_pointwise_eval's output shape.

    Per-dim heatmap is taken directly from worker reports — each dim is
    fully owned by ONE worker (round-robin shard), so the worker's own
    per-dim PLCC/SRCC/MAE is already correctly pooled over that dim's
    samples.

    Overall = MACRO-AVERAGE over per-dim metrics (each dim weighted
    equally). Macro-avg is the right summary for a 16-dim capability
    eval: a dim with 100 rows shouldn't outweigh a dim with 30 rows
    just because it has more rows. This also matches what _run_one
    reports now (single-card path was switched to macro-avg too), so
    multi-GPU and single-card numbers are directly comparable.
    """
    heatmap: dict[str, dict] = {}
    skipped_no_video: dict[str, int] = {}
    skipped_score_failed: dict[str, int] = {}
    skipped_no_test: list[str] = []

    for p in partials:
        if p.get("empty"):
            continue
        r = p.get("result") or {}
        for dim, stats in (r.get("heatmap") or {}).items():
            heatmap[dim] = stats
        sk = r.get("skipped") or {}
        for d, n in (sk.get("no_local_video") or {}).items():
            skipped_no_video[d] = skipped_no_video.get(d, 0) + n
        for d, n in (sk.get("score_failed") or {}).items():
            skipped_score_failed[d] = skipped_score_failed.get(d, 0) + n
        skipped_no_test.extend(sk.get("no_test_file") or [])

    # Macro-average across dims (each dim weighted equally).
    if heatmap:
        n_d = len(heatmap)
        overall = {
            "plcc":  round(sum(s.get("plcc", 0.0) for s in heatmap.values()) / n_d, 4),
            "srcc":  round(sum(s.get("srcc", 0.0) for s in heatmap.values()) / n_d, 4),
            "mae":   round(sum(s.get("mae",  0.0) for s in heatmap.values()) / n_d, 4),
            "n":     sum(int(s.get("n", 0)) for s in heatmap.values()),
            "n_dims": n_d,
            "macro_average": True,
        }
    else:
        overall = {"plcc": 0.0, "srcc": 0.0, "mae": 0.0, "n": 0,
                   "n_dims": 0, "macro_average": True}

    ranked = sorted(heatmap.items(), key=lambda kv: kv[1].get("plcc", 0),
                    reverse=True)

    return {
        "overall": overall,
        "heatmap": {d: heatmap[d] for d in dims_order if d in heatmap},
        "ranked":  [d for d, _ in ranked],
        "skipped": {
            "no_test_file":   sorted(set(skipped_no_test)),
            "no_local_video": skipped_no_video,
            "score_failed":   skipped_score_failed,
        },
    }


def _run_one_multigpu(
    ckpt_dir: Path,
    *,
    num_workers: int,
    use_gold_rubric: bool,
    max_samples_per_dim: int | None,
    base_model_path: str | None,
    attn_implementation: str,
    score_max_new_tokens: int,
    gold_rubrics_jsonl: Path,
    write_artifacts: bool = True,
) -> dict:
    """Multi-GPU sibling of ``_run_one``. Spawns ``num_workers`` children;
    each takes one shard of the 16 active dims and runs on a distinct GPU."""
    import tempfile
    import torch.multiprocessing as mp
    from functools import partial

    # Import lazily so the active-dim list lives next to the rest of the
    # eval pipeline rather than at module top (avoid loading heavy deps
    # on import when this function isn't called).
    from rm_training.stage2_grpo.eval_pointwise import ACTIVE_DIMENSIONS

    dims_full = list(ACTIVE_DIMENSIONS)
    dim_shards = _split_dims(dims_full, num_workers)
    actual_workers = len(dim_shards)
    if actual_workers < num_workers:
        print(f"[eval_ckpt] {num_workers} workers requested but only "
              f"{actual_workers} dim shards — capping.",
              file=sys.stderr, flush=True)

    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="eval_ckpt_partials_") as tmpdir:
        # Pre-fetch once on the parent before spawning so all children see
        # the same cache state.
        try:
            prefetch_pointwise_videos(ROOT, max_samples_per_dim=max_samples_per_dim)
        except Exception as e:
            print(f"[eval_ckpt] parent prefetch failed: {e} — continuing.",
                  file=sys.stderr, flush=True)

        print(f"[eval_ckpt] spawning {actual_workers} workers across "
              f"cuda:0..cuda:{actual_workers-1}",
              file=sys.stderr, flush=True)

        # mp.spawn passes (rank, *args) → target. We bind every other
        # argument via functools.partial so the target function only
        # has to accept `rank` positionally.
        bound = partial(
            _worker_entry,
            world_size=actual_workers,
            ckpt_dir_s=str(ckpt_dir),
            use_gold_rubric=use_gold_rubric,
            max_samples_per_dim=max_samples_per_dim,
            base_model_path=base_model_path,
            attn_implementation=attn_implementation,
            score_max_new_tokens=score_max_new_tokens,
            gold_rubrics_jsonl_s=str(gold_rubrics_jsonl),
            dim_shards=dim_shards,
            out_partial_dir_s=tmpdir,
        )
        mp.spawn(bound, nprocs=actual_workers, join=True, daemon=False)

        partials: list[dict] = []
        for rank in range(actual_workers):
            fp = Path(tmpdir, f"partial_{rank}.json")
            if fp.exists():
                try:
                    partials.append(json.loads(fp.read_text(encoding="utf-8")))
                except Exception as e:
                    print(f"[eval_ckpt] could not read {fp}: {e}",
                          file=sys.stderr, flush=True)

    wall = round(time.time() - t0, 2)
    report = _merge_partials(partials, dims_order=dims_full)
    overall = report.get("overall", {})
    print(
        f"[eval_ckpt] {ckpt_dir.name} [multi-gpu x{actual_workers}]: "
        f"plcc={overall.get('plcc', 0):+.3f}  "
        f"srcc={overall.get('srcc', 0):+.3f}  "
        f"mae={overall.get('mae', 0):.3f}  "
        f"n={overall.get('n', 0)}  "
        f"use_gold={use_gold_rubric}  wall={wall}s",
        file=sys.stderr, flush=True,
    )

    if write_artifacts:
        out_path = ckpt_dir / "val.json"
        payload = {
            "ckpt_dir":            str(ckpt_dir),
            "use_gold_rubric":     use_gold_rubric,
            "max_samples_per_dim": max_samples_per_dim,
            "wall_s":              wall,
            "num_workers":         actual_workers,
            "result":              report,
        }
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"[eval_ckpt] wrote {out_path}", file=sys.stderr, flush=True)

    return {
        "ckpt": str(ckpt_dir),
        "overall": overall,
        "wall_s": wall,
    }


# ---------------------------------------------------------------------------
# 4. CLI.
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Offline pointwise val on saved Stage-2 GRPO checkpoint(s). "
            "Replaces the in-train --val_every hook for runs where val's "
            "memory pressure / lockstep cost is undesirable."
        )
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--ckpt_dir", type=str, default=None,
                     help="Path to a single checkpoint dir to evaluate "
                          "(e.g. ckpts/.../checkpoint-30 or "
                          "ckpts/.../checkpoint-latest).")
    grp.add_argument("--output_dir", type=str, default=None,
                     help="Path to a training output dir; evaluates EVERY "
                          "checkpoint-N under it in step order and writes a "
                          "val_summary.jsonl with one row per ckpt.")
    p.add_argument("--max_samples_per_dim", type=int, default=10,
                   help="Cap rows per dim. Matches the trainer default. "
                        "Set to 0 or negative for the full sweep "
                        "(slow — ~30 min/ckpt).")
    p.add_argument("--use_gold_rubric", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Use per-dim gold rubrics from the training jsonl "
                        "(matches the trainer's --val_use_gold default). "
                        "Pass --no-use_gold_rubric to use student-generated "
                        "rubrics — slower and more representative of how "
                        "deployment will behave end-to-end.")
    p.add_argument("--gold_rubrics", type=str,
                   default="data/rm_train/gold_rubrics_18d.clean.jsonl",
                   help="Path to the training gold-rubrics jsonl, used to "
                        "extract per-dim gold rubrics for use_gold_rubric=True.")
    p.add_argument("--base_model_path", type=str, default=None,
                   help="Path to the base Qwen2.5-VL-7B-Instruct weights. "
                        "REQUIRED for checkpoints saved before 2026-06-14 "
                        "— those omit the image processor config and "
                        "AutoProcessor.from_pretrained crashes without it. "
                        "We copy preprocessor_config.json + friends from "
                        "this path into the ckpt dir on first load. "
                        "Newer checkpoints carry these files already and "
                        "this argument is optional / harmless for them.")
    p.add_argument("--attn_implementation", type=str, default="flash_attention_2",
                   help="Forwarded to StudentWrapper. Match the training run "
                        "for numerical parity.")
    p.add_argument("--score_max_new_tokens", type=int, default=64,
                   help="Score-side generation budget. 64 matches the "
                        "in-train val override.")
    p.add_argument("--num_workers", type=int, default=1,
                   help="Number of GPU workers to run in parallel. 1 = "
                        "single-card serial. >1 = multi-GPU: dims are "
                        "sharded round-robin across cuda:0..cuda:N-1, "
                        "each worker loads its own 7B replica (~14 GB). "
                        "Workers ship their raw (S, label) records back "
                        "to the parent which pools them and computes the "
                        "canonical PLCC/SRCC — numerically identical to "
                        "single-worker output (modulo tied-rank "
                        "tiebreaks in Spearman, which are seed-stable).")
    args = p.parse_args()

    max_samples = args.max_samples_per_dim
    if max_samples is not None and max_samples <= 0:
        max_samples = None    # full sweep
    gold_jsonl = Path(args.gold_rubrics)
    if not gold_jsonl.is_absolute():
        gold_jsonl = (ROOT / gold_jsonl).resolve()

    if args.ckpt_dir is not None:
        # Resolve symlink (checkpoint-latest → checkpoint-N).
        ck = Path(args.ckpt_dir).resolve()
        if args.num_workers > 1:
            _run_one_multigpu(
                ck,
                num_workers=args.num_workers,
                use_gold_rubric=args.use_gold_rubric,
                max_samples_per_dim=max_samples,
                base_model_path=args.base_model_path,
                attn_implementation=args.attn_implementation,
                score_max_new_tokens=args.score_max_new_tokens,
                gold_rubrics_jsonl=gold_jsonl,
            )
        else:
            _run_one(
                ck,
                use_gold_rubric=args.use_gold_rubric,
                max_samples_per_dim=max_samples,
                base_model_path=args.base_model_path,
                attn_implementation=args.attn_implementation,
                score_max_new_tokens=args.score_max_new_tokens,
                gold_rubrics_jsonl=gold_jsonl,
            )
    else:
        out_dir = Path(args.output_dir).resolve()
        ckpts = _enumerate_ckpts(out_dir)
        if not ckpts:
            raise SystemExit(
                f"[eval_ckpt] no checkpoint-N dirs found under {out_dir}. "
                f"Expected sub-dirs named 'checkpoint-30', 'checkpoint-60', ..."
            )
        print(f"[eval_ckpt] sweeping {len(ckpts)} checkpoints under {out_dir}",
              file=sys.stderr, flush=True)
        summary_path = out_dir / "val_summary.jsonl"
        # Append-mode keeps prior runs around; if you want a fresh sweep
        # delete val_summary.jsonl first.
        with summary_path.open("a", encoding="utf-8") as fh:
            for ck in ckpts:
                # Skip ckpts that already have a val.json with the same
                # config — saves time on resume. Detection is conservative:
                # any val.json whose use_gold_rubric / max_samples match.
                cached = ck / "val.json"
                if cached.exists():
                    try:
                        prior = json.loads(cached.read_text(encoding="utf-8"))
                        if (prior.get("use_gold_rubric") == args.use_gold_rubric
                                and prior.get("max_samples_per_dim") == max_samples):
                            print(f"[eval_ckpt] skip {ck.name} — val.json "
                                  f"already present with matching config",
                                  file=sys.stderr, flush=True)
                            continue
                    except Exception:
                        pass     # malformed prior → re-run
                if args.num_workers > 1:
                    row = _run_one_multigpu(
                        ck,
                        num_workers=args.num_workers,
                        use_gold_rubric=args.use_gold_rubric,
                        max_samples_per_dim=max_samples,
                        base_model_path=args.base_model_path,
                        attn_implementation=args.attn_implementation,
                        score_max_new_tokens=args.score_max_new_tokens,
                        gold_rubrics_jsonl=gold_jsonl,
                    )
                else:
                    row = _run_one(
                        ck,
                        use_gold_rubric=args.use_gold_rubric,
                        max_samples_per_dim=max_samples,
                        base_model_path=args.base_model_path,
                        attn_implementation=args.attn_implementation,
                        score_max_new_tokens=args.score_max_new_tokens,
                        gold_rubrics_jsonl=gold_jsonl,
                    )
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
        print(f"[eval_ckpt] sweep complete → {summary_path}",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
