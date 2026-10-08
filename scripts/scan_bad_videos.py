#!/usr/bin/env python3
"""Offline scan: find videos that crash decord / qwen-vl-utils.

Why
---
Training (rm_training/stage2_grpo/student_wrapper.py) decodes each video via
``qwen_vl_utils.process_vision_info`` with ``fps=cfg.fps, max_pixels=360*420``.
Some videos in the dataset trigger native-level SIGSEGV inside decord/ffmpeg
during decode — the segfault kills the whole rank, which kills the whole
distributed training run. Try/except can't catch SIGSEGV (it's a signal,
not a Python exception), so the only way to make training robust is:

  (a) scan all videos offline in subprocess isolation → mark the bad ones,
  (b) emit a cleaned jsonl that excludes any pair touching a bad video,
  (c) point training at the cleaned jsonl.

This script is (a) + (b). Run on the training machine (same env, same
qwen-vl-utils version) so the segfault repro is identical.

Path rebase
-----------
The jsonl stores ABSOLUTE paths from whatever host wrote it (typically
``/group/40063/jacuzzitang/projects/RewardVerse/...``). Training rewrites
those at load time via ``rm_training.stage2_grpo.data_collator
.rebase_video_paths`` so the basename is resolved under the current
project root. We use the SAME helper here so the scanner sees the
exact paths training sees — otherwise on a different host the jsonl
paths look "missing" and we'd mark every video bad.

Usage
-----
    python scripts/scan_bad_videos.py \\
        --jsonl data/rm_train/gold_rubrics_18d.clean.jsonl \\
        --out_jsonl data/rm_train/gold_rubrics_18d.clean.scanned.jsonl \\
        --bad_json data/rm_train/bad_videos.json \\
        --workers 4 \\
        --timeout 60

Worker mode (internal — invoked by the driver via subprocess)::

    python scripts/scan_bad_videos.py --worker <video_path>

Exit codes (worker mode):
    0   = decoded fine
    1   = decoded but raised a Python exception (caught, video usable but flaky)
    -N  = killed by signal N (e.g. SIGSEGV → -11). Driver treats as bad.
    other nonzero = generic failure
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# ---------------------------------------------------------------------------
# Worker: decode one video using the same path qwen-vl-utils uses at train.
# Runs in its own subprocess so a SIGSEGV here is isolated.
# ---------------------------------------------------------------------------
def _worker_decode_one(video_path: str, fps: float, max_pixels: int) -> int:
    try:
        from qwen_vl_utils import process_vision_info  # type: ignore
    except ImportError:
        print(f"[worker] qwen_vl_utils not installed", file=sys.stderr)
        return 2

    msgs = [{
        "role": "user",
        "content": [
            {"type": "video", "video": str(video_path),
             "fps": fps, "max_pixels": max_pixels},
            {"type": "text", "text": "ping"},
        ],
    }]
    try:
        image_inputs, video_inputs = process_vision_info(msgs)
        # Touch the tensor so any lazy decode work fires.
        if video_inputs is not None:
            for v in video_inputs:
                if hasattr(v, "shape"):
                    _ = tuple(v.shape)
                elif hasattr(v, "__len__"):
                    _ = len(v)
    except Exception as e:
        print(f"[worker] {video_path}: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Driver: spawn one subprocess per video, collect statuses.
# ---------------------------------------------------------------------------
def _scan_one(video_path: str, fps: float, max_pixels: int,
              timeout: float, this_script: str) -> dict:
    """Run the worker in a fresh subprocess. Returns status dict.

    stderr/stdout handling
    ----------------------
    decord/ffmpeg can spam thousands of ``Invalid NAL unit size`` /
    ``partial file`` / ``missing picture`` warnings during a SUCCESSFUL
    decode of a slightly-truncated mp4 (we saw this in train_warmup.log).
    If we used ``stderr=PIPE``, that 64 KB OS pipe buffer would fill,
    block the worker, and ``subprocess.run(timeout=…)`` would mis-fire
    as a "bad video" timeout. We redirect both streams to a tempfile so
    the worker can write freely; on failure we read back the tail.

    Cold-start cost
    ---------------
    The worker imports qwen_vl_utils + transformers + decord on each
    invocation (~2-10 s on warm IO, more on a cold cache). With
    ``--workers 4`` and 400+ videos this is the bulk of wall-clock,
    but it's the price of true subprocess isolation. Keep
    ``--timeout`` ≥ 90 s on slow filesystems.
    """
    import tempfile
    t0 = time.time()
    cmd = [sys.executable, this_script, "--worker", video_path,
           "--fps", str(fps), "--max_pixels", str(max_pixels)]
    # Buffer stderr+stdout to a tempfile, NOT a pipe — pipes have a 64 KB
    # cap and ffmpeg's NAL-unit warnings can blow past that on a single
    # truncated video, hanging the worker.
    with tempfile.TemporaryFile(mode="w+b") as logbuf:
        try:
            proc = subprocess.run(
                cmd,
                stdout=logbuf,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            return {"path": video_path, "status": "timeout",
                    "wall": timeout}
        wall = time.time() - t0
        if rc == 0:
            return {"path": video_path, "status": "ok", "wall": wall}
        # On failure read the LAST 500 bytes of combined output for
        # diagnostics. We don't read the whole thing — could be megabytes.
        logbuf.seek(0, os.SEEK_END)
        size = logbuf.tell()
        logbuf.seek(max(0, size - 500))
        tail = logbuf.read().decode("utf-8", errors="replace")
        if rc < 0:
            return {"path": video_path, "status": "signal",
                    "signal": -rc, "wall": wall, "stderr": tail}
        return {"path": video_path, "status": "error", "rc": rc,
                "wall": wall, "stderr": tail}


def _is_bad(result: dict) -> bool:
    """Worker exit 1 means caught Python exception — still risky but not
    a process-killer. Treat signal/timeout/error as bad. Anything that
    raised a Python exception we ALSO treat as bad: training caller paths
    will (per our patch) try/except them, but at scan time we're trying
    to remove the flaky ones up front."""
    return result["status"] != "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", default=None,
                    help="(internal) decode one video and exit with status")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max_pixels", type=int, default=360 * 420)
    ap.add_argument("--jsonl", default="data/rm_train/gold_rubrics_18d.clean.jsonl")
    ap.add_argument("--out_jsonl",
                    default="data/rm_train/gold_rubrics_18d.clean.scanned.jsonl")
    ap.add_argument("--bad_json", default="data/rm_train/bad_videos.json")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel scan workers (each spawns its own subprocess)")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-video subprocess timeout in seconds. The "
                         "worker has to import qwen_vl_utils + "
                         "transformers + decord (~5-15 s on cold IO) "
                         "before decoding, so 60 s is too tight on "
                         "shared filesystems. 120 s is a safe default.")
    ap.add_argument("--limit", type=int, default=0,
                    help="scan only first N unique videos (debug)")
    args = ap.parse_args()

    # Worker branch.
    if args.worker is not None:
        sys.exit(_worker_decode_one(args.worker, args.fps, args.max_pixels))

    # Driver branch.
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.is_file():
        print(f"[scan] ERROR: jsonl not found: {jsonl_path}", file=sys.stderr)
        return 4

    # Project root = parent of the script's dir (i.e. RewardVerse/).
    # Same convention as ``data_collator.GoldRubricDataset.from_jsonl``
    # (Path(jsonl).resolve().parents[2]) — we just resolve from the
    # script's location instead so the user can pass any jsonl path.
    project_root = Path(__file__).resolve().parents[1]
    print(f"[scan] project_root = {project_root}", file=sys.stderr)

    # Rebase paths the SAME way training does, so we scan the files the
    # trainer would actually open (not the original /group/... paths
    # from whichever host wrote the jsonl).
    sys.path.insert(0, str(project_root))
    try:
        from rm_training.stage2_grpo.data_collator import rebase_video_paths
    except ImportError as e:
        print(f"[scan] ERROR: cannot import rebase_video_paths "
              f"({e}). Run this script from the project root.",
              file=sys.stderr)
        return 5

    # Collect rows, then rebase, then collect unique rebased paths.
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rebase_video_paths(rows, project_root)
    unique = sorted({r[k] for r in rows for k in ("y_pos", "y_neg") if k in r})
    if args.limit > 0:
        unique = unique[:args.limit]

    # Sanity-check that rebase worked: paths should now resolve.
    n_missing = sum(1 for p in unique if not Path(p).is_file())
    if n_missing == len(unique):
        print(f"[scan] ERROR: ALL {len(unique)} videos missing after "
              f"rebase. Check that videos exist under {project_root}.",
              file=sys.stderr)
        return 6
    if n_missing:
        print(f"[scan] WARN: {n_missing}/{len(unique)} videos missing on "
              f"disk after rebase; they will be marked bad.",
              file=sys.stderr)

    print(f"[scan] {len(rows)} rows reference {len(unique)} unique videos")
    print(f"[scan] workers={args.workers} timeout={args.timeout}s "
          f"fps={args.fps} max_pixels={args.max_pixels}")

    this_script = os.path.abspath(__file__)

    results: list[dict] = []
    n_done = 0
    n_bad = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(_scan_one, p, args.fps, args.max_pixels,
                        args.timeout, this_script): p
            for p in unique
        }
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            n_done += 1
            if _is_bad(r):
                n_bad += 1
                short = r["path"].split("/")[-1]
                msg = r.get("status")
                if msg == "signal":
                    msg = f"signal={r['signal']}"
                elif msg in ("error", "timeout"):
                    msg = f"{msg}={r.get('rc', r.get('wall'))}"
                print(f"[scan] BAD {short}: {msg}")
            if n_done % 25 == 0 or n_done == len(unique):
                rate = n_done / max(time.time() - t0, 1e-6)
                eta = (len(unique) - n_done) / max(rate, 1e-6)
                print(f"[scan] {n_done}/{len(unique)} bad={n_bad} "
                      f"rate={rate:.2f}/s eta={eta:.0f}s")

    bad = [r for r in results if _is_bad(r)]
    bad_paths = {r["path"] for r in bad}

    # Write bad-videos report.
    bad_out = Path(args.bad_json)
    bad_out.parent.mkdir(parents=True, exist_ok=True)
    with bad_out.open("w") as f:
        json.dump({
            "scanned": len(unique),
            "bad": len(bad),
            "fps": args.fps,
            "max_pixels": args.max_pixels,
            "timeout": args.timeout,
            "details": sorted(bad, key=lambda r: r["path"]),
        }, f, indent=2, ensure_ascii=False)
    print(f"[scan] wrote {bad_out} ({len(bad)} bad videos)")

    # Emit cleaned jsonl: keep rows where neither y_pos nor y_neg is bad.
    # We write the ORIGINAL (un-rebased) row dicts so the cleaned jsonl
    # stays portable across hosts — training's rebase pass handles the
    # absolute-path rewrite at load time. To do that we re-load the raw
    # rows (mutated rows in memory have rebased paths after the call to
    # rebase_video_paths above; we need the originals).
    raw_rows: list[dict] = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_rows.append(json.loads(line))
    # Build a parallel "is row bad?" mask using the rebased path set.
    # Each raw row's y_pos/y_neg, after rebase, must not be in bad_paths.
    n_kept = 0
    n_dropped = 0
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dropped_dims: dict[str, int] = {}
    # rows[] (mutated copy) and raw_rows[] are aligned 1:1.
    with out_path.open("w") as f:
        for raw, reb in zip(raw_rows, rows):
            yp = reb.get("y_pos")
            yn = reb.get("y_neg")
            if yp in bad_paths or yn in bad_paths:
                n_dropped += 1
                d = raw.get("dimension", "?")
                dropped_dims[d] = dropped_dims.get(d, 0) + 1
                continue
            f.write(json.dumps(raw, ensure_ascii=False) + "\n")
            n_kept += 1
    print(f"[scan] wrote {out_path}: kept={n_kept} dropped={n_dropped}")
    if dropped_dims:
        per_dim = " ".join(f"{k}={v}" for k, v in sorted(dropped_dims.items()))
        print(f"[scan] dropped per dim: {per_dim}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
