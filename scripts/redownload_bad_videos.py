#!/usr/bin/env python3
"""Re-download videos that the offline scan flagged as bad.

Why
---
``scripts/scan_bad_videos.py`` writes ``bad_videos.json`` listing the
local files that crash decord/ffmpeg during decode. Two failure modes
land here:

  * the file on disk is genuinely corrupt (truncated download, partial
    cache, mis-named copy) → re-fetching from the COS URL fixes it,
  * the source video itself is broken at the encoder level → re-download
    won't help; we'll detect that via the post-download re-verify pass
    and report which ones are still bad.

Re-downloading is preferable to dropping the rows because Stage ② is
data-starved (480 rows / 16 dims = 30 pairs/dim) and dropping ~10
"bad" entries from the under-covered dims (logic 17, rhythm 22,
action 29, expression 29) loses a noticeable chunk of supervision.

How it finds the URL
--------------------
The jsonl row has BOTH the local path (``y_pos``/``y_neg``) and the
COS URL (``url_pos``/``url_neg``) — they're aligned by index (pos↔pos,
neg↔neg). We rebase the row's local paths the same way training does,
match each against the bad-set, then fetch from the corresponding URL.

Usage
-----
Default workflow — re-download then re-verify in subprocess isolation::

    python scripts/redownload_bad_videos.py \\
        --jsonl data/rm_train/gold_rubrics_18d.clean.jsonl \\
        --bad_json data/rm_train/bad_videos.json

After it finishes, re-run ``scripts/scan_bad_videos.py`` to confirm 0
videos still segfault. Anything that's still bad after re-download is
encoder-level corrupt at the source — those genuinely need to be
dropped (or fixed upstream).

Options::

    --dry_run        only print what would be downloaded, don't fetch
    --no_verify      skip the post-download re-verify pass
    --workers N      parallel downloads (default 4)
    --timeout N      per-download HTTP timeout in seconds (default 120)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _download_one(url: str, dst: Path, timeout: float,
                  user_agent: str, min_bytes: int) -> dict:
    """Fetch ``url`` to ``dst``. Returns status dict.

    Sanity checks
    -------------
    * HTTP non-200 → error status raised by urlopen, caught below.
    * Body smaller than ``min_bytes`` → treat as failed download, do
      NOT replace the existing file (it might be the only thing we
      have). 1 KB is a generous floor — any real mp4 has >> 1 KB.
    """
    t0 = time.time()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=timeout) as resp, \
                tmp.open("wb") as f:
            total = 0
            while True:
                chunk = resp.read(1 << 16)  # 64 KB
                if not chunk:
                    break
                f.write(chunk)
                total += len(chunk)
        if total < min_bytes:
            tmp.unlink(missing_ok=True)
            return {"url": url, "dst": str(dst), "ok": False,
                    "error": f"body too small ({total} < {min_bytes} bytes)",
                    "wall": time.time() - t0}
        tmp.replace(dst)
        return {"url": url, "dst": str(dst), "ok": True,
                "bytes": total, "wall": time.time() - t0}
    except (urllib.error.URLError, urllib.error.HTTPError,
            TimeoutError, OSError) as e:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        return {"url": url, "dst": str(dst), "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "wall": time.time() - t0}


def _verify_one(video_path: str, fps: float, max_pixels: int,
                timeout: float, scan_script: str) -> dict:
    """Re-decode in subprocess isolation, like scan_bad_videos.py does.

    Buffers worker output to a tempfile (not a pipe) — see ``_scan_one``
    in scan_bad_videos.py for why this matters."""
    import tempfile
    t0 = time.time()
    cmd = [sys.executable, scan_script, "--worker", video_path,
           "--fps", str(fps), "--max_pixels", str(max_pixels)]
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
        logbuf.seek(0, os.SEEK_END)
        size = logbuf.tell()
        logbuf.seek(max(0, size - 500))
        tail = logbuf.read().decode("utf-8", errors="replace")
        if rc < 0:
            return {"path": video_path, "status": "signal",
                    "signal": -rc, "wall": wall, "stderr": tail}
        return {"path": video_path, "status": "error", "rc": rc,
                "wall": wall, "stderr": tail}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl",
                    default="data/rm_train/gold_rubrics_18d.clean.jsonl",
                    help="source jsonl with url_pos/url_neg fields")
    ap.add_argument("--bad_json",
                    default="data/rm_train/bad_videos.json",
                    help="output of scan_bad_videos.py")
    ap.add_argument("--report_json",
                    default="data/rm_train/redownload_report.json",
                    help="write per-file outcome here")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="HTTP timeout per file (seconds)")
    ap.add_argument("--verify_timeout", type=float, default=60.0,
                    help="decode-verify timeout per file (seconds)")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max_pixels", type=int, default=360 * 420)
    ap.add_argument("--dry_run", action="store_true",
                    help="print plan, do not fetch")
    ap.add_argument("--no_verify", action="store_true",
                    help="skip post-download decode verification")
    ap.add_argument("--user_agent",
                    default="rewardverse-redownload/1.0",
                    help="HTTP User-Agent header")
    ap.add_argument("--min_bytes", type=int, default=1024,
                    help="reject downloads smaller than this many bytes "
                         "(guards against CDN returning 200 with empty "
                         "body)")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    print(f"[redl] project_root = {project_root}", file=sys.stderr)

    sys.path.insert(0, str(project_root))
    try:
        from rm_training.stage2_grpo.data_collator import rebase_video_paths
    except ImportError as e:
        print(f"[redl] ERROR: cannot import rebase_video_paths "
              f"({e}). Run from project root.", file=sys.stderr)
        return 5

    jsonl_path = Path(args.jsonl)
    bad_json_path = Path(args.bad_json)
    if not jsonl_path.is_file():
        print(f"[redl] ERROR: jsonl not found: {jsonl_path}", file=sys.stderr)
        return 4
    if not bad_json_path.is_file():
        print(f"[redl] ERROR: bad_json not found: {bad_json_path}",
              file=sys.stderr)
        return 4

    # Load bad-set.
    with bad_json_path.open() as f:
        bad_doc = json.load(f)
    bad_paths = {d["path"] for d in bad_doc.get("details", [])}
    if not bad_paths:
        print(f"[redl] {bad_json_path} reports 0 bad videos. Nothing "
              f"to re-download.")
        return 0
    print(f"[redl] {len(bad_paths)} bad videos to re-download "
          f"(per {bad_json_path.name})")

    # Build path → URL map by walking the jsonl with rebased paths.
    raw_rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_rows.append(json.loads(line))
    rebased_rows = [dict(r) for r in raw_rows]
    rebase_video_paths(rebased_rows, project_root)

    path_to_url: dict[str, str] = {}
    for raw, reb in zip(raw_rows, rebased_rows):
        for side in ("pos", "neg"):
            local = reb.get(f"y_{side}")
            url = raw.get(f"url_{side}")
            if local and url and local not in path_to_url:
                path_to_url[local] = url

    # Resolve which bad paths we have URLs for.
    plan = []
    no_url = []
    for p in sorted(bad_paths):
        url = path_to_url.get(p)
        if url is None:
            no_url.append(p)
        else:
            plan.append((p, url))

    if no_url:
        print(f"[redl] WARN: {len(no_url)} bad paths have no matching "
              f"URL in the jsonl — they'll be skipped:", file=sys.stderr)
        for p in no_url[:10]:
            print(f"  - {p}", file=sys.stderr)
        if len(no_url) > 10:
            print(f"  ... ({len(no_url) - 10} more)", file=sys.stderr)

    print(f"[redl] {len(plan)} files queued for download "
          f"(workers={args.workers} timeout={args.timeout}s)")
    if args.dry_run:
        for p, url in plan[:20]:
            print(f"  WOULD FETCH {url}\n          → {p}")
        if len(plan) > 20:
            print(f"  ... ({len(plan) - 20} more)")
        return 0

    # Download in parallel.
    download_results = []
    n_done = 0
    n_ok = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(_download_one, url, Path(p), args.timeout,
                        args.user_agent, args.min_bytes): (p, url)
            for p, url in plan
        }
        for fut in as_completed(futs):
            r = fut.result()
            download_results.append(r)
            n_done += 1
            if r["ok"]:
                n_ok += 1
                print(f"[redl] OK  {Path(r['dst']).name} "
                      f"({r['bytes'] / 1024:.0f} KB, {r['wall']:.1f}s)")
            else:
                print(f"[redl] FAIL {Path(r['dst']).name}: {r['error']}")
            if n_done % 5 == 0 or n_done == len(plan):
                rate = n_done / max(time.time() - t0, 1e-6)
                eta = (len(plan) - n_done) / max(rate, 1e-6)
                print(f"[redl] {n_done}/{len(plan)} ok={n_ok} "
                      f"rate={rate:.2f}/s eta={eta:.0f}s")

    # Verify pass — re-decode each freshly-downloaded file in subprocess
    # isolation, exactly like scan_bad_videos.py.
    verify_results = []
    if not args.no_verify:
        good_dl = [r for r in download_results if r["ok"]]
        if good_dl:
            print(f"[redl] verifying {len(good_dl)} re-downloaded videos…")
            scan_script = os.path.abspath(os.path.join(
                os.path.dirname(__file__), "scan_bad_videos.py"))
            if not os.path.isfile(scan_script):
                print(f"[redl] WARN: cannot find {scan_script} — "
                      f"skipping verify pass", file=sys.stderr)
            else:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futs = {
                        pool.submit(_verify_one, r["dst"], args.fps,
                                    args.max_pixels, args.verify_timeout,
                                    scan_script): r["dst"]
                        for r in good_dl
                    }
                    for fut in as_completed(futs):
                        v = fut.result()
                        verify_results.append(v)
                        if v["status"] != "ok":
                            sig = v.get("signal") or v.get("rc") \
                                or v.get("status")
                            print(f"[redl] STILL BAD after re-download: "
                                  f"{Path(v['path']).name} ({sig})")

    n_dl_fail = len([r for r in download_results if not r["ok"]])
    n_still_bad = len([v for v in verify_results if v["status"] != "ok"])
    n_recovered = n_ok - n_still_bad

    summary = {
        "total_bad_input": len(bad_paths),
        "no_url_skipped": len(no_url),
        "downloads_attempted": len(plan),
        "downloads_ok": n_ok,
        "downloads_failed": n_dl_fail,
        "verified_ok": n_recovered,
        "still_bad_after_redl": n_still_bad,
        "fps": args.fps,
        "max_pixels": args.max_pixels,
        "downloads": download_results,
        "verify": verify_results,
        "no_url_paths": no_url,
    }
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_json, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print(f"[redl] === summary ===")
    print(f"  bad in scan        : {len(bad_paths)}")
    print(f"  no URL in jsonl    : {len(no_url)}  (skipped)")
    print(f"  download attempted : {len(plan)}")
    print(f"  download ok        : {n_ok}")
    print(f"  download failed    : {n_dl_fail}")
    if not args.no_verify:
        print(f"  recovered (verified): {n_recovered}")
        print(f"  STILL bad          : {n_still_bad}")
    print(f"  report             : {args.report_json}")
    if not args.no_verify and n_still_bad == 0 and n_dl_fail == 0:
        print()
        print(f"[redl] All bad videos recovered. Re-run "
              f"scripts/scan_bad_videos.py to confirm 0 segfaults, then "
              f"point training at the original (unfiltered) jsonl.")
    elif n_still_bad or n_dl_fail:
        print()
        print(f"[redl] Some videos still unusable after re-download.")
        print(f"       Either run scan_bad_videos.py to emit a "
              f"cleaned jsonl that drops the residual bad rows, or "
              f"investigate them manually (see "
              f"{args.report_json}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
