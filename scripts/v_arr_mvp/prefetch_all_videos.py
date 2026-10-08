#!/usr/bin/env python
"""Prefetch every remote .mp4 referenced by RewardVerse v_arr_mvp datasets.

Sweeps a configurable list of dataset JSON files (default: every per-dim
``v2_prompt_pair.json`` and ``test_pointwise_<dim>.json`` /
``test_pointwise_<dim>_v2.json``), collects all unique ``http(s)://...mp4``
URLs, and downloads them concurrently into the prefetch cache.

Why this script (vs the original prefetch_videos.py): the latter only
sweeps a single src_dir at a time and is single-threaded. After the
2026-06-18 data extension we have ~2k videos to fetch across 16 dims,
which would take 2-3h serially. This script:

* discovers files automatically across the dim folder tree,
* parallelises downloads with a thread pool (default 10 workers),
* uses the **same** filename convention as build_pair_data.cache_path_prefetch
  (``sha1[:10]_<basename>.mp4``) so resolve_local() finds them.

Usage
-----

    python scripts/v_arr_mvp/prefetch_all_videos.py
    # Custom concurrency / dirs:
    python scripts/v_arr_mvp/prefetch_all_videos.py \
        --concurrency 16 \
        --cache_dir data/v_arr_mvp/cache_videos
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def is_remote_video(s: object) -> bool:
    return (
        isinstance(s, str)
        and s.startswith(("http://", "https://"))
        and ".mp4" in s.lower()
    )


def collect_urls(obj) -> set[str]:
    found: set[str] = set()

    def _walk(x):
        if is_remote_video(x):
            found.add(x)
        elif isinstance(x, dict):
            for v in x.values():
                _walk(v)
        elif isinstance(x, list):
            for v in x:
                _walk(v)

    _walk(obj)
    return found


def cache_path_prefetch(url: str, cache_dir: Path) -> Path:
    """Same convention as ``rm_training/stage0_diagnosis/build_pair_data.py``
    so resolve_local() will find these files.
    """
    parsed = urllib.parse.urlparse(url)
    base = os.path.basename(parsed.path) or "video.mp4"
    if not base.lower().endswith(".mp4"):
        base += ".mp4"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return cache_dir / f"{digest}_{base}"


def download_one(url: str, dst: Path, *,
                 timeout: int = 180, force: bool = False) -> tuple[bool, str]:
    """Returns (ok, message). Skips when dst already non-empty."""
    if dst.exists() and dst.stat().st_size > 0 and not force:
        return True, "skip-cached"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "RewardVerse-prefetch/2.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp, tmp.open("wb") as f:
            while True:
                chunk = resp.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                f.write(chunk)
        if tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            return False, "empty-body"
        tmp.rename(dst)
        return True, f"ok ({dst.stat().st_size / 1024 / 1024:.1f} MiB)"
    except Exception as e:
        tmp.unlink(missing_ok=True)
        return False, f"err: {e}"


def discover_dataset_files(roots: list[Path]) -> list[Path]:
    """Return the list of dataset JSON files to scan for video URLs."""
    found: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if root.is_file():
            if root.suffix == ".json" or root.suffix == ".jsonl":
                if root not in seen:
                    seen.add(root); found.append(root)
            continue
        if not root.is_dir():
            continue
        # Standard v_arr_mvp dim folders.
        for sub in sorted(root.iterdir()):
            if not sub.is_dir() or not sub.name.endswith("_mvp"):
                continue
            for fname in (
                "v2_prompt_pair.json",
                f"test_pointwise_{sub.name[:-4]}.json",
                f"test_pointwise_{sub.name[:-4]}_v2.json",
                "shared_test_pointwise.json",  # vq legacy
            ):
                fp = sub / fname
                if fp.is_file() and fp not in seen:
                    seen.add(fp); found.append(fp)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan_root", action="append", default=None,
                    help="dir to scan for *_mvp dataset jsons (default: "
                         "data/v_arr_mvp). Can be passed multiple times.")
    ap.add_argument("--cache_dir", default=str(ROOT / "data/v_arr_mvp/cache_videos"),
                    help="output directory for cached videos.")
    ap.add_argument("--concurrency", type=int, default=10,
                    help="number of parallel HTTP downloads (default 10).")
    ap.add_argument("--force", action="store_true",
                    help="re-download even if cached.")
    ap.add_argument("--manifest", default=None,
                    help="optional path to write a JSON summary of "
                         "(url -> local_path / failure_reason).")
    args = ap.parse_args()

    scan_roots = [Path(p) for p in (args.scan_root or [str(ROOT / "data/v_arr_mvp")])]
    cache_dir = Path(args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = discover_dataset_files(scan_roots)
    print(f"[scan] {len(files)} dataset files under {[str(p) for p in scan_roots]}",
          file=sys.stderr)

    all_urls: set[str] = set()
    for fp in files:
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] {fp}: {e}", file=sys.stderr)
            continue
        urls = collect_urls(data)
        all_urls.update(urls)
    all_urls_sorted = sorted(all_urls)
    print(f"[scan] {len(all_urls_sorted)} unique remote .mp4 URLs",
          file=sys.stderr)

    # Pre-skip already-cached URLs so the progress counter is meaningful.
    pending = []
    cached = 0
    for url in all_urls_sorted:
        dst = cache_path_prefetch(url, cache_dir)
        if dst.exists() and dst.stat().st_size > 0 and not args.force:
            cached += 1
        else:
            pending.append((url, dst))
    print(f"[scan] {cached} already cached; {len(pending)} to fetch "
          f"({args.concurrency} workers)", file=sys.stderr)

    results: dict[str, str] = {}
    failures: dict[str, str] = {}

    def _job(item):
        url, dst = item
        ok, msg = download_one(url, dst, force=args.force)
        return url, dst, ok, msg

    if pending:
        # Use as_completed instead of map() so the progress counter
        # reflects DOWNLOAD finishes in real time, not the input-order
        # finish queue. With map() a single slow URL stalls the visible
        # progress until everything ahead of it has completed too.
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as ex:
            futs = [ex.submit(_job, item) for item in pending]
            done = 0
            for fut in concurrent.futures.as_completed(futs):
                url, dst, ok, msg = fut.result()
                done += 1
                if ok:
                    results[url] = str(dst)
                else:
                    failures[url] = msg
                if done % 10 == 0 or done == len(pending):
                    print(f"[fetch] {done}/{len(pending)}  ok={len(results)}  "
                          f"fail={len(failures)}", file=sys.stderr, flush=True)

    # Always include already-cached entries in results map.
    for url in all_urls_sorted:
        if url in failures:
            continue
        if url not in results:
            dst = cache_path_prefetch(url, cache_dir)
            if dst.exists() and dst.stat().st_size > 0:
                results[url] = str(dst)

    print(f"[done] cached={len(results)}  failed={len(failures)}  "
          f"total={len(all_urls_sorted)}", file=sys.stderr)
    if failures:
        print("[done] first 5 failures:", file=sys.stderr)
        for url, why in list(failures.items())[:5]:
            print(f"  {why}  {url}", file=sys.stderr)

    if args.manifest:
        out = {
            "cache_dir": str(cache_dir),
            "n_total": len(all_urls_sorted),
            "n_cached": len(results),
            "n_failed": len(failures),
            "url_to_path": results,
            "failures": failures,
        }
        Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
        Path(args.manifest).write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[done] manifest -> {args.manifest}", file=sys.stderr)

    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
