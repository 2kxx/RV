#!/usr/bin/env python
"""Prefetch all remote mp4 videos referenced by the RewardVerse MVP datasets to a
local cache, and write 'localised' copies of the dataset JSON files where every
http(s) mp4 URL is replaced by its corresponding local path.

Why: Tencent venus llmproxy (the OpenAI-compatible gateway used in this MVP)
only accepts static images (JPEG/PNG/WebP/GIF). It cannot ingest mp4 URLs. We
therefore must use the `uniform_frames` video-input strategy, which extracts N
JPEG/PNG frames per clip via decord. decord, however, can only read local
files. Hence: download once, rewrite dataset to point at local paths.

Usage
-----
    python scripts/v_arr_mvp/prefetch_videos.py
    # or to customise:
    python scripts/v_arr_mvp/prefetch_videos.py \
        --src_dir data/v_arr_mvp/visual_quality_mvp \
        --cache_dir data/v_arr_mvp/cache_videos \
        --out_dir  data/v_arr_mvp/visual_quality_mvp_local
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_FILES = [
    "v1_prompt_ywin.json",
    "v2_prompt_pair.json",
    "v3_ywin_only.json",
    "v4_pair_only.json",
]


def is_remote_video(s: object) -> bool:
    return (
        isinstance(s, str)
        and s.startswith(("http://", "https://"))
        and ".mp4" in s.lower()
    )


def collect_urls(obj) -> list[str]:
    found: list[str] = []

    def _walk(x):
        if is_remote_video(x):
            found.append(x)
        elif isinstance(x, dict):
            for v in x.values():
                _walk(v)
        elif isinstance(x, list):
            for v in x:
                _walk(v)

    _walk(obj)
    return found


def local_path_for(url: str, cache_dir: Path) -> Path:
    """Stable, filesystem-safe local filename for a remote URL."""
    parsed = urllib.parse.urlparse(url)
    base = os.path.basename(parsed.path) or "video.mp4"
    if not base.lower().endswith(".mp4"):
        base += ".mp4"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    safe = f"{digest}_{base}"
    return cache_dir / safe


def download(url: str, dst: Path, force: bool = False) -> None:
    if dst.exists() and not force and dst.stat().st_size > 0:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "RewardVerse-prefetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            f.write(chunk)
    tmp.rename(dst)


def rewrite(obj, url_to_path: dict[str, str]):
    if isinstance(obj, str):
        return url_to_path.get(obj, obj) if is_remote_video(obj) else obj
    if isinstance(obj, dict):
        return {k: rewrite(v, url_to_path) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite(v, url_to_path) for v in obj]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="data/v_arr_mvp/visual_quality_mvp")
    ap.add_argument("--cache_dir", default="data/v_arr_mvp/cache_videos")
    ap.add_argument("--out_dir", default="data/v_arr_mvp/visual_quality_mvp_local")
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if cached file exists.")
    args = ap.parse_args()

    src_dir = Path(args.src_dir)
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect every unique URL across all variant files.
    all_urls: set[str] = set()
    loaded: dict[str, object] = {}
    for fname in args.files:
        p = src_dir / fname
        if not p.exists():
            print(f"[warn] missing: {p}", file=sys.stderr)
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        loaded[fname] = data
        all_urls.update(collect_urls(data))

    print(f"[info] {len(all_urls)} unique remote mp4 URLs across {len(loaded)} files")

    # 2. Download each URL once.
    url_to_path: dict[str, str] = {}
    for i, url in enumerate(sorted(all_urls), 1):
        dst = local_path_for(url, cache_dir).resolve()
        try:
            download(url, dst, force=args.force)
            url_to_path[url] = str(dst)
            size_mb = dst.stat().st_size / 1024 / 1024
            print(f"[ok]   ({i}/{len(all_urls)}) {dst.name}  ({size_mb:.1f} MiB)")
        except Exception as e:
            print(f"[fail] ({i}/{len(all_urls)}) {url} -> {e}", file=sys.stderr)

    # 3. Rewrite each dataset JSON.
    for fname, data in loaded.items():
        out_path = out_dir / fname
        new_data = rewrite(data, url_to_path)
        out_path.write_text(
            json.dumps(new_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[ok]   wrote {out_path}")

    missing = sorted(set(all_urls) - set(url_to_path))
    if missing:
        print(f"[warn] {len(missing)} URLs failed to download; the rewritten "
              f"dataset still contains those original URLs and will fail at "
              f"frame-extraction time. First 3: {missing[:3]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
