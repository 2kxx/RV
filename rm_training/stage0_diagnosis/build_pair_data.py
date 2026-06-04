"""rm_training.stage0_diagnosis.build_pair_data
============================================
Stage ⓪ data preparation — convert RewardVerse's existing 18-dim pair
files into the *exp0* JSONL schema expected by ``exp0_base_diagnosis.py``.

Two outputs:
  1. ``data/rm_train/exp0/exp0_pairs.jsonl``       (D1 / D2)
     {id, instruction, video_pos, video_neg, dimension}
  2. ``data/rm_train/exp0/exp0_dim_pairs.jsonl``   (D3)
     same schema; ``dimension`` is *required* and points at the
     RewardVerse axis whose pair seed produced the sample.

What we adapt from existing data
--------------------------------
Source: ``data/v_arr_mvp/<dim>_mvp/v2_prompt_pair.json``  (505 pairs / 18 dims)

Original schema::

    {"id": "varr_vq_000",
     "query": "<full prompt with [shot_0_prompt] ...>",
     "response": ["<URL_a>", "<URL_b>"],
     "label_rank": [1, 2] | [2, 1] | [1, 1],
     "task_type": "...",
     "metadata": {...}}

Mapping rules
-------------
* ``instruction`` ← ``query``
* ``label_rank == [1, 2]`` → ``video_pos = response[0]``, ``video_neg = response[1]``
* ``label_rank == [2, 1]`` → ``video_pos = response[1]``, ``video_neg = response[0]``
* ``label_rank == [1, 1]`` → **dropped** (D1 has no signal on ties)
* ``dimension`` ← parent folder ``<dim>_mvp`` (strip ``_mvp``)
* URLs ← rewritten to local paths if a cached copy exists; otherwise
  left as URLs (``--require_local`` flag will drop unresolved rows).

Local cache discovery
---------------------
We try TWO naming conventions in priority order so already-downloaded
videos are reused:

  (a) ``runs/baselines/_video_cache/<sha256(url)[:32]>.mp4``
      (used by ``baseline_videoscore.py`` / ``baseline_videoscore2.py``;
      currently has 236 pre-cached videos)
  (b) ``data/v_arr_mvp/cache_videos/<sha1(url)[:10]>_<basename>.mp4``
      (used by ``prefetch_videos.py``)

Whichever exists first wins. Use ``--prefetch_missing`` to download the
rest into convention (b).

CLI
---
::

    # 1) dry-run: see how many pairs survive cache resolution
    python rm_training/stage0_diagnosis/build_pair_data.py --dry_run

    # 2) actually emit the JSONL files (drops URLs that resolve to no local file)
    python rm_training/stage0_diagnosis/build_pair_data.py \\
        --pair_glob "data/v_arr_mvp/*_mvp/v2_prompt_pair.json" \\
        --out_dir   data/rm_train/exp0 \\
        --require_local

    # 3) full run: also fetch any missing video into cache_videos/
    python rm_training/stage0_diagnosis/build_pair_data.py \\
        --prefetch_missing \\
        --require_local
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. Cache convention helpers (mirror baseline_videoscore.py + prefetch_videos.py).
# ---------------------------------------------------------------------------
def cache_path_baseline(url: str, cache_dir: Path) -> Path:
    """``baseline_videoscore.py`` convention: ``sha256(url)[:32].mp4``."""
    return cache_dir / (hashlib.sha256(url.encode("utf-8")).hexdigest()[:32] + ".mp4")


def cache_path_prefetch(url: str, cache_dir: Path) -> Path:
    """``prefetch_videos.py`` convention: ``sha1[:10]_<basename>.mp4``."""
    parsed = urllib.parse.urlparse(url)
    base = os.path.basename(parsed.path) or "video.mp4"
    if not base.lower().endswith(".mp4"):
        base += ".mp4"
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return cache_dir / f"{digest}_{base}"


def resolve_local(
    url_or_path: str,
    *,
    baseline_cache: Path,
    prefetch_cache: Path,
) -> str | None:
    """Return an existing local path for ``url_or_path``, or None if absent.

    Order:
      1. If it's already a local file, return as-is.
      2. Try baseline cache (sha256[:32]).
      3. Try prefetch cache (sha1[:10]_<basename>).
      4. Otherwise, fail.
    """
    if not isinstance(url_or_path, str):
        return None
    # Local already?
    if not url_or_path.startswith(("http://", "https://")):
        return url_or_path if Path(url_or_path).is_file() else None
    # Cache convention (a).
    p_a = cache_path_baseline(url_or_path, baseline_cache)
    if p_a.is_file() and p_a.stat().st_size > 0:
        return str(p_a)
    # Cache convention (b).
    p_b = cache_path_prefetch(url_or_path, prefetch_cache)
    if p_b.is_file() and p_b.stat().st_size > 0:
        return str(p_b)
    return None


def prefetch_one(url: str, dst: Path, *, force: bool = False, timeout: int = 120) -> bool:
    """Download ``url`` to ``dst``. Skips if dst already non-empty.

    Returns True iff the file is on disk and >0 bytes after this call.
    """
    if dst.is_file() and dst.stat().st_size > 0 and not force:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, tmp.open("wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        tmp.rename(dst)
        return dst.stat().st_size > 0
    except Exception as e:
        print(f"[prefetch] FAIL {url}: {e}", file=sys.stderr)
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False


# ---------------------------------------------------------------------------
# 2. v_arr_mvp → exp0 row adapter.
# ---------------------------------------------------------------------------
def _dim_from_folder(folder_name: str) -> str:
    """``visual_quality_mvp`` → ``visual_quality``."""
    if folder_name.endswith("_mvp"):
        return folder_name[:-4]
    return folder_name


def adapt_pair_record(
    rec: dict,
    *,
    dimension: str,
    baseline_cache: Path,
    prefetch_cache: Path,
    require_local: bool,
    drop_ties: bool = True,
) -> dict | None:
    """Map one v_arr_mvp pair record into the exp0 schema.

    Returns ``None`` if the record should be dropped (tie / unresolved
    video / malformed).
    """
    pid = rec.get("id")
    instruction = rec.get("query") or rec.get("prompt") or ""
    responses = rec.get("response") or []
    label_rank = rec.get("label_rank")
    if not pid or len(responses) < 2 or not isinstance(label_rank, (list, tuple)):
        return None

    a, b = responses[0], responses[1]
    try:
        ra, rb = int(label_rank[0]), int(label_rank[1])
    except (TypeError, ValueError):
        return None
    if ra == rb:
        # Tie: D1 cannot get a signal here.
        return None if drop_ties else None
    if ra < rb:
        v_pos, v_neg = a, b
    else:
        v_pos, v_neg = b, a

    # Resolve to local files.
    lp_pos = resolve_local(v_pos, baseline_cache=baseline_cache, prefetch_cache=prefetch_cache)
    lp_neg = resolve_local(v_neg, baseline_cache=baseline_cache, prefetch_cache=prefetch_cache)

    # If require_local, drop pairs that don't fully resolve.
    if require_local and (lp_pos is None or lp_neg is None):
        return None

    return {
        "id":          pid,
        "instruction": instruction,
        "video_pos":   lp_pos or v_pos,    # fall back to URL if not requiring local
        "video_neg":   lp_neg or v_neg,
        "dimension":   dimension,
        # Carry the originals for full traceability.
        "url_pos":     v_pos,
        "url_neg":     v_neg,
    }


# ---------------------------------------------------------------------------
# 2b. Pointwise-supplement: rebuild pairs for short dims from
#     ``test_pointwise_<dim>.json`` (top-score vs bottom-score within
#     the same dim). Used to lift logic / rhythm / soundscape from
#     17/22/16 native pair seeds up to ~30.
# ---------------------------------------------------------------------------
def _pair_from_pointwise(
    pointwise_file: Path,
    *,
    dimension:      str,
    target_pairs:   int,
    score_gap_min:  float,
    baseline_cache: Path,
    prefetch_cache: Path,
    require_local:  bool,
    existing_ids:   set[str] | None = None,
) -> list[dict]:
    """Pair-mine a pointwise (single-video, label_score) test file.

    Strategy (deterministic):
      1. Sort pointwise records by label_score descending.
      2. Pair the i-th from the top with the i-th from the bottom
         (largest score gap first), iterate until ``target_pairs`` is
         reached or the next pair's gap drops below ``score_gap_min``.
      3. Skip pairs whose ID combo already shows up in ``existing_ids``
         (so the union with v2_prompt_pair stays unique).

    Pointwise schema (from data/v_arr_mvp/<dim>_mvp/test_pointwise_<dim>.json)::

        {"id": "varr_<dim>_test_v2_NNN",
         "query": "...",
         "response": "<URL or local path>",   # SINGLE video
         "label_score": <float, e.g. 1.0..5.0>,
         "dimension": "<dim>"}
    """
    if not pointwise_file.is_file():
        return []
    try:
        rows = json.loads(pointwise_file.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[pw->pair] failed to read {pointwise_file}: {e}", file=sys.stderr)
        return []
    if not isinstance(rows, list) or not rows:
        return []

    # Keep only rows with a usable label_score and a single response path.
    good: list[tuple[float, dict]] = []
    for r in rows:
        try:
            score = float(r.get("label_score"))
        except (TypeError, ValueError):
            continue
        resp = r.get("response")
        # `response` may be a single URL/path string, or a 1-element list.
        if isinstance(resp, list):
            resp = resp[0] if resp else None
        if not isinstance(resp, str):
            continue
        good.append((score, {**r, "_resolved_video": resp}))
    if len(good) < 2:
        return []

    good.sort(key=lambda kv: kv[0], reverse=True)   # top first
    n = len(good)
    out: list[dict] = []
    seen_ids = set(existing_ids or set())

    # Pair (top, bottom), (top-1, bottom+1), ..., until we hit target_pairs
    # or the score gap closes.
    for i in range(n // 2):
        top_score, top_rec = good[i]
        bot_score, bot_rec = good[n - 1 - i]
        gap = top_score - bot_score
        if gap < score_gap_min:
            break
        if len(out) >= target_pairs:
            break

        pid = f"{dimension}_pwpair_{i:03d}"
        if pid in seen_ids:
            continue

        v_pos = top_rec["_resolved_video"]
        v_neg = bot_rec["_resolved_video"]
        lp_pos = resolve_local(v_pos, baseline_cache=baseline_cache, prefetch_cache=prefetch_cache)
        lp_neg = resolve_local(v_neg, baseline_cache=baseline_cache, prefetch_cache=prefetch_cache)
        if require_local and (lp_pos is None or lp_neg is None):
            continue

        # Use whichever record has a non-empty query as the instruction;
        # they should agree because pointwise records share the prompt.
        instr = (top_rec.get("query") or bot_rec.get("query")
                 or top_rec.get("prompt") or "")
        out.append({
            "id":          pid,
            "instruction": instr,
            "video_pos":   lp_pos or v_pos,
            "video_neg":   lp_neg or v_neg,
            "dimension":   dimension,
            "url_pos":     v_pos,
            "url_neg":     v_neg,
            "label_score_pos": top_score,
            "label_score_neg": bot_score,
            "score_gap":   gap,
            "source":      "pointwise_supplement",
            "source_pointwise_file": str(pointwise_file),
        })
        seen_ids.add(pid)

    return out


def _select_pointwise_file(dim_dir: Path, dimension: str) -> Path | None:
    """Prefer ``test_pointwise_<dim>_v2.json`` if present, else fall back
    to ``test_pointwise_<dim>.json``."""
    p_v2 = dim_dir / f"test_pointwise_{dimension}_v2.json"
    if p_v2.is_file():
        return p_v2
    p_v1 = dim_dir / f"test_pointwise_{dimension}.json"
    if p_v1.is_file():
        return p_v1
    return None


# ---------------------------------------------------------------------------
# 3. Driver.
# ---------------------------------------------------------------------------
def build(args: argparse.Namespace) -> None:
    pair_files = sorted((ROOT).glob(args.pair_glob))
    if not pair_files:
        # Allow the glob to be already absolute.
        pair_files = sorted(Path(".").glob(args.pair_glob))
    if not pair_files:
        raise SystemExit(f"[build] no files matched glob {args.pair_glob}")
    print(f"[build] found {len(pair_files)} pair files", file=sys.stderr)

    baseline_cache = (ROOT / args.baseline_cache_dir).resolve()
    prefetch_cache = (ROOT / args.prefetch_cache_dir).resolve()
    out_dir        = (ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: collect URLs we'd want to prefetch (only if user asked).
    urls_to_fetch: set[str] = set()
    if args.prefetch_missing:
        for pf in pair_files:
            try:
                rows = json.loads(pf.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[build] skip unreadable {pf}: {e}", file=sys.stderr)
                continue
            for r in rows:
                for u in (r.get("response") or []):
                    if not isinstance(u, str) or not u.startswith(("http://", "https://")):
                        continue
                    if resolve_local(u, baseline_cache=baseline_cache,
                                       prefetch_cache=prefetch_cache) is None:
                        urls_to_fetch.add(u)
        if urls_to_fetch:
            print(f"[build] prefetching {len(urls_to_fetch)} missing videos -> "
                  f"{prefetch_cache}", file=sys.stderr)
            prefetch_cache.mkdir(parents=True, exist_ok=True)
            for i, u in enumerate(sorted(urls_to_fetch), 1):
                ok = prefetch_one(u, cache_path_prefetch(u, prefetch_cache))
                if i % 20 == 0 or not ok:
                    print(f"  [{i}/{len(urls_to_fetch)}] {'OK' if ok else 'FAIL'} {u[:80]}",
                          file=sys.stderr)

    # Pass 2: adapt every record.
    rows_out: list[dict] = []
    drop_stats: Counter[str] = Counter()
    by_dim: dict[str, int] = defaultdict(int)

    for pf in pair_files:
        dim = _dim_from_folder(pf.parent.name)
        try:
            recs = json.loads(pf.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[build] skip unreadable {pf}: {e}", file=sys.stderr)
            continue
        for r in recs:
            label = r.get("label_rank")
            if isinstance(label, (list, tuple)) and len(label) == 2:
                try:
                    if int(label[0]) == int(label[1]):
                        drop_stats["tie"] += 1
                except (TypeError, ValueError):
                    pass
            adapted = adapt_pair_record(
                r, dimension=dim,
                baseline_cache=baseline_cache,
                prefetch_cache=prefetch_cache,
                require_local=args.require_local,
            )
            if adapted is None:
                if r.get("label_rank") == [1, 1]:
                    pass  # already counted under "tie"
                else:
                    drop_stats["unresolved_or_malformed"] += 1
                continue
            rows_out.append(adapted)
            by_dim[dim] += 1

    print(f"[build] (pair-files) kept {len(rows_out)} pairs; drop stats: {dict(drop_stats)}",
          file=sys.stderr)
    print(f"[build] (pair-files) per-dim counts:", file=sys.stderr)
    for d, n in sorted(by_dim.items()):
        print(f"  {d:<22s} {n:4d}", file=sys.stderr)

    # ── Pass 1b: pointwise supplement for short dimensions.
    if args.pointwise_supplement:
        print(f"[build] pointwise supplement enabled  target_per_dim={args.target_per_dim}",
              file=sys.stderr)
        existing_ids: set[str] = {r["id"] for r in rows_out}
        sup_rows: list[dict] = []
        sup_by_dim: dict[str, int] = defaultdict(int)
        for pf in pair_files:
            dim = _dim_from_folder(pf.parent.name)
            current = by_dim.get(dim, 0)
            need = max(0, args.target_per_dim - current)
            if need <= 0:
                continue
            pw_file = _select_pointwise_file(pf.parent, dim)
            if pw_file is None:
                continue
            extra = _pair_from_pointwise(
                pw_file,
                dimension      = dim,
                target_pairs   = need,
                score_gap_min  = args.score_gap_min,
                baseline_cache = baseline_cache,
                prefetch_cache = prefetch_cache,
                require_local  = args.require_local,
                existing_ids   = existing_ids,
            )
            sup_rows.extend(extra)
            sup_by_dim[dim] += len(extra)
            existing_ids.update(r["id"] for r in extra)
            print(f"  {dim:<22s} need={need:>2d}  supplemented={len(extra):>2d}  "
                  f"from {pw_file.name}", file=sys.stderr)
        rows_out.extend(sup_rows)
        for d, n in sup_by_dim.items():
            by_dim[d] += n
        print(f"[build] (after supplement) total pairs={len(rows_out)}",
              file=sys.stderr)
        for d, n in sorted(by_dim.items()):
            print(f"  {d:<22s} {n:4d}", file=sys.stderr)

    if args.dry_run:
        print("[build] --dry_run set; not writing files.", file=sys.stderr)
        return

    # ── exp0_pairs.jsonl: D1 / D2 sample (cap per dim if needed for cost).
    pairs_path = out_dir / "exp0_pairs.jsonl"
    n_per_dim = args.cap_per_dim_d1
    if n_per_dim and n_per_dim > 0:
        # take top-N from each dim deterministically.
        d1_rows: list[dict] = []
        seen: dict[str, int] = defaultdict(int)
        for r in rows_out:
            d = r["dimension"]
            if seen[d] < n_per_dim:
                d1_rows.append(r); seen[d] += 1
    else:
        d1_rows = rows_out
    with pairs_path.open("w", encoding="utf-8") as f:
        for r in d1_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[build] wrote {len(d1_rows):4d} -> {pairs_path}", file=sys.stderr)

    # ── exp0_dim_pairs.jsonl: D3 needs explicit dimension; we inherit from folder.
    dim_path = out_dir / "exp0_dim_pairs.jsonl"
    with dim_path.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[build] wrote {len(rows_out):4d} -> {dim_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 4. CLI.
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair_glob",
                   default="data/v_arr_mvp/*_mvp/v2_prompt_pair.json",
                   help="glob (relative to repo root) matching per-dim pair JSON files")
    p.add_argument("--out_dir",
                   default="data/rm_train/exp0")
    p.add_argument("--baseline_cache_dir",
                   default="runs/baselines/_video_cache",
                   help="where baseline_videoscore.py stores videos (sha256[:32].mp4)")
    p.add_argument("--prefetch_cache_dir",
                   default="data/v_arr_mvp/cache_videos",
                   help="where prefetch_videos.py stores videos (sha1[:10]_basename.mp4)")
    p.add_argument("--require_local", action="store_true",
                   help="drop pairs whose videos can't be resolved to a local file")
    p.add_argument("--prefetch_missing", action="store_true",
                   help="download missing videos into prefetch_cache_dir before adapting")
    p.add_argument("--cap_per_dim_d1", type=int, default=0,
                   help="if >0, exp0_pairs.jsonl caps each dim to this many "
                        "(default: full set, no cap)")
    p.add_argument("--pointwise_supplement", action="store_true",
                   help="for any dim whose v2_prompt_pair has fewer than "
                        "--target_per_dim pairs (logic=17, rhythm=22, "
                        "soundscape=16), mine extra pairs from the per-dim "
                        "test_pointwise_<dim>.json by score top-vs-bottom.")
    p.add_argument("--target_per_dim", type=int, default=30,
                   help="per-dim target pair count when --pointwise_supplement is on")
    p.add_argument("--score_gap_min", type=float, default=1.0,
                   help="minimum label_score gap when mining pairs from pointwise data")
    p.add_argument("--dry_run", action="store_true",
                   help="report stats only, don't write JSONL")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    build(args)


if __name__ == "__main__":
    main()
