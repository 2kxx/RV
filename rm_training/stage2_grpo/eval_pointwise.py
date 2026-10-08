"""rm_training.stage2_grpo.eval_pointwise
=====================================
Pointwise PLCC/SRCC evaluation against the EvalVerse hold-out test set.

Shared by:
  • stage0  D6 diagnosis (zero-shot baseline)
  • stage2  train_grpo val hook (per-N-step training validation)
  • stage3  pointwise_rm_inference + eval_on_vgrb (deployment-time)

What this module owns
---------------------
* Test-set discovery: the per-dim ``test_pointwise_<dim>_v2.json``
  hold-out (with ``label_score`` ∈ [1, 5]) — separate from the
  ``v2_prompt_pair.json`` Stage ① teacher-iteration training set.
* Local-cache resolution + opt-in URL prefetch (mirrors stage0
  build_pair_data.py conventions).
* Pure-Python PLCC + SRCC + MAE (no scipy dep).
* A driver function ``run_pointwise_eval`` that takes a ``score_fn``
  callable and walks the test set, returning per-dim heatmap +
  overall summary in the same shape stage0 D6 already emits.

Why a callable score_fn (instead of a StudentWrapper)
-----------------------------------------------------
stage0 wraps a ``QwenVLJudge`` (no LoRA, no DimensionContext);
stage2 wraps a ``StudentWrapper`` (LoRA-on, with DimensionContext);
stage3 might wrap something else (vLLM endpoint, distilled checkpoint).
By taking a plain ``score_fn(dim, caption, video_path) -> float`` we
keep this module free of model-specific imports and let each caller
adapt its own forward path.

Schema returned by ``run_pointwise_eval``
-----------------------------------------
::

    {
      "overall": {"plcc":..., "srcc":..., "mae":..., "n":..., "S_mean":..., "label_mean":...},
      "heatmap": {"<dim>": {... same fields, plus n_total, n_skipped}},
      "ranked":  ["best_dim_by_plcc", ..., "worst_dim_by_plcc"],
      "skipped": {"no_test_file": [...], "no_local_video": {...}, "score_failed": {...}}
    }
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Callable

__all__ = [
    "DIMENSIONS_18",
    "EXCLUDED_DIMENSIONS",
    "ACTIVE_DIMENSIONS",
    "pearson",
    "spearman",
    "summarize_label_corr",
    "resolve_test_pointwise_path",
    "resolve_local_video",
    "prefetch_url",
    "prefetch_pointwise_videos",
    "run_pointwise_eval",
]


# ---------------------------------------------------------------------------
# 0. Canonical 18-axis taxonomy + active subset.
#
# DIMENSIONS_18: the full EvalVerse-second-level taxonomy. Kept as a frozen
#                reference so paper Table 1 row counts stay reproducible.
# EXCLUDED_DIMENSIONS: dims we deliberately skip in stage2 training and
#                eval. Currently:
#   • soundscape — Qwen2.5-VL is vision-only and cannot hear audio. Stage0
#                D6 confirmed structural failure (PLCC = -0.04, S stdev
#                0.34 with no signal in the rubric). No amount of rubric
#                rewriting or BT training will recover audio-grounded
#                judgement on a vision-only backbone; we'd need a
#                video+audio model (Qwen-Audio-Omni / GPT-4o-mini-audio).
#   • vocal     — Same audio-modality reason as soundscape: vocal scoring
#                requires hearing dialogue/voice-over delivery, which a
#                vision-only backbone cannot ground.
# ACTIVE_DIMENSIONS: DIMENSIONS_18 minus excluded — what stage2 actually
#                trains and evaluates on (currently 16 dims).
#
# Mirror of stage0's DIMENSIONS but lifted here so callers don't need to
# import stage0_diagnosis (which imports transformers / torch and is heavy
# on cold start).
# ---------------------------------------------------------------------------
DIMENSIONS_18 = [
    "visual_quality", "consistency", "grounding", "character", "action",
    "lens", "lighting", "composition", "scene", "chromaticity",
    "expression", "materiality", "logic", "pacing", "progression",
    "rhythm", "soundscape", "vocal",
]

EXCLUDED_DIMENSIONS = frozenset({"soundscape", "vocal"})

ACTIVE_DIMENSIONS = [d for d in DIMENSIONS_18 if d not in EXCLUDED_DIMENSIONS]


# ---------------------------------------------------------------------------
# 1. Pure-Python correlation primitives (verified against scipy in stage0).
# ---------------------------------------------------------------------------
def pearson(xs: list[float], ys: list[float]) -> float:
    """Pearson ρ. Returns 0.0 on degenerate input (n<2, length mismatch,
    or zero variance on either side)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    denom = (vx * vy) ** 0.5
    return cov / denom if denom > 0 else 0.0


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman ρ via Pearson on rank-transformed values. Handles ties
    with average-rank, so a constant input degenerates to 0 (not NaN)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return 0.0
    def _avg_rank(vs: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vs[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks
    return pearson(_avg_rank(xs), _avg_rank(ys))


def summarize_label_corr(records: list[dict]) -> dict:
    """records: list of ``{"S": float, "label": float, ...}`` → metrics dict."""
    n = len(records)
    if n == 0:
        return {"n": 0, "plcc": 0.0, "srcc": 0.0, "mae": 0.0,
                "S_mean": 0.0, "label_mean": 0.0}
    xs = [float(r["S"])     for r in records]
    ys = [float(r["label"]) for r in records]
    return {
        "n":          n,
        "plcc":       round(pearson(xs, ys), 4),
        "srcc":       round(spearman(xs, ys), 4),
        "mae":        round(sum(abs(x - y) for x, y in zip(xs, ys)) / n, 4),
        "S_mean":     round(sum(xs) / n, 4),
        "label_mean": round(sum(ys) / n, 4),
    }


# ---------------------------------------------------------------------------
# 2. Test-set discovery + local-video resolution.
# ---------------------------------------------------------------------------
def resolve_test_pointwise_path(dim: str, repo_root: Path | str) -> Path | None:
    """Pick the right test_pointwise file for ``dim``.

    Priority:
      1. ``data/v_arr_mvp/<dim>_mvp/test_pointwise_<dim>_v2.json``
      2. ``data/v_arr_mvp/<dim>_mvp/test_pointwise_<dim>.json``
      3. ``data/v_arr_mvp/visual_quality_mvp/shared_test_pointwise.json``
         (only the visual_quality dim falls through to the shared file)

    Returns None when nothing exists; caller decides whether to skip
    or warn.
    """
    root = Path(repo_root)
    d = root / "data" / "v_arr_mvp" / f"{dim}_mvp"
    cands = [
        d / f"test_pointwise_{dim}_v2.json",
        d / f"test_pointwise_{dim}.json",
    ]
    if dim == "visual_quality":
        cands.append(d / "shared_test_pointwise.json")
    for fp in cands:
        if fp.is_file():
            return fp
    return None


def resolve_local_video(url_or_path: str,
                        baseline_cache: Path,
                        prefetch_cache: Path) -> str | None:
    """Walk both v_arr_mvp cache conventions to find the local mp4.

    Mirrors ``stage0_diagnosis.build_pair_data.resolve_local`` so any
    video stored either by ``baseline_videoscore.py`` (sha256[:32].mp4)
    or by ``prefetch_videos.py`` (sha1[:10]_basename.mp4) gets picked up.

    Returns None when neither convention matches a file on disk.
    """
    if not isinstance(url_or_path, str):
        return None
    if not url_or_path.startswith(("http://", "https://")):
        return url_or_path if Path(url_or_path).is_file() else None
    p_a = baseline_cache / (hashlib.sha256(url_or_path.encode()).hexdigest()[:32] + ".mp4")
    if p_a.is_file() and p_a.stat().st_size > 0:
        return str(p_a)
    parsed = urllib.parse.urlparse(url_or_path)
    base = os.path.basename(parsed.path) or "video.mp4"
    if not base.lower().endswith(".mp4"):
        base += ".mp4"
    digest = hashlib.sha1(url_or_path.encode()).hexdigest()[:10]
    p_b = prefetch_cache / f"{digest}_{base}"
    if p_b.is_file() and p_b.stat().st_size > 0:
        return str(p_b)
    return None


def prefetch_url(url: str, dst: Path, *, timeout: int = 120) -> bool:
    """Stream a URL to ``dst``. Idempotent on already-non-empty files."""
    if dst.is_file() and dst.stat().st_size > 0:
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


def _prefetch_dst_for(url: str, prefetch_cache: Path) -> Path:
    """Map a URL to its prefetch_cache filename (sha1[:10]_basename.mp4)."""
    parsed = urllib.parse.urlparse(url)
    base = os.path.basename(parsed.path) or "video.mp4"
    if not base.lower().endswith(".mp4"):
        base += ".mp4"
    digest = hashlib.sha1(url.encode()).hexdigest()[:10]
    return prefetch_cache / f"{digest}_{base}"


def prefetch_pointwise_videos(
    repo_root:           Path | str,
    *,
    dimensions:          list[str] | None = None,
    max_samples_per_dim: int | None = None,
    log:                 Callable[[str], None] | None = None,
) -> dict:
    """Single-process prefetch of every URL the val sweep will touch.

    Why this exists
    ---------------
    ``run_pointwise_eval`` is invoked by EVERY rank under ZeRO-3 (all ranks
    must participate in the ``_summon_full_params_ctx`` collective inside
    ``score_fn``). When ``prefetch_missing=True`` was on, every rank
    independently raced on the SAME ``<sha1>_<basename>.mp4.part`` temp
    file: rank A would urlopen-write-rename, rank B would then fail its
    own rename with ``[Errno 2] .part not found``, and the resulting
    "rank A sees the file, rank B doesn't" inconsistency made the per-rank
    ``score_fn`` call counts diverge → the next ``_summon_full_params``
    all_gather deadlocked until the NCCL watchdog timed out.

    Solution: have rank-0 prefetch all missing videos BEFORE the val loop,
    barrier the world, then run ``run_pointwise_eval(prefetch_missing=False)``.
    Every rank now resolves the same file set deterministically.

    Returns a tally dict (mostly for logging) with counts of cache-hit /
    fetched / failed URLs, keyed by dim and overall.
    """
    root = Path(repo_root)
    dims = list(dimensions) if dimensions else list(ACTIVE_DIMENSIONS)
    baseline_cache = root / "runs" / "baselines" / "_video_cache"
    prefetch_cache = root / "data" / "v_arr_mvp" / "cache_videos"
    _log = log or (lambda s: print(s, file=sys.stderr))

    tally = {"hit": 0, "fetched": 0, "failed": 0, "skipped": 0,
             "by_dim": defaultdict(lambda: {"hit": 0, "fetched": 0, "failed": 0})}

    for dim in dims:
        fp = resolve_test_pointwise_path(dim, root)
        if fp is None:
            continue
        try:
            rows = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            _log(f"[prefetch-pw] {fp.name} unreadable: {e}")
            continue
        if max_samples_per_dim is not None and max_samples_per_dim > 0:
            rows = rows[: int(max_samples_per_dim)]
        for r in rows:
            url = r.get("response")
            if not isinstance(url, str):
                tally["skipped"] += 1
                continue
            local = resolve_local_video(url, baseline_cache, prefetch_cache)
            if local is not None:
                tally["hit"] += 1
                tally["by_dim"][dim]["hit"] += 1
                continue
            if not url.startswith(("http://", "https://")):
                # Local path that doesn't exist — nothing we can fetch.
                tally["failed"] += 1
                tally["by_dim"][dim]["failed"] += 1
                continue
            dst = _prefetch_dst_for(url, prefetch_cache)
            if prefetch_url(url, dst):
                tally["fetched"] += 1
                tally["by_dim"][dim]["fetched"] += 1
            else:
                tally["failed"] += 1
                tally["by_dim"][dim]["failed"] += 1

    _log(
        f"[prefetch-pw] hit={tally['hit']} fetched={tally['fetched']} "
        f"failed={tally['failed']} skipped={tally['skipped']}"
    )
    return tally


# ---------------------------------------------------------------------------
# 3. The evaluator. ``score_fn`` must accept (dim, caption, video_path)
#    and return a finite float in [1.0, 5.0]. Any exception inside score_fn
#    is caught and counted under ``skipped.score_failed``.
# ---------------------------------------------------------------------------
def run_pointwise_eval(
    score_fn:        Callable[[str, str, str], float],
    repo_root:       Path | str,
    *,
    dimensions:      list[str] | None = None,
    max_samples_per_dim: int | None = None,
    prefetch_missing: bool = True,
    progress_every:  int = 20,
    log:             Callable[[str], None] | None = None,
    include_records: bool = False,
) -> dict:
    """Run the EvalVerse hold-out PLCC/SRCC sweep over (up to) 18 dims.

    Args
    ----
    score_fn
        ``(dim: str, caption: str, video_path: str) -> float``. The
        returned float is the RM's predicted score on Likert 1-5.
    repo_root
        Path to the RewardVerse repo root (where ``data/v_arr_mvp/`` and
        ``runs/baselines/_video_cache/`` live).
    dimensions
        List of dim slugs to evaluate. None means "all 18 in the canonical
        order". For a fast val hook, pass a sub-list.
    max_samples_per_dim
        Cap per dim. None = take every row in the test file. For a fast
        val hook during training, set to e.g. 5 (× 18 dims = 90 forwards
        ≈ 5-10 min instead of the full ~3 h).
    prefetch_missing
        Download URL-only videos into the prefetch cache before scoring.
        Disable for offline runs.
    progress_every
        Emit a "[N videos] running plcc=…" line every N videos. 0 to silence.
    log
        Optional ``str -> None`` for progress messages. Defaults to
        ``print(…, file=sys.stderr)``.

    Returns
    -------
    dict matching stage0 D6's ``report["D6"]`` shape (overall + heatmap +
    ranked + skipped breakdown).
    """
    root = Path(repo_root)
    dims = list(dimensions) if dimensions else list(ACTIVE_DIMENSIONS)
    baseline_cache = root / "runs" / "baselines" / "_video_cache"
    prefetch_cache = root / "data" / "v_arr_mvp" / "cache_videos"
    _log = log or (lambda s: print(s, file=sys.stderr))

    by_dim:           dict[str, list[dict]] = defaultdict(list)
    skip_no_test:     dict[str, int] = {}
    skip_no_video:    dict[str, int] = defaultdict(int)
    skip_score_fail:  dict[str, int] = defaultdict(int)
    n_total = 0

    for dim in dims:
        fp = resolve_test_pointwise_path(dim, root)
        if fp is None:
            skip_no_test[dim] = 1
            continue

        try:
            rows = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            _log(f"[eval-pw] {fp.name} unreadable: {e}")
            skip_no_test[dim] = 1
            continue
        if max_samples_per_dim is not None and max_samples_per_dim > 0:
            rows = rows[: int(max_samples_per_dim)]

        for r in rows:
            url   = r.get("response")
            label = r.get("label_score")
            if not isinstance(url, str) or label is None:
                continue

            local = resolve_local_video(url, baseline_cache, prefetch_cache)
            if local is None and prefetch_missing and url.startswith(("http://", "https://")):
                dst = _prefetch_dst_for(url, prefetch_cache)
                if prefetch_url(url, dst):
                    local = str(dst)
            if local is None:
                skip_no_video[dim] += 1
                continue

            caption = r.get("query") or r.get("metadata", {}).get("prompt_zh") or ""
            try:
                S = float(score_fn(dim, caption, local))
            except Exception as e:
                _log(f"[eval-pw] score_fn failed on {r.get('id')}: {e}")
                skip_score_fail[dim] += 1
                continue
            # Guard against NaN / inf from a misbehaving score_fn.
            if not (S == S and -1e6 < S < 1e6):
                skip_score_fail[dim] += 1
                continue

            by_dim[dim].append({
                "id":    r.get("id"),
                "dim":   dim,
                "S":     S,
                "label": float(label),
            })
            n_total += 1
            if progress_every and n_total % progress_every == 0:
                flat = [rec for lst in by_dim.values() for rec in lst]
                stat = summarize_label_corr(flat)
                _log(f"[eval-pw] [{n_total}] running plcc={stat['plcc']:+.3f} "
                     f"srcc={stat['srcc']:+.3f} mae={stat['mae']:.3f}")

    # Heatmap in canonical dim order; absent dims simply skipped.
    heatmap: dict[str, dict] = {}
    for dim in dims:
        if dim in by_dim:
            heatmap[dim] = summarize_label_corr(by_dim[dim])
            heatmap[dim]["n_skipped_video"] = skip_no_video.get(dim, 0)
            heatmap[dim]["n_skipped_score"] = skip_score_fail.get(dim, 0)

    flat_all = [rec for lst in by_dim.values() for rec in lst]
    overall  = summarize_label_corr(flat_all)
    ranked   = sorted(heatmap.items(), key=lambda kv: kv[1]["plcc"], reverse=True)

    out = {
        "overall":  overall,
        "heatmap":  heatmap,
        "ranked":   [d for d, _ in ranked],
        "skipped":  {
            "no_test_file":   list(skip_no_test.keys()),
            "no_local_video": dict(skip_no_video),
            "score_failed":   dict(skip_score_fail),
        },
    }
    if include_records:
        # Raw (S, label, dim, id) tuples per dim, for callers that need
        # to re-pool across shards (e.g. multi-GPU eval). Pearson/Spearman
        # are NOT linearly aggregatable, so any shard-merge has to
        # recompute from these rows.
        out["records"] = {dim: list(by_dim[dim]) for dim in by_dim}
    return out
