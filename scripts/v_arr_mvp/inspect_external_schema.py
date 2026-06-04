#!/usr/bin/env python3
"""RewardVerse — external benchmark schema inspector (DRYRUN, no sampling).

Purpose
-------
Before we burn cycles on a 200-sample stratified fetch, we need to KNOW
the real shape of each HF row — particularly:

  * What does the ``video`` / ``video_a`` / ``video_b`` field contain?
    A URL string? A local-cache path? A ``datasets.Video`` object that
    decodes lazily? A bytes blob?
  * Which score / label keys actually exist on this build of the
    dataset (HF datasets get re-versioned without notice)?

This script
  1. Pulls the FIRST ``--n`` rows from a target HF dataset (default 5).
  2. Prints, per row, every top-level key with its
     (type, length, repr-snippet).
  3. Dumps the raw rows to ``<out_dir>/<short>_dryrun_rows.jsonl`` so
     the user can grep further.
  4. Special-cases ``video*`` fields: prints whether each is
     str / dict / bytes / a non-string object with a ``"path"``
     attribute (the typical signature of a ``datasets.Video`` lazy
     decoder).

Run examples
------------
    # VideoGen-RewardBench inspect
    python scripts/v_arr_mvp/inspect_external_schema.py \\
        --target videogen_rewardbench \\
        --out_dir /workspace/user_code/40063/jacuzzitang/datasets/_v_arr_external_subsets/_dryrun
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_assets import (  # type: ignore  # noqa: E402
    ensure_dataset_local,
    resolve_local_dataset_dir,
)


# ---------------------------------------------------------------------------
# Target registry.
# ---------------------------------------------------------------------------
TARGETS: dict[str, dict[str, Any]] = {
    "videogen_rewardbench": {
        # Try both vendor namespaces; whichever resolves first wins.
        "repos": [
            "KlingAIResearch/VideoGen-RewardBench",
            "KwaiVGI/VideoGen-RewardBench",
        ],
        # VGRB has historically been a single default config; leave None
        # and rely on HF auto-pick. If it ever splits into multiple
        # configs, set this string explicitly.
        "config":      None,
        # NOTE: VGRB names its only split 'eval' (not 'test' / 'train').
        "split":       "eval",
        "video_keys":  (
            "video_a", "video_b",
            "preferred_video", "rejected_video",
            "chosen_video", "win_video", "winner_video",
            "lose_video", "loser_video",
        ),
    },
}


# ---------------------------------------------------------------------------
# Field shape diagnostics.
# ---------------------------------------------------------------------------
def _short_repr(v: Any, n: int = 120) -> str:
    """Return a single-line, length-capped repr for display."""
    try:
        s = repr(v)
    except Exception as e:  # pragma: no cover
        s = f"<unrepresentable: {e}>"
    s = s.replace("\n", " ")
    if len(s) > n:
        s = s[:n] + f"...<+{len(s) - n} chars>"
    return s


def _shape_of(v: Any) -> str:
    """Compact shape descriptor (type + size hint)."""
    if v is None:
        return "None"
    t = type(v).__name__
    if isinstance(v, (list, tuple)):
        return f"{t}(len={len(v)})"
    if isinstance(v, dict):
        return f"{t}(keys={sorted(v.keys())[:6]}{'...' if len(v) > 6 else ''})"
    if isinstance(v, (bytes, bytearray)):
        return f"{t}(nbytes={len(v)})"
    if isinstance(v, str):
        return f"{t}(len={len(v)})"
    # Could be a datasets.Video lazy decoder, an np.ndarray, etc.
    extra = []
    if hasattr(v, "path"):
        try:
            extra.append(f"path={v.path!r}")
        except Exception:
            pass
    if hasattr(v, "shape"):
        try:
            extra.append(f"shape={v.shape}")
        except Exception:
            pass
    if hasattr(v, "dtype"):
        try:
            extra.append(f"dtype={v.dtype}")
        except Exception:
            pass
    return f"{t}({', '.join(extra)})" if extra else t


def _is_video_like_key(k: str) -> bool:
    return any(tok in k.lower() for tok in ("video", "url", "path"))


# ---------------------------------------------------------------------------
# HF loader (lifted from fetch_external_benchmarks.py but explicit).
# ---------------------------------------------------------------------------
def _try_load_one_repo(
    repo: str, *, split: str, n: int, allow_download: bool,
    config: str | None = None,
) -> list[dict] | None:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("[error] `datasets` library not available — pip install datasets",
              file=sys.stderr)
        return None

    # Snapshot to .../jacuzzitang/datasets/<short_name>/ if missing.
    local_dir = None
    try:
        local_dir = ensure_dataset_local(repo, allow_download=allow_download)
    except FileNotFoundError as exc:
        print(f"[warn] {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"[warn] failed to populate local dir for {repo!r}: {exc}",
              file=sys.stderr)

    # Build kwargs once so we can pass `name=config` only when set.
    # (HF treats name=None and "missing" differently for some loaders.)
    common_kwargs: dict[str, Any] = {"split": split, "streaming": True}
    if config:
        common_kwargs["name"] = config

    if local_dir is not None:
        # Strategy A: load directly from the snapshot directory. For
        # multi-config datasets this STILL requires `name=` because the
        # snapshot mirrors the HF layout (annotated/ and real/ as
        # sibling dirs of parquet shards).
        try:
            ds = load_dataset(
                str(local_dir), trust_remote_code=True, **common_kwargs,
            )
            print(f"[info] {repo}: loaded from local snapshot {local_dir} "
                  f"(config={config!r}, split={split!r})", flush=True)
            rows: list[dict] = []
            for i, row in enumerate(ds):
                if i >= n:
                    break
                rows.append(dict(row))
            return rows
        except Exception as exc:
            print(f"[warn] local-snapshot load failed for {repo}: {exc}",
                  file=sys.stderr)

        # Strategy B: VideoFeedback ships per-config parquet at
        #   <snap>/<config>/{train,test}-00000-of-00001.parquet
        # so when load_dataset(local_dir, name=...) confuses itself we
        # can fall back to pointing it directly at the parquet file.
        if config:
            cand = local_dir / config / f"{split}-00000-of-00001.parquet"
            if cand.exists():
                try:
                    ds = load_dataset(
                        "parquet",
                        data_files=str(cand),
                        split="train",      # parquet loader uses 'train'
                        streaming=True,
                    )
                    print(f"[info] {repo}: loaded direct parquet "
                          f"{cand}", flush=True)
                    rows = []
                    for i, row in enumerate(ds):
                        if i >= n:
                            break
                        rows.append(dict(row))
                    return rows
                except Exception as exc:
                    print(f"[warn] direct-parquet load failed for {cand}: "
                          f"{exc}", file=sys.stderr)

    # Last-resort streaming from HF.
    try:
        ds = load_dataset(repo, **common_kwargs)
    except Exception as exc:
        print(f"[warn] streaming {repo!r} failed: {exc}", file=sys.stderr)
        return None
    print(f"[info] {repo}: streamed directly from HuggingFace "
          f"(config={config!r}, split={split!r})", flush=True)
    rows = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        rows.append(dict(row))
    return rows


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------
def inspect_target(target: str, n: int, out_dir: Path,
                   allow_download: bool) -> int:
    if target not in TARGETS:
        print(f"[error] unknown target {target!r}; choose from "
              f"{list(TARGETS)}", file=sys.stderr)
        return 2
    cfg = TARGETS[target]
    rows: list[dict] | None = None
    chosen_repo: str | None = None
    for repo in cfg["repos"]:
        rows = _try_load_one_repo(
            repo,
            split=cfg["split"],
            n=n,
            allow_download=allow_download,
            config=cfg.get("config"),
        )
        if rows:
            chosen_repo = repo
            break
    if not rows:
        print(f"[error] could not fetch any rows for target={target!r}.",
              file=sys.stderr)
        return 3

    print()
    print("=" * 72)
    print(f" target  : {target}")
    print(f" repo    : {chosen_repo}")
    print(f" config  : {cfg.get('config')}")
    print(f" split   : {cfg['split']}")
    print(f" n_rows  : {len(rows)}")
    print(f" snapshot: {resolve_local_dataset_dir(chosen_repo or '')}")
    print("=" * 72)

    # 1) Per-row dump.
    for i, row in enumerate(rows):
        print()
        print(f"--- row[{i}] ---  total_keys={len(row)}")
        for k in sorted(row.keys()):
            v = row[k]
            tag = "  [V]" if _is_video_like_key(k) else "     "
            print(f"{tag} {k:<32s} :: {_shape_of(v):<40s} = "
                  f"{_short_repr(v)}")

    # 2) Persist raw rows for offline analysis.
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{target}_dryrun_rows.jsonl"

    def _to_jsonable(v: Any) -> Any:
        """Best-effort downgrade so json.dumps doesn't blow up on
        bytes / Video / np.ndarray. Lossy by design — this is for
        debugging, not for downstream eval."""
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        if isinstance(v, (bytes, bytearray)):
            return f"<bytes nbytes={len(v)}>"
        if isinstance(v, (list, tuple)):
            return [_to_jsonable(x) for x in v]
        if isinstance(v, dict):
            return {str(kk): _to_jsonable(vv) for kk, vv in v.items()}
        # datasets.Video lazy decoder commonly exposes .path
        if hasattr(v, "path"):
            try:
                return {"_lazy_type": type(v).__name__, "path": v.path}
            except Exception:
                pass
        try:
            return {"_lazy_type": type(v).__name__, "repr": _short_repr(v, 200)}
        except Exception:
            return f"<{type(v).__name__} (unrepresentable)>"

    with out_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(
                {k: _to_jsonable(v) for k, v in row.items()},
                ensure_ascii=False,
            ) + "\n")
    print()
    print(f"[ok] wrote {len(rows)} dryrun rows to {out_file}")

    # 3) Quick verdict on video field shape.
    print()
    print("--- video-field verdict ---")
    for vk in cfg["video_keys"]:
        kinds: dict[str, int] = {}
        for row in rows:
            if vk in row:
                v = row[vk]
                kind = type(v).__name__
                if isinstance(v, str):
                    kind = "str(http)" if v.startswith("http") else "str(local?)"
                elif hasattr(v, "path"):
                    kind = f"{type(v).__name__}(.path)"
                kinds[kind] = kinds.get(kind, 0) + 1
        if kinds:
            print(f"  {vk:<24s} -> {kinds}")
        # else: silent — that key just doesn't exist in this dataset.

    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True,
                   choices=sorted(TARGETS.keys()))
    p.add_argument("--n", type=int, default=5,
                   help="how many head rows to inspect (default 5)")
    p.add_argument("--out_dir", required=True,
                   help="where to write the *_dryrun_rows.jsonl file.")
    p.add_argument("--no_download", action="store_true",
                   help="refuse to auto-download dataset files (air-gap).")
    args = p.parse_args()
    sys.exit(inspect_target(
        target=args.target,
        n=args.n,
        out_dir=Path(args.out_dir),
        allow_download=not args.no_download,
    ))


if __name__ == "__main__":
    main()
