"""RewardVerse — local-asset directory layout.

All baseline experiments share two on-disk caches that live OUTSIDE this
repo so they survive ``git clean`` and can be reused across sister projects:

  * ``../../models/<short_name>/``   — HuggingFace model snapshots
  * ``../../datasets/<short_name>/`` — HuggingFace dataset snapshots / dumps

This module centralises four things so the 4 baseline / fetcher scripts
don't repeat themselves:

  1. The canonical resolver for ``<repo_id>`` -> on-disk directory path
     (``resolve_local_model_dir`` / ``resolve_local_dataset_dir``).
  2. A short-name registry that gives well-known assets (VideoScore-v1.1,
     VideoFeedback, ...) stable directory names regardless of HF repo
     renames.
  3. Idempotent ``ensure_*_local`` helpers that ``snapshot_download``
     into the target directory if it doesn't already contain a real
     model / dataset (and otherwise no-op).
  4. A ``--no_download`` opt-out so air-gapped environments fail fast
     when the directory is empty rather than trying to hit HuggingFace.

Why a registry instead of just hashing the repo id?
  Vendors rename HF repos sometimes (KwaiVGI -> KlingAIResearch is a
  recent example for VideoGen-RewardBench). With a curated short-name
  alias we keep the on-disk directory stable, and only need to update
  this one file when a vendor renames their repo.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Roots — relative to the RewardVerse project root.
#
#   _REPO_ROOT      = /group/<acct>/projects/RewardVerse   (this repo)
#   ../..           = /group/<acct>/                       (two levels up)
#
# So ``../../models/`` and ``../../datasets/`` are sister directories of
# ``projects/``, NOT subdirectories of it. This matches the user's
# requested layout (``../../models/`` / ``../../datasets/`` from RewardVerse)
# and lines up with the cluster's group-shared cache convention.
# ---------------------------------------------------------------------------
_REPO_ROOT      = Path(__file__).resolve().parents[2]
_GROUP_ROOT     = _REPO_ROOT.parent.parent  # two-up from the repo root
MODELS_ROOT     = _GROUP_ROOT / "models"
DATASETS_ROOT   = _GROUP_ROOT / "datasets"


# ---------------------------------------------------------------------------
# Registry — well-known repo_id -> short_name.
# ---------------------------------------------------------------------------

# Models (used by the local-VLM baselines).
MODEL_ALIASES: dict[str, str] = {
    # VideoScore family.
    "TIGER-Lab/VideoScore-v1.1":   "VideoScore-v1.1",
    "TIGER-Lab/VideoScore":        "VideoScore",          # v1 fallback
    "TIGER-AI-Lab/VideoScore2":    "VideoScore2",
    "TIGER-Lab/VideoScore2":       "VideoScore2",          # alt namespace
    # UnifiedReward.
    "CodeGoat24/UnifiedReward-7B": "UnifiedReward-7B",
    "CodeGoat24/UnifiedReward":    "UnifiedReward",
}

# Datasets (used by fetch_external_benchmarks.py).
DATASET_ALIASES: dict[str, str] = {
    "TIGER-Lab/VideoFeedback":              "VideoFeedback",
    "KlingAIResearch/VideoGen-RewardBench": "VideoGen-RewardBench",
    "KwaiVGI/VideoGen-RewardBench":         "VideoGen-RewardBench",
}


def _short_name_from_repo(repo_id: str, table: dict[str, str]) -> str:
    """``Vendor/Name`` -> ``Name`` (use registry override if any)."""
    if repo_id in table:
        return table[repo_id]
    return repo_id.rsplit("/", 1)[-1]


# ---------------------------------------------------------------------------
# Resolvers — pure path math, no I/O.
# ---------------------------------------------------------------------------
def resolve_local_model_dir(repo_id: str) -> Path:
    """Where ``repo_id``'s snapshot SHOULD live on disk.

    Does not check whether the directory exists or has any contents; for
    that, see ``ensure_model_local`` below.
    """
    return MODELS_ROOT / _short_name_from_repo(repo_id, MODEL_ALIASES)


def resolve_local_dataset_dir(repo_id: str) -> Path:
    """Where ``repo_id``'s snapshot SHOULD live on disk."""
    return DATASETS_ROOT / _short_name_from_repo(repo_id, DATASET_ALIASES)


# ---------------------------------------------------------------------------
# I/O helpers.
# ---------------------------------------------------------------------------
def _dir_has_model(p: Path) -> bool:
    """Heuristic: a directory is "ready" if it contains at least ONE of the
    typical HF model files (config.json, *.safetensors, *.bin)."""
    if not p.is_dir():
        return False
    for f in p.rglob("*"):
        if f.name == "config.json" or f.suffix in {".safetensors", ".bin"}:
            return True
    return False


def _dir_has_dataset(p: Path) -> bool:
    """Heuristic: a dataset directory is "ready" if it contains at least ONE
    parquet / arrow / json / jsonl file."""
    if not p.is_dir():
        return False
    for f in p.rglob("*"):
        if f.suffix in {".parquet", ".arrow", ".json", ".jsonl"}:
            return True
    return False


def ensure_model_local(repo_id: str, *,
                       allow_download: bool = True,
                       revision: str | None = None,
                       allow_patterns: list[str] | None = None,
                       ignore_patterns: list[str] | None = None) -> Path:
    """Resolve, and (idempotently) populate, the local model directory.

    Idempotent contract:
      1. If the local dir already passes ``_dir_has_model``, return its
         path (no download attempt).
      2. Else, if ``allow_download`` is True, ``snapshot_download`` the
         repo into the local dir; return its path.
      3. Else, raise ``FileNotFoundError`` so air-gapped runs fail fast
         instead of silently re-downloading every time.

    The download uses ``local_dir`` (NOT the symlink-style cache) so the
    directory is self-contained and movable across machines.
    """
    target = resolve_local_model_dir(repo_id)
    if _dir_has_model(target):
        return target

    if not allow_download:
        raise FileNotFoundError(
            f"local model dir is empty / incomplete: {target}\n"
            f"  expected to find a snapshot of {repo_id!r} there.\n"
            "  Pass --no_download=False (or run with HF reachable) to "
            "auto-download, or pre-populate the directory manually."
        )

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Auto-download requires `huggingface_hub`. Install via "
            "`pip install huggingface_hub`, or pre-populate "
            f"{target} manually."
        ) from e

    target.mkdir(parents=True, exist_ok=True)
    print(f"[local_assets] downloading {repo_id!r} -> {target} ...",
          flush=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    return target


def ensure_dataset_local(repo_id: str, *,
                         allow_download: bool = True,
                         revision: str | None = None,
                         allow_patterns: list[str] | None = None,
                         ignore_patterns: list[str] | None = None) -> Path:
    """Same contract as ``ensure_model_local`` but for HF datasets."""
    target = resolve_local_dataset_dir(repo_id)
    if _dir_has_dataset(target):
        return target

    if not allow_download:
        raise FileNotFoundError(
            f"local dataset dir is empty / incomplete: {target}\n"
            f"  expected to find a snapshot of {repo_id!r} there.\n"
            "  Re-run with downloads enabled, or pre-populate the dir."
        )

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as e:
        raise ImportError(
            "Auto-download requires `huggingface_hub`. Install via "
            "`pip install huggingface_hub`, or pre-populate "
            f"{target} manually."
        ) from e

    target.mkdir(parents=True, exist_ok=True)
    print(f"[local_assets] downloading dataset {repo_id!r} -> {target} ...",
          flush=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(target),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    return target


# ---------------------------------------------------------------------------
# CLI smoke-test.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Repo root     :", _REPO_ROOT)
    print("Group root    :", _GROUP_ROOT)
    print("Models root   :", MODELS_ROOT)
    print("Datasets root :", DATASETS_ROOT)
    print()
    print("Model directory layout:")
    for repo, short in MODEL_ALIASES.items():
        p = MODELS_ROOT / short
        ready = "READY  " if _dir_has_model(p) else "empty  "
        print(f"  {ready} {repo:<35} -> {p}")
    print()
    print("Dataset directory layout:")
    for repo, short in DATASET_ALIASES.items():
        p = DATASETS_ROOT / short
        ready = "READY  " if _dir_has_dataset(p) else "empty  "
        print(f"  {ready} {repo:<40} -> {p}")
