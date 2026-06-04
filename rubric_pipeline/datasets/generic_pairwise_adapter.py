# -*- coding: utf-8 -*-
"""Generic pairwise preference adapter for RewardVerse MVP.

Accepts a variety of upstream pairwise preference schemas and emits the
Auto-Rubric canonical sample:

    {
      "query": "<prompt>",
      "response": ["<path_to_winner>", "<path_to_loser>"],
      "label_rank": [1, 2],
      "task_type": "t2v",
      "metadata": {...}
    }

Recognized upstream shapes (best-effort, in priority order):
  1. Already canonical ARR pairwise (response + label_rank).
  2. Explicit winner/loser fields: win_video / lose_video,
     preferred / rejected, chosen / rejected.
  3. Multi-candidate with winner index (candidates + winner).
  4. Scored candidates per prompt (TIGER-Lab/VideoFeedback-style): expanded by
     pairing the highest-scored and lowest-scored candidate within the same
     prompt, gated by `min_score_gap`.

This adapter intentionally does NOT depend on EvalVerse fields, so it is
independent of `evalverse_adapter.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping


# ---------------------------- Field aliases ---------------------------------

_QUERY_KEYS = (
    "query",
    "prompt",
    "caption",
    "text",
    "instruction",
)

_PATH_KEYS = ("path", "video_path", "video", "file", "url", "video_url")

_WIN_KEYS = (
    "win_video",
    "preferred",
    "chosen",
    "winner_video",
    "Vw",
    "video_w",
    "positive_video",
)

_LOSE_KEYS = (
    "lose_video",
    "rejected",
    "loser_video",
    "Vl",
    "video_l",
    "negative_video",
)

_RESPONSE_LIST_KEYS = (
    "response",
    "responses",
    "videos",
    "video_paths",
    "candidates",
)


# ---------------------------- Helpers ---------------------------------------


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, (str, list, tuple, dict, Mapping)) and len(value) == 0:
            continue
        return value
    return None


def _resolve_path(record: Mapping[str, Any], item: Any) -> str | None:
    if item is None:
        return None
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        for key in _PATH_KEYS:
            value = item.get(key)
            if value:
                return str(value)
        video_id = _first_non_empty(item.get("video_id"), item.get("id"))
        root = _first_non_empty(record.get("video_root"), record.get("root"))
        ext = item.get("ext") or "mp4"
        if video_id and root:
            return str(Path(root) / f"{video_id}.{ext}")
    return None


def _query_of(record: Mapping[str, Any]) -> str:
    return str(_first_non_empty(*(record.get(k) for k in _QUERY_KEYS)) or "")


def _metadata_of(record: Mapping[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    for key in ("id", "sample_id", "uid"):
        value = record.get(key)
        if value and "id" not in metadata:
            metadata["id"] = str(value)
    for key in ("split", "source", "generator_a", "generator_b"):
        value = record.get(key)
        if value:
            metadata.setdefault(key, value)
    if extra:
        metadata.update(extra)
    return metadata


def _as_canonical_pairwise(
    query: str,
    win: str,
    lose: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "query": query,
        "response": [win, lose],
        "label_rank": [1, 2],
        "task_type": "t2v",
        "metadata": metadata,
    }


# ---------------------------- Conversion -------------------------------------


def convert_generic_pairwise_record(
    record: Mapping[str, Any],
    min_score_gap: float = 1.0,
) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Convert one upstream record to canonical ARR pairwise sample(s)."""
    query = _query_of(record)
    metadata = _metadata_of(record)

    # 1. Already-canonical pairwise (response list + label_rank).
    response = record.get("response") or record.get("responses")
    label_rank = record.get("label_rank")
    if isinstance(response, list) and len(response) >= 2 and label_rank:
        first_two = [str(p) for p in response[:2]]
        try:
            rank_pair = [int(label_rank[0]), int(label_rank[1])]
        except (TypeError, ValueError, IndexError):
            rank_pair = [1, 2]
        if rank_pair == [1, 2]:
            win, lose = first_two[0], first_two[1]
        elif rank_pair == [2, 1]:
            win, lose = first_two[1], first_two[0]
        else:
            return None
        return _as_canonical_pairwise(query, win, lose, metadata)

    # 2. Explicit winner / loser fields.
    win_raw = _first_non_empty(*(record.get(k) for k in _WIN_KEYS))
    lose_raw = _first_non_empty(*(record.get(k) for k in _LOSE_KEYS))
    if win_raw and lose_raw:
        win = _resolve_path(record, win_raw)
        lose = _resolve_path(record, lose_raw)
        if win and lose:
            return _as_canonical_pairwise(query, win, lose, metadata)

    # 3. Multi-candidate + winner index.
    candidates = record.get("candidates") or record.get("videos")
    winner_idx = record.get("winner")
    if isinstance(candidates, list) and len(candidates) >= 2 and winner_idx is not None:
        try:
            winner_idx = int(winner_idx)
        except (TypeError, ValueError):
            return None
        if 0 <= winner_idx < len(candidates):
            loser_idx = 1 - winner_idx if len(candidates) == 2 else next(
                (i for i in range(len(candidates)) if i != winner_idx),
                None,
            )
            if loser_idx is None:
                return None
            win = _resolve_path(record, candidates[winner_idx])
            lose = _resolve_path(record, candidates[loser_idx])
            if win and lose:
                return _as_canonical_pairwise(query, win, lose, metadata)

    # 4. Scored per-candidate (VideoFeedback-style). Score keys vary:
    #    - "score": overall, or
    #    - "scores": dict, fallback to max of values, or
    #    - "rating": single number.
    if isinstance(candidates, list) and len(candidates) >= 2:
        scored: list[tuple[float, str]] = []
        for c in candidates:
            if not isinstance(c, Mapping):
                continue
            score = c.get("score")
            if score is None and isinstance(c.get("scores"), Mapping):
                vals = [float(v) for v in c["scores"].values() if isinstance(v, (int, float))]
                score = sum(vals) / len(vals) if vals else None
            if score is None:
                score = c.get("rating")
            if score is None:
                continue
            path = _resolve_path(record, c)
            if not path:
                continue
            scored.append((float(score), path))
        if len(scored) >= 2:
            scored.sort(key=lambda x: x[0])
            best_score, best_path = scored[-1]
            worst_score, worst_path = scored[0]
            if best_score - worst_score >= float(min_score_gap):
                meta = {**metadata, "score_gap": best_score - worst_score}
                return _as_canonical_pairwise(query, best_path, worst_path, meta)

    return None


def load_generic_pairwise_dataset(
    path: str | Path,
    min_score_gap: float = 1.0,
) -> list[dict[str, Any]]:
    """Load JSON / JSONL and convert to canonical ARR pairwise samples."""
    path = Path(path)
    if path.suffix == ".jsonl":
        raw = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        raw = loaded if isinstance(loaded, list) else [loaded]

    samples: list[dict[str, Any]] = []
    for record in raw:
        converted = convert_generic_pairwise_record(record, min_score_gap=min_score_gap)
        if converted is None:
            continue
        if isinstance(converted, list):
            samples.extend(converted)
        else:
            samples.append(converted)
    return samples


def save_arr_dataset(samples: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    """Save canonical ARR samples as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = list(samples)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


__all__ = [
    "convert_generic_pairwise_record",
    "load_generic_pairwise_dataset",
    "save_arr_dataset",
]
