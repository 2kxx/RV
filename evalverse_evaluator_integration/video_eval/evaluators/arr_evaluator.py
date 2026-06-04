# -*- coding: utf-8 -*-
"""Auto-Rubric (ARR) pointwise evaluator for the EvalVerseEvaluator framework.

This file is a drop-in evaluator: copy it into your EvalVerseEvaluator checkout
at `video_eval/evaluators/arr_evaluator.py`, then import it from
`video_eval/evaluators/__init__.py`, and register it in `configs/eval.yaml`.

Design overview
---------------
- Inherits from EvalVerseEvaluator's BaseEvaluator and is registered via
  EVALUATORS.register_module().
- Wraps an Auto-Rubric `Judger` from the ARR repo (2kxx/ARR / RewardVerse MVP):
    * loads a pre-baked rubric text file (recommended for benchmark runs),
    * routes the EvalVerseEvaluator `VideoData` into an ARR canonical sample,
    * calls the ARR VLM judge in pointwise mode (returns score 1-5),
    * normalises the score to [0, 5] and emits the framework's dict shape.
- Each EvalVerseEvaluator dimension is wired to its own rubric file; the same
  evaluator class is reused per dimension by changing `dimension_name`.
- `task_name` from `VideoData.metadata` selects the ARR task family
  (`t2v` vs `r2v`) and decides which reference asset to attach.

Configuration sources (priority order: high -> low)
---------------------------------------------------
1. Constructor kwargs: `arr_config_path`, `arr_rubrics_file`, `arr_rubrics_dir`,
   `arr_repo_root`, `score_range` ("1-5" or "0-5").
2. Environment variables: `ARR_CONFIG_PATH`, `ARR_RUBRICS_FILE`,
   `ARR_RUBRICS_DIR`, `ARR_REPO_ROOT`, `ARR_SCORE_RANGE`.
3. Convention: if `ARR_RUBRICS_DIR` is set, the rubric file is resolved as
   `<ARR_RUBRICS_DIR>/<dimension_name>.txt`.

Score normalisation
-------------------
ARR uses an integer 1..5 scale internally. EvalVerseEvaluator expects [0, 5]
with 5 = best. Default (`score_range="1-5"`) clamps ARR's raw score to [0, 5]
since 1..5 already fits. Set `score_range="0-5"` to apply
`(arr_score - 1) / 4 * 5` instead.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

try:
    # The framework provides these in EvalVerseEvaluator.
    from video_eval.core import BaseEvaluator, EVALUATORS  # type: ignore
    from video_eval.data.benchmark_loader import VideoData  # type: ignore

    _FRAMEWORK_AVAILABLE = True
except ImportError:
    # Fallback so this file can be imported (e.g. for lint / smoke tests) when
    # EvalVerseEvaluator is not on sys.path. The framework will import the file
    # against the real base classes when deployed.
    _FRAMEWORK_AVAILABLE = False

    class BaseEvaluator:  # type: ignore[no-redef]
        def __init__(self, dimension_name: str = "", model: Any = None, **kwargs: Any) -> None:
            self.dimension_name = dimension_name
            self.model = model
            self.kwargs = kwargs

    class _DummyRegistry:
        def register_module(self):  # type: ignore[no-untyped-def]
            def _decorator(cls):
                return cls

            return _decorator

    EVALUATORS = _DummyRegistry()  # type: ignore[assignment]
    VideoData = Any  # type: ignore[assignment,misc]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Asyncio helper: ARR's Judger is async; the framework may or may not run us
# inside an existing event loop. Run-anywhere wrapper.
# ---------------------------------------------------------------------------


def _run_coro(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Inside a running loop: hand off to a dedicated thread that owns its own loop.
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result()


# ---------------------------------------------------------------------------
# Task-name -> ARR canonical sample mapping.
# ---------------------------------------------------------------------------


_T2V_TASKS = {"t2v", "text_to_video", "text-to-video"}
_R2V_TASK_FIELDS: Dict[str, str] = {
    "subject_r2v": "subject_image",
    "scene_r2v": "scene_image",
    "object_r2v": "object_image",
}


def _build_arr_sample(video_data: Any) -> Dict[str, Any]:
    """Translate a framework `VideoData` to an ARR canonical pointwise sample.

    Returns a dict whose `task_type` field drives ARR's media routing
    (`t2v` vs `r2v`), and whose `metadata.task_name` is preserved for logs.
    """
    metadata = (getattr(video_data, "metadata", None) or {}).copy() if hasattr(video_data, "metadata") else {}
    task_name = str(metadata.get("task_name", "t2v")).lower()
    prompt = getattr(video_data, "prompt", None) or ""
    video_path = getattr(video_data, "video_path", None)

    sample: Dict[str, Any] = {
        "query": prompt,
        "response": video_path,
        "metadata": metadata,
    }

    ref_images = list(getattr(video_data, "ref_image_paths", []) or [])
    ref_video = getattr(video_data, "ref_video_path", None)
    gt_video = getattr(video_data, "gt_video_path", None)

    if task_name in _R2V_TASK_FIELDS and ref_images:
        sample["task_type"] = "r2v"
        sample[_R2V_TASK_FIELDS[task_name]] = ref_images[0]
    elif task_name == "camera_r2v" and ref_video:
        sample["task_type"] = "r2v"
        sample["ground_truth_video"] = ref_video
    elif gt_video:
        # If the upstream supplies a GT video on any task, use it as the
        # R2V reference channel.
        sample["task_type"] = "r2v"
        sample["ground_truth_video"] = gt_video
    elif task_name in _T2V_TASKS:
        sample["task_type"] = "t2v"
    else:
        sample["task_type"] = "t2v"
    return sample


# ---------------------------------------------------------------------------
# Configuration resolution helpers.
# ---------------------------------------------------------------------------


def _resolve_arr_repo_root(explicit: Optional[str]) -> Optional[str]:
    return explicit or os.environ.get("ARR_REPO_ROOT")


def _resolve_arr_config_path(explicit: Optional[str]) -> str:
    path = explicit or os.environ.get("ARR_CONFIG_PATH")
    if not path:
        raise ValueError(
            "ARRPointwiseEvaluator requires arr_config_path (kwarg) or "
            "ARR_CONFIG_PATH (env var).",
        )
    return path


def _resolve_rubrics_file(
    explicit_file: Optional[str],
    explicit_dir: Optional[str],
    dimension_name: str,
) -> Optional[str]:
    if explicit_file:
        return explicit_file
    env_file = os.environ.get("ARR_RUBRICS_FILE")
    if env_file:
        return env_file
    rubrics_dir = explicit_dir or os.environ.get("ARR_RUBRICS_DIR")
    if rubrics_dir:
        candidate = Path(rubrics_dir) / f"{dimension_name}.txt"
        if candidate.exists():
            return str(candidate)
    return None


# ---------------------------------------------------------------------------
# Lazy import of the ARR Judger. We do this inside the constructor so that
# users can drop this file into EvalVerseEvaluator and only pay the import
# cost when the evaluator is actually constructed.
# ---------------------------------------------------------------------------


def _import_arr_judger(arr_repo_root: Optional[str]):
    if arr_repo_root and arr_repo_root not in sys.path:
        sys.path.insert(0, arr_repo_root)
    from judger import Judger  # type: ignore  # pylint: disable=import-error

    return Judger


@EVALUATORS.register_module()
class ARRPointwiseEvaluator(BaseEvaluator):
    """Pointwise EvalVerseEvaluator dimension backed by the ARR Auto-Rubric judge."""

    def __init__(
        self,
        dimension_name: str = "arr_dimension",
        model: Any = None,
        # ARR-specific kwargs.
        arr_config_path: Optional[str] = None,
        arr_rubrics_file: Optional[str] = None,
        arr_rubrics_dir: Optional[str] = None,
        arr_repo_root: Optional[str] = None,
        score_range: str = "1-5",
        skip_without_rubrics: bool = True,
        base_url: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(dimension_name, model, **kwargs)
        self._score_range = score_range
        self._skip_without_rubrics = skip_without_rubrics

        arr_repo_root = _resolve_arr_repo_root(arr_repo_root)
        try:
            self._JudgerCls = _import_arr_judger(arr_repo_root)
        except Exception as exc:  # pragma: no cover - exercised only at deploy time
            logger.error("Failed to import ARR Judger: %s", exc)
            raise

        self._arr_config_path = _resolve_arr_config_path(arr_config_path)
        self._arr_rubrics_file = _resolve_rubrics_file(
            arr_rubrics_file, arr_rubrics_dir, dimension_name
        )
        self._arr_base_url = base_url or os.environ.get("ARR_BASE_URL")
        self._judger = None

        logger.info(
            "ARRPointwiseEvaluator init: dimension=%s config=%s rubrics_file=%s",
            dimension_name,
            self._arr_config_path,
            self._arr_rubrics_file,
        )

    # ------------------------------------------------------------------
    # Lifecycle: lazy Judger init so the constructor is cheap.
    # ------------------------------------------------------------------
    def _ensure_judger(self) -> Any:
        if self._judger is not None:
            return self._judger
        judger = self._JudgerCls(
            config_path=self._arr_config_path,
            rubrics_file=self._arr_rubrics_file,
            base_url=self._arr_base_url,
        )
        _run_coro(judger.initialize())
        self._judger = judger
        return judger

    # ------------------------------------------------------------------
    # EvalVerseEvaluator hooks.
    # ------------------------------------------------------------------
    def preprocess(self, video_data: Any, **kwargs: Any) -> Any:
        return _build_arr_sample(video_data)

    def evaluate(self, video_data: Any, **kwargs: Any) -> Dict[str, Any]:
        # 0. Required fields.
        if getattr(video_data, "video_path", None) is None:
            return {
                "dimension": self.dimension_name,
                "score": None,
                "details": {"skipped": True, "reason": "missing video_path"},
            }

        # 1. If the dimension is rubric-bound and we can't find the rubric file,
        #    either skip (default) or proceed with auto-rubric generation.
        if self._skip_without_rubrics and not self._arr_rubrics_file:
            return {
                "dimension": self.dimension_name,
                "score": None,
                "details": {
                    "skipped": True,
                    "reason": "no arr_rubrics_file for this dimension",
                    "hint": "set ARR_RUBRICS_DIR or pass arr_rubrics_file kwarg",
                },
            }

        # 2. Construct sample and call ARR.
        sample = self.preprocess(video_data, **kwargs)

        try:
            judger = self._ensure_judger()
            result = _run_coro(judger._evaluate_item(sample))  # pylint: disable=protected-access
        except Exception as exc:  # pragma: no cover - depends on network / VLM
            logger.exception("ARR pointwise evaluation failed: %s", exc)
            return {
                "dimension": self.dimension_name,
                "score": 0.0,
                "details": {"error": str(exc), "task_type": sample.get("task_type")},
            }

        raw_score = getattr(result, "score", None)
        reason = getattr(result, "reason", "") or ""
        if raw_score is None:
            return {
                "dimension": self.dimension_name,
                "score": 0.0,
                "details": {
                    "error": "no score from ARR",
                    "raw_result_repr": repr(result),
                    "task_type": sample.get("task_type"),
                },
            }

        score = self._normalise_score(float(raw_score))
        return {
            "dimension": self.dimension_name,
            "score": score,
            "details": {
                "arr_raw_score": float(raw_score),
                "arr_reason": reason,
                "task_type": sample.get("task_type"),
                "task_name": (sample.get("metadata") or {}).get("task_name"),
                "arr_config_path": self._arr_config_path,
                "arr_rubrics_file": self._arr_rubrics_file,
            },
        }

    # ------------------------------------------------------------------
    # Score normalisation.
    # ------------------------------------------------------------------
    def _normalise_score(self, raw: float) -> float:
        if self._score_range == "0-5":
            mapped = (raw - 1.0) / 4.0 * 5.0
        else:
            mapped = raw
        return max(0.0, min(5.0, mapped))


__all__ = [
    "ARRPointwiseEvaluator",
    "_build_arr_sample",
]
