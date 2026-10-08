"""V3-zeroshot adapter — base Qwen2.5-VL-7B-Instruct with gold rubrics.

Same forward path as :mod:`rm_training.stage2_grpo.eval_ckpt` **without**
``--no-use_gold_rubric``: load the base model via ``StudentWrapper``, call
``_make_val_score_fn(use_gold_rubric=True)`` so scoring uses the per-dim
gold rubric from ``data/rm_train/gold_rubrics_18d.jsonl`` (no student
rubric generation). Returns a scalar in [1, 5].

This isolates the effect of *RGPO training* from the effect of *rubric
generation*: v3_zeroshot uses the fixed gold rubric, so its variance
comes only from the base backbone's scoring quality — no rubric drift.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Sequence

import torch

from scripts.downstream_grpo.src.reward_adapters import RewardRequest


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASE_MODEL = "/efs/jacuzzitang/models/Qwen2.5-VL-7B-Instruct"
DEFAULT_GOLD_JSONL = "data/rm_train/gold_rubrics_18d.jsonl"


class V3ZeroshotAdapter:
    """StudentWrapper-backed pointwise scorer, base Qwen2.5-VL-7B-Instruct.

    Uses the same ``_make_val_score_fn(use_gold_rubric=True)`` path as
    ``eval_ckpt.py`` **without** ``--no-use_gold_rubric``. Concurrency is
    intentionally 1 because ``score_video`` is a single-GPU forward and
    the model is replicated on one card.
    """

    name = "v3_zeroshot"
    # ``model_path`` defaults to base Qwen2.5-VL-7B-Instruct. Subclasses
    # (RewardVerseAdapter) override this to point at a trained ckpt dir.
    _model_path_env = "V3_BASE_MODEL"
    _default_model_path = DEFAULT_BASE_MODEL
    # Arm C uses gold rubric (fixed scaffold); RewardVerseAdapter flips
    # this to False so the trained student generates its own rubric first.
    _default_use_gold_rubric = True

    def __init__(
        self,
        *,
        device: str = "cuda",
        use_gold_rubric: bool | None = None,
        attn_implementation: str = "sdpa",
        score_max_new_tokens: int = 64,
        gold_rubrics_jsonl: str | None = None,
        model_path: str | None = None,
        base_model_path: str | None = None,
    ) -> None:
        if not str(REPO_ROOT) in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        from rm_training.stage1_teacher.task_descriptions import load_task_specs
        from rm_training.stage2_grpo.data_collator import GoldRubricDataset
        from rm_training.stage2_grpo.eval_ckpt import _load_student_from_ckpt
        from rm_training.stage2_grpo.train_grpo import _build_gold_by_dim, _make_val_score_fn

        # Concrete resolution.
        self.device = str(device) if str(device).startswith("cuda") else "cuda:0"
        self.use_gold_rubric = bool(
            self._default_use_gold_rubric if use_gold_rubric is None else use_gold_rubric
        )

        mp = model_path or os.environ.get(self._model_path_env) or self._default_model_path
        if not Path(mp).exists():
            raise FileNotFoundError(
                f"[{self.name}] model path not found: {mp}. "
                f"Set ${self._model_path_env} or pass model_path=."
            )
        self.model_path = mp

        # Only needed for ckpt-mode (backfilling processor cfg on old ckpts).
        # For the base model, base_model_path == model_path is fine and the
        # backfill is a no-op.
        self.base_model_path = base_model_path or mp

        gold_jsonl = Path(gold_rubrics_jsonl or (REPO_ROOT / DEFAULT_GOLD_JSONL))
        if not gold_jsonl.exists():
            raise FileNotFoundError(
                f"[{self.name}] gold rubrics jsonl not found: {gold_jsonl}."
            )

        print(f"[{self.name}] loading student from {self.model_path} on {self.device}...",
              flush=True)
        self._student = _load_student_from_ckpt(
            Path(self.model_path),
            base_model_path=self.base_model_path,
            attn_implementation=attn_implementation,
            score_max_new_tokens=int(score_max_new_tokens),
            device=self.device,
        )
        print(f"[{self.name}] student ready.", flush=True)

        task_specs = load_task_specs(REPO_ROOT)
        ds = GoldRubricDataset.from_jsonl(str(gold_jsonl), rebase_root=REPO_ROOT)
        gold_by_dim = _build_gold_by_dim(ds)

        # ``score_fn`` matches the training/val contract exactly.
        self._score_fn = _make_val_score_fn(
            self._student, task_specs,
            use_gold_rubric=self.use_gold_rubric,
            gold_by_dim=gold_by_dim,
        )

    def _score_one(self, req: RewardRequest) -> float:
        try:
            s = self._score_fn(req.dimension, req.prompt, req.video_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[{self.name}] score_fn raised on {req.video_path} "
                  f"(dim={req.dimension}): {exc}", flush=True)
            return 3.0
        if s is None or (isinstance(s, float) and (s != s)):
            # NaN or None → neutral fallback (equivalent behaviour to
            # eval_pointwise's "score_failed" bucket).
            print(f"[{self.name}] non-finite score for {req.video_path} "
                  f"(dim={req.dimension}); returning 3.0", flush=True)
            return 3.0
        return float(s)

    def score_batch(self, batch: Sequence[RewardRequest]) -> torch.Tensor:
        # Serial: the underlying forward already pins ONE 7B replica on ONE
        # card. Parallel calls just contend on the same GPU. The rewardverse
        # HTTP server serializes concurrent HTTP requests with a lock, so
        # this loop is the only place execution actually happens.
        scores = [self._score_one(r) for r in batch]
        return torch.tensor(scores, dtype=torch.float32)

    def close(self) -> None:
        try:
            del self._score_fn
        except AttributeError:
            pass
        try:
            model = getattr(self._student, "model", None)
            if model is not None:
                del model
            del self._student
        except AttributeError:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
