"""RewardVerse adapter — trained RGPO checkpoint with student-generated rubrics.

Same forward path as V3-zeroshot (StudentWrapper + _make_val_score_fn) but
with two flips:
  * ``model_path`` points at a trained Stage-2 GRPO checkpoint dir instead
    of the base Qwen2.5-VL-7B-Instruct.
  * ``use_gold_rubric=False`` — the trained student generates its own
    rubric first, then scores against it. This mirrors
    ``eval_ckpt.py --no-use_gold_rubric`` and is the deployment-equivalent
    path: the RL policy will consume whatever rubric the deployed RM
    actually produces, not the gold one.

Environment
-----------
* ``RM_CKPT`` (or ``model_path=`` kwarg): path to the trained ckpt dir, e.g.
  ``ckpts/student_grpo_8gpu_zero3_joint_v3/checkpoint-45``. Falls back to
  V3-zeroshot semantics (base model + gold rubric) if unset — useful for
  smoke tests before the RGPO checkpoint lands.
* ``V3_BASE_MODEL``: base Qwen2.5-VL-7B-Instruct path, needed to backfill
  missing processor cfg files into older checkpoints (see
  ``eval_ckpt._backfill_processor_files``).
"""

from __future__ import annotations

import os
from pathlib import Path

from scripts.downstream_grpo.src.reward_models.v3_zeroshot_adapter import (
    DEFAULT_BASE_MODEL,
    V3ZeroshotAdapter,
)


class RewardVerseAdapter(V3ZeroshotAdapter):
    """StudentWrapper-backed scorer using a trained RGPO checkpoint.

    Differences from :class:`V3ZeroshotAdapter`:
      * ``model_path`` resolves from ``$RM_CKPT`` (a trained ckpt dir).
      * ``use_gold_rubric`` defaults to ``False`` so the trained student
        generates its own rubric per query (matches deployment).

    If ``$RM_CKPT`` is unset we transparently delegate to the base model
    (and keep the gold-rubric behaviour of V3-zeroshot) — smoke tests
    can run before the checkpoint materialises, and the arm silently
    degrades to V3 (logged, not raised).
    """

    name = "rewardverse"
    _model_path_env = "RM_CKPT"
    _default_model_path = ""  # sentinel: "unset"
    # RewardVerse (trained student) generates rubrics itself.
    _default_use_gold_rubric = False

    def __init__(self, **kwargs) -> None:
        rm_ckpt = kwargs.pop("model_path", None) or os.environ.get("RM_CKPT", "").strip()

        if not rm_ckpt:
            print(
                f"[{type(self).name}] $RM_CKPT unset; falling back to base "
                f"Qwen2.5-VL-7B + gold-rubric (V3-zeroshot semantics). "
                "Set $RM_CKPT to your trained ckpt dir to enable the real "
                "RewardVerse arm (student-generated rubrics).",
                flush=True,
            )
            rm_ckpt = DEFAULT_BASE_MODEL
            # Delegating to base model → also switch back to gold rubric,
            # since a base (untrained) student's self-generated rubrics
            # are noisy and defeat the point of the fallback.
            kwargs.setdefault("use_gold_rubric", True)

        if not Path(rm_ckpt).exists():
            raise FileNotFoundError(
                f"[rewardverse] RM_CKPT does not exist: {rm_ckpt}. Set $RM_CKPT "
                "or pass model_path= to a valid checkpoint dir."
            )

        # base_model_path is needed for backfilling missing processor files
        # into old-format checkpoints; defaults to the base model env var.
        base = (
            kwargs.pop("base_model_path", None)
            or os.environ.get("V3_BASE_MODEL")
            or DEFAULT_BASE_MODEL
        )

        super().__init__(model_path=rm_ckpt, base_model_path=base, **kwargs)
