"""rm_training.stage2_grpo.valid_filter
====================================
Detect "cheat" rubrics that produce ≈0 score discrimination between y⁺
and y⁻ (i.e. ``|S⁺ − S⁻| ≤ eps``). Such rubrics get clipped reward and
push the running ``valid_ratio`` down, which in turn relaxes the
λ-annealing schedule so gold alignment stays active longer.

This module is deliberately tiny + side-effect-free. It does NOT touch
torch / numpy beyond plain Python floats so it can be imported in any
context (CPU dummy mode, multi-process workers, ablation scripts).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

__all__ = ["is_valid_score_pair", "ValidRatioTracker"]


def is_valid_score_pair(s_pos: float, s_neg: float, *, eps: float = 0.05) -> bool:
    """Return True iff the rubric actually discriminates the two videos.

    NaN / inf inputs always fail (a rubric that crashed the parser is
    by definition not a useful one).
    """
    if not (math.isfinite(s_pos) and math.isfinite(s_neg)):
        return False
    return abs(s_pos - s_neg) > eps


@dataclass
class ValidRatioTracker:
    """Running EMA of the ``valid_ratio`` over the last ``window`` GRPO steps.

    Used in two places:
      • ``reward_fn`` clips r_pref to 0 for invalid samples.
      • ``train_grpo`` reads ``current()`` to compute λ(t) = λ₀·(1-ema).
    """
    window: int = 256
    alpha:  float = 0.05
    _hits:  deque = None      # type: ignore[assignment]
    _ema:   float = 0.0       # exponential moving average; init 0 → cold start

    def __post_init__(self) -> None:
        self._hits = deque(maxlen=self.window)

    def update(self, valid_flags: list[bool]) -> float:
        """Push a batch of bools and return the new EMA."""
        if not valid_flags:
            return self._ema
        batch_ratio = sum(1.0 for v in valid_flags if v) / len(valid_flags)
        # EMA: smooth across batches so noise from G=8 rollouts doesn't
        # whiplash the λ schedule.
        if not self._hits:
            self._ema = batch_ratio
        else:
            self._ema = (1.0 - self.alpha) * self._ema + self.alpha * batch_ratio
        self._hits.append(batch_ratio)
        return self._ema

    def current(self) -> float:
        """Return the running ratio without updating."""
        return self._ema

    def lambda_t(self, lambda_0: float = 0.3) -> float:
        """λ(t) = λ₀ · (1 − valid_ratio_ema)  (clipped to [0, λ₀])."""
        l = lambda_0 * (1.0 - self._ema)
        return max(0.0, min(lambda_0, l))
