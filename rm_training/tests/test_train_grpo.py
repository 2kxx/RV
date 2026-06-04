"""Unit tests for ``rm_training.stage2_grpo.train_grpo`` policy gradient.

Validates the end-to-end backward path in **real** mode (i.e. with a
torch model and an optimiser) using a toy LM whose interface mimics
the subset of ``Qwen2.5-VL`` that ``compute_log_probs`` touches:

    .device                 — torch device
    .parameters()           — yields trainable LoRA-like params
    .train() / .eval()      — set training mode
    .__call__(input_ids=…)  — returns object with ``.logits`` (B, T, V)
    .disable_adapter()      — ctx manager that turns the LoRA off

We then run ``policy_update`` over 4 RubricRollouts with a mix of
positive / negative advantages and assert:

  • all 4 are applied (no skips);
  • the trainable adapter weights actually change after one step;
  • the dummy-mode short-circuit still returns zeros and never touches
    the optimiser.

The purpose is to catch regressions in compute_log_probs ↔ policy_update
plumbing on CPU, before paying the 7B-on-GPU debugging tax.
"""

from __future__ import annotations

try:
    import pytest                                    # noqa: F401  (optional)
except ImportError:                                  # pragma: no cover
    pytest = None                                    # type: ignore

import torch

from rm_training.stage2_grpo.student_wrapper import (
    RubricRollout,
    StudentConfig,
    StudentWrapper,
)
from rm_training.stage2_grpo.train_grpo import policy_update


class _DisableAdapterCtx:
    """ctx manager mimicking peft's ``model.disable_adapter()``."""
    def __init__(self, m): self.m = m
    def __enter__(self):
        self._mask = self.m._adapter_mask
        self.m._adapter_mask = 0.0
    def __exit__(self, *exc):
        self.m._adapter_mask = self._mask
        return False


class _ToyLM(torch.nn.Module):
    """Tiny stand-in for Qwen2.5-VL that exercises the same interface."""
    def __init__(self, vocab: int = 32, hid: int = 8):
        super().__init__()
        self.emb     = torch.nn.Embedding(vocab, hid)
        self.lm      = torch.nn.Linear(hid, vocab, bias=False)
        self.adapter = torch.nn.Linear(hid, hid, bias=False)
        torch.nn.init.zeros_(self.adapter.weight)        # init at LoRA-disabled
        self._adapter_mask = 1.0
        for p in self.emb.parameters():     p.requires_grad = False
        for p in self.lm.parameters():      p.requires_grad = False
        for p in self.adapter.parameters(): p.requires_grad = True

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, input_ids):
        h = self.emb(input_ids)
        h = h + self._adapter_mask * self.adapter(h)
        return type("O", (), {"logits": self.lm(h)})()

    def disable_adapter(self):
        return _DisableAdapterCtx(self)


def _toy_student() -> StudentWrapper:
    s = StudentWrapper(StudentConfig(model_path=None, dummy=True))
    s._dummy = False                                 # pretend real
    s._model = _ToyLM()
    return s


# ---------------------------------------------------------------------------
# Real-mode (toy LM) policy gradient.
# ---------------------------------------------------------------------------
def test_policy_update_steps_trainable_params():
    student = _toy_student()
    rollouts = [
        RubricRollout(text="r",
                      prompt_ids=torch.tensor([1, 2, 3], dtype=torch.long),
                      sampled_ids=torch.tensor([5, 7, 9], dtype=torch.long))
        for _ in range(4)
    ]
    advantages = [0.5, -0.5, 0.7, -0.3]

    optim = torch.optim.AdamW(list(student.trainable_parameters()), lr=1e-2)
    w_before = student._model.adapter.weight.detach().clone()

    out = policy_update(
        student, rollouts, advantages,
        optimizer=optim, beta_kl=0.04,
    )

    w_after = student._model.adapter.weight.detach().clone()
    assert out["applied"] == 4
    assert out["skipped"] == 0
    # Adapter must have moved — otherwise grad never flowed through.
    assert (w_after - w_before).abs().sum().item() > 0.0
    # Loss should be a finite scalar (not NaN / inf).
    assert isinstance(out["loss"], float)
    assert out["loss"] == out["loss"]                # not NaN


def test_policy_update_kl_zero_when_adapter_neutral():
    """Adapter init at 0 → policy ≡ reference → KL ≈ 0 on the first step."""
    student = _toy_student()
    rollouts = [
        RubricRollout(text="r",
                      prompt_ids=torch.tensor([1, 2, 3], dtype=torch.long),
                      sampled_ids=torch.tensor([5, 7, 9], dtype=torch.long))
    ]
    optim = torch.optim.AdamW(list(student.trainable_parameters()), lr=1e-2)
    out = policy_update(
        student, rollouts, [0.5],
        optimizer=optim, beta_kl=0.04,
    )
    assert abs(out["kl"]) < 1e-6


def test_policy_update_kl_grows_after_step():
    """After a real backward step, policy ≠ reference → KL becomes positive.

    We intentionally use **asymmetric** advantages so the per-rollout
    losses don't cancel after stack().mean(): with [1.0, -1.0] the
    averaged grad would be ~0 and the adapter would stay at zero.
    """
    student = _toy_student()
    rollouts = [
        RubricRollout(text="a",
                      prompt_ids=torch.tensor([1, 2, 3], dtype=torch.long),
                      sampled_ids=torch.tensor([5, 7, 9], dtype=torch.long)),
        RubricRollout(text="b",
                      prompt_ids=torch.tensor([4, 5, 6], dtype=torch.long),
                      sampled_ids=torch.tensor([2, 3, 4], dtype=torch.long)),
    ]
    optim = torch.optim.AdamW(list(student.trainable_parameters()), lr=1e-1)
    # First step pushes adapter away from zero.
    out1 = policy_update(student, rollouts, [1.0, 0.7],
                         optimizer=optim, beta_kl=0.0)
    assert out1["applied"] == 2
    moved = student._model.adapter.weight.detach().abs().sum().item()
    assert moved > 1e-4, f"adapter must move on step 1; got {moved}"
    # Second step should now see a non-zero KL.
    out2 = policy_update(student, rollouts, [1.0, 0.7],
                         optimizer=optim, beta_kl=0.04)
    assert out2["kl"] != 0.0


# ---------------------------------------------------------------------------
# Dummy short-circuit.
# ---------------------------------------------------------------------------
def test_policy_update_dummy_short_circuits():
    student = StudentWrapper(StudentConfig(model_path=None, dummy=True))
    out = policy_update(
        student, [None, None, None], [0.0, 0.5, -0.5],
        optimizer=None, beta_kl=0.04,
    )
    assert out["loss"]    == 0.0
    assert out["applied"] == 0
    assert out["skipped"] == 3


def test_policy_update_handles_empty_sampled_ids():
    """If a rollout's sampled_ids is empty (degenerate), we skip it cleanly."""
    student = _toy_student()
    rollouts = [
        RubricRollout(text="empty",
                      prompt_ids=torch.tensor([1, 2, 3], dtype=torch.long),
                      sampled_ids=torch.tensor([],       dtype=torch.long)),
        RubricRollout(text="ok",
                      prompt_ids=torch.tensor([1, 2, 3], dtype=torch.long),
                      sampled_ids=torch.tensor([5, 7],   dtype=torch.long)),
    ]
    optim = torch.optim.AdamW(list(student.trainable_parameters()), lr=1e-2)
    out = policy_update(student, rollouts, [0.5, 0.5],
                        optimizer=optim, beta_kl=0.04)
    assert out["applied"] == 1
    assert out["skipped"] == 1
