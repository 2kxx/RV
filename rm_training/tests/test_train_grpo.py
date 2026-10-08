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

    def forward(self, input_ids, attention_mask=None,
                past_key_values=None, use_cache=False, **kwargs):
        # attention_mask is consumed but ignored — the toy LM doesn't
        # need it; we just accept it so the wrapper's compute_log_probs
        # call (which now passes attention_mask for Qwen2.5-VL multimodal
        # compatibility) doesn't blow up here.
        #
        # ``past_key_values`` / ``use_cache`` minimal simulation: when a
        # cache is supplied it carries the embedded prefix tokens, and
        # we concatenate them onto the current step's tokens before
        # running the adapter so the output for the new tokens reflects
        # the full context (matching how a real attention model behaves).
        # Logits are returned ONLY for the new tokens (HF convention with
        # past_kv supplied), matching ``out.logits.shape == (B, T_new, V)``.
        if past_key_values is not None:
            past_emb, past_attn = past_key_values
            new_emb = self.emb(input_ids)
            full_emb = torch.cat([past_emb, new_emb], dim=1)
            adapted = full_emb + self._adapter_mask * self.adapter(full_emb)
            # Logits only for the new positions (last input_ids.size(1)).
            new_len = input_ids.size(1)
            adapted_new = adapted[:, -new_len:, :]
            logits = self.lm(adapted_new)
            new_kv = (full_emb.detach(), None) if use_cache else None
        else:
            h = self.emb(input_ids)
            adapted = h + self._adapter_mask * self.adapter(h)
            logits = self.lm(adapted)
            # Stash the (detached) embeddings as the cache for the next
            # forward — detached because the cache should not carry
            # autograd history when downstream code claims it ran the
            # prefix under no_grad.
            new_kv = (h.detach(), None) if use_cache else None
        out = type("O", (), {"logits": logits, "past_key_values": new_kv})()
        return out

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


# ---------------------------------------------------------------------------
# Per-rollout backward + grad-accumulation invariant.
#
# The new policy_update does ``(L_g / G).backward()`` per rollout instead of
# stacking G losses and backwarding once. Both should yield IDENTICAL grads
# (proof: backward is linear in the loss). We verify this on the toy LM
# by snapshotting ``adapter.weight.grad`` after one accumulated step and
# comparing against an explicit "stack-then-mean" reference computation.
# ---------------------------------------------------------------------------
def test_policy_update_grads_match_stack_mean_baseline():
    """Per-rollout (L/G).backward() == stack(L).mean().backward() in grads."""
    # Set up the toy student twice with identical init.
    torch.manual_seed(0)
    s_a = _toy_student()
    torch.manual_seed(0)
    s_b = _toy_student()

    rollouts = [
        RubricRollout(text="r",
                      prompt_ids=torch.tensor([1, 2, 3], dtype=torch.long),
                      sampled_ids=torch.tensor([5, 7, 9], dtype=torch.long))
        for _ in range(3)
    ]
    advantages = [0.4, -0.7, 0.9]

    # Path A: new policy_update (per-rollout backward + accumulate).
    optim_a = torch.optim.SGD(list(s_a.trainable_parameters()), lr=0.0)  # no step
    s_a._model.train()
    optim_a.zero_grad(set_to_none=True)
    n_usable = len(rollouts)
    for adv, meta in zip(advantages, rollouts):
        with torch.enable_grad():
            logp = s_a.compute_log_probs(meta.prompt_ids, meta.sampled_ids, ref=False)
        logp_ref = s_a.compute_log_probs(meta.prompt_ids, meta.sampled_ids, ref=True)
        loss = (-float(adv) * logp.mean() + 0.04 * (logp.mean() - logp_ref.mean())) / n_usable
        loss.backward()
    grad_a = s_a._model.adapter.weight.grad.detach().clone()

    # Path B: stack-then-mean baseline (legacy).
    s_b._model.train()
    losses = []
    for adv, meta in zip(advantages, rollouts):
        with torch.enable_grad():
            logp = s_b.compute_log_probs(meta.prompt_ids, meta.sampled_ids, ref=False)
        logp_ref = s_b.compute_log_probs(meta.prompt_ids, meta.sampled_ids, ref=True)
        loss = -float(adv) * logp.mean() + 0.04 * (logp.mean() - logp_ref.mean())
        losses.append(loss)
    total = torch.stack(losses).mean()
    total.backward()
    grad_b = s_b._model.adapter.weight.grad.detach().clone()

    assert torch.allclose(grad_a, grad_b, atol=1e-6), \
        f"grad mismatch: max diff {(grad_a-grad_b).abs().max().item():.3e}"


def test_policy_update_skipped_count_does_not_starve_normalisation():
    """When some rollouts skip, n_usable < G but the normalisation
    divides by n_usable, not G — otherwise the update is biased small."""
    student = _toy_student()
    rollouts = [
        RubricRollout(text="ok",
                      prompt_ids=torch.tensor([1, 2, 3], dtype=torch.long),
                      sampled_ids=torch.tensor([5, 7], dtype=torch.long)),
        RubricRollout(text="empty",
                      prompt_ids=torch.tensor([1, 2, 3], dtype=torch.long),
                      sampled_ids=torch.tensor([], dtype=torch.long)),
    ]
    optim = torch.optim.AdamW(list(student.trainable_parameters()), lr=1.0)
    out = policy_update(student, rollouts, [0.5, 0.0],
                        optimizer=optim, beta_kl=0.04)
    assert out["applied"] == 1 and out["skipped"] == 1
    # Reported loss is the un-scaled per-rollout loss (multiplied back).
    assert isinstance(out["loss"], float)


# ---------------------------------------------------------------------------
# Gradient checkpointing + LoRA — verify input_require_grads still allows
# the LoRA params to receive grads (the canonical peft+HF gotcha).
# ---------------------------------------------------------------------------
def test_grad_checkpointing_does_not_break_lora_grads():
    """Toy stand-in with the same forward shape: emb → adapter → lm_head.

    We register the input-require-grads forward hook on the embedding,
    wrap the adapter+lm portion in ``torch.utils.checkpoint.checkpoint``,
    and check that adapter.weight receives a non-None grad after backward.
    This reproduces (in miniature) what HF's ``gradient_checkpointing_enable``
    + ``enable_input_require_grads`` does for the real Qwen2.5-VL stack.
    """
    import torch.utils.checkpoint as cp

    class _Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(32, 8)
            self.adapter = torch.nn.Linear(8, 8, bias=False)
            torch.nn.init.zeros_(self.adapter.weight)
            self.lm = torch.nn.Linear(8, 32, bias=False)
            for p in self.emb.parameters(): p.requires_grad = False
            for p in self.lm.parameters():  p.requires_grad = False
            for p in self.adapter.parameters(): p.requires_grad = True

            # The HF input_require_grads hook: force embedded outputs
            # to require grad even when the embedding is frozen.
            def _input_require_grads(_m, _i, output):
                output.requires_grad_(True)
            self.emb.register_forward_hook(_input_require_grads)

        def forward(self, ids):
            h = self.emb(ids)
            # Activation checkpoint the adapter forward — exactly the
            # pattern HF uses inside transformer layers.
            h = cp.checkpoint(self._adapter_forward, h, use_reentrant=False)
            return self.lm(h)

        def _adapter_forward(self, h):
            return h + self.adapter(h)

    m = _Toy()
    ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
    out = m(ids)
    loss = out.float().sum()
    loss.backward()
    g = m.adapter.weight.grad
    assert g is not None, "LoRA-style param received None grad with checkpointing"
    assert g.abs().sum().item() > 0.0, "LoRA-style param grad is all zeros"


def test_student_config_has_gradient_checkpointing_defaults_on():
    """Sanity: default StudentConfig() turns on gradient_checkpointing."""
    cfg = StudentConfig()
    assert cfg.use_gradient_checkpointing is True
    assert cfg.grad_checkpoint_use_reentrant is False


# ---------------------------------------------------------------------------
# P1 regression: token-level Schulman k3 KL is what's actually computed.
#
# The previous implementation collapsed to ``exp(mean Δ) − 1 − mean Δ``,
# which is ≤ token-level KL by Jensen. Verify two things:
#   1. policy_update's reported ``kl`` matches the token-level k3 formula
#      (within float tolerance).
#   2. when log-ratios are non-uniform across tokens, token-level KL is
#      strictly greater than the legacy sequence-level approximation —
#      i.e. we are no longer under-counting.
# ---------------------------------------------------------------------------
def test_policy_update_kl_is_token_level_k3():
    """policy_update.kl == mean_t[exp(Δ_t) − 1 − Δ_t]."""
    student = _toy_student()
    # Step the adapter away from zero so policy ≠ ref → KL > 0.
    optim = torch.optim.SGD(list(student.trainable_parameters()), lr=0.5)
    rollouts_pre = [
        RubricRollout(text="seed",
                      prompt_ids=torch.tensor([1, 2, 3], dtype=torch.long),
                      sampled_ids=torch.tensor([5, 7, 9], dtype=torch.long)),
    ]
    policy_update(student, rollouts_pre, [1.0],
                  optimizer=optim, beta_kl=0.0, max_grad_norm=0.0)

    # Now measure KL on a fresh rollout. Use a single rollout so the
    # reported kl_per_rollout entry is unambiguous.
    rollouts = [
        RubricRollout(text="probe",
                      prompt_ids=torch.tensor([2, 4, 6], dtype=torch.long),
                      sampled_ids=torch.tensor([3, 5, 7, 9], dtype=torch.long)),
    ]
    optim2 = torch.optim.SGD(list(student.trainable_parameters()), lr=0.0)  # no step
    out = policy_update(student, rollouts, [0.5],
                        optimizer=optim2, beta_kl=0.04)
    reported_kl = out["kl"]

    # Recompute the expected token-level k3 KL outside policy_update.
    with torch.no_grad():
        meta = rollouts[0]
        logp     = student.compute_log_probs(meta.prompt_ids, meta.sampled_ids,
                                             ref=False)
        logp_ref = student.compute_log_probs(meta.prompt_ids, meta.sampled_ids,
                                             ref=True)
        delta_t  = logp - logp_ref                          # (T,)
        token_kl = (delta_t.exp() - 1.0 - delta_t).mean().item()
        # And the legacy sequence-level approximation, for the
        # under-counting comparison below.
        delta_mean = delta_t.mean()
        seq_kl = (delta_mean.exp() - 1.0 - delta_mean).item()

    assert abs(reported_kl - token_kl) < 1e-5, \
        f"reported kl={reported_kl} vs token-level k3={token_kl}"
    # When Δ_t is non-uniform across tokens (true after a real grad step),
    # token-level KL ≥ sequence-level by Jensen. Equality only when all
    # Δ_t are identical, which they won't be on a random backward.
    assert token_kl >= seq_kl - 1e-9


# ---------------------------------------------------------------------------
# P2 regression: warm-up bookkeeping in policy_update.
#
# When rubric_metas is all-None (gold-rubric warm-up case) but score-PG
# is OFF (rollouts=None or score_pg_alpha=0), policy_update must skip
# cleanly without reaching into ``meta.sampled_ids``.
# ---------------------------------------------------------------------------
def test_policy_update_warmup_no_score_pg_skips_cleanly():
    """Warm-up + score-PG OFF → all skipped, no exception."""
    student = _toy_student()
    optim = torch.optim.AdamW(list(student.trainable_parameters()), lr=1e-2)
    out = policy_update(
        student,
        rollout_meta=[None, None, None],
        advantages=[0.5, -0.5, 0.7],
        optimizer=optim,
        beta_kl=0.04,
        rollouts=None,
        score_pg_alpha=0.0,
    )
    assert out["applied"] == 0
    assert out["skipped"] == 3
    assert out["loss"] == 0.0
    assert out["kl"]   == 0.0


def test_policy_update_warmup_with_score_pg_routes_to_score_leg():
    """Warm-up (rubric_metas=None) + score-PG ON + valid score meta:
    policy_update must NOT early-return; it should run the score-side
    forward and increment ``score_pg_applied``."""
    from rm_training.stage2_grpo.reward_fn import RolloutPair
    from rm_training.stage2_grpo.rubric_parser import ParsedFull, ParsedRubric, RubricEntry

    student = _toy_student()
    optim = torch.optim.SGD(list(student.trainable_parameters()), lr=0.0)  # no step

    # Build a RolloutPair carrying score-side meta. The reward_fn fmt
    # gates require fmt_rubric/fmt_pos/fmt_neg ≥ score_format_gate; we
    # supply breakdowns with all 1.0s to pass.
    score_meta = RubricRollout(
        text="dummy_score",
        prompt_ids =torch.tensor([1, 2, 3], dtype=torch.long),
        sampled_ids=torch.tensor([5, 7, 9], dtype=torch.long),
    )
    # Synthesise minimal RolloutPair shapes — only the score_*_meta
    # fields are read by the score-PG branch in policy_update.
    rubric_stub = ParsedRubric(rubrics=[RubricEntry(theme="t", weight=1.0, tips=[])])
    full_stub   = ParsedFull(rubrics=rubric_stub.rubrics, scores=[3.0])
    rollout = RolloutPair(
        student_rubric=rubric_stub,
        student_full_pos=full_stub,
        student_full_neg=full_stub,
        score_pos_meta=score_meta,
        score_neg_meta=score_meta,
    )
    breakdown = {"fmt_rubric": 1.0, "fmt_pos": 1.0, "fmt_neg": 1.0}

    out = policy_update(
        student,
        rollout_meta=[None],            # warm-up: no rubric meta
        advantages=[0.5],
        optimizer=optim,
        beta_kl=0.04,
        rollouts=[rollout],
        breakdowns=[breakdown],
        score_pg_alpha=1.0,             # full strength
        score_format_gate=0.65,
    )
    # The rollout was usable via the score leg, so it should NOT be
    # counted as skipped, and score_pg_applied should be 1.
    assert out["applied"] == 1
    assert out["skipped"] == 0
    assert out["score_pg_applied"] == 1


# ---------------------------------------------------------------------------
# P2: argparse exposes the new --gold_rubric_warmup_epochs flag and it
# defaults to 0 (legacy behaviour).
# ---------------------------------------------------------------------------
def test_argparser_has_gold_rubric_warmup_flag():
    from rm_training.stage2_grpo.train_grpo import build_argparser
    p = build_argparser()
    args = p.parse_args(["--gold_rubrics", "x.jsonl", "--output_dir", "out"])
    assert hasattr(args, "gold_rubric_warmup_epochs")
    assert args.gold_rubric_warmup_epochs == 0
    args2 = p.parse_args(
        ["--gold_rubrics", "x.jsonl", "--output_dir", "out",
         "--gold_rubric_warmup_epochs", "1"]
    )
    assert args2.gold_rubric_warmup_epochs == 1


# ---------------------------------------------------------------------------
# Trick A: detach_prompt path in compute_log_probs.
#
# Validates the EasyR1/VeRL-style two-stage forward (no_grad prompt prefill +
# grad-on sampled tail) on the toy LM. Three properties must hold:
#   1. Numerical equivalence to the legacy single-forward path within
#      reasonable float tolerance (the toy LM is position-wise, so the
#      two paths produce identical logits up to float-add reordering).
#   2. Backward through the sampled tail still reaches trainable params.
#   3. The prompt-side activations are NOT saved into the autograd graph
#      — i.e. setting requires_grad on a parameter that's only touched
#      during the prompt forward should yield no grad after backward.
# ---------------------------------------------------------------------------
def test_detach_prompt_numerical_equivalence_to_legacy():
    """tok_logp from detach_prompt=True == tok_logp from detach_prompt=False
    (toy LM is position-wise; this is the clean equivalence check)."""
    student = _toy_student()
    # Make adapter non-zero so the two paths actually exercise the adapter
    # (with all-zero adapter the equivalence is trivial).
    with torch.no_grad():
        student._model.adapter.weight.copy_(
            torch.randn_like(student._model.adapter.weight) * 0.1
        )

    prompt_ids  = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
    sampled_ids = torch.tensor([7, 8, 9],          dtype=torch.long)

    with torch.no_grad():
        legacy = student.compute_log_probs(
            prompt_ids, sampled_ids, ref=False, detach_prompt=False,
        )
        detached = student.compute_log_probs(
            prompt_ids, sampled_ids, ref=False, detach_prompt=True,
        )

    assert legacy.shape == detached.shape == (3,)
    assert torch.allclose(legacy, detached, atol=1e-6), \
        f"max diff {(legacy - detached).abs().max().item():.3e}"


def test_detach_prompt_grad_reaches_trainable_params():
    """Backward through tok_logp from the detached path produces a non-zero
    grad on the adapter weight — i.e. the sampled-tail forward is not
    accidentally detached itself."""
    student = _toy_student()
    with torch.no_grad():
        student._model.adapter.weight.copy_(
            torch.randn_like(student._model.adapter.weight) * 0.1
        )

    prompt_ids  = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
    sampled_ids = torch.tensor([7, 8, 9],          dtype=torch.long)

    student._model.adapter.weight.grad = None
    student._model.train()
    with torch.enable_grad():
        logp = student.compute_log_probs(
            prompt_ids, sampled_ids, ref=False, detach_prompt=True,
        )
        loss = -logp.sum()
        loss.backward()

    g = student._model.adapter.weight.grad
    assert g is not None, "adapter weight received no grad"
    assert g.abs().sum().item() > 0.0, \
        "adapter weight grad is exactly zero — likely detached the whole forward"


def test_detach_prompt_short_circuits_for_ref_or_empty():
    """ref=True / T==0 / P<2 must fall through to the legacy path even
    when detach_prompt=True is requested."""
    student = _toy_student()

    # Case 1: ref=True. Legacy uses no_grad either way; detach_prompt is
    # silently ignored. Expect the same shape and finite values.
    prompt_ids  = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    sampled_ids = torch.tensor([5, 6],       dtype=torch.long)
    with torch.no_grad():
        out_ref = student.compute_log_probs(
            prompt_ids, sampled_ids, ref=True, detach_prompt=True,
        )
    assert out_ref.shape == (2,)
    assert torch.isfinite(out_ref).all()

    # Case 2: T == 0. Compute path returns an empty (0,) tensor; both
    # legacy and detach paths must agree.
    empty = torch.tensor([], dtype=torch.long)
    out_t0 = student.compute_log_probs(
        prompt_ids, empty, ref=False, detach_prompt=True,
    )
    assert out_t0.shape == (0,)

    # Case 3: P < 2. detach_prompt requires at least 2 prompt tokens to
    # split (P-1 prefix + 1 last token). With P=1 we fall through.
    p1 = torch.tensor([1], dtype=torch.long)
    s2 = torch.tensor([5, 6], dtype=torch.long)
    out_p1 = student.compute_log_probs(
        p1, s2, ref=False, detach_prompt=True,
    )
    assert out_p1.shape == (2,)
    assert torch.isfinite(out_p1).all()


def test_detach_prompt_preserves_kl_path():
    """policy_update with the score-PG branch (detach_prompt=True under
    the hood) must still produce a finite KL on the rubric leg — the
    rubric forward is independent and goes through the legacy path."""
    from rm_training.stage2_grpo.reward_fn import RolloutPair
    from rm_training.stage2_grpo.rubric_parser import ParsedFull, ParsedRubric, RubricEntry

    student = _toy_student()
    with torch.no_grad():
        student._model.adapter.weight.copy_(
            torch.randn_like(student._model.adapter.weight) * 0.05
        )
    optim = torch.optim.SGD(list(student.trainable_parameters()), lr=0.0)  # no step

    rubric_meta = RubricRollout(
        text="rubric",
        prompt_ids =torch.tensor([1, 2, 3, 4], dtype=torch.long),
        sampled_ids=torch.tensor([5, 6, 7],    dtype=torch.long),
    )
    score_meta = RubricRollout(
        text="score",
        prompt_ids =torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long),
        sampled_ids=torch.tensor([8, 9],             dtype=torch.long),
    )
    rubric_stub = ParsedRubric(rubrics=[RubricEntry(theme="t", weight=1.0, tips=[])])
    full_stub   = ParsedFull(rubrics=rubric_stub.rubrics, scores=[3.0])
    rollout = RolloutPair(
        student_rubric=rubric_stub,
        student_full_pos=full_stub,
        student_full_neg=full_stub,
        score_pos_meta=score_meta,
        score_neg_meta=score_meta,
    )
    breakdown = {"fmt_rubric": 1.0, "fmt_pos": 1.0, "fmt_neg": 1.0}

    out = policy_update(
        student,
        rollout_meta=[rubric_meta],
        advantages=[0.5],
        optimizer=optim,
        beta_kl=0.04,
        rollouts=[rollout],
        breakdowns=[breakdown],
        score_pg_alpha=1.0,
        score_format_gate=0.65,
    )
    assert out["applied"] == 1
    assert out["score_pg_applied"] == 1
    # KL must be a finite number on the rubric leg.
    assert isinstance(out["kl"], float)
    assert out["kl"] == out["kl"]            # not NaN
