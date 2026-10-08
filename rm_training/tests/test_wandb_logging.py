"""Unit tests for the optional wandb integration in
``rm_training.stage2_grpo.train_grpo``.

We don't want the test suite to require a wandb account or network, so
the wandb module is monkey-patched with a tiny in-memory recorder.
The tests verify:

1. Without ``--wandb_project``, ``_wandb_init`` returns None and the
   log helpers are no-ops.
2. With ``--wandb_project`` set + a fake wandb installed,
   ``_wandb_init`` returns a usable run object, ``_wandb_log_step``
   forwards the metric dict, and ``_wandb_log_val`` produces both
   per-dim scalars and a wandb.Table.
3. Setting RANK env > 0 silently skips init (DDP-friendly).
4. ``_wandb_log_step`` survives a heatmap-less val (overall only).
5. The CLI parses the new --wandb_* flags.
"""

from __future__ import annotations

import os
import sys

try:
    import pytest                                    # noqa: F401  (optional)
except ImportError:                                  # pragma: no cover
    pytest = None                                    # type: ignore

from rm_training.stage2_grpo.reward_fn import BatchRewardConfig
from rm_training.stage2_grpo.train_grpo import (
    _wandb_finish,
    _wandb_init,
    _wandb_log_step,
    _wandb_log_val,
    build_argparser,
)


# ---------------------------------------------------------------------------
# Fake wandb. Records every .log() / .finish() call so tests can assert on
# the recorded payload. Mirrors enough of the public API for our helpers.
# ---------------------------------------------------------------------------
class _FakeRun:
    def __init__(self, **init_kwargs):
        self.init_kwargs = init_kwargs
        self.url         = "http://wandb.fake/run/fake"
        self.name        = init_kwargs.get("name", "fake")
        self.summary     = type("S", (), {"update": lambda self, d: None})()
        self.logs:       list[tuple[dict, int | None]] = []
        self.finished    = False

    def log(self, payload, step=None):
        self.logs.append((dict(payload), step))

    def finish(self):
        self.finished = True


class _FakeWandb:
    Run = _FakeRun

    def __init__(self):
        self.last_run = None

    def init(self, **kwargs):
        self.last_run = _FakeRun(**kwargs)
        return self.last_run

    class Table:
        def __init__(self, columns):
            self.columns = list(columns)
            self.rows: list[list] = []
        def add_data(self, *row):
            self.rows.append(list(row))


def _install_fake_wandb():
    fake = _FakeWandb()
    sys.modules["wandb"] = fake
    return fake


# ---------------------------------------------------------------------------
# Tiny argparse stand-in that mimics the train_grpo CLI shape we need.
# ---------------------------------------------------------------------------
def _make_args(**overrides):
    base = dict(
        wandb_project = None,
        wandb_name    = None,
        wandb_entity  = None,
        wandb_offline = False,
        wandb_tags    = None,
        output_dir    = "/tmp/wb_test",
    )
    base.update(overrides)
    return type("A", (), base)()


# ---------------------------------------------------------------------------
# 1. No project → init returns None.
# ---------------------------------------------------------------------------
def test_wandb_init_returns_none_when_disabled():
    cfg = BatchRewardConfig()
    run = _wandb_init(_make_args(wandb_project=None), cfg)
    assert run is None


def test_wandb_log_step_noop_when_run_is_none():
    # Should not raise even with a populated metrics dict.
    _wandb_log_step(None, {"step": 0, "loss": 1.2, "rewards": [0.1, 0.2]})


def test_wandb_log_val_noop_when_run_is_none():
    _wandb_log_val(None, 0, {"plcc": 0.5, "srcc": 0.4, "mae": 1.0,
                              "n": 10, "S_mean": 3.0, "label_mean": 2.5},
                   {"action": {"plcc": 0.4}})


# ---------------------------------------------------------------------------
# 2. With fake wandb installed → init returns run, log writes payload.
# ---------------------------------------------------------------------------
def test_wandb_init_writes_through_to_fake():
    fake = _install_fake_wandb()
    cfg = BatchRewardConfig()
    args = _make_args(wandb_project="rv-rm", wandb_name="exp1",
                      wandb_tags=["smoke"])
    run = _wandb_init(args, cfg)
    assert run is not None
    assert run.init_kwargs["project"] == "rv-rm"
    assert run.init_kwargs["name"]    == "exp1"
    assert run.init_kwargs["tags"]    == ["smoke"]
    # cfg fields prefixed under reward_cfg.* in config.
    cfg_dict = run.init_kwargs["config"]
    assert any(k.startswith("reward_cfg.") for k in cfg_dict)


def test_wandb_log_step_forwards_payload():
    _install_fake_wandb()
    cfg = BatchRewardConfig()
    run = _wandb_init(_make_args(wandb_project="rv-rm"), cfg)
    metrics = {
        "step":          5,
        "wall_s":        12.3,
        "dimension":     "action",
        "rewards":       [0.1, 0.2, 0.3, 0.4],
        "pref_mean":     0.6,
        "align_mean":    0.4,
        "fmt_mean":      0.9,
        "valid_count":   3,
        "lambda_t":      0.18,
        "beta_t":        0.42,
        "loss":          0.05,
        "pg_loss":       0.04,
        "kl":            0.001,
        "applied":       4,
        "skipped":       0,
        "valid_ratio_ema": 0.55,
    }
    _wandb_log_step(run, metrics)
    assert len(run.logs) == 1
    payload, step = run.logs[0]
    assert step == 5
    assert payload["train/dim"]    == "action"
    # New format: per-rollout stats live under prefix_{mean,std,min,max}.
    assert abs(payload["train/reward_mean"] - 0.25) < 1e-9   # mean of [0.1,0.2,0.3,0.4]
    assert payload["train/reward_max"] == 0.4
    assert payload["train/reward_min"] == 0.1
    assert payload["train/loss"]   == 0.05
    assert payload["train/lambda_t"] == 0.18


def test_wandb_log_val_emits_overall_macro_per_dim_and_table():
    _install_fake_wandb()
    cfg = BatchRewardConfig()
    run = _wandb_init(_make_args(wandb_project="rv-rm"), cfg)
    overall = {"plcc": 0.4, "srcc": 0.3, "mae": 1.1, "n": 30,
               "S_mean": 4.0, "label_mean": 3.5}
    heatmap = {
        "action":    {"plcc": 0.2, "srcc": 0.1, "mae": 1.4, "n": 10,
                       "S_mean": 4.1, "label_mean": 3.6},
        "character": {"plcc": 0.6, "srcc": 0.5, "mae": 0.8, "n": 10,
                       "S_mean": 4.0, "label_mean": 3.5},
    }
    _wandb_log_val(run, step=100, overall=overall, heatmap=heatmap, tag="val")
    # Two log() calls: scalars + per-step table.
    assert len(run.logs) >= 1
    flat = [p for p, _ in run.logs]
    # Find the scalar payload (has 'val/plcc' key).
    scalar_payload = next((p for p in flat if "val/plcc" in p), None)
    assert scalar_payload is not None
    assert scalar_payload["val/plcc"]       == 0.4
    assert scalar_payload["val/macro_plcc"] == (0.2 + 0.6) / 2
    assert scalar_payload["val_dim/action/plcc"]    == 0.2
    assert scalar_payload["val_dim/character/plcc"] == 0.6


def test_wandb_finish_idempotent():
    _install_fake_wandb()
    cfg = BatchRewardConfig()
    run = _wandb_init(_make_args(wandb_project="rv-rm"), cfg)
    _wandb_finish(run)
    assert run.finished is True
    # Calling again on the same run is a no-op (no exception).
    _wandb_finish(run)
    # Calling on None is a no-op.
    _wandb_finish(None)


# ---------------------------------------------------------------------------
# 3. RANK > 0 → init silently returns None (DDP friendly).
# ---------------------------------------------------------------------------
def test_wandb_init_skips_on_non_zero_rank():
    _install_fake_wandb()
    os.environ["RANK"] = "3"
    try:
        cfg = BatchRewardConfig()
        run = _wandb_init(_make_args(wandb_project="rv-rm"), cfg)
        assert run is None
    finally:
        del os.environ["RANK"]


# ---------------------------------------------------------------------------
# 4. log_val survives missing heatmap.
# ---------------------------------------------------------------------------
def test_wandb_log_val_without_heatmap():
    _install_fake_wandb()
    cfg = BatchRewardConfig()
    run = _wandb_init(_make_args(wandb_project="rv-rm"), cfg)
    overall = {"plcc": 0.1, "srcc": 0.0, "mae": 1.5, "n": 5,
               "S_mean": 3.0, "label_mean": 3.0}
    _wandb_log_val(run, step=50, overall=overall, heatmap=None, tag="val")
    payload = next((p for p, _ in run.logs if "val/plcc" in p), None)
    assert payload is not None
    assert "val/macro_plcc" not in payload  # no heatmap → no macro


# ---------------------------------------------------------------------------
# 5. CLI parses the new flags.
# ---------------------------------------------------------------------------
def test_cli_accepts_wandb_flags():
    p = build_argparser()
    args = p.parse_args([
        "--gold_rubrics", "x",
        "--output_dir",   "y",
        "--wandb_project", "rv-rm",
        "--wandb_name",    "smoke",
        "--wandb_offline",
        "--wandb_tags",    "experiment", "ddp",
    ])
    assert args.wandb_project == "rv-rm"
    assert args.wandb_name    == "smoke"
    assert args.wandb_offline is True
    assert args.wandb_tags    == ["experiment", "ddp"]


def test_cli_wandb_disabled_by_default():
    p = build_argparser()
    args = p.parse_args(["--gold_rubrics", "x", "--output_dir", "y"])
    assert args.wandb_project is None
    assert args.wandb_offline is False
