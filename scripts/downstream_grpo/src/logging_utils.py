"""Structured jsonl + optional wandb logger for downstream GRPO."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional
    wandb = None


class RunLogger:
    """Emit one jsonl line per event; mirror to wandb if available."""

    def __init__(
        self,
        run_dir: str | Path,
        arm: str,
        *,
        wandb_project: str | None = None,
        wandb_run_name: str | None = None,
        rank: int = 0,
    ) -> None:
        self.rank = rank
        self.arm = arm
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / f"{arm}.jsonl"
        if self.rank == 0:
            self._fh = self.jsonl_path.open("a", encoding="utf-8", buffering=1)
        else:
            self._fh = None

        self._wandb = None
        if self.rank == 0 and wandb is not None and wandb_project and os.getenv("WANDB_MODE") != "disabled":
            try:
                self._wandb = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name or arm,
                    dir=str(self.run_dir),
                    reinit=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[logger] wandb init failed: {exc}; continuing with jsonl only")
                self._wandb = None

    def log(self, event: str, step: int, **kw: Any) -> None:
        if self.rank != 0:
            return
        rec = {"event": event, "step": step, "arm": self.arm, **kw}
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if self._wandb is not None and event != "config":
            scalar = {f"{event}/{k}": v for k, v in kw.items() if isinstance(v, (int, float))}
            if scalar:
                self._wandb.log(scalar, step=step)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
        if self._wandb is not None:
            self._wandb.finish()
