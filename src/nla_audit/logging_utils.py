"""Metrics go to a local CSV (always) and Weights & Biases (when a key is configured)."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(
        self,
        csv_path: str | Path,
        *,
        run_name: str,
        config: dict[str, Any] | None = None,
        use_wandb: bool | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._fields: list[str] | None = None
        if self.csv_path.exists():
            with self.csv_path.open(newline="", encoding="utf-8") as f:
                header = next(csv.reader(f), None)
                self._fields = header or None
        if use_wandb is None:
            use_wandb = bool(os.environ.get("WANDB_API_KEY"))
        self._wandb = None
        if use_wandb:
            import wandb

            self._wandb = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "open-nla-audits"),
                name=run_name,
                config=config or {},
                resume="allow",
                id=run_name.replace("/", "-"),
            )

    def log(self, row: dict[str, Any], step: int | None = None) -> None:
        row = dict(row)
        if step is not None:
            row.setdefault("step", step)
        if self._fields is None:
            self._fields = list(row)
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._fields).writeheader()
        extra = [k for k in row if k not in self._fields]
        if extra:
            raise KeyError(f"new columns {extra} not in CSV header {self._fields}")
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._fields).writerow(row)
        if self._wandb is not None:
            self._wandb.log(row, step=step)

    def close(self) -> None:
        if self._wandb is not None:
            self._wandb.finish()
