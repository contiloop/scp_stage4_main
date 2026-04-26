"""Persistent run state.

Tracks current round, current main checkpoint id + path, per-step completion
flags, and most recent SFT/preference dataset versions. JSON file with
atomic write so a mid-run crash leaves a valid file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class RunState:
    FILE = "state.json"

    def __init__(self, run_dir: str | Path):
        self.path = Path(run_dir) / self.FILE
        if self.path.exists():
            self._data = json.loads(self.path.read_text())
        else:
            self._data = {
                "current_round": 0,
                "checkpoints": {},          # role -> {id, path}
                "completed_steps": {},      # round_id -> [step names]
                "datasets": {},             # version label -> path
            }
            self._write()

    def _write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False))
        os.replace(tmp, self.path)

    @property
    def current_round(self) -> int:
        return int(self._data["current_round"])

    def advance_round(self, round_id: int) -> None:
        self._data["current_round"] = int(round_id)
        self._write()

    def set_checkpoint(self, role: str, ckpt_id: str, path: str | None = None) -> None:
        self._data["checkpoints"][role] = {"id": ckpt_id, "path": path}
        self._write()

    def get_checkpoint(self, role: str) -> dict | None:
        return self._data["checkpoints"].get(role)

    def mark_step_done(self, round_id: int, step: str) -> None:
        steps = self._data["completed_steps"].setdefault(str(round_id), [])
        if step not in steps:
            steps.append(step)
            self._write()

    def is_step_done(self, round_id: int, step: str) -> bool:
        return step in self._data["completed_steps"].get(str(round_id), [])

    def register_dataset(self, version: str, path: str) -> None:
        self._data["datasets"][version] = path
        self._write()

    def snapshot(self) -> dict[str, Any]:
        return json.loads(json.dumps(self._data))
