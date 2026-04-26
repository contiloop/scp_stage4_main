"""JSONL append-first event ledger + round-end Parquet compaction.

Runtime: every event hits events.jsonl with fsync.
Round end: filter events.jsonl by (run_id, round_id, event_type) -> parquet.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

from .schemas import EVENT_TYPES, validate_event
from .storage import LocalStorage


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.", time.gmtime()) + f"{int((time.time()%1)*1e6):06d}Z"


class Ledger:
    """Append events to a single JSONL; compact per round on demand."""

    EVENTS_FILE = "events.jsonl"

    def __init__(self, run_dir: str | Path, run_id: str, cloud_uri: str | None = None):
        self.storage = LocalStorage(run_dir, cloud_uri=cloud_uri)
        self.run_dir = self.storage.root
        self.run_id = run_id

    def log(self, event: dict) -> dict:
        ev = dict(event)
        ev.setdefault("run_id", self.run_id)
        ev.setdefault("ts", _now_iso())
        validate_event(ev)
        self.storage.append_jsonl(self.EVENTS_FILE, ev)
        return ev

    def iter_events(self) -> Iterable[dict]:
        yield from self.storage.read_jsonl(self.EVENTS_FILE)

    def existing_keys(self, event_type: str, round_id: int, key: str = "example_id") -> set:
        out = set()
        for ev in self.iter_events():
            if ev.get("event_type") == event_type and ev.get("round_id") == round_id:
                v = ev.get(key)
                if v is not None:
                    out.add(v)
        return out

    def compact_round(self, round_id: int) -> dict[str, Path]:
        """Write per-event-type parquet files for one round. Returns paths."""
        import pandas as pd

        buckets: dict[str, list[dict]] = {t: [] for t in EVENT_TYPES}
        for ev in self.iter_events():
            if ev.get("round_id") != round_id:
                continue
            t = ev.get("event_type")
            if t in buckets:
                buckets[t].append(ev)

        out: dict[str, Path] = {}
        round_subdir = f"rounds/round_{round_id:03d}"
        for t, rows in buckets.items():
            if not rows:
                continue
            df = pd.DataFrame(rows)
            rel = f"{round_subdir}/{t}s.parquet"
            out[t] = self.storage.write_parquet(rel, df)
        return out

    def write_round_summary(self, round_id: int, summary: dict) -> Path:
        import json
        rel = f"rounds/round_{round_id:03d}/round_summary.json"
        data = json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8")
        return self.storage.write_atomic(rel, data)
