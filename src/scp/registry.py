"""Lightweight example registry.

Stores only what's needed to follow a sentence across rounds:
example_id, source_text, source_hash, dataset_version, assigned_round,
optional metadata. Backed by a single Parquet file under data/examples.parquet.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

REQUIRED_COLS = ["example_id", "source_text", "source_hash",
                 "dataset_version", "assigned_round"]


def source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ExampleRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self._df = pd.read_parquet(self.path)
        else:
            self._df = pd.DataFrame(columns=REQUIRED_COLS + ["metadata"])

    def add_many(self, rows: Iterable[dict]) -> int:
        new_rows = []
        existing = set(self._df["example_id"]) if len(self._df) else set()
        for r in rows:
            if r["example_id"] in existing:
                continue
            r = dict(r)
            r.setdefault("source_hash", source_hash(r["source_text"]))
            md = r.get("metadata", {})
            r["metadata"] = json.dumps(md, ensure_ascii=False) if md else ""
            new_rows.append(r)
        if not new_rows:
            return 0
        self._df = pd.concat([self._df, pd.DataFrame(new_rows)], ignore_index=True)
        return len(new_rows)

    def assign_round(self, example_ids: list[str], round_id: int) -> int:
        mask = self._df["example_id"].isin(example_ids)
        self._df.loc[mask, "assigned_round"] = round_id
        return int(mask.sum())

    def iter_round(self, round_id: int) -> Iterable[dict]:
        sub = self._df[self._df["assigned_round"] == round_id]
        for _, row in sub.iterrows():
            yield row.to_dict()

    def __len__(self) -> int:
        return len(self._df)

    def save(self) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self._df.to_parquet(tmp, index=False)
        tmp.replace(self.path)
