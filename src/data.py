"""
Unified monolingual source loader for SCP Stage A.

All stages of the algorithm consume the same record schema:
    {
        "sample_id": str,
        "source_text": str,
        "source_lang_iso": str,
        "target_lang_iso": str,
        "reference_text": str | None,   # only present for test.csv, used for analysis
        "metadata": dict,               # sector, source_type, ticker, ...
    }

A config switch (kind: csv | hf) selects the backend. Adding Reuters or
anti-forgetting corpora later is a matter of dropping a new YAML into
configs/data/ — no code changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

from omegaconf import DictConfig

from .common import resolve_workspace_path


REQUIRED_FIELDS = ("sample_id", "source_text", "source_lang_iso", "target_lang_iso")


def _coerce(val: Any) -> str:
    return "" if val is None else str(val).strip()


def _metadata_columns(cfg: DictConfig) -> dict[str, str]:
    keys = (
        "sector_column",
        "source_type_column",
        "sub_sector_column",
        "ticker_column",
    )
    return {k.removesuffix("_column"): str(cfg[k]) for k in keys if cfg.get(k)}


def _iter_csv(cfg: DictConfig) -> Iterator[dict[str, Any]]:
    import pandas as pd

    path = resolve_workspace_path(cfg.path)
    df = pd.read_csv(path, encoding="utf-8-sig")
    id_col = str(cfg.id_column)
    src_col = str(cfg.source_column)
    ref_col = cfg.get("reference_column")
    meta_cols = _metadata_columns(cfg)
    src_iso = str(cfg.source_lang_iso)
    tgt_iso = str(cfg.target_lang_iso)

    max_rows = cfg.get("max_rows")
    if max_rows is not None:
        df = df.head(int(max_rows))

    for _, row in df.iterrows():
        source_text = _coerce(row.get(src_col))
        if not source_text:
            continue
        record: dict[str, Any] = {
            "sample_id": _coerce(row.get(id_col)) or f"csv:{int(_)}",
            "source_text": source_text,
            "source_lang_iso": src_iso,
            "target_lang_iso": tgt_iso,
            "reference_text": _coerce(row.get(ref_col)) if ref_col else "",
            "metadata": {k: _coerce(row.get(col)) for k, col in meta_cols.items()},
        }
        yield record


def _iter_hf(cfg: DictConfig) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset(
        path=str(cfg.path),
        name=cfg.get("name"),
        split=str(cfg.get("split", "train")),
        streaming=bool(cfg.get("streaming", False)),
    )
    src_field = str(cfg.get("source_text_field", "text"))
    id_field = cfg.get("id_field")
    src_iso = str(cfg.source_lang_iso)
    tgt_iso = str(cfg.target_lang_iso)
    max_rows = cfg.get("max_rows")

    for idx, row in enumerate(ds):
        if max_rows is not None and idx >= int(max_rows):
            break
        source_text = _coerce(row.get(src_field))
        if not source_text:
            continue
        sample_id = (
            _coerce(row.get(id_field)) if id_field and id_field in row else f"hf:{cfg.path}:{idx}"
        )
        yield {
            "sample_id": sample_id or f"hf:{cfg.path}:{idx}",
            "source_text": source_text,
            "source_lang_iso": src_iso,
            "target_lang_iso": tgt_iso,
            "reference_text": "",
            "metadata": {k: _coerce(v) for k, v in row.items() if k != src_field},
        }


def iter_records(cfg: DictConfig) -> Iterator[dict[str, Any]]:
    kind = str(cfg.kind).strip().lower()
    if kind == "csv":
        yield from _iter_csv(cfg)
    elif kind == "hf":
        yield from _iter_hf(cfg)
    else:
        raise ValueError(f"Unsupported data.kind: {cfg.kind!r}. Expected 'csv' or 'hf'.")


def load_records(cfg: DictConfig) -> list[dict[str, Any]]:
    records = list(iter_records(cfg))
    for r in records[:3]:
        for field in REQUIRED_FIELDS:
            if field not in r:
                raise RuntimeError(f"record missing required field {field!r}: keys={list(r.keys())}")
    return records


def partition_disjoint(
    records: list[dict[str, Any]],
    num_subsets: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    """Algorithm 1 line 1: partition D into K disjoint subsets D_1 ... D_K."""
    import random

    if num_subsets <= 0:
        raise ValueError("num_subsets must be positive")
    if num_subsets == 1:
        return [list(records)]

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    chunks: list[list[dict[str, Any]]] = [[] for _ in range(num_subsets)]
    for i, rec in enumerate(shuffled):
        chunks[i % num_subsets].append(rec)
    return chunks
