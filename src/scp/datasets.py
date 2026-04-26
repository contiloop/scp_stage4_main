"""SFT and Preference dataset builders.

Reads ledger events (or compacted parquets) and writes versioned datasets to
`datasets/{name}_{version}.parquet`. Action-based filtering is config-driven.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

from .ledger import Ledger
from .schemas import PAIR_TYPES, TEACHER_ACTIONS


@dataclass
class SftBuildConfig:
    include_actions: list[str] = field(
        default_factory=lambda: ["minor_edit", "major_edit", "rewrite"]
    )
    drop_actions: list[str] = field(default_factory=lambda: ["invalid"])


@dataclass
class PrefBuildConfig:
    pair_types: list[str] = field(default_factory=lambda: ["free_pair", "collapse_pair"])
    require_action_in: list[str] = field(
        default_factory=lambda: ["minor_edit", "major_edit", "rewrite"]
    )
    high_confidence_min_delta_qe: float = 0.05


def _events(ledger: Ledger, etype: str, round_id: int | None) -> Iterable[dict]:
    for ev in ledger.iter_events():
        if ev.get("event_type") != etype:
            continue
        if round_id is not None and ev.get("round_id") != round_id:
            continue
        yield ev


def build_sft(
    ledger: Ledger,
    round_id: int | None,
    cfg: SftBuildConfig,
    out_path: str | Path,
    version: str,
) -> tuple[Path, int]:
    rows = []
    for ev in _events(ledger, "teacher_edit", round_id):
        action = ev.get("teacher_action")
        if action in cfg.drop_actions:
            continue
        if cfg.include_actions and action not in cfg.include_actions:
            continue
        rows.append({
            "example_id": ev["example_id"],
            "round_id": ev["round_id"],
            "source_text": ev.get("source_text"),
            "target_text": ev.get("teacher_output"),
            "target_source": "teacher_edit",
            "teacher_action": action,
            "sft_dataset_version": version,
        })
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    return out_path, len(rows)


def build_preference(
    ledger: Ledger,
    round_id: int | None,
    cfg: PrefBuildConfig,
    out_path: str | Path,
    version: str,
) -> tuple[Path, int]:
    teacher_by_id: dict[tuple[int, str], dict] = {}
    student_by_id: dict[tuple[int, str], dict] = {}
    probe_by_id: dict[tuple[int, str], dict] = {}
    collapse_by_id: dict[tuple[int, str], dict] = {}

    for ev in ledger.iter_events():
        rid = ev.get("round_id")
        eid = ev.get("example_id")
        if round_id is not None and rid != round_id:
            continue
        key = (rid, eid)
        et = ev.get("event_type")
        if et == "teacher_edit":
            teacher_by_id[key] = ev
        elif et == "student_generation":
            student_by_id[key] = ev
        elif et == "probe_generation":
            probe_by_id[key] = ev
        elif et == "collapse_analysis":
            collapse_by_id[key] = ev

    rows = []
    for key, t in teacher_by_id.items():
        if cfg.require_action_in and t.get("teacher_action") not in cfg.require_action_in:
            continue
        rid, eid = key
        student = student_by_id.get(key)
        probe = probe_by_id.get(key)
        collapse = collapse_by_id.get(key, {})
        delta_qe = float(collapse.get("delta_qe") or 0.0)
        confidence = "high" if delta_qe >= cfg.high_confidence_min_delta_qe else "low"

        if "free_pair" in cfg.pair_types and student is not None:
            rows.append(_pair_row(t, student, "student_output", "free_pair",
                                   rid, eid, confidence, version))
        if "collapse_pair" in cfg.pair_types and probe is not None:
            rows.append(_pair_row(t, probe, "probe_output", "collapse_pair",
                                   rid, eid, confidence, version))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    return out_path, len(rows)


def _pair_row(teacher_ev, rejected_ev, rejected_source, pair_type,
              rid, eid, confidence, version):
    rejected_text = (rejected_ev.get("student_output")
                     if rejected_source == "student_output"
                     else rejected_ev.get("probe_output"))
    assert pair_type in PAIR_TYPES
    return {
        "example_id": eid,
        "round_id": rid,
        "source_text": teacher_ev.get("source_text"),
        "chosen": teacher_ev.get("teacher_output"),
        "rejected": rejected_text,
        "chosen_source": "teacher_edit",
        "rejected_source": rejected_source,
        "pair_type": pair_type,
        "confidence": confidence,
        "eligible_for_training": confidence == "high",
        "pref_dataset_version": version,
    }
