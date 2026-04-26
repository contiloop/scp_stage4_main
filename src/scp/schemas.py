"""Event schemas for SCP ledger.

All events share: event_type, run_id, round_id, example_id, ts.
Stored as JSONL lines; compacted into per-event-type Parquet at round end.
"""
from __future__ import annotations

EVENT_TYPES = (
    "student_generation",
    "pre_probe_score",
    "probe_training",
    "probe_generation",
    "collapse_analysis",
    "teacher_edit",
    "sft_item",
    "preference_candidate",
    "round_summary",
    "budget_event",
    "dev_eval",
)

TEACHER_ACTIONS = ("no_change", "minor_edit", "major_edit", "rewrite", "invalid")

PAIR_TYPES = ("free_pair", "collapse_pair", "refreshed_pair")


REQUIRED_KEYS = {"event_type", "run_id", "round_id", "example_id", "ts"}


def validate_event(event: dict) -> None:
    missing = REQUIRED_KEYS - event.keys()
    if missing:
        raise ValueError(f"event missing required keys: {missing}")
    if event["event_type"] not in EVENT_TYPES:
        raise ValueError(f"unknown event_type: {event['event_type']}")
    if event["event_type"] == "teacher_edit":
        action = event.get("teacher_action")
        if action is not None and action not in TEACHER_ACTIONS:
            raise ValueError(f"invalid teacher_action: {action}")
    if event["event_type"] == "preference_candidate":
        pt = event.get("pair_type")
        if pt is not None and pt not in PAIR_TYPES:
            raise ValueError(f"invalid pair_type: {pt}")
