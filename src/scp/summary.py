"""Round summary aggregator.

Reads ledger events for one round and emits the metrics the dashboard needs.
"""
from __future__ import annotations

from .ledger import Ledger


def summarize_round(ledger: Ledger, round_id: int) -> dict:
    counts = {
        "num_examples": 0,
        "collapse_candidate_count": 0,
        "selected_for_teacher_count": 0,
        "teacher_call_count": 0,
        "teacher_cached_count": 0,
        "teacher_no_change_count": 0,
        "teacher_minor_edit_count": 0,
        "teacher_major_edit_count": 0,
        "teacher_rewrite_count": 0,
        "teacher_invalid_count": 0,
        "actual_correction_count": 0,
        "sft_items_created": 0,
        "preference_candidates_created": 0,
    }
    api_cost = 0.0
    delta_qes: list[float] = []

    seen_examples: set[str] = set()

    for ev in ledger.iter_events():
        if ev.get("round_id") != round_id:
            continue
        et = ev.get("event_type")
        eid = ev.get("example_id")
        if eid is not None:
            seen_examples.add(eid)

        if et == "collapse_analysis":
            if ev.get("collapse_flag"):
                counts["collapse_candidate_count"] += 1
            if ev.get("selected_for_teacher"):
                counts["selected_for_teacher_count"] += 1
            if ev.get("delta_qe") is not None:
                delta_qes.append(float(ev["delta_qe"]))

        elif et == "teacher_edit":
            counts["teacher_call_count"] += 1
            if ev.get("cached"):
                counts["teacher_cached_count"] += 1
            else:
                api_cost += float(ev.get("cost_usd") or 0.0)
            action = ev.get("teacher_action")
            key = f"teacher_{action}_count" if action else None
            if key and key in counts:
                counts[key] += 1
            if action in ("minor_edit", "major_edit", "rewrite"):
                counts["actual_correction_count"] += 1

        elif et == "sft_item":
            if ev.get("used_for_sft", True):
                counts["sft_items_created"] += 1

        elif et == "preference_candidate":
            if ev.get("eligible_for_training", True):
                counts["preference_candidates_created"] += 1

    counts["num_examples"] = len(seen_examples)
    counts["api_cost_usd"] = round(api_cost, 6)
    if delta_qes:
        delta_qes_sorted = sorted(delta_qes)
        counts["avg_delta_qe"] = sum(delta_qes) / len(delta_qes)
        counts["p90_delta_qe"] = delta_qes_sorted[int(0.9 * (len(delta_qes) - 1))]
    if counts["actual_correction_count"]:
        counts["cost_per_useful_correction"] = (
            api_cost / counts["actual_correction_count"]
        )
    counts["round_id"] = round_id
    return counts
