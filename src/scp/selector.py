"""Teacher-call selector.

Inputs: list of {example_id, selection_score, collapse_flag, rule_worsened, ...}
Output: ordered list of selected example_ids + reasons, capped by budget.
Diversity clustering deferred (config flag stub).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SelectorConfig:
    min_selection_score: float = 0.0
    require_collapse_flag: bool = False
    max_per_round: int | None = None
    diversity_clustering: bool = False  # phase 2


def select(rows: list[dict], cfg: SelectorConfig) -> list[dict]:
    eligible = []
    for r in rows:
        if cfg.require_collapse_flag and not r.get("collapse_flag"):
            continue
        if float(r.get("selection_score", 0.0)) < cfg.min_selection_score:
            continue
        eligible.append(r)
    eligible.sort(key=lambda r: r.get("selection_score", 0.0), reverse=True)
    if cfg.max_per_round is not None:
        eligible = eligible[: cfg.max_per_round]

    out = []
    for r in eligible:
        reasons = []
        if r.get("rule_worsened"):
            reasons.append("rule_flag")
        if r.get("delta_qe", 0.0) >= 0.05:
            reasons.append("delta_qe_high")
        if r.get("collapse_flag"):
            reasons.append("collapse")
        out.append({
            "example_id": r["example_id"],
            "selection_score": r.get("selection_score"),
            "selection_reason": reasons,
        })
    return out
