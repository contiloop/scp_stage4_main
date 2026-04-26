"""Per-round and global teacher API budget tracker.

Truth source = teacher_edit events in the ledger (cost_usd field).
This module provides cap enforcement; it does not replace the ledger.
"""
from __future__ import annotations

from dataclasses import dataclass

from .ledger import Ledger


@dataclass
class BudgetCaps:
    per_round_usd: float | None = None
    global_usd: float | None = None
    per_round_calls: int | None = None
    global_calls: int | None = None


class BudgetTracker:
    def __init__(self, ledger: Ledger, caps: BudgetCaps):
        self.ledger = ledger
        self.caps = caps

    def _scan(self, round_id: int | None = None) -> tuple[float, int]:
        cost = 0.0
        calls = 0
        for ev in self.ledger.iter_events():
            if ev.get("event_type") != "teacher_edit":
                continue
            if ev.get("cached"):
                continue  # cache hits don't consume budget
            if round_id is not None and ev.get("round_id") != round_id:
                continue
            cost += float(ev.get("cost_usd") or 0.0)
            calls += 1
        return cost, calls

    def round_spent(self, round_id: int) -> tuple[float, int]:
        return self._scan(round_id)

    def global_spent(self) -> tuple[float, int]:
        return self._scan(None)

    def can_call(self, round_id: int, est_cost: float = 0.0) -> tuple[bool, str]:
        rcost, rcalls = self.round_spent(round_id)
        gcost, gcalls = self.global_spent()
        c = self.caps
        if c.per_round_usd is not None and rcost + est_cost > c.per_round_usd:
            return False, f"round_usd_cap reached ({rcost:.4f}+{est_cost:.4f}>{c.per_round_usd})"
        if c.global_usd is not None and gcost + est_cost > c.global_usd:
            return False, f"global_usd_cap reached ({gcost:.4f}+{est_cost:.4f}>{c.global_usd})"
        if c.per_round_calls is not None and rcalls + 1 > c.per_round_calls:
            return False, f"round_calls_cap reached ({rcalls}+1>{c.per_round_calls})"
        if c.global_calls is not None and gcalls + 1 > c.global_calls:
            return False, f"global_calls_cap reached ({gcalls}+1>{c.global_calls})"
        return True, "ok"
