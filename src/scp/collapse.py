"""Collapse analysis: compare pre-probe and post-probe scores.

Produces a row per example combining q_before, q_after, delta_qe, rule
worsening flag, and a single selection_score the selector can rank by.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CollapseConfig:
    qe_drop_threshold: float = 0.05      # delta_qe above this -> collapse_flag
    weight_delta_qe: float = 1.0
    weight_rule_worsened: float = 0.5
    weight_qe_after_low: float = 0.0     # penalize low post-probe QE directly


def analyze(pre: dict, post: dict, cfg: CollapseConfig) -> dict:
    q_before = float(pre.get("qe_score", 0.0))
    q_after = float(post.get("qe_score", 0.0))
    delta_qe = q_before - q_after  # positive = degraded under probe

    rule_before = set(pre.get("rule_flags", []) or [])
    rule_after = set(post.get("rule_flags", []) or [])
    rule_worsened = bool(rule_after - rule_before)

    selection_score = (
        cfg.weight_delta_qe * max(0.0, delta_qe)
        + cfg.weight_rule_worsened * (1.0 if rule_worsened else 0.0)
        + cfg.weight_qe_after_low * max(0.0, 1.0 - q_after)
    )
    collapse_flag = delta_qe >= cfg.qe_drop_threshold or rule_worsened

    return {
        "q_before": q_before,
        "q_after": q_after,
        "delta_qe": delta_qe,
        "rule_before": sorted(rule_before),
        "rule_after": sorted(rule_after),
        "rule_worsened": rule_worsened,
        "collapse_flag": collapse_flag,
        "selection_score": selection_score,
    }
