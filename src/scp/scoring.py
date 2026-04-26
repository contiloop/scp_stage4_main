"""Cheap scorers + QE protocol with COMET-Kiwi subprocess adapter.

QE design
---------
COMET-Kiwi pins transformers<5, our training stack uses transformers>=5.
Following the existing project pattern (see src/qe.py + notebook cell 1.5),
COMET runs in a SEPARATE venv pointed to by $COMET_PYTHON, called via
subprocess. This module wraps `src/qe.py:CometKiwiSubprocessScorer` so the
new SCP code path uses the exact same backend the notebook uses.

QE is BATCH-friendly: the QE Protocol exposes `score_batch(pairs)` and the
composite scorer's `score_many(pairs)` runs cheap scorers per-example but
QE in a single batched subprocess call. Per-example `score(src, hyp)`
still works but invokes a 1-element batch (slow for COMET).

Fallback: HeuristicQE (no model) when $COMET_PYTHON is unset.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Protocol


_NUM_RE = re.compile(r"\d[\d.,]*")


class QEScorer(Protocol):
    name: str
    def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]: ...


# ----------------------------- cheap scorers ----------------------------- #

@dataclass
class LengthRatioScorer:
    min_ratio: float = 0.5
    max_ratio: float = 2.0
    name: str = "length_ratio"

    def score(self, source: str, hypothesis: str) -> dict:
        ratio = max(1, len(hypothesis)) / max(1, len(source))
        return {"length_ratio": ratio,
                "length_ratio_flag": ratio < self.min_ratio or ratio > self.max_ratio}


@dataclass
class NumberMatchScorer:
    name: str = "number_match"

    def score(self, source: str, hypothesis: str) -> dict:
        s = sorted(_NUM_RE.findall(source))
        h = sorted(_NUM_RE.findall(hypothesis))
        return {"number_match": s == h, "number_count_src": len(s),
                "number_count_hyp": len(h)}


@dataclass
class RuleFlagScorer:
    """rules: [{name, pattern, on: source|hypothesis|both, flag_if: present|absent}]"""
    rules: list[dict]
    name: str = "rule_flags"

    def score(self, source: str, hypothesis: str) -> dict:
        flags: list[str] = []
        for r in self.rules:
            on = r["on"]
            if on == "both":
                hit = bool(re.search(r["pattern"], source)) or bool(re.search(r["pattern"], hypothesis))
            else:
                target = source if on == "source" else hypothesis
                hit = bool(re.search(r["pattern"], target or ""))
            if r.get("flag_if", "present") == "present" and hit:
                flags.append(r["name"])
            elif r.get("flag_if") == "absent" and not hit:
                flags.append(r["name"])
        return {"rule_flags": flags}


# ------------------------------- QE backends ----------------------------- #

class HeuristicQE:
    name = "heuristic_qe"
    def score_batch(self, pairs):
        out = []
        for src, hyp in pairs:
            ratio = max(1, len(hyp)) / max(1, len(src))
            out.append(max(0.0, 1.0 - min(abs(ratio - 1.0), 1.0)))
        return out


class CometKiwiQE:
    """Adapter over src/qe.py:CometKiwiSubprocessScorer.

    Construct via build_qe_from_config; never instantiated directly when
    $COMET_PYTHON is unset (constructor will raise).
    """
    def __init__(self, model_name: str, fallback_model_name: str | None = None,
                 batch_size: int = 32, gpus: int = 1, python_bin: str | None = None):
        from src.qe import CometKiwiSubprocessScorer  # lazy: keeps tests light
        py = python_bin or os.environ.get("COMET_PYTHON", "").strip() or None
        if not py:
            raise RuntimeError("CometKiwiQE requires $COMET_PYTHON or python_bin=...")
        self._inner = CometKiwiSubprocessScorer(
            python_bin=py,
            model_name=model_name,
            fallback_model_name=fallback_model_name,
            batch_size=batch_size,
            gpus=gpus,
        )
        self.name = self._inner.name

    def score_batch(self, pairs):
        return self._inner.score(pairs)


# ------------------------------ composite -------------------------------- #

class CompositeScorer:
    def __init__(self, scorers: Iterable, qe: QEScorer | None = None):
        self.scorers = list(scorers)
        self.qe = qe

    def _cheap(self, source: str, hypothesis: str) -> dict:
        out: dict = {}
        for s in self.scorers:
            out.update(s.score(source, hypothesis))
        return out

    def score(self, source: str, hypothesis: str) -> dict:
        out = self._cheap(source, hypothesis)
        if self.qe is not None:
            out["qe_score"] = float(self.qe.score_batch([(source, hypothesis)])[0])
        return out

    def score_many(self, pairs: list[tuple[str, str]]) -> list[dict]:
        rows = [self._cheap(s, h) for s, h in pairs]
        if self.qe is not None and pairs:
            qes = self.qe.score_batch(pairs)
            for r, q in zip(rows, qes):
                r["qe_score"] = float(q)
        return rows


# ------------------------------- factory --------------------------------- #

def build_qe_from_config(qe_cfg: dict | None) -> QEScorer | None:
    """qe_cfg = {type: heuristic|comet_kiwi, ...}"""
    if not qe_cfg:
        return None
    t = qe_cfg.get("type")
    if t == "heuristic":
        return HeuristicQE()
    if t == "comet_kiwi":
        params = {k: v for k, v in qe_cfg.items() if k != "type"}
        try:
            return CometKiwiQE(**params)
        except RuntimeError as e:
            import warnings
            warnings.warn(f"COMET QE unavailable ({e}); falling back to HeuristicQE")
            return HeuristicQE()
    raise ValueError(f"unknown qe type: {t}")


def build_scorer_from_config(cfg: dict) -> CompositeScorer:
    type_map = {
        "length_ratio": LengthRatioScorer,
        "number_match": NumberMatchScorer,
        "rule_flags": RuleFlagScorer,
    }
    scorers = []
    for s in cfg.get("scorers", []):
        params = {k: v for k, v in s.items() if k != "type"}
        scorers.append(type_map[s["type"]](**params))
    return CompositeScorer(scorers, qe=build_qe_from_config(cfg.get("qe")))
