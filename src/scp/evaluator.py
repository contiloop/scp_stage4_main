"""Dev-set evaluator for round-over-round tracking.

Composable metrics:
  - CometKiwiMetric: reference-free, batched via the existing subprocess
  - ChrfMetric: reference-based, sentence-mean (sacrebleu)
  - LengthRatioStat: hyp/src char ratio mean
  - Add more via the Metric Protocol.

The evaluator takes a `generate(source) -> str` callable so any runner
(student / probe / SFT-tuned) can be benchmarked against the same dev set.
Results are returned as a dict and logged to the ledger as a `dev_eval` event.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

import pandas as pd

from .ledger import Ledger
from .scoring import CometKiwiQE, HeuristicQE


class Metric(Protocol):
    name: str
    def compute(self, records: list[dict]) -> dict: ...


@dataclass
class CometKiwiMetric:
    """records: list[{source, hypothesis}] -> {cometkiwi: mean_score}"""
    model_name: str = "Unbabel/wmt22-cometkiwi-da"
    fallback_model_name: str | None = None
    batch_size: int = 32
    gpus: int = 1
    name: str = "cometkiwi"

    def compute(self, records):
        try:
            qe = CometKiwiQE(self.model_name, self.fallback_model_name,
                             self.batch_size, self.gpus)
        except Exception:
            qe = HeuristicQE()
        pairs = [(r["source_text"], r["hypothesis"]) for r in records]
        scores = qe.score_batch(pairs) if pairs else []
        return {f"{self.name}_mean": float(sum(scores) / max(1, len(scores))),
                f"{self.name}_n": len(scores),
                f"{self.name}_backend": qe.name}


@dataclass
class ChrfMetric:
    """Reference-based sentence-mean chrF. Skips records with no reference."""
    word_order: int = 2
    name: str = "chrf"

    def compute(self, records):
        try:
            import sacrebleu
        except ImportError:
            return {f"{self.name}_mean": None, f"{self.name}_n": 0,
                    f"{self.name}_skipped": "sacrebleu not installed"}
        scored = []
        for r in records:
            ref = r.get("reference")
            hyp = r.get("hypothesis")
            if not ref or not hyp:
                continue
            scored.append(sacrebleu.sentence_chrf(hyp, [ref],
                                                   word_order=self.word_order).score)
        return {f"{self.name}_mean": (sum(scored) / len(scored)) if scored else None,
                f"{self.name}_n": len(scored),
                f"{self.name}_word_order": self.word_order}


@dataclass
class LengthRatioStat:
    name: str = "length_ratio"

    def compute(self, records):
        ratios = []
        for r in records:
            s = max(1, len(r.get("source_text", "")))
            h = max(1, len(r.get("hypothesis", "")))
            ratios.append(h / s)
        if not ratios:
            return {f"{self.name}_mean": None, f"{self.name}_n": 0}
        return {f"{self.name}_mean": sum(ratios) / len(ratios),
                f"{self.name}_n": len(ratios)}


def load_dev_set(path: str | Path) -> list[dict]:
    """Accepts parquet/jsonl/csv with columns: source_text, [reference].
    Legacy `source` column is auto-renamed to `source_text`."""
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in (".jsonl", ".ndjson"):
        df = pd.read_json(p, lines=True)
    else:
        df = pd.read_csv(p)
    if "source_text" not in df.columns:
        if "source" in df.columns:
            df = df.rename(columns={"source": "source_text"})
        else:
            raise ValueError(f"dev set {p} missing 'source_text' column")
    return df.to_dict(orient="records")


class DevEvaluator:
    def __init__(self, metrics: Iterable[Metric]):
        self.metrics = list(metrics)

    def evaluate(self, *, dev_set: list[dict],
                 generate: Callable[[str], str],
                 ledger: Ledger | None = None,
                 round_id: int | None = None,
                 checkpoint_id: str | None = None) -> dict:
        records = []
        for ex in dev_set:
            src = ex["source_text"]
            hyp = generate(src)
            records.append({"source_text": src, "hypothesis": hyp,
                            "reference": ex.get("reference")})
        out: dict = {"checkpoint_id": checkpoint_id,
                     "n_examples": len(records)}
        for m in self.metrics:
            out.update(m.compute(records))
        if ledger is not None and round_id is not None:
            ledger.log({
                "event_type": "dev_eval", "round_id": round_id,
                "example_id": "_dev_set_", "checkpoint_id": checkpoint_id,
                **out,
            })
        return out


def build_evaluator_from_config(cfg: dict) -> DevEvaluator:
    """cfg = {metrics: [{type, ...}, ...]}"""
    type_map = {"cometkiwi": CometKiwiMetric, "chrf": ChrfMetric,
                "length_ratio": LengthRatioStat}
    metrics = []
    for m in cfg.get("metrics", []):
        params = {k: v for k, v in m.items() if k != "type"}
        metrics.append(type_map[m["type"]](**params))
    return DevEvaluator(metrics)
