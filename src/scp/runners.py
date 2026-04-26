"""Runner Protocol interfaces for GPU-bound steps.

Concrete impls (transformers/peft/trl wrappers) live elsewhere and are wired
in at run time. These Protocols pin the contract so the orchestrator can be
written and tested with stubs first.
"""
from __future__ import annotations

from typing import Iterable, Protocol


class StudentGenRunner(Protocol):
    def generate(self, source: str, *, decoding: dict, checkpoint_id: str) -> dict:
        """Return {output, input_tokens, output_tokens, latency_ms}."""
        ...


class ProbeTrainer(Protocol):
    def train(self, *, base_checkpoint_id: str, data_slice: list[dict],
              probe_cfg: dict, round_id: int) -> dict:
        """Train an aggressive probe LoRA. Returns {probe_lora_id, path, training_meta}.
        The caller is responsible for discarding the LoRA after probe generation."""
        ...


class ProbeGenRunner(Protocol):
    def generate(self, source: str, *, decoding: dict,
                 base_checkpoint_id: str, probe_lora_id: str) -> dict:
        ...


class SftTrainer(Protocol):
    def train(self, *, base_checkpoint_id: str, dataset_path: str,
              training_cfg: dict, round_id: int) -> dict:
        """Returns {new_checkpoint_id, path, metrics}."""
        ...


class PrefTrainer(Protocol):
    def train(self, *, base_checkpoint_id: str, dataset_path: str,
              training_cfg: dict, stage_label: str) -> dict:
        ...


class Evaluator(Protocol):
    def evaluate(self, *, checkpoint_id: str, dev_set_path: str) -> dict:
        """Returns metric dict (chrf, comet, ...)."""
        ...


# --- Stubs for non-GPU smoke tests ---

class EchoStudent:
    def generate(self, source, *, decoding, checkpoint_id):
        return {"output": f"[stu:{checkpoint_id}] {source}",
                "input_tokens": len(source.split()),
                "output_tokens": len(source.split()),
                "latency_ms": 1}


class EchoProbe:
    def generate(self, source, *, decoding, base_checkpoint_id, probe_lora_id):
        # crude degradation: drop second half
        words = source.split()
        return {"output": f"[probe:{probe_lora_id}] " + " ".join(words[: max(1, len(words)//2)]),
                "input_tokens": len(words), "output_tokens": len(words)//2,
                "latency_ms": 1}


class StubProbeTrainer:
    def train(self, *, base_checkpoint_id, data_slice, probe_cfg, round_id):
        return {"probe_lora_id": f"probe_r{round_id}", "path": None,
                "training_meta": {"base": base_checkpoint_id,
                                  "n_samples": len(data_slice),
                                  "config": probe_cfg}}
