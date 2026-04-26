"""Real GPU runners: unsloth + peft + trl adapters.

Implements the Protocols in scp.runners. All heavyweight imports are lazy
(inside __init__ / methods) so the orchestrator + Echo stubs remain importable
on machines without unsloth/torch/trl.

Mirrors scp_stage3_it/src/train.py patterns:
- unsloth.FastLanguageModel + full_finetuning kwarg
- peft LoRA via FastLanguageModel.get_peft_model
- trl SFTTrainer for SFT (and probe overfit)
- trl DPOTrainer for preference

Adapter style: keep one model instance loaded; swap LoRA adapters when the
caller asks for a different checkpoint_id. Probe LoRA is loaded fresh each
round and its weights are deleted from disk after step_probe_gen.
"""
from __future__ import annotations

import os
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .orchestrator import Runners


# --------------------------- shared state -------------------------------- #

@dataclass
class RealRunnerCfg:
    """Hyperparameters injected into the unsloth backend.

    Anything that varies per-experiment lives here so callers don't have to
    rebuild the runners. Per-call decoding/training overrides still come
    through the runner method kwargs.
    """
    base_model_id: str
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    dtype: str | None = None  # let unsloth pick
    device_map: str = "auto"
    checkpoints_dir: str = "./scp_runs/_shared/checkpoints"
    probe_dir: str = "./scp_runs/_shared/probe"
    sft_lora: dict = field(default_factory=lambda: {
        "r": 32, "alpha": 64, "dropout": 0.05, "use_rslora": True, "use_dora": False,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "bias": "none",
    })
    probe_lora: dict = field(default_factory=lambda: {
        "r": 16, "alpha": 32, "dropout": 0.0, "use_rslora": False, "use_dora": False,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "bias": "none",
    })


class _UnslothBackend:
    """Single shared backend instance: loads the base model once, then swaps
    LoRA adapters via load_adapter / set_adapter."""

    def __init__(self, cfg: RealRunnerCfg):
        self.cfg = cfg
        self._model = None
        self._tokenizer = None
        self._FastLanguageModel = None
        self._loaded_adapters: dict[str, str] = {}  # adapter_id -> path
        self._active_adapter: str | None = None

    def _ensure(self):
        if self._model is not None:
            return
        import unsloth  # noqa: F401  (must precede transformers for kernel patches)
        from unsloth import FastLanguageModel
        self._FastLanguageModel = FastLanguageModel
        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.cfg.base_model_id,
            max_seq_length=self.cfg.max_seq_length,
            dtype=self.cfg.dtype,
            load_in_4bit=self.cfg.load_in_4bit,
        )

    def model_for_inference(self, adapter_id: str | None):
        self._ensure()
        if adapter_id and adapter_id != "M0":
            self._activate_adapter(adapter_id)
        else:
            self._deactivate_adapter()
        self._FastLanguageModel.for_inference(self._model)
        return self._model, self._tokenizer

    def model_for_training(self, base_adapter_id: str | None):
        self._ensure()
        if base_adapter_id and base_adapter_id != "M0":
            self._activate_adapter(base_adapter_id)
        else:
            self._deactivate_adapter()
        self._FastLanguageModel.for_training(self._model)
        return self._model, self._tokenizer

    def _activate_adapter(self, adapter_id: str):
        if adapter_id == self._active_adapter:
            return
        path = self._loaded_adapters.get(adapter_id)
        if path is None:
            raise KeyError(
                f"adapter {adapter_id!r} not registered with backend; "
                "call register_adapter(adapter_id, path) first")
        if hasattr(self._model, "load_adapter") and adapter_id not in getattr(self._model, "peft_config", {}):
            self._model.load_adapter(path, adapter_name=adapter_id)
        self._model.set_adapter(adapter_id)
        self._active_adapter = adapter_id

    def _deactivate_adapter(self):
        if self._active_adapter is None:
            return
        if hasattr(self._model, "disable_adapters"):
            self._model.disable_adapters()
        self._active_adapter = None

    def register_adapter(self, adapter_id: str, path: str) -> None:
        self._loaded_adapters[adapter_id] = path

    def forget_adapter(self, adapter_id: str) -> None:
        if hasattr(self._model, "delete_adapter"):
            try:
                self._model.delete_adapter(adapter_id)
            except Exception:
                pass
        self._loaded_adapters.pop(adapter_id, None)
        if self._active_adapter == adapter_id:
            self._active_adapter = None


# ----------------------------- generation -------------------------------- #

class UnslothStudent:
    def __init__(self, backend: _UnslothBackend):
        self.backend = backend

    def generate(self, source: str, *, decoding: dict, checkpoint_id: str) -> dict:
        model, tok = self.backend.model_for_inference(checkpoint_id)
        return _generate(model, tok, source, decoding)


class UnslothProbeGen:
    def __init__(self, backend: _UnslothBackend):
        self.backend = backend

    def generate(self, source: str, *, decoding: dict,
                 base_checkpoint_id: str, probe_lora_id: str) -> dict:
        # base_checkpoint_id intentionally ignored: probe runs on the *base*
        # model + probe LoRA only, never on top of the main SFT adapter.
        # That matches the "collapse measurement must not be polluted by main
        # LoRA" rule from the design discussion.
        model, tok = self.backend.model_for_inference(probe_lora_id)
        return _generate(model, tok, source, decoding)


def _generate(model, tokenizer, source: str, decoding: dict) -> dict:
    import torch
    inputs = tokenizer(source, return_tensors="pt", truncation=True).to(model.device)
    gen_kwargs = {
        "max_new_tokens": int(decoding.get("max_new_tokens", 512)),
        "do_sample": bool(decoding.get("do_sample", False)),
        "temperature": float(decoding.get("temperature", 1.0)),
        "top_p": float(decoding.get("top_p", 1.0)),
    }
    if decoding.get("top_k") is not None:
        gen_kwargs["top_k"] = int(decoding["top_k"])
    if decoding.get("repetition_penalty") is not None:
        gen_kwargs["repetition_penalty"] = float(decoding["repetition_penalty"])
    if decoding.get("seed") is not None:
        torch.manual_seed(int(decoding["seed"]))
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    text = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return {
        "output": text,
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "output_tokens": int(out.shape[1] - inputs["input_ids"].shape[1]),
        "latency_ms": int((time.time() - t0) * 1000),
    }


# ------------------------------ training --------------------------------- #

class UnslothProbeTrainer:
    """Aggressive overfit on the round's data slice -> a probe LoRA that
    exposes which examples collapse easily under that update."""

    def __init__(self, backend: _UnslothBackend):
        self.backend = backend

    def train(self, *, base_checkpoint_id: str, data_slice: list[dict],
              probe_cfg: dict, round_id: int) -> dict:
        from datasets import Dataset
        from trl import SFTConfig, SFTTrainer

        model, tok = self.backend.model_for_training(base_checkpoint_id)
        lora_kwargs = dict(self.backend.cfg.probe_lora)
        lora_kwargs.update({k: v for k, v in probe_cfg.items()
                             if k in ("r", "alpha", "dropout",
                                      "target_modules", "use_rslora", "use_dora")})
        # apply LoRA on top of base for probe
        FastLanguageModel = self.backend._FastLanguageModel
        peft_model = FastLanguageModel.get_peft_model(
            model,
            r=lora_kwargs["r"], lora_alpha=lora_kwargs["alpha"],
            lora_dropout=lora_kwargs["dropout"],
            target_modules=lora_kwargs["target_modules"],
            bias=lora_kwargs["bias"],
            use_rslora=lora_kwargs.get("use_rslora", False),
            use_dora=lora_kwargs.get("use_dora", False),
            use_gradient_checkpointing="unsloth",
            random_state=int(probe_cfg.get("seed", 42)),
        )

        ds = Dataset.from_list([{"text": ex.get("source") or ex.get("text", "")}
                                 for ex in data_slice])
        out_dir = Path(self.backend.cfg.probe_dir) / f"round_{round_id:03d}_{uuid.uuid4().hex[:6]}"
        out_dir.mkdir(parents=True, exist_ok=True)
        sft = SFTTrainer(
            model=peft_model,
            args=SFTConfig(
                output_dir=str(out_dir),
                per_device_train_batch_size=int(probe_cfg.get("batch_size", 4)),
                gradient_accumulation_steps=int(probe_cfg.get("grad_accum", 1)),
                num_train_epochs=float(probe_cfg.get("epochs", 1)),
                max_steps=int(probe_cfg.get("steps", 50)),
                learning_rate=float(probe_cfg.get("lr", 1e-3)),
                logging_steps=10,
                save_strategy="no",
                report_to="none",
                bf16=bool(probe_cfg.get("bf16", True)),
                seed=int(probe_cfg.get("seed", 42)),
                dataset_text_field="text",
                packing=False,
            ),
            train_dataset=ds,
            processing_class=getattr(tok, "tokenizer", tok),
        )
        sft.train()
        adapter_id = f"probe_r{round_id}"
        peft_model.save_pretrained(str(out_dir))
        self.backend.register_adapter(adapter_id, str(out_dir))
        return {
            "probe_lora_id": adapter_id,
            "path": str(out_dir),
            "training_meta": {"base": base_checkpoint_id,
                              "n_samples": len(data_slice),
                              "config": lora_kwargs,
                              "steps": int(probe_cfg.get("steps", 50))},
        }

    def discard(self, probe_lora_id: str, path: str | None) -> None:
        self.backend.forget_adapter(probe_lora_id)
        if path:
            shutil.rmtree(path, ignore_errors=True)


class UnslothSftTrainer:
    """Per-round SFT: trains a fresh LoRA on top of base (or on top of the
    previous main adapter if `stack_on_previous=True`) using the SFT dataset
    parquet built by step_build_sft."""

    def __init__(self, backend: _UnslothBackend, stack_on_previous: bool = False):
        self.backend = backend
        self.stack_on_previous = stack_on_previous

    def train(self, *, base_checkpoint_id: str, dataset_path: str,
              training_cfg: dict, round_id: int) -> dict:
        from datasets import Dataset
        import pandas as pd
        from trl import SFTConfig, SFTTrainer

        df = pd.read_parquet(dataset_path)
        if df.empty:
            return {"new_checkpoint_id": base_checkpoint_id,
                    "path": None, "metrics": {"skipped": "empty dataset"}}

        prompt_template = training_cfg.get(
            "prompt_template", "{source}\n\n번역:\n{target}")
        ds = Dataset.from_list([
            {"text": prompt_template.format(
                source=row["source_text"], target=row["target_text"])}
            for _, row in df.iterrows()
        ])

        base = base_checkpoint_id if self.stack_on_previous else None
        model, tok = self.backend.model_for_training(base)
        FastLanguageModel = self.backend._FastLanguageModel
        lora = self.backend.cfg.sft_lora
        peft_model = FastLanguageModel.get_peft_model(
            model,
            r=int(training_cfg.get("lora_r", lora["r"])),
            lora_alpha=int(training_cfg.get("lora_alpha", lora["alpha"])),
            lora_dropout=float(training_cfg.get("lora_dropout", lora["dropout"])),
            target_modules=lora["target_modules"], bias=lora["bias"],
            use_rslora=lora.get("use_rslora", False),
            use_dora=lora.get("use_dora", False),
            use_gradient_checkpointing="unsloth",
            random_state=int(training_cfg.get("seed", 42)),
        )

        adapter_id = f"M{round_id}_sft"
        out_dir = Path(self.backend.cfg.checkpoints_dir) / adapter_id
        out_dir.mkdir(parents=True, exist_ok=True)
        sft = SFTTrainer(
            model=peft_model,
            args=SFTConfig(
                output_dir=str(out_dir),
                per_device_train_batch_size=int(training_cfg.get("batch_size", 4)),
                gradient_accumulation_steps=int(training_cfg.get("grad_accum", 4)),
                num_train_epochs=float(training_cfg.get("epochs", 1)),
                max_steps=int(training_cfg.get("max_steps", -1)),
                learning_rate=float(training_cfg.get("lr", 2e-4)),
                warmup_ratio=float(training_cfg.get("warmup_ratio", 0.03)),
                lr_scheduler_type=str(training_cfg.get("lr_scheduler", "cosine")),
                logging_steps=int(training_cfg.get("logging_steps", 10)),
                save_strategy="no",
                report_to=training_cfg.get("report_to", "none"),
                bf16=bool(training_cfg.get("bf16", True)),
                seed=int(training_cfg.get("seed", 42)),
                dataset_text_field="text",
                packing=False,
            ),
            train_dataset=ds,
            processing_class=getattr(tok, "tokenizer", tok),
        )
        result = sft.train()
        peft_model.save_pretrained(str(out_dir))
        self.backend.register_adapter(adapter_id, str(out_dir))
        return {
            "new_checkpoint_id": adapter_id,
            "path": str(out_dir),
            "metrics": dict(result.metrics) if hasattr(result, "metrics") else {},
        }


class UnslothPrefTrainer:
    """DPO-style preference training. Inputs the pref parquet built by
    step_build_pref (chosen/rejected/source_text columns)."""

    def __init__(self, backend: _UnslothBackend):
        self.backend = backend

    def train(self, *, base_checkpoint_id: str, dataset_path: str,
              training_cfg: dict, stage_label: str) -> dict:
        from datasets import Dataset
        import pandas as pd
        from trl import DPOConfig, DPOTrainer

        df = pd.read_parquet(dataset_path)
        if "eligible_for_training" in df.columns:
            df = df[df["eligible_for_training"].astype(bool)]
        if df.empty:
            return {"new_checkpoint_id": base_checkpoint_id,
                    "path": None, "metrics": {"skipped": "empty pref dataset"}}

        ds = Dataset.from_list([
            {"prompt": row["source_text"],
             "chosen": row["chosen"], "rejected": row["rejected"]}
            for _, row in df.iterrows()
        ])

        model, tok = self.backend.model_for_training(base_checkpoint_id)
        adapter_id = f"{base_checkpoint_id}__{stage_label}"
        out_dir = Path(self.backend.cfg.checkpoints_dir) / adapter_id
        out_dir.mkdir(parents=True, exist_ok=True)

        dpo = DPOTrainer(
            model=model,
            ref_model=None,  # peft + DPO uses base as reference automatically
            args=DPOConfig(
                output_dir=str(out_dir),
                per_device_train_batch_size=int(training_cfg.get("batch_size", 2)),
                gradient_accumulation_steps=int(training_cfg.get("grad_accum", 8)),
                num_train_epochs=float(training_cfg.get("epochs", 1)),
                max_steps=int(training_cfg.get("max_steps", -1)),
                learning_rate=float(training_cfg.get("lr", 5e-6)),
                beta=float(training_cfg.get("beta", 0.1)),
                logging_steps=int(training_cfg.get("logging_steps", 10)),
                save_strategy="no",
                report_to=training_cfg.get("report_to", "none"),
                bf16=bool(training_cfg.get("bf16", True)),
                seed=int(training_cfg.get("seed", 42)),
            ),
            train_dataset=ds,
            processing_class=getattr(tok, "tokenizer", tok),
        )
        result = dpo.train()
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(str(out_dir))
        self.backend.register_adapter(adapter_id, str(out_dir))
        return {
            "new_checkpoint_id": adapter_id,
            "path": str(out_dir),
            "metrics": dict(result.metrics) if hasattr(result, "metrics") else {},
        }


# ------------------------------ factory ---------------------------------- #

def make_runners(real_cfg: RealRunnerCfg) -> Runners:
    """Build a Runners bundle including SFT/Pref trainers wired to the same
    shared backend so the base model is loaded only once per process."""
    backend = _UnslothBackend(real_cfg)
    return Runners(
        student=UnslothStudent(backend),
        probe_trainer=UnslothProbeTrainer(backend),
        probe_gen=UnslothProbeGen(backend),
        sft_trainer=UnslothSftTrainer(backend),
        pref_trainer=UnslothPrefTrainer(backend),
    )


def make_runners_from_dict(d: dict) -> Runners:
    cfg = RealRunnerCfg(**{k: v for k, v in d.items()
                            if k in RealRunnerCfg.__dataclass_fields__})
    return make_runners(cfg)
