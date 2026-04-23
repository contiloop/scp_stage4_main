"""
End-to-end orchestrator for SCP Stage A (probe + measure).

Usage pattern in the notebook:

    ctx = build_context(cfg)
    t_before = generate_self_translations(ctx, ctx.probe_records)
    q_before = score_qe(ctx, ctx.probe_records, t_before)

    # attach probe LoRA, train on (records, t_before), then regenerate
    attach_and_train_probe(ctx, t_before)
    t_after = generate_self_translations(ctx, ctx.probe_records)
    q_after = score_qe(ctx, ctx.probe_records, t_after)

    delta_qe = [b - a for b, a in zip(q_before, q_after)]

The caller is responsible for resetting the model (reload_base) between
different sweep combos so each probe run is independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from . import data as data_mod
from .generate import generate_translations
from .model_io import (
    attach_probe_lora,
    load_base_model,
    prepare_tokenizer_for_generation,
    prepare_tokenizer_for_training,
    set_inference_mode,
    set_training_mode,
)
from .probe import build_probe_samples, train_probe
from .qe import build_qe_primary, build_qe_reference


@dataclass
class RunContext:
    cfg: DictConfig
    model: Any
    tokenizer: Any
    ModelClass: Any
    backend: str
    prompt_templates: list[str]
    response_prefix: str
    records: list[dict]
    probe_records: list[dict]
    qe_primary: Any = None
    qe_reference: Any | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)


def _subset_for_probe(records: list[dict], size: int | None, seed: int) -> list[dict]:
    if size is None or int(size) <= 0 or int(size) >= len(records):
        return list(records)
    import random

    rng = random.Random(seed)
    return rng.sample(records, int(size))


def build_context(cfg: DictConfig, *, load_qe: bool = True) -> RunContext:
    templates = cfg.prompts.get("templates")
    if not templates:
        raise RuntimeError("cfg.prompts.templates is empty; check prompts config.")
    prompt_templates = [str(t) for t in templates]
    response_prefix = str(cfg.probe.response_prefix)

    records = data_mod.load_records(cfg.data)
    probe_records = _subset_for_probe(
        records,
        size=cfg.probe.get("probe_subset_size"),
        seed=int(cfg.probe.get("probe_subset_seed", cfg.seed)),
    )

    model, tokenizer, ModelClass, backend = load_base_model(
        model_name=str(cfg.model.pretrained_model_name_or_path),
        max_seq_length=int(cfg.model.max_seq_length),
        preferred_backend=str(cfg.model.get("backend", "auto")),
        load_in_4bit=bool(cfg.model.quantization.load_in_4bit),
        local_files_only=bool(cfg.model.get("local_files_only", False)),
    )

    qe_primary = build_qe_primary(cfg.qe) if load_qe else None
    qe_reference = build_qe_reference(cfg.qe) if load_qe else None

    return RunContext(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        ModelClass=ModelClass,
        backend=backend,
        prompt_templates=prompt_templates,
        response_prefix=response_prefix,
        records=records,
        probe_records=probe_records,
        qe_primary=qe_primary,
        qe_reference=qe_reference,
    )


def generate_self_translations(
    ctx: RunContext,
    records: list[dict],
) -> list[dict]:
    prepare_tokenizer_for_generation(ctx.tokenizer)
    set_inference_mode(ctx.model, ctx.backend)
    return generate_translations(
        model=ctx.model,
        tokenizer=ctx.tokenizer,
        records=records,
        prompt_templates=ctx.prompt_templates,
        response_prefix=ctx.response_prefix,
        template_seed=int(ctx.cfg.seed),
        fixed_template_index=None,
        gen_cfg=ctx.cfg.generation,
    )


def score_qe_primary(ctx: RunContext, records: list[dict], translations: list[dict]) -> list[float]:
    pairs = [(r["source_text"], t["hypothesis"]) for r, t in zip(records, translations)]
    return ctx.qe_primary.score(pairs)


def score_qe_reference(
    ctx: RunContext, records: list[dict], translations: list[dict]
) -> list[float] | None:
    if ctx.qe_reference is None:
        return None
    triplets = [
        (r["source_text"], t["hypothesis"], r.get("reference_text", ""))
        for r, t in zip(records, translations)
        if r.get("reference_text")
    ]
    if not triplets:
        return None
    return ctx.qe_reference.score(triplets)


def attach_and_train_probe(
    ctx: RunContext,
    self_translations: list[dict],
    *,
    probe_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach a fresh probe LoRA, train on (probe_records, self_translations), return train metrics.

    `probe_overrides` lets the sweep loop override rank/lr/epochs per combo without
    mutating the original config.
    """
    pcfg = ctx.cfg.probe
    overrides = probe_overrides or {}
    rank = int(overrides.get("rank", pcfg.rank))
    alpha = int(overrides.get("alpha", pcfg.alpha))
    dropout = float(overrides.get("dropout", pcfg.dropout))
    learning_rate = float(overrides.get("learning_rate", pcfg.learning_rate))
    num_train_epochs = int(overrides.get("num_train_epochs", pcfg.num_train_epochs))

    ctx.model = attach_probe_lora(
        ctx.model,
        ctx.ModelClass,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=list(pcfg.target_modules),
        bias=str(pcfg.bias),
        use_rslora=bool(pcfg.use_rslora),
        use_dora=bool(pcfg.use_dora),
        gradient_checkpointing=bool(ctx.cfg.model.gradient_checkpointing),
        seed=int(ctx.cfg.seed),
        vision_backend=(ctx.backend == "vision"),
    )

    prepare_tokenizer_for_training(ctx.tokenizer)
    set_training_mode(ctx.model, ctx.backend)

    samples = build_probe_samples(
        tokenizer=ctx.tokenizer,
        records=ctx.probe_records,
        self_translations=[t["hypothesis"] for t in self_translations],
        prompt_templates=ctx.prompt_templates,
        response_prefix=ctx.response_prefix,
        template_seed=int(ctx.cfg.seed),
        fixed_template_index=None,
        max_length=int(ctx.cfg.model.max_seq_length),
        add_eos_to_target=bool(pcfg.add_eos_to_target),
        drop_overlength=True,
    )

    metrics = train_probe(
        model=ctx.model,
        tokenizer=ctx.tokenizer,
        samples=samples,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=int(pcfg.per_device_train_batch_size),
        gradient_accumulation_steps=int(pcfg.gradient_accumulation_steps),
        max_grad_norm=float(pcfg.max_grad_norm),
        warmup_ratio=float(pcfg.warmup_ratio),
        lr_scheduler_type=str(pcfg.lr_scheduler_type),
        weight_decay=float(pcfg.weight_decay),
    )
    metrics.update(
        {
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "num_train_epochs": num_train_epochs,
            "n_samples": len(samples),
        }
    )
    return metrics


def reload_base_model(ctx: RunContext) -> RunContext:
    """Reset model to base; required between independent probe combos."""
    from .common import free_vram

    del ctx.model
    free_vram()
    model, tokenizer, ModelClass, backend = load_base_model(
        model_name=str(ctx.cfg.model.pretrained_model_name_or_path),
        max_seq_length=int(ctx.cfg.model.max_seq_length),
        preferred_backend=str(ctx.cfg.model.get("backend", "auto")),
        load_in_4bit=bool(ctx.cfg.model.quantization.load_in_4bit),
        local_files_only=bool(ctx.cfg.model.get("local_files_only", False)),
    )
    ctx.model = model
    ctx.tokenizer = tokenizer
    ctx.ModelClass = ModelClass
    ctx.backend = backend
    return ctx
