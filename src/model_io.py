"""
Model / tokenizer loading + probe-LoRA lifecycle helpers.

Design notes:
  - Uses Unsloth's FastLanguageModel / FastVisionModel, matching stage3_it.
  - Probe LoRA is attached via get_peft_model and MUST be removable after each
    probe run so the base model is observed again. We discard the peft-wrapped
    module and reload the base when the caller asks to reset.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch

from .common import (
    UNSLOTH_BACKEND_LANGUAGE,
    UNSLOTH_BACKEND_VISION,
    free_vram,
    resolve_unsloth_backend_order,
)


def load_base_model(
    model_name: str,
    max_seq_length: int,
    *,
    preferred_backend: str = "auto",
    load_in_4bit: bool = False,
    local_files_only: bool = False,
) -> tuple[Any, Any, Any, str]:
    """Load base model via Unsloth. Returns (model, tokenizer, ModelClass, backend)."""
    from unsloth import FastLanguageModel, FastVisionModel

    candidates, reason = resolve_unsloth_backend_order(
        model_name, preferred_backend, local_files_only
    )
    print(f"[model_io] backend preference: {candidates[0]} (reason: {reason})")

    errors: list[str] = []
    for backend in candidates:
        ModelClass = FastVisionModel if backend == UNSLOTH_BACKEND_VISION else FastLanguageModel
        kwargs: dict[str, Any] = {
            "model_name": model_name,
            "max_seq_length": int(max_seq_length),
            "dtype": None,
            "load_in_4bit": bool(load_in_4bit),
        }
        sig_params = inspect.signature(ModelClass.from_pretrained).parameters
        if "full_finetuning" in sig_params:
            kwargs["full_finetuning"] = False
        try:
            model, tokenizer = ModelClass.from_pretrained(**kwargs)
            print(f"[model_io] loaded ({backend}): {model_name}")
            return model, tokenizer, ModelClass, backend
        except Exception as exc:
            errors.append(f"{backend}: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        "Failed to load model with any Unsloth backend.\n"
        f"model={model_name}\nerrors:\n" + "\n".join(errors)
    )


def prepare_tokenizer_for_generation(tokenizer) -> Any:
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def prepare_tokenizer_for_training(tokenizer) -> Any:
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def attach_probe_lora(
    model,
    ModelClass,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: list[str],
    bias: str = "none",
    use_rslora: bool = False,
    use_dora: bool = False,
    gradient_checkpointing: bool = True,
    seed: int = 42,
    vision_backend: bool = False,
):
    """Wrap the base model with a fresh PEFT LoRA adapter configured for probing."""
    kwargs: dict[str, Any] = {
        "r": int(rank),
        "lora_alpha": int(alpha),
        "lora_dropout": float(dropout),
        "target_modules": [str(m) for m in target_modules],
        "bias": str(bias),
        "use_rslora": bool(use_rslora),
        "use_dora": bool(use_dora),
        "use_gradient_checkpointing": "unsloth" if gradient_checkpointing else True,
        "random_state": int(seed),
    }
    if vision_backend:
        kwargs.update(
            {
                "finetune_vision_layers": False,
                "finetune_language_layers": True,
                "finetune_attention_modules": True,
                "finetune_mlp_modules": True,
            }
        )
    return ModelClass.get_peft_model(model, **kwargs)


def set_inference_mode(model, backend: str) -> None:
    """Enable KV cache and flip to inference routines."""
    if hasattr(model, "config"):
        model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True
    if backend == UNSLOTH_BACKEND_LANGUAGE:
        try:
            from unsloth import FastLanguageModel

            FastLanguageModel.for_inference(model)
        except Exception as exc:
            print(f"[model_io] for_inference failed: {type(exc).__name__}: {exc}")
    model.eval()


def set_training_mode(model, backend: str) -> None:
    if hasattr(model, "config"):
        model.config.use_cache = False
    if backend == UNSLOTH_BACKEND_LANGUAGE:
        try:
            from unsloth import FastLanguageModel

            FastLanguageModel.for_training(model)
        except Exception as exc:
            print(f"[model_io] for_training failed: {type(exc).__name__}: {exc}")
    model.train()


def reset_base_model(model_name: str, max_seq_length: int, **kwargs):
    """Hard reset: reload the base model fresh. Use between probe sweep runs."""
    free_vram()
    return load_base_model(model_name, max_seq_length, **kwargs)
