"""
Shared utilities for the SCP stage4 PoC.

Reuses the same backend-resolution and dtype helpers as scp_stage3_it so that
identical base models load the same way in both training and probing.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNSLOTH_BACKEND_AUTO = "auto"
UNSLOTH_BACKEND_VISION = "vision"
UNSLOTH_BACKEND_LANGUAGE = "language"
UNSLOTH_BACKENDS = {
    UNSLOTH_BACKEND_AUTO,
    UNSLOTH_BACKEND_VISION,
    UNSLOTH_BACKEND_LANGUAGE,
}

_MULTIMODAL_MODEL_TYPES = {
    "gemma3",
    "gemma3n",
    "gemma4",
    "idefics",
    "idefics2",
    "idefics3",
    "llava",
    "mllama",
    "paligemma",
    "qwen2vl",
    "qwen25vl",
    "qwen3vl",
}


def resolve_workspace_path(path_like: str | Path) -> Path:
    path = Path(str(path_like))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_unsloth_backend(value: Any) -> str:
    text = str(value).strip().lower() if value is not None else UNSLOTH_BACKEND_AUTO
    aliases = {
        "": UNSLOTH_BACKEND_AUTO,
        "auto": UNSLOTH_BACKEND_AUTO,
        "text": UNSLOTH_BACKEND_LANGUAGE,
        "language": UNSLOTH_BACKEND_LANGUAGE,
        "llm": UNSLOTH_BACKEND_LANGUAGE,
        "vision": UNSLOTH_BACKEND_VISION,
        "vlm": UNSLOTH_BACKEND_VISION,
    }
    normalized = aliases.get(text, text)
    if normalized not in UNSLOTH_BACKENDS:
        raise ValueError(
            f"Unsupported model backend '{value}'. Expected one of: "
            f"{sorted(UNSLOTH_BACKENDS)}"
        )
    return normalized


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _normalize_model_type(model_type: Any) -> str:
    return str(model_type).strip().lower().replace("-", "").replace("_", "").replace(".", "")


def _is_multimodal_config(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False
    if _normalize_model_type(config.get("model_type")) in _MULTIMODAL_MODEL_TYPES:
        return True
    multimodal_keys = (
        "audio_config",
        "image_seq_length",
        "vision_config",
        "vision_tower",
    )
    if any(key in config for key in multimodal_keys):
        return True
    architectures = " ".join(str(name) for name in config.get("architectures") or [])
    if "ImageTextToText" in architectures or "ConditionalGeneration" in architectures:
        model_type = _normalize_model_type(config.get("model_type"))
        if model_type.startswith("gemma") or model_type in _MULTIMODAL_MODEL_TYPES:
            return True
    return False


def resolve_unsloth_backend_order(
    path_or_repo: str | Path,
    preferred_backend: Any = None,
    local_files_only: bool = False,
) -> tuple[list[str], str]:
    requested = normalize_unsloth_backend(preferred_backend)
    if requested != UNSLOTH_BACKEND_AUTO:
        fallback = (
            UNSLOTH_BACKEND_LANGUAGE
            if requested == UNSLOTH_BACKEND_VISION
            else UNSLOTH_BACKEND_VISION
        )
        return [requested, fallback], f"config backend={requested}"

    try:
        from huggingface_hub import hf_hub_download

        try:
            downloaded = hf_hub_download(
                repo_id=str(path_or_repo),
                filename="config.json",
                repo_type="model",
                local_files_only=bool(local_files_only),
            )
            if _is_multimodal_config(_load_json_if_exists(Path(downloaded))):
                return [UNSLOTH_BACKEND_VISION, UNSLOTH_BACKEND_LANGUAGE], "remote config indicates multimodal"
        except Exception:
            pass
    except Exception:
        pass

    return [UNSLOTH_BACKEND_LANGUAGE, UNSLOTH_BACKEND_VISION], "default text-first"


def resolve_torch_dtype(name: str | None):
    if name is None:
        return None
    key = str(name).strip().lower()
    if key in {"", "auto"}:
        return None
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{name}'")
    return mapping[key]


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    return [value]


def suppress_noisy_library_logs() -> None:
    for logger_name in (
        "httpx",
        "httpcore",
        "huggingface_hub",
        "transformers",
        "datasets",
    ):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def cfg_to_container(cfg: DictConfig | dict) -> dict:
    if isinstance(cfg, dict):
        return cfg
    return OmegaConf.to_container(cfg, resolve=True)


def pick_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def free_vram() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
