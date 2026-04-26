"""YAML + Pydantic config for SCP runs.

Round-specific overrides: top-level fields are defaults; `round_overrides`
is a dict {round_id: partial_config} merged at runtime via `for_round`.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class StorageCfg(BaseModel):
    workdir: str
    cloud_uri: str | None = None
    write_jsonl_events: bool = True
    write_parquet_tables: bool = True
    sync_every_round: bool = False


class LoggingCfg(BaseModel):
    wandb_enabled: bool = False
    wandb_project: str | None = None
    weave_enabled: bool = False
    weave_project: str | None = None
    weave_trace_teacher_only: bool = True


class RecoveryCfg(BaseModel):
    skip_existing_teacher_calls: bool = True
    skip_existing_generations: bool = True
    atomic_writes: bool = True


class DecodingCfg(BaseModel):
    temperature: float = 0.0
    do_sample: bool = False
    top_p: float = 1.0
    top_k: int | None = None
    max_new_tokens: int = 512
    repetition_penalty: float | None = None
    seed: int | None = None


class StudentCfg(BaseModel):
    model_id: str
    decoding: DecodingCfg = Field(default_factory=DecodingCfg)


class ProbeCfg(BaseModel):
    base_model_id: str | None = None
    lora_id: str | None = None
    decoding: DecodingCfg = Field(default_factory=DecodingCfg)


class TeacherCfg(BaseModel):
    provider: str  # "openai" | "anthropic" | ...
    model: str
    prompt_version: str
    prompt_path: str
    system_prompt: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: int = 120
    max_retries: int = 3


class ModelsCfg(BaseModel):
    student: StudentCfg
    probe: ProbeCfg = Field(default_factory=ProbeCfg)
    teacher: TeacherCfg


class ScpConfig(BaseModel):
    run_id: str
    seed: int = 42
    storage: StorageCfg
    logging: LoggingCfg = Field(default_factory=LoggingCfg)
    recovery: RecoveryCfg = Field(default_factory=RecoveryCfg)
    models: ModelsCfg
    round_overrides: dict[int, dict[str, Any]] = Field(default_factory=dict)
    extras: dict[str, Any] = Field(default_factory=dict)

    def for_round(self, round_id: int) -> "ScpConfig":
        override = self.round_overrides.get(round_id)
        if not override:
            return self
        merged = _deep_merge(self.model_dump(), override)
        merged["round_overrides"] = {}
        return ScpConfig(**merged)

    def hash(self) -> str:
        payload = self.model_dump_json().encode()
        return hashlib.sha256(payload).hexdigest()


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path) -> ScpConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return ScpConfig(**raw)


def write_resolved_config(cfg: ScpConfig, run_dir: Path) -> None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(), sort_keys=False)
    )
    (run_dir / "config_hash.txt").write_text(cfg.hash())
