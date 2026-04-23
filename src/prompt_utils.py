"""
Prompt rendering helpers identical in behavior to scp_stage3_it/src/preprocess.py.

The stage-3 SFT model was trained with:
  - 5 alternating prompt templates (translation_dynamic_5.yaml)
  - deterministic template selection via stable_template_index(sample_key, seed)
  - response_prefix appended with a leading newline

To avoid a train/probe prompt distribution gap we reuse the same pipeline here.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import yaml

DEFAULT_LANG_NAME_MAP = {
    "en": "English",
    "ko": "Korean",
    "ja": "Japanese",
    "zh": "Chinese",
}
DEFAULT_LANG_LOCALE_MAP = {
    "en": "en-US",
    "ko": "ko-KR",
    "ja": "ja-JP",
    "zh": "zh-CN",
}


def load_prompt_templates(path: str | Path) -> list[str]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "templates" not in data:
        raise ValueError(f"Prompt config missing 'templates': {path}")
    templates = data["templates"]
    if not isinstance(templates, list) or not templates:
        raise ValueError(f"Prompt config 'templates' must be non-empty list: {path}")
    return [str(t) for t in templates]


def stable_template_index(sample_key: str, template_count: int, seed: int) -> int:
    if template_count <= 0:
        raise ValueError("template_count must be positive")
    key = f"{seed}|{sample_key}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return value % template_count


def resolve_lang_descriptors(
    src_iso: str,
    tgt_iso: str,
    lang_name_map: dict[str, str] | None = None,
    lang_locale_map: dict[str, str] | None = None,
) -> tuple[str, str, str, str]:
    name_map = lang_name_map or DEFAULT_LANG_NAME_MAP
    locale_map = lang_locale_map or DEFAULT_LANG_LOCALE_MAP
    s = str(src_iso or "source").strip().lower()
    t = str(tgt_iso or "target").strip().lower()
    return (
        name_map.get(s, s.upper()),
        name_map.get(t, t.upper()),
        locale_map.get(s, s),
        locale_map.get(t, t),
    )


def render_prompt(
    source_text: str,
    template: str,
    response_prefix: str,
    src_iso: str,
    tgt_iso: str,
    lang_name_map: dict[str, str] | None = None,
    lang_locale_map: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Return (prompt_full, prompt_prefix_with_response_marker)."""
    src_lang_name, tgt_lang_name, src_locale, tgt_locale = resolve_lang_descriptors(
        src_iso, tgt_iso, lang_name_map, lang_locale_map
    )
    prompt_body = template.format(
        src_lang_name=src_lang_name,
        tgt_lang_name=tgt_lang_name,
        src_locale=src_locale,
        tgt_locale=tgt_locale,
        src=source_text,
    )
    prefix = f"{prompt_body}\n{response_prefix}"
    return prompt_body, prefix


def render_for_sample(
    *,
    sample_key: str,
    source_text: str,
    templates: Iterable[str],
    response_prefix: str,
    template_seed: int,
    src_iso: str,
    tgt_iso: str,
    fixed_template_index: int | None = None,
) -> tuple[str, int]:
    """Return (prompt_prefix_ready_for_generation, template_index)."""
    templates = list(templates)
    if fixed_template_index is not None:
        idx = int(fixed_template_index)
        if idx < 0 or idx >= len(templates):
            raise ValueError(f"fixed_template_index out of range: {idx}")
    else:
        idx = stable_template_index(sample_key, len(templates), template_seed)
    _, prefix = render_prompt(
        source_text=source_text,
        template=templates[idx],
        response_prefix=response_prefix,
        src_iso=src_iso,
        tgt_iso=tgt_iso,
    )
    return prefix, idx


def clean_hypothesis(text: str, response_prefix: str) -> str:
    out = str(text or "").strip()
    rp = str(response_prefix or "").strip()
    if rp and rp in out:
        out = out.rsplit(rp, 1)[-1].strip()
    for p in (rp, "<KO>", "<ko>", "[KO]", "KO:", "번역:", "Translation:"):
        if p and out.startswith(p):
            out = out[len(p):].strip()
    return out
