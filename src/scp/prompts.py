"""Prompt loading + content hashing.

Use prompt_version as a human label, prompt_hash as the integrity key.
Same version with different content -> different hash -> different cache entry.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def load_prompt(path: str | Path) -> str:
    return Path(path).read_text()


def hash_prompt(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def copy_prompt_to_run(src: str | Path, run_dir: str | Path) -> Path:
    src = Path(src)
    dst_dir = Path(run_dir) / "prompts"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst
