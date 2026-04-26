"""Bootstrap a separate venv for unbabel-comet (mirrors notebook cell 1.5).

Why
---
COMET pins transformers<5; training stack uses transformers>=5. They cannot
share a Python process. We create ~/.venvs/comet (or a custom path), install
unbabel-comet into it, and export $COMET_PYTHON so src/qe.py and
scp.scoring.CometKiwiQE pick it up via subprocess.

Idempotent: reuses existing venv if Python binary already present.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


DEFAULT_VENV = Path.home() / ".venvs" / "comet"


def ensure_comet_venv(
    venv_dir: str | Path = DEFAULT_VENV,
    comet_spec: str = "unbabel-comet>=2.2.7",
    install: bool = True,
    extra_packages: list[str] | None = None,
    export_env: bool = True,
) -> str | None:
    """Create venv if missing, install COMET, return path to its python.

    Set install=False to only verify/export without re-installing.
    Returns None if venv missing and install=False.
    """
    venv_dir = Path(venv_dir).expanduser()
    py = venv_dir / "bin" / "python"
    if not py.exists():
        if not install:
            return None
        print(f"[qe-setup] creating venv: {venv_dir}")
        subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
        subprocess.check_call([str(py), "-m", "pip", "install", "-q", "-U", "pip"])
        pkgs = [comet_spec] + (extra_packages or [])
        subprocess.check_call([str(py), "-m", "pip", "install", "-q", *pkgs])
    if export_env:
        os.environ["COMET_PYTHON"] = str(py)
        print(f"[qe-setup] COMET_PYTHON = {py}")
    return str(py)


def comet_python() -> str | None:
    p = os.environ.get("COMET_PYTHON", "").strip()
    return p or None


def smoke_check(python_bin: str | None = None) -> bool:
    """Verify the COMET venv can import comet."""
    py = python_bin or comet_python()
    if not py or not Path(py).exists():
        return False
    proc = subprocess.run(
        [py, "-c", "import comet; print(comet.__version__)"],
        capture_output=True, text=True,
    )
    if proc.returncode == 0:
        print(f"[qe-setup] comet ok: {proc.stdout.strip()}")
        return True
    print(f"[qe-setup] comet import failed:\n{proc.stderr}")
    return False
