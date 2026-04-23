"""
Quality estimation backends for the collapse signal.

Primary: reference-free COMET-Kiwi on (source, hypothesis).

Compatibility problem
---------------------
`unbabel-comet` 2.2.7 (latest) pins `transformers>=4.17,<5.0`. Our training
stack requires `transformers==5.5.4` (unsloth + stage3 SFT). Therefore we
CANNOT import COMET in the same Python process as unsloth.

Workaround: a subprocess scorer. The user sets `COMET_PYTHON` to the python
binary of a separate venv that has `unbabel-comet` installed. We write
(src, hyp) pairs to a temp JSONL, launch that interpreter with a short
driver script, and read back scores as JSON. No Python-level import of
COMET happens in the unsloth process.

Fallback: a chrF-vs-reference scorer (sentence-level) for the test.csv path,
and a dummy zero-scorer otherwise, so the notebook still runs end-to-end.

QE scorers expose:
    .name: str
    .score(pairs: list[tuple[str, str]]) -> list[float]

Reference scorers expose:
    .name: str
    .score(triplets: list[tuple[str, str, str]]) -> list[float]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import warnings
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Fallback scorers (always safe to construct).
# --------------------------------------------------------------------------- #


class _DummyQE:
    def __init__(self, reason: str) -> None:
        self.reason = reason
        self.name = f"dummy (reason: {reason})"

    def score(self, pairs):
        return [0.0 for _ in pairs]


class _ChrfRefQE:
    """Fallback QE proxy when COMET is unavailable AND references exist.

    Treats sentence-chrF(hyp, ref) as a proxy for QE. This is REFERENCE-BASED,
    so it only works on test.csv. It is NOT the same signal as COMET-Kiwi —
    a genuine reference-free QE is required for production SCP on monolingual
    corpora. Use this only to unblock PoC exploration.
    """

    def __init__(self, references_by_src: dict[str, str], word_order: int = 2) -> None:
        self.refs = references_by_src
        self.word_order = int(word_order)
        self.name = f"chrf_ref (word_order={word_order}) [fallback, not reference-free]"

    def score(self, pairs):
        try:
            import sacrebleu
        except ImportError:
            return [0.0 for _ in pairs]
        scores = []
        for src, hyp in pairs:
            ref = self.refs.get(src, "")
            if not ref or not hyp:
                scores.append(0.0)
                continue
            val = sacrebleu.sentence_chrf(hyp, [ref], word_order=self.word_order).score
            scores.append(float(val) / 100.0)  # rescale to 0..1 range
        return scores


# --------------------------------------------------------------------------- #
# Subprocess COMET driver.
# --------------------------------------------------------------------------- #


_COMET_DRIVER_SCRIPT = textwrap.dedent(
    """
    import json, sys
    from comet import download_model, load_from_checkpoint

    args = json.loads(sys.stdin.read())
    model_name = args["model_name"]
    fallback   = args.get("fallback_model_name")
    batch_size = int(args.get("batch_size", 16))
    gpus       = int(args.get("gpus", 1))
    mode       = args.get("mode", "qe")  # "qe" for src,mt or "ref" for src,mt,ref
    payload    = args["payload"]

    try:
        ckpt = download_model(model_name)
    except Exception as exc:
        if not fallback:
            raise
        sys.stderr.write(f"[comet-subprocess] primary {model_name} failed: {exc}; falling back to {fallback}\\n")
        ckpt = download_model(fallback)
        model_name = fallback

    model = load_from_checkpoint(ckpt)
    result = model.predict(payload, batch_size=batch_size, gpus=gpus, progress_bar=False)
    json.dump({"model_name": model_name, "scores": [float(x) for x in result["scores"]]}, sys.stdout)
    """
).strip()


def _run_comet_subprocess(
    python_bin: str,
    model_name: str,
    fallback_model_name: str | None,
    batch_size: int,
    gpus: int,
    mode: str,
    payload: list[dict[str, str]],
) -> dict[str, Any]:
    if not Path(python_bin).exists():
        raise FileNotFoundError(f"COMET_PYTHON does not exist: {python_bin}")

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(_COMET_DRIVER_SCRIPT)
        driver_path = fh.name

    args = {
        "model_name": model_name,
        "fallback_model_name": fallback_model_name,
        "batch_size": int(batch_size),
        "gpus": int(gpus),
        "mode": mode,
        "payload": payload,
    }
    try:
        proc = subprocess.run(
            [python_bin, driver_path],
            input=json.dumps(args),
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        try:
            os.unlink(driver_path)
        except OSError:
            pass

    if proc.returncode != 0:
        raise RuntimeError(
            f"COMET subprocess failed (rc={proc.returncode}).\n"
            f"stderr:\n{proc.stderr}\n"
            f"stdout:\n{proc.stdout[:500]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse COMET subprocess output: {exc}\nraw: {proc.stdout[:500]}")


class CometKiwiSubprocessScorer:
    """Reference-free QE via COMET-Kiwi running in a separate venv."""

    def __init__(
        self,
        python_bin: str,
        model_name: str,
        fallback_model_name: str | None = None,
        batch_size: int = 16,
        gpus: int = 1,
    ) -> None:
        self.python_bin = python_bin
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.batch_size = int(batch_size)
        self.gpus = int(gpus)
        self.name = f"comet_kiwi_subprocess ({model_name}, via {python_bin})"

    def score(self, pairs):
        if not pairs:
            return []
        payload = [{"src": str(src), "mt": str(mt)} for src, mt in pairs]
        result = _run_comet_subprocess(
            self.python_bin,
            self.model_name,
            self.fallback_model_name,
            self.batch_size,
            self.gpus,
            "qe",
            payload,
        )
        # The model actually used may differ if fallback was triggered.
        self.model_name = result.get("model_name", self.model_name)
        return [float(x) for x in result["scores"]]


class CometRefSubprocessScorer:
    """Reference-based COMET via the same subprocess mechanism."""

    def __init__(
        self,
        python_bin: str,
        model_name: str,
        batch_size: int = 16,
        gpus: int = 1,
    ) -> None:
        self.python_bin = python_bin
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.gpus = int(gpus)
        self.name = f"comet_ref_subprocess ({model_name}, via {python_bin})"

    def score(self, triplets):
        if not triplets:
            return []
        payload = [{"src": str(s), "mt": str(h), "ref": str(r)} for s, h, r in triplets]
        result = _run_comet_subprocess(
            self.python_bin,
            self.model_name,
            None,
            self.batch_size,
            self.gpus,
            "ref",
            payload,
        )
        return [float(x) for x in result["scores"]]


# --------------------------------------------------------------------------- #
# In-process COMET (kept for forward compatibility; will fail on transformers 5.x).
# --------------------------------------------------------------------------- #


class CometKiwiInProcessScorer:
    """Direct import. Only use if `transformers<5.0` or a future unbabel-comet
    bumps its pin. Will raise on 5.x."""

    def __init__(self, model_name: str, fallback_model_name: str | None, batch_size: int, gpus: int) -> None:
        from comet import download_model, load_from_checkpoint

        try:
            ckpt = download_model(model_name)
        except Exception:
            if not fallback_model_name:
                raise
            warnings.warn(f"Primary QE model unavailable; falling back to {fallback_model_name}")
            ckpt = download_model(fallback_model_name)
            model_name = fallback_model_name

        self.model = load_from_checkpoint(ckpt)
        self.name = f"comet_kiwi_inprocess ({model_name})"
        self.batch_size = int(batch_size)
        self.gpus = int(gpus)

    def score(self, pairs):
        if not pairs:
            return []
        payload = [{"src": s, "mt": m} for s, m in pairs]
        result = self.model.predict(payload, batch_size=self.batch_size, gpus=self.gpus, progress_bar=False)
        return [float(x) for x in result["scores"]]


# --------------------------------------------------------------------------- #
# Factories consumed by src/scp_a.py.
# --------------------------------------------------------------------------- #


def _comet_python_bin() -> str | None:
    val = os.environ.get("COMET_PYTHON", "").strip()
    return val or None


def build_qe_primary(cfg) -> Any:
    """Return a reference-free QE scorer.

    Resolution order:
      1. COMET_PYTHON is set -> CometKiwiSubprocessScorer
      2. transformers<5.0 in current env -> CometKiwiInProcessScorer
      3. dummy scorer (zeros) with a loud warning
    """
    backend = str(cfg.primary.backend).strip().lower()
    if backend != "comet_kiwi":
        return _DummyQE(f"unknown backend {backend!r}")

    model_name = str(cfg.primary.model_name)
    fallback_model_name = (
        str(cfg.primary.fallback_model_name) if cfg.primary.get("fallback_model_name") else None
    )
    batch_size = int(cfg.primary.batch_size)
    gpus = int(cfg.primary.gpus)

    comet_py = _comet_python_bin()
    if comet_py:
        print(f"[qe] using COMET subprocess backend via {comet_py}")
        return CometKiwiSubprocessScorer(
            python_bin=comet_py,
            model_name=model_name,
            fallback_model_name=fallback_model_name,
            batch_size=batch_size,
            gpus=gpus,
        )

    # No COMET_PYTHON: attempt in-process import. This will almost certainly
    # fail on transformers==5.5.4 (unbabel-comet pins transformers<5.0) but we
    # keep the path so a future unpinned release just works.
    try:
        import transformers as _tx

        major = int(str(_tx.__version__).split(".")[0])
    except Exception:
        major = 0
    if major >= 5:
        warnings.warn(
            "COMET_PYTHON is not set and transformers>=5 in this env; "
            "unbabel-comet is incompatible and cannot be imported in-process. "
            "Falling back to a dummy zero-scorer. See notebook bootstrap cell "
            "for the separate-venv setup."
        )
        return _DummyQE("COMET_PYTHON unset and transformers>=5")

    try:
        return CometKiwiInProcessScorer(model_name, fallback_model_name, batch_size, gpus)
    except ImportError as exc:
        return _DummyQE(f"comet not installed in-process: {exc}")
    except Exception as exc:
        return _DummyQE(f"comet init failed: {exc}")


def build_qe_reference(cfg) -> Any | None:
    """Return a reference-based scorer for analysis. None if disabled."""
    ref_cfg = cfg.get("reference_based")
    if not ref_cfg:
        return None
    comet_ref = ref_cfg.get("comet_ref")
    if not comet_ref or not bool(comet_ref.get("enabled", False)):
        return None

    model_name = str(comet_ref.model_name)
    batch_size = int(cfg.primary.batch_size)
    gpus = int(cfg.primary.gpus)
    comet_py = _comet_python_bin()
    if comet_py:
        return CometRefSubprocessScorer(comet_py, model_name, batch_size, gpus)
    # In-process path (same caveat as above).
    try:
        from comet import download_model, load_from_checkpoint

        class _InProc:
            def __init__(self):
                self.model = load_from_checkpoint(download_model(model_name))
                self.name = f"comet_ref_inprocess ({model_name})"

            def score(self, triplets):
                payload = [{"src": s, "mt": h, "ref": r} for s, h, r in triplets]
                return [float(x) for x in self.model.predict(payload, batch_size=batch_size, gpus=gpus, progress_bar=False)["scores"]]

        return _InProc()
    except Exception:
        return None


def build_chrf_fallback(records) -> _ChrfRefQE | None:
    """Build a reference-based chrF fallback from records that carry references."""
    refs = {
        r["source_text"]: r["reference_text"]
        for r in records
        if r.get("reference_text")
    }
    if not refs:
        return None
    return _ChrfRefQE(refs, word_order=2)
