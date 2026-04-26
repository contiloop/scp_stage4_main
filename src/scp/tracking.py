"""W&B + Weave adapters. Both are optional and fail-soft.

Design
------
- Source of truth = local JSONL/Parquet ledger. These adapters are
  observability sinks; if they fail or are disabled, training continues.
- Weave is used ONLY to trace teacher API calls (per project policy in the
  earlier design discussion). Per-event tracing would be too noisy/expensive
  on Vast.
- W&B is used for round summary metrics + (optional) artifact sync.
"""
from __future__ import annotations

from typing import Any, Callable


# ------------------------------ Weave ----------------------------------- #

def make_weave_tracer(project: str | None, *, enabled: bool = True
                      ) -> Callable[[dict, dict], None] | None:
    """Return a tracer callable matching TeacherClient(weave_tracer=...).

    Returns None when disabled or weave isn't importable. The TeacherClient
    already swallows tracer exceptions, so callers don't need extra try/except.
    """
    if not enabled or not project:
        return None
    try:
        import weave  # type: ignore
    except Exception as e:
        import warnings
        warnings.warn(f"weave not available ({e}); tracing disabled")
        return None
    weave.init(project)

    @weave.op()
    def trace_teacher_call(request: dict, response: dict) -> dict:
        return response

    def tracer(request: dict, response: dict) -> None:
        try:
            trace_teacher_call(request, response)
        except Exception:
            pass
    return tracer


# ------------------------------- W&B ------------------------------------ #

class WandbReporter:
    """Round-summary + arbitrary metric sink. No-op when disabled."""

    def __init__(self, *, enabled: bool, project: str | None,
                 run_id: str | None, config: dict | None = None,
                 group: str | None = None):
        self.run = None
        if not enabled:
            return
        try:
            import wandb  # type: ignore
        except Exception as e:
            import warnings
            warnings.warn(f"wandb not available ({e}); reporting disabled")
            return
        self._wandb = wandb
        self.run = wandb.init(
            project=project, name=run_id, group=group,
            config=config or {}, reinit=True, resume="allow",
        )

    def log_round_summary(self, round_id: int, summary: dict) -> None:
        if self.run is None:
            return
        flat = {f"round/{k}": v for k, v in summary.items()
                if isinstance(v, (int, float))}
        flat["round_id"] = round_id
        self.run.log(flat, step=round_id)

    def log_metrics(self, round_id: int, metrics: dict, prefix: str = "metric") -> None:
        if self.run is None:
            return
        self.run.log(
            {f"{prefix}/{k}": v for k, v in metrics.items()
             if isinstance(v, (int, float))},
            step=round_id,
        )

    def log_artifact(self, name: str, path: str, type_: str = "dataset") -> None:
        if self.run is None:
            return
        try:
            art = self._wandb.Artifact(name=name, type=type_)
            art.add_file(path)
            self.run.log_artifact(art)
        except Exception:
            pass

    def finish(self) -> None:
        if self.run is not None:
            try:
                self.run.finish()
            except Exception:
                pass
            self.run = None
