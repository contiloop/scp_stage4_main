"""Round orchestrator. Schedule-driven and resumable.

The schedule is config-driven (`extras.schedule`) so ablations stay
declarative. Each step is idempotent; the orchestrator just iterates and
records `mark_step_done`. SFT/Pref training are triggered separately via
`extras.training_schedule` (per-round labels) - they're optional in MVP.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import steps
from .budget import BudgetCaps, BudgetTracker
from .cache import TeacherCache
from .config import ScpConfig
from .ledger import Ledger
from .registry import ExampleRegistry
from .runners import (PrefTrainer, ProbeGenRunner, ProbeTrainer, SftTrainer,
                      StudentGenRunner)
from .scoring import CompositeScorer, build_scorer_from_config
from .state import RunState
from .teacher import TeacherClient
from .tracking import WandbReporter, make_weave_tracer


DEFAULT_SCHEDULE = [
    "student_gen", "pre_score", "probe_train", "probe_gen",
    "collapse_select", "teacher", "build_sft", "build_pref",
    "dev_eval", "summarize",
]


@dataclass
class Runners:
    student: StudentGenRunner
    probe_trainer: ProbeTrainer
    probe_gen: ProbeGenRunner
    sft_trainer: SftTrainer | None = None
    pref_trainer: PrefTrainer | None = None
    # `dev_generate` lets the dev evaluator use any callable(source)->str.
    # Default uses the student runner with the current main checkpoint.
    dev_generate: Callable[[str], str] | None = None


class Orchestrator:
    def __init__(self, *, cfg: ScpConfig, run_dir, runners: Runners,
                 registry: ExampleRegistry):
        self.cfg = cfg
        self.run_dir = run_dir
        self.runners = runners
        self.registry = registry

        self.state = RunState(run_dir)
        self.ledger = Ledger(run_dir, cfg.run_id, cloud_uri=cfg.storage.cloud_uri)
        self.cache = TeacherCache(f"{run_dir}/teacher_cache.sqlite")

        weave = make_weave_tracer(
            cfg.logging.weave_project,
            enabled=cfg.logging.weave_enabled,
        )
        self.teacher = TeacherClient(cfg.models.teacher, cache=self.cache,
                                     ledger=self.ledger, weave_tracer=weave)
        b_cfg = cfg.extras.get("budget", {})
        self.budget = BudgetTracker(self.ledger, BudgetCaps(**{
            k: v for k, v in b_cfg.items()
            if k in BudgetCaps.__dataclass_fields__
        }))
        self.scorer: CompositeScorer = build_scorer_from_config(
            cfg.extras.get("scoring", {}))
        self.wandb = WandbReporter(
            enabled=cfg.logging.wandb_enabled,
            project=cfg.logging.wandb_project,
            run_id=cfg.run_id,
            config=cfg.model_dump(),
        )

    # ---- step dispatch ---- #

    def _run_step(self, name: str, round_id: int) -> dict | None:
        s = steps
        common = dict(cfg=self.cfg, ledger=self.ledger, state=self.state,
                      round_id=round_id)
        if name == "student_gen":
            return s.step_student_gen(**common, registry=self.registry,
                                      runner=self.runners.student)
        if name == "pre_score":
            return s.step_pre_score(cfg=self.cfg, ledger=self.ledger,
                                    registry=self.registry, scorer=self.scorer,
                                    round_id=round_id)
        if name == "probe_train":
            return s.step_probe_train(**common, registry=self.registry,
                                      trainer=self.runners.probe_trainer)
        if name == "probe_gen":
            return s.step_probe_gen(**common, registry=self.registry,
                                    runner=self.runners.probe_gen)
        if name == "collapse_select":
            return s.step_collapse_select(cfg=self.cfg, ledger=self.ledger,
                                          registry=self.registry,
                                          scorer=self.scorer, round_id=round_id)
        if name == "teacher":
            return s.step_teacher(cfg=self.cfg, ledger=self.ledger,
                                  registry=self.registry,
                                  teacher=self.teacher, budget=self.budget,
                                  round_id=round_id)
        if name == "build_sft":
            return s.step_build_sft(**common)
        if name == "build_pref":
            return s.step_build_pref(**common)
        if name == "dev_eval":
            gen = self.runners.dev_generate
            if gen is None:
                # default: use student runner against current checkpoint
                ckpt = (self.state.get_checkpoint("main") or {"id": "M0"})["id"]
                decoding = self.cfg.models.student.decoding.model_dump()
                runner = self.runners.student
                gen = lambda src: runner.generate(src, decoding=decoding,
                                                  checkpoint_id=ckpt)["output"]
            return s.step_dev_eval(**common, generate=gen)
        if name == "summarize":
            return s.step_summarize(ledger=self.ledger, round_id=round_id,
                                    wandb_reporter=self.wandb)
        if name == "sft_train":
            if self.runners.sft_trainer is None:
                return {"skipped": "no_sft_trainer"}
            return s.step_sft_train(**common, trainer=self.runners.sft_trainer)
        if name == "pref_train":
            if self.runners.pref_trainer is None:
                return {"skipped": "no_pref_trainer"}
            return s.step_pref_train(**common, trainer=self.runners.pref_trainer)
        raise ValueError(f"unknown step: {name}")

    def _training_steps_for(self, round_id: int) -> list[str]:
        ts = self.cfg.extras.get("training_schedule") or {}
        # per-round entries can be a list of step names (e.g. ["sft_train"])
        # or a dict {round: [...]}; YAML keys may be ints or strings
        if isinstance(ts, dict):
            return list(ts.get(round_id) or ts.get(str(round_id)) or [])
        return []

    def run_round(self, round_id: int, schedule: list[str] | None = None
                  ) -> dict[str, dict | None]:
        schedule = schedule or self.cfg.extras.get("schedule") or DEFAULT_SCHEDULE
        schedule = list(schedule) + self._training_steps_for(round_id)
        self.state.advance_round(round_id)
        results: dict[str, dict | None] = {}
        for step in schedule:
            if self.state.is_step_done(round_id, step):
                results[step] = {"skipped": "already done"}
                continue
            results[step] = self._run_step(step, round_id)
            self.state.mark_step_done(round_id, step)
        if self.cfg.storage.sync_every_round and self.cfg.storage.cloud_uri:
            try:
                self.ledger.storage.sync(commit_message=f"round {round_id}")
            except Exception as e:
                print(f"[sync] round {round_id} failed: {e}")
        return results

    def run(self, rounds: list[int]) -> dict[int, dict]:
        out = {}
        try:
            for r in rounds:
                out[r] = self.run_round(r)
        finally:
            self.cache.close()
            self.wandb.finish()
        return out
