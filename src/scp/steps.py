"""SCP step functions. Each step is idempotent and resumable.

Idempotency rule: every step consults `ledger.existing_keys(event_type, round_id)`
and skips example_ids already logged for the round. The teacher step also goes
through TeacherCache so re-runs across machines never duplicate API spend.

Steps return a small status dict for the orchestrator to log + summarise.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .budget import BudgetTracker
from .cache import TeacherCache
from .collapse import CollapseConfig, analyze as analyze_collapse
from .config import ScpConfig
from .datasets import (PrefBuildConfig, SftBuildConfig,
                       build_preference, build_sft)
from .evaluator import DevEvaluator, build_evaluator_from_config, load_dev_set
from .ledger import Ledger
from .registry import ExampleRegistry
from .runners import (Evaluator, PrefTrainer, ProbeGenRunner, ProbeTrainer,
                      SftTrainer, StudentGenRunner)
from .scoring import CompositeScorer, build_scorer_from_config
from .selector import SelectorConfig, select
from .state import RunState
from .summary import summarize_round
from .teacher import TeacherClient


# --------------------------- helpers ---------------------------- #

def _examples_for_round(registry: ExampleRegistry, round_id: int) -> list[dict]:
    return list(registry.iter_round(round_id))


def _select_cfg(cfg: ScpConfig, key: str, default: dict | None = None) -> dict:
    return cfg.extras.get(key, default or {})


# --------------------------- steps ---------------------------- #

def step_student_gen(*, cfg: ScpConfig, ledger: Ledger, state: RunState,
                     registry: ExampleRegistry, runner: StudentGenRunner,
                     round_id: int) -> dict:
    ckpt = state.get_checkpoint("main") or {"id": "M0", "path": None}
    decoding = cfg.models.student.decoding.model_dump()
    done = ledger.existing_keys("student_generation", round_id)
    n_new = 0
    for ex in _examples_for_round(registry, round_id):
        eid = ex["example_id"]
        if eid in done:
            continue
        gen = runner.generate(ex["source_text"], decoding=decoding,
                              checkpoint_id=ckpt["id"])
        ledger.log({
            "event_type": "student_generation", "round_id": round_id,
            "example_id": eid, "source_text": ex["source_text"],
            "checkpoint_id": ckpt["id"],
            "prompt_version": "student_translate_v1",
            "decoding": decoding, "student_output": gen["output"],
            "input_tokens": gen.get("input_tokens"),
            "output_tokens": gen.get("output_tokens"),
            "latency_ms": gen.get("latency_ms"),
        })
        n_new += 1
    state.mark_step_done(round_id, "student_gen")
    return {"new": n_new, "skipped": len(done)}


def _student_outputs(ledger: Ledger, round_id: int) -> dict[str, str]:
    out = {}
    for ev in ledger.iter_events():
        if (ev.get("event_type") == "student_generation"
                and ev.get("round_id") == round_id):
            out[ev["example_id"]] = ev.get("student_output", "")
    return out


def _probe_outputs(ledger: Ledger, round_id: int) -> dict[str, str]:
    out = {}
    for ev in ledger.iter_events():
        if (ev.get("event_type") == "probe_generation"
                and ev.get("round_id") == round_id):
            out[ev["example_id"]] = ev.get("probe_output", "")
    return out


def step_pre_score(*, cfg: ScpConfig, ledger: Ledger, registry: ExampleRegistry,
                   scorer: CompositeScorer, round_id: int) -> dict:
    done = ledger.existing_keys("pre_probe_score", round_id)
    student_outs = _student_outputs(ledger, round_id)
    pending = [(ex["example_id"], ex["source_text"], student_outs.get(ex["example_id"], ""))
               for ex in _examples_for_round(registry, round_id)
               if ex["example_id"] not in done and ex["example_id"] in student_outs]
    rows = scorer.score_many([(s, h) for _, s, h in pending])
    for (eid, _src, _hyp), pre in zip(pending, rows):
        ledger.log({"event_type": "pre_probe_score", "round_id": round_id,
                    "example_id": eid, "student_output_ref": eid, **pre})
    return {"new": len(pending), "skipped": len(done)}


def step_probe_train(*, cfg: ScpConfig, ledger: Ledger, state: RunState,
                     registry: ExampleRegistry, trainer: ProbeTrainer,
                     round_id: int) -> dict:
    if any(ev.get("event_type") == "probe_training"
           and ev.get("round_id") == round_id for ev in ledger.iter_events()):
        return {"skipped": True}
    base_ckpt = (state.get_checkpoint("main") or {"id": "M0"})["id"]
    probe_cfg = _select_cfg(cfg, "probe_training", {"rank": 8, "steps": 50, "lr": 1e-3})
    data_slice = [{"example_id": ex["example_id"], "source": ex["source_text"]}
                  for ex in _examples_for_round(registry, round_id)]
    meta = trainer.train(base_checkpoint_id=base_ckpt, data_slice=data_slice,
                         probe_cfg=probe_cfg, round_id=round_id)
    state.set_checkpoint("probe", meta["probe_lora_id"], meta.get("path"))
    ledger.log({"event_type": "probe_training", "round_id": round_id,
                "example_id": "_round_", **meta})
    return {"new": 1, "probe_lora_id": meta["probe_lora_id"]}


def step_probe_gen(*, cfg: ScpConfig, ledger: Ledger, state: RunState,
                   registry: ExampleRegistry, runner: ProbeGenRunner,
                   round_id: int) -> dict:
    base_ckpt = (state.get_checkpoint("main") or {"id": "M0"})["id"]
    probe = state.get_checkpoint("probe")
    if probe is None:
        raise RuntimeError("probe checkpoint not set; run step_probe_train first")
    decoding = cfg.models.probe.decoding.model_dump()
    done = ledger.existing_keys("probe_generation", round_id)
    n_new = 0
    for ex in _examples_for_round(registry, round_id):
        eid = ex["example_id"]
        if eid in done:
            continue
        gen = runner.generate(ex["source_text"], decoding=decoding,
                              base_checkpoint_id=base_ckpt,
                              probe_lora_id=probe["id"])
        ledger.log({"event_type": "probe_generation", "round_id": round_id,
                    "example_id": eid, "base_checkpoint_id": base_ckpt,
                    "probe_lora_id": probe["id"],
                    "probe_output": gen["output"], "decoding": decoding})
        n_new += 1
    return {"new": n_new, "skipped": len(done)}


def step_collapse_select(*, cfg: ScpConfig, ledger: Ledger,
                         registry: ExampleRegistry, scorer: CompositeScorer,
                         round_id: int) -> dict:
    done = ledger.existing_keys("collapse_analysis", round_id)
    probe_outs = _probe_outputs(ledger, round_id)
    pre_by_id = {}
    for ev in ledger.iter_events():
        if (ev.get("event_type") == "pre_probe_score"
                and ev.get("round_id") == round_id):
            pre_by_id[ev["example_id"]] = ev

    pending = [ex for ex in _examples_for_round(registry, round_id)
               if ex["example_id"] not in done
               and ex["example_id"] in probe_outs
               and ex["example_id"] in pre_by_id]
    if not pending:
        return {"new": 0, "skipped": len(done)}

    post_pairs = [(ex["source_text"], probe_outs[ex["example_id"]]) for ex in pending]
    post_rows = scorer.score_many(post_pairs)

    cc_kwargs = {k: v for k, v in _select_cfg(cfg, "collapse").items()
                 if k in CollapseConfig.__dataclass_fields__}
    cc = CollapseConfig(**cc_kwargs)
    coll_rows = []
    for ex, post in zip(pending, post_rows):
        coll = analyze_collapse(pre_by_id[ex["example_id"]], post, cc)
        coll_rows.append({"example_id": ex["example_id"], **coll})

    sel_kwargs = {k: v for k, v in _select_cfg(cfg, "selector").items()
                  if k in SelectorConfig.__dataclass_fields__}
    sel = select(coll_rows, SelectorConfig(**sel_kwargs))
    sel_lookup = {s["example_id"]: s for s in sel}
    selected_ids = set(sel_lookup)

    for row in coll_rows:
        eid = row["example_id"]
        sm = sel_lookup.get(eid)
        ledger.log({"event_type": "collapse_analysis", "round_id": round_id,
                    "example_id": eid, **row,
                    "selected_for_teacher": eid in selected_ids,
                    "selection_reason": sm["selection_reason"] if sm else []})
    return {"new": len(coll_rows), "selected": len(selected_ids),
            "skipped": len(done)}


def step_teacher(*, cfg: ScpConfig, ledger: Ledger, registry: ExampleRegistry,
                 teacher: TeacherClient, budget: BudgetTracker,
                 round_id: int) -> dict:
    done = ledger.existing_keys("teacher_edit", round_id)
    student_outs = _student_outputs(ledger, round_id)
    probe_outs = _probe_outputs(ledger, round_id)
    selected: dict[str, dict] = {}
    for ev in ledger.iter_events():
        if (ev.get("event_type") == "collapse_analysis"
                and ev.get("round_id") == round_id
                and ev.get("selected_for_teacher")):
            selected[ev["example_id"]] = ev

    n_called = 0; n_skipped_budget = 0
    for ex in _examples_for_round(registry, round_id):
        eid = ex["example_id"]
        if eid in done or eid not in selected:
            continue
        ok, reason = budget.can_call(round_id)
        if not ok:
            n_skipped_budget += 1
            ledger.log({"event_type": "budget_event", "round_id": round_id,
                        "example_id": eid, "skipped_reason": reason})
            continue
        teacher.edit(round_id=round_id, example_id=eid,
                     source_text=ex["source_text"],
                     student_output=student_outs.get(eid, ""),
                     probe_output=probe_outs.get(eid))
        n_called += 1
    return {"called": n_called, "budget_skipped": n_skipped_budget,
            "already_done": len(done)}


def step_build_sft(*, cfg: ScpConfig, ledger: Ledger, state: RunState,
                   round_id: int, version: str | None = None) -> dict:
    out_dir = Path(state.path).parent / "datasets"
    version = version or f"sft_r{round_id}_v1"
    out_path = out_dir / f"{version}.parquet"
    sft_kwargs = _select_cfg(cfg, "sft_build")
    sft_cfg = SftBuildConfig(**{k: v for k, v in sft_kwargs.items()
                                if k in SftBuildConfig.__dataclass_fields__})
    p, n = build_sft(ledger, round_id, sft_cfg, out_path, version)
    state.register_dataset(version, str(p))
    return {"version": version, "rows": n, "path": str(p)}


def step_build_pref(*, cfg: ScpConfig, ledger: Ledger, state: RunState,
                    round_id: int, version: str | None = None) -> dict:
    out_dir = Path(state.path).parent / "datasets"
    version = version or f"pref_r{round_id}_v1"
    out_path = out_dir / f"{version}.parquet"
    pref_kwargs = _select_cfg(cfg, "pref_build")
    pref_cfg = PrefBuildConfig(**{k: v for k, v in pref_kwargs.items()
                                  if k in PrefBuildConfig.__dataclass_fields__})
    p, n = build_preference(ledger, round_id, pref_cfg, out_path, version)
    state.register_dataset(version, str(p))
    return {"version": version, "rows": n, "path": str(p)}


def step_dev_eval(*, cfg: ScpConfig, ledger: Ledger, state: RunState,
                  round_id: int, generate: Callable[[str], str]) -> dict | None:
    eval_cfg = cfg.extras.get("dev_eval")
    if not eval_cfg or not eval_cfg.get("enabled", True):
        return None
    dev_path = eval_cfg.get("dev_set_path")
    if not dev_path:
        return None
    dev = load_dev_set(dev_path)
    evaluator = build_evaluator_from_config(eval_cfg)
    ckpt = state.get_checkpoint("main") or {"id": "M0"}
    res = evaluator.evaluate(dev_set=dev, generate=generate, ledger=ledger,
                             round_id=round_id, checkpoint_id=ckpt["id"])
    return res


def step_sft_train(*, cfg: ScpConfig, ledger: Ledger, state: RunState,
                   round_id: int, trainer: SftTrainer,
                   dataset_version: str | None = None,
                   training_cfg: dict | None = None) -> dict | None:
    version = dataset_version or f"sft_r{round_id}_v1"
    ds_path = state.snapshot().get("datasets", {}).get(version)
    if not ds_path:
        return {"skipped": "no_dataset"}
    base = (state.get_checkpoint("main") or {"id": "M0"})["id"]
    tcfg = training_cfg or cfg.extras.get("sft_training", {})
    res = trainer.train(base_checkpoint_id=base, dataset_path=ds_path,
                        training_cfg=tcfg, round_id=round_id)
    state.set_checkpoint("main", res["new_checkpoint_id"], res.get("path"))
    ledger.log({"event_type": "round_summary", "round_id": round_id,
                "example_id": "_sft_", "kind": "sft_train",
                "new_checkpoint_id": res["new_checkpoint_id"],
                "metrics": res.get("metrics", {})})
    return {"new_checkpoint_id": res["new_checkpoint_id"]}


def step_pref_train(*, cfg: ScpConfig, ledger: Ledger, state: RunState,
                    round_id: int, trainer: PrefTrainer,
                    dataset_version: str | None = None,
                    stage_label: str | None = None,
                    training_cfg: dict | None = None) -> dict | None:
    version = dataset_version or f"pref_r{round_id}_v1"
    ds_path = state.snapshot().get("datasets", {}).get(version)
    if not ds_path:
        return {"skipped": "no_dataset"}
    base = (state.get_checkpoint("main") or {"id": "M0"})["id"]
    label = stage_label or f"dpo_r{round_id}"
    tcfg = training_cfg or cfg.extras.get("pref_training", {})
    res = trainer.train(base_checkpoint_id=base, dataset_path=ds_path,
                        training_cfg=tcfg, stage_label=label)
    state.set_checkpoint("main", res["new_checkpoint_id"], res.get("path"))
    ledger.log({"event_type": "round_summary", "round_id": round_id,
                "example_id": "_pref_", "kind": "pref_train",
                "new_checkpoint_id": res["new_checkpoint_id"],
                "metrics": res.get("metrics", {})})
    return {"new_checkpoint_id": res["new_checkpoint_id"]}


def step_summarize(*, ledger: Ledger, round_id: int,
                   wandb_reporter=None) -> dict:
    summary = summarize_round(ledger, round_id)
    ledger.compact_round(round_id)
    ledger.write_round_summary(round_id, summary)
    if wandb_reporter is not None:
        wandb_reporter.log_round_summary(round_id, summary)
    return summary
