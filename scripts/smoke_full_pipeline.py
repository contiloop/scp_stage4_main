"""End-to-end SCP pipeline smoke test using stubs (no GPU, echo teacher).

Exercises: registry -> student -> cheap score -> probe train (stub) ->
probe gen -> collapse -> select -> teacher (cached) -> SFT/Pref builders ->
round summary.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scp import (
    BudgetCaps, BudgetTracker, CollapseConfig, ExampleRegistry, Ledger,
    LengthRatioScorer, NumberMatchScorer, PrefBuildConfig, RuleFlagScorer,
    RunState, SelectorConfig, SftBuildConfig, StubProbeTrainer,
    TeacherCache, analyze_collapse, build_preference, build_sft,
    build_scorer_from_config, copy_prompt_to_run, load_config, select,
    source_hash, summarize_round, write_resolved_config,
)
from src.scp.runners import EchoProbe, EchoStudent
from src.scp.teacher import TeacherClient


def main():
    cfg = load_config(ROOT / "configs/scp/debug.yaml")
    run_dir = Path(cfg.storage.workdir.replace("debug_001", "debug_full"))
    write_resolved_config(cfg, run_dir)
    copy_prompt_to_run(cfg.models.teacher.prompt_path, run_dir)

    state = RunState(run_dir)
    ledger = Ledger(run_dir, cfg.run_id)
    cache = TeacherCache(run_dir / "teacher_cache.sqlite")
    teacher = TeacherClient(cfg.models.teacher, cache=cache, ledger=ledger)
    budget = BudgetTracker(ledger, BudgetCaps(per_round_usd=10.0, per_round_calls=100))

    # 1. registry
    registry = ExampleRegistry(run_dir / "data/examples.parquet")
    raw = [
        ("ex_000001", "The cat sat on the mat near the door at 9:30 am."),
        ("ex_000002", "Open the second door and check the box labeled A1."),
        ("ex_000003", "Total payment is 1,250.50 USD due on 2026-05-01."),
        ("ex_000004", "Press the red button to start."),
    ]
    registry.add_many([
        {"example_id": eid, "source_text": s, "source_hash": source_hash(s),
         "dataset_version": "mono_v1", "assigned_round": 1}
        for eid, s in raw
    ])
    registry.save()

    round_id = 1
    state.advance_round(round_id)
    state.set_checkpoint("main", "M0")

    # 2-3. student gen + cheap score
    student = EchoStudent()
    probe_runner = EchoProbe()
    probe_trainer = StubProbeTrainer()

    # Pull scorer config from extras; default to heuristic QE for the smoke test.
    scoring_cfg = cfg.extras.get("scoring") or {
        "scorers": [
            {"type": "length_ratio", "min_ratio": 0.3, "max_ratio": 3.0},
            {"type": "number_match"},
            {"type": "rule_flags", "rules": [
                {"name": "missing_period", "on": "hypothesis",
                 "pattern": r"[.!?]$", "flag_if": "absent"},
            ]},
        ],
        "qe": {"type": "heuristic"},
    }
    scorer = build_scorer_from_config(scoring_cfg)
    print(f"[scoring] qe={getattr(scorer.qe, 'name', None)}")

    student_outs: dict[str, str] = {}
    examples = list(registry.iter_round(round_id))
    for ex in examples:
        eid, src = ex["example_id"], ex["source_text"]
        gen = student.generate(src, decoding=cfg.models.student.decoding.model_dump(),
                               checkpoint_id="M0")
        student_outs[eid] = gen["output"]
        ledger.log({
            "event_type": "student_generation", "round_id": round_id,
            "example_id": eid, "source_text": src, "checkpoint_id": "M0",
            "prompt_version": "student_translate_v1",
            "decoding": cfg.models.student.decoding.model_dump(),
            "student_output": gen["output"],
            "input_tokens": gen["input_tokens"], "output_tokens": gen["output_tokens"],
            "latency_ms": gen["latency_ms"],
        })

    pre_pairs = [(ex["source_text"], student_outs[ex["example_id"]]) for ex in examples]
    pre_rows = scorer.score_many(pre_pairs)
    pre_scores: dict[str, dict] = {}
    for ex, pre in zip(examples, pre_rows):
        eid = ex["example_id"]
        pre_scores[eid] = pre
        ledger.log({
            "event_type": "pre_probe_score", "round_id": round_id,
            "example_id": eid, "student_output_ref": eid, **pre,
        })

    # 4. probe trainer (stub, then immediately discard conceptually)
    probe_meta = probe_trainer.train(
        base_checkpoint_id="M0",
        data_slice=[{"example_id": eid, "source": s} for eid, s in raw],
        probe_cfg={"rank": 8, "steps": 50, "lr": 1e-3},
        round_id=round_id,
    )
    ledger.log({
        "event_type": "probe_training", "round_id": round_id,
        "example_id": "_round_", **probe_meta,
    })

    # 5-6. probe gen + post score + collapse (post scoring batched)
    cc_cfg = cfg.extras.get("collapse", {})
    cc = CollapseConfig(**{k: v for k, v in cc_cfg.items()
                           if k in CollapseConfig.__dataclass_fields__})
    probe_outs: dict[str, str] = {}
    for ex in examples:
        eid, src = ex["example_id"], ex["source_text"]
        pgen = probe_runner.generate(
            src, decoding=cfg.models.probe.decoding.model_dump(),
            base_checkpoint_id="M0", probe_lora_id=probe_meta["probe_lora_id"],
        )
        probe_outs[eid] = pgen["output"]
        ledger.log({
            "event_type": "probe_generation", "round_id": round_id,
            "example_id": eid, "base_checkpoint_id": "M0",
            "probe_lora_id": probe_meta["probe_lora_id"],
            "probe_output": pgen["output"],
            "decoding": cfg.models.probe.decoding.model_dump(),
        })

    post_pairs = [(ex["source_text"], probe_outs[ex["example_id"]]) for ex in examples]
    post_rows = scorer.score_many(post_pairs)
    collapse_rows: list[dict] = []
    for ex, post in zip(examples, post_rows):
        eid = ex["example_id"]
        coll = analyze_collapse(pre_scores[eid], post, cc)
        collapse_rows.append({"example_id": eid, **coll})

    # 7. selector
    sel = select(collapse_rows, SelectorConfig(min_selection_score=0.0,
                                                require_collapse_flag=False,
                                                max_per_round=10))
    selected_ids = {s["example_id"] for s in sel}
    sel_lookup = {s["example_id"]: s for s in sel}

    for row in collapse_rows:
        eid = row["example_id"]
        sel_meta = sel_lookup.get(eid)
        ledger.log({
            "event_type": "collapse_analysis", "round_id": round_id,
            "example_id": eid, **row,
            "selected_for_teacher": eid in selected_ids,
            "selection_reason": sel_meta["selection_reason"] if sel_meta else [],
        })

    # 8. teacher edits with budget gate
    probe_outputs = {ev["example_id"]: ev["probe_output"]
                     for ev in ledger.iter_events()
                     if ev.get("event_type") == "probe_generation"
                     and ev.get("round_id") == round_id}

    for ex in registry.iter_round(round_id):
        eid, src = ex["example_id"], ex["source_text"]
        if eid not in selected_ids:
            continue
        ok, reason = budget.can_call(round_id)
        if not ok:
            print(f"budget gate stop: {reason}")
            break
        teacher.edit(
            round_id=round_id, example_id=eid,
            source_text=src, student_output=student_outs[eid],
            probe_output=probe_outputs.get(eid),
        )

    # 9-10. SFT + Preference dataset builders
    sft_path, n_sft = build_sft(
        ledger, round_id, SftBuildConfig(),
        run_dir / "datasets/sft_r1_v1.parquet", "sft_r1_v1",
    )
    pref_path, n_pref = build_preference(
        ledger, round_id, PrefBuildConfig(),
        run_dir / "datasets/pref_r1_v1.parquet", "pref_r1_v1",
    )
    state.register_dataset("sft_r1_v1", str(sft_path))
    state.register_dataset("pref_r1_v1", str(pref_path))

    # 11. compact + summary
    paths = ledger.compact_round(round_id)
    summary = summarize_round(ledger, round_id)
    summary["sft_items_created_dataset"] = n_sft
    summary["pref_pairs_in_dataset"] = n_pref
    sp = ledger.write_round_summary(round_id, summary)

    state.mark_step_done(round_id, "round_complete")
    cache.close()

    print(f"selected={len(selected_ids)} sft={n_sft} pref={n_pref}")
    print(f"summary -> {sp}")
    print(f"compacted: {list(paths.keys())}")
    print(f"summary content: {summary}")


if __name__ == "__main__":
    main()
