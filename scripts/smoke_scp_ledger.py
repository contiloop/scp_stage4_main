"""Smoke test: config -> ledger -> teacher (echo) -> compact -> read back.

Run: python scripts/smoke_scp_ledger.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scp import Ledger, TeacherCache, load_config
from src.scp.config import write_resolved_config
from src.scp.prompts import copy_prompt_to_run
from src.scp.teacher import TeacherClient


def main():
    cfg = load_config(ROOT / "configs/scp/debug.yaml")
    run_dir = Path(cfg.storage.workdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(cfg, run_dir)
    copy_prompt_to_run(cfg.models.teacher.prompt_path, run_dir)

    ledger = Ledger(run_dir, cfg.run_id)
    cache = TeacherCache(run_dir / "teacher_cache.sqlite")
    teacher = TeacherClient(cfg.models.teacher, cache=cache, ledger=ledger)

    examples = [
        ("ex_000001", "The cat sat on the mat.", "고양이가 매트에 앉았다.", "고양이 매트."),
        ("ex_000002", "Open the door.", "문을 열어라.", "문 열다."),
    ]
    round_id = 1

    skip = ledger.existing_keys("teacher_edit", round_id) if cfg.recovery.skip_existing_teacher_calls else set()

    for ex_id, src, stu, probe in examples:
        ledger.log({
            "event_type": "student_generation",
            "round_id": round_id,
            "example_id": ex_id,
            "source_text": src,
            "checkpoint_id": "M0",
            "prompt_version": "student_translate_v1",
            "decoding": cfg.models.student.decoding.model_dump(),
            "student_output": stu,
        })
        if ex_id in skip:
            print(f"skip cached teacher: {ex_id}")
            continue
        res = teacher.edit(
            round_id=round_id,
            example_id=ex_id,
            source_text=src,
            student_output=stu,
            probe_output=probe,
        )
        print(f"{ex_id}: cached={res.cached} action={res.teacher_action} latency={res.latency_ms}ms")

    paths = ledger.compact_round(round_id)
    print("compacted:")
    for k, p in paths.items():
        print(f"  {k}: {p}")

    summary = {
        "round_id": round_id,
        "num_examples": len(examples),
        "teacher_call_count": len(examples) - len(skip),
    }
    sp = ledger.write_round_summary(round_id, summary)
    print(f"summary: {sp}")

    cache.close()


if __name__ == "__main__":
    main()
