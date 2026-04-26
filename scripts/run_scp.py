"""SCP orchestrator CLI.

Usage:
  python scripts/run_scp.py --config configs/scp/scp_v1.yaml --rounds 1
  python scripts/run_scp.py --config configs/scp/scp_v1.yaml --rounds 1-3 --resume

GPU runners must be wired explicitly. This CLI ships with Echo stubs so the
non-GPU path works; replace via --runner-impl <python.module:factory>.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scp import (ExampleRegistry, Orchestrator, Runners, copy_prompt_to_run,
                     load_config, source_hash, write_resolved_config)
from src.scp.runners import EchoProbe, EchoStudent, StubProbeTrainer


def parse_rounds(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def load_runners(spec: str | None, cfg) -> Runners:
    if not spec:
        return Runners(student=EchoStudent(), probe_trainer=StubProbeTrainer(),
                       probe_gen=EchoProbe())
    module, fn = spec.split(":")
    factory = getattr(importlib.import_module(module), fn)
    runner_cfg = cfg.extras.get("real_runners", {})
    try:
        return factory(runner_cfg)
    except TypeError:
        return factory()


def maybe_seed_registry(registry: ExampleRegistry, seed_jsonl: str | None,
                        round_id: int) -> int:
    if not seed_jsonl or len(registry):
        return 0
    import json
    rows = []
    with open(seed_jsonl) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            src = obj["source_text"]
            rows.append({
                "example_id": obj.get("example_id", f"ex_{i:06d}"),
                "source_text": src, "source_hash": source_hash(src),
                "dataset_version": obj.get("dataset_version", "mono_v1"),
                "assigned_round": obj.get("assigned_round", round_id),
            })
    n = registry.add_many(rows)
    registry.save()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rounds", default="1", help="e.g. 1 or 1-3 or 1,2,5")
    ap.add_argument("--runners", default=None,
                    help="module:factory returning a Runners instance")
    ap.add_argument("--seed-examples", default=None,
                    help="optional JSONL to seed the registry on first run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(cfg.storage.workdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(cfg, run_dir)
    copy_prompt_to_run(cfg.models.teacher.prompt_path, run_dir)

    registry = ExampleRegistry(run_dir / "data/examples.parquet")
    rounds = parse_rounds(args.rounds)
    n_seeded = maybe_seed_registry(registry, args.seed_examples, rounds[0])
    if n_seeded:
        print(f"[registry] seeded {n_seeded} examples")

    runners = load_runners(args.runners, cfg)
    orch = Orchestrator(cfg=cfg, run_dir=run_dir, runners=runners,
                        registry=registry)
    out = orch.run(rounds)
    for r, results in out.items():
        print(f"\n=== round {r} ===")
        for step, res in results.items():
            print(f"  {step}: {res}")


if __name__ == "__main__":
    main()
