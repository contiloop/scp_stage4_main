"""Stub factory for testing training_schedule wiring without GPU."""
from src.scp.orchestrator import Runners
from src.scp.runners import EchoProbe, EchoStudent, StubProbeTrainer


class StubSft:
    def train(self, *, base_checkpoint_id, dataset_path, training_cfg, round_id):
        return {"new_checkpoint_id": f"M{round_id}_sft", "path": None,
                "metrics": {"stub": True, "base": base_checkpoint_id}}


class StubPref:
    def train(self, *, base_checkpoint_id, dataset_path, training_cfg, stage_label):
        return {"new_checkpoint_id": f"{base_checkpoint_id}__{stage_label}",
                "path": None, "metrics": {"stub": True}}


def make_stub_runners(_cfg=None):
    return Runners(student=EchoStudent(), probe_trainer=StubProbeTrainer(),
                   probe_gen=EchoProbe(), sft_trainer=StubSft(),
                   pref_trainer=StubPref())
