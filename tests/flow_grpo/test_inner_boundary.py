"""Production checkpoint format preserves an unfinished behavior batch.

This CPU optimizer test does not stand in for the required full model ZeRO run.
"""
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from omegaconf import OmegaConf
from starVLA.rl.flow_grpo.checkpoint import save_boundary, resume_boundary
from starVLA.rl.flow_grpo.data import SceneStream
from starVLA.rl.flow_grpo.reproducibility import processor_identity


class Processor:
    def save_pretrained(self, path):
        Path(path).mkdir()
        (Path(path) / "processor_config.json").write_text('{"size":1024}')


class CPUState:
    device = torch.device("cpu")
    is_main_process = True
    process_index = 0
    num_processes = 1

    def __init__(self, policy, optimizer):
        self.policy, self.optimizer = policy, optimizer

    def unwrap_model(self, actor):
        return actor

    def save_state(self, path, **kwargs):
        torch.save(
            {
                "model": self.policy.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            Path(path) / "state.pt",
        )

    def load_state(self, path):
        state = torch.load(Path(path) / "state.pt", weights_only=True)
        self.policy.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])


def test_exact_inner_epoch_resume_and_negative_identities(tmp_path):
    policy = torch.nn.Linear(2, 1)
    policy.qwen_vl_interface = SimpleNamespace(processor=Processor())
    actor = SimpleNamespace(policy=policy)
    opt = torch.optim.AdamW(policy.parameters(), lr=0.01)
    accelerator = CPUState(policy, opt)
    cfg = {
        "runtime": {"output_dir": str(tmp_path)},
        "algorithm": {"logprob_reduction": "flow_grpo_dimension_mean"},
    }
    sft = OmegaConf.create({"datasets": {"vla_data": {"act_norm": 1}}})
    stream = SceneStream(None, ["a", "b"], 42, cursor=1)
    processor_root = tmp_path / "source_processor"
    Processor().save_pretrained(processor_root)
    provenance = {
        "processor": processor_identity(processor_root),
        "numerics": {"tf32": False},
    }
    old = torch.zeros(1, 2, 3)
    pending = {
        "next_inner": 1,
        "buffers": [
            {
                "old_logprob": old.clone(),
                "advantages": torch.tensor([[-1.0, 1.0]]),
                "policy_version": 7,
            }
        ],
        "replay": [{"_flow_sample_seed": 12}],
    }

    def step():
        opt.zero_grad()
        loss = (policy(torch.rand(3, 2)) ** 2).sum()
        loss.backward()
        opt.step()

    torch.manual_seed(99)
    step()
    checkpoint = save_boundary(
        accelerator, actor, cfg, sft, {}, provenance, 1, 7, [stream], pending
    )
    step()
    expected = copy.deepcopy(policy.state_dict())
    moments = copy.deepcopy(opt.state_dict())
    update, version, resumed = resume_boundary(
        checkpoint, accelerator, actor, cfg, provenance, [stream]
    )
    assert (update, version, resumed["next_inner"]) == (1, 7, 1)
    assert torch.equal(resumed["buffers"][0]["old_logprob"], old)
    assert stream.cursor == 1
    step()
    for key in expected:
        assert torch.equal(policy.state_dict()[key], expected[key])
    for index, state in opt.state_dict()["state"].items():
        for key, value in state.items():
            assert torch.equal(value, moments["state"][index][key])
    bad = copy.deepcopy(provenance)
    bad["numerics"]["tf32"] = True
    with pytest.raises(RuntimeError, match="exact resume identity changed"):
        resume_boundary(checkpoint, accelerator, actor, cfg, bad, [stream])
    (processor_root / "processor_config.json").write_text('{"size":1025}')
    bad = {**provenance, "processor": processor_identity(processor_root)}
    with pytest.raises(RuntimeError, match="processor"):
        resume_boundary(checkpoint, accelerator, actor, cfg, bad, [stream])
    changed_stream = SceneStream(None, ["a", "c"], 42)
    with pytest.raises(RuntimeError, match="sampler resume contract"):
        resume_boundary(
            checkpoint, accelerator, actor, cfg, provenance, [changed_stream]
        )
