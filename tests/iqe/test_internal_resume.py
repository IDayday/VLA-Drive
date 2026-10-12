from copy import deepcopy
import json
import os
import subprocess
import sys
import torch
from iqe.internal_baseline.scorer import InternalScorer, initialization_rng
from iqe.internal_baseline.sampling import TwoStreamSampler
from iqe.internal_baseline.objectives import terms, denominators
from iqe.query_base import source_trajectory_contract
from iqe.training.trainer import train


def test_independent_process_initialization_hash():
    code = "from iqe.internal_baseline.scorer import initialization_rng,InternalScorer; from iqe.query_base import source_trajectory_contract; from iqe.evaluation.retention import module_hash\nwith initialization_rng(31415):\n m=InternalScorer(8,8,source_trajectory_contract(),hidden=16)\nprint(module_hash(m))"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    a = subprocess.check_output([sys.executable, "-c", code], env=env, text=True).splitlines()[-1]
    b = subprocess.check_output([sys.executable, "-c", code], env=env, text=True).splitlines()[-1]
    assert a == b


def test_real_trainer_two_stream_exact_resume_with_dropout(tmp_path):
    cfg = {"lr": 1e-4, "weight_decay": .01, "global_batch_size": 128, "micro_batch_size": 32,
           "max_optimizer_steps": 4, "warmup_steps": 1, "lr_min_ratio": .1, "grad_clip_norm": 1.,
           "validation_every_steps": 2, "checkpoint_every_steps": 2}
    with initialization_rng(1):
        x = {"world": torch.randn(3, 6, 8), "action": torch.randn(3, 4, 8), "current_ego": torch.randn(3, 4),
             "world_valid": torch.ones(3, 6, dtype=torch.bool), "action_valid": torch.ones(3, 4, dtype=torch.bool)}
        trajectories = torch.randn(3, 2, 8, 3)
    def new():
        with initialization_rng(42):
            m = InternalScorer(8, 8, source_trajectory_contract(), hidden=16, dropout=.1)
        sampler = TwoStreamSampler(["0", "1", "2"], [.95, .02, 0.], natural_seed=5, comparison_seed=6, pointwise_updates=1, bank_hash="b")
        return m, sampler
    def fetch(entries, device):
        ix = [int(e["scene_id"]) for e in entries]
        return {"context": {k: v[ix] for k, v in x.items()}, "trajectories": trajectories[ix],
                "scores": torch.tensor([[0., .95], [.5, .52], [.6, .6]])[ix],
                "valid": torch.ones(len(ix), 2, dtype=torch.bool), "components": torch.full((len(ix), 2, 6), .5),
                "component_valid": torch.ones(len(ix), 2, 6, dtype=torch.bool),
                "natural": torch.tensor([e["natural"] for e in entries]),
                "comparison": torch.tensor([e["comparison"] for e in entries])}
    def loss(m, b):
        return terms(m(b["context"], b["trajectories"]), b)
    weights = {"value": 1., "components": .2, "ranking": .2}
    whole, ws = new()
    a = train(whole, loss, fetch, ws, cfg, {"run": "unit"}, tmp_path/"whole", weights=weights, seed=17, denominator_function=denominators)
    split, ss = new()
    train(split, loss, fetch, ss, cfg, {"run": "unit"}, tmp_path/"split", weights=weights, seed=17, denominator_function=denominators, stop_after=2)
    restored, rs = new()
    b = train(restored, loss, fetch, rs, cfg, {"run": "unit"}, tmp_path/"split", weights=weights, seed=17, denominator_function=denominators, resume="auto")
    assert [h["sample_ids"] for h in a[2:]] == [h["sample_ids"] for h in b]
    assert [h["loss"] for h in a[2:]] == [h["loss"] for h in b]
    assert [h["gradient_norm"] for h in a[2:]] == [h["gradient_norm"] for h in b]
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in whole.state_dict().items())
    assert ws.state_dict() == rs.state_dict()
    ca = torch.load(tmp_path/"whole/step_000004.pt", weights_only=False)
    cb = torch.load(tmp_path/"split/step_000004.pt", weights_only=False)
    assert torch.equal(ca["rng_by_rank"][0]["cpu"], cb["rng_by_rank"][0]["cpu"])
