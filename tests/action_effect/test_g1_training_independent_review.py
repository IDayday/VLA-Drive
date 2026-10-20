"""Independent CPU checks; historical defects stay tied to their source SHA.

These tests do not simulate trajectories or constitute a real GPU qualification.
"""
from copy import deepcopy
import os
import random
import signal
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

from tools.action_effect import training
from tools.action_effect.common import artifact, digest, write_json
from starVLA.model.modules.action_effect.effect_model import FeedbackLabelSpec
from starVLA.model.modules.action_effect.effect_losses import (
    effect_loss_contributions, combine_contributions, reduce_contributions,
)


torch.set_num_threads(2)
WEIGHTS = dict(value=1., ranking=.2, events=.2, temporal=.2, uncertainty=.05, repair=.2)


def historical_training():
    """Load the reviewed historical implementation without changing the tree."""
    name = "tools.action_effect._independent_review_training_f43eca75"
    module = types.ModuleType(name)
    module.__package__ = "tools.action_effect"
    sys.modules[name] = module
    source = subprocess.check_output(
        ["git", "show", "f43eca756bc22663d0a2c2c7d65ed14d4b923934:tools/action_effect/training.py"], text=True)
    exec(compile(source, "training.py@f43eca75", "exec"), module.__dict__)
    return module


def recipe():
    return {"training": dict(learning_rate=1e-3, weight_decay=.01, warmup_updates=2,
                            batch_scenes=4, microbatch_scenes=2, gradient_clip=1.,
                            validation_every=2, max_updates=6)}


def fit(module, root, model, *, resume=False, stop_after=None, pair_rng=None):
    seen = []
    pair_rng = pair_rng if pair_rng is not None else torch.Generator().manual_seed(913)

    def step(m, ids, index):
        pairs = torch.randint(17, (len(ids),), generator=pair_rng)
        seen.append((ids, pairs.tolist()))
        x = torch.tensor(ids).float()[:, None] + pairs.float()[:, None] / 100
        x = x + .01 * torch.randn_like(x)
        y = float(random.random()) + float(np.random.rand())
        loss = (m(x) - y).square().mean()
        loss.backward()
        return {"loss": float(loss.detach())}

    def validate(m, index):
        # Validation must not consume any production training RNG stream.
        random.random(); np.random.rand(); torch.randn(7)
        return {"step": index}

    result = module.run_training(model, output=root, identity={"source_commit": "independent_fixture"},
        dataset_size=11, step_fn=step, validation_fn=validate, cfg=recipe(), device="cpu", seed=31,
        max_updates=6, stop_after=stop_after, resume=resume, stateful={"pair_rng": pair_rng})
    return result, seen


def test_historical_pause_after_periodic_receipt_is_an_actual_conflict(tmp_path, monkeypatch):
    legacy = historical_training()
    original = legacy.write_json

    def write_then_signal(path, value, **kwargs):
        original(path, value, **kwargs)
        if str(path).endswith("step_000002.COMPLETE.json"):
            os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(legacy, "write_json", write_then_signal)
    with pytest.raises(ValueError, match="Immutable content conflict"):
        fit(legacy, tmp_path, torch.nn.Linear(1, 1))
    receipt = legacy.read_json(tmp_path / "step_000002.COMPLETE.json")
    assert receipt["termination_reason"] == "RUNNING_CHECKPOINT"
    assert len(list(tmp_path.glob("step_*.pt"))) == 1


def test_historical_orphan_after_payload_blocks_resaved_same_step(tmp_path, monkeypatch):
    legacy = historical_training()
    original = legacy.write_json

    def crash_before_receipt(path, value, **kwargs):
        if str(path).endswith("step_000004.COMPLETE.json"):
            raise RuntimeError("injected: payload persisted, receipt absent")
        return original(path, value, **kwargs)

    monkeypatch.setattr(legacy, "write_json", crash_before_receipt)
    with pytest.raises(RuntimeError, match="payload persisted"):
        fit(legacy, tmp_path, torch.nn.Linear(1, 1))
    orphan = (tmp_path / "step_000004.pt").read_bytes()
    assert not (tmp_path / "step_000004.COMPLETE.json").exists()
    monkeypatch.setattr(legacy, "write_json", original)
    with pytest.raises(ValueError, match="Immutable content conflict"):
        fit(legacy, tmp_path, torch.nn.Linear(1, 1), resume=True)
    assert (tmp_path / "step_000004.pt").read_bytes() == orphan


def assert_tree_equal(first, second):
    if torch.is_tensor(first):
        assert torch.equal(first, second)
    elif isinstance(first, np.ndarray):
        assert np.array_equal(first, second)
    elif isinstance(first, dict):
        assert set(first) == set(second)
        for key in first: assert_tree_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second): assert_tree_equal(a, b)
    else:
        assert first == second


def test_current_same_runtime_resume_restores_scene_pair_all_rng_and_adam(tmp_path):
    torch.manual_seed(5)
    initial = torch.nn.Linear(1, 1)
    full, split = deepcopy(initial), deepcopy(initial)
    a, ids_a = fit(training, tmp_path / "full", full)
    _, ids_b = fit(training, tmp_path / "split", split, stop_after=3)
    random.random(); np.random.rand(5); torch.randn(11)
    resumed = torch.nn.Linear(1, 1)
    b, ids_c = fit(training, tmp_path / "split", resumed, resume=True,
                   pair_rng=torch.Generator().manual_seed(999))
    assert ids_a == ids_b + ids_c
    for p, q in zip(full.parameters(), resumed.parameters()): assert torch.equal(p, q)
    first = training.resolve_checkpoint(tmp_path / "full", a["run_identity"])
    second = training.resolve_checkpoint(tmp_path / "split", b["run_identity"])
    for key in ("model", "optimizer", "scheduler", "sampler", "consumed_scenes", "rng", "stateful"):
        assert_tree_equal(first[key], second[key])
    (tmp_path / "split" / "RESULT.json").unlink()
    before = training.rng_state()
    c, ids_d = fit(training, tmp_path / "split", resumed, resume=True)
    assert ids_d == [] and c["segment_updates"] == 0 and c["status"] == "COMPLETE"
    assert_tree_equal(before, training.rng_state())
    assert c["checkpoint"] == second["_resolved_artifact"]


def targets(n=5):
    spec = FeedbackLabelSpec.from_label_arrays({"names": ["score", "progress_raw_meters", "event_time"],
                                               "units": ["0..1", "m", "s"]})
    before = torch.tensor([[0., 2., .5], [.5, 3., 2.], [.2, 7., 1.], [.3, 9., 2.], [.8, 10., 3.]])
    after = before + torch.tensor([[.95, 1., 0.], [-.02, 2., -.1], [0., 0., 0.], [.1, -2., .5], [-.3, 3., 0.]])
    valid = torch.tensor([[True, True, False], [True, False, True], [False, False, False],
                          [True, False, False], [True, True, True]])
    result = dict(before=before, after=after, before_valid=valid, after_valid=valid.clone(),
        event_targets=torch.tensor([[0, 1, 2]] * n),
        event_valid=torch.tensor([[True] * 3, [True, False, False], [False] * 3, [True] * 3, [True] * 3]),
        temporal_targets=torch.zeros(n, 41, 3), temporal_valid=torch.ones(n, 41, 3, dtype=torch.bool),
        repair_targets=torch.tensor([1., 0., 0., 1., 0.]), repair_valid=torch.tensor([True, False, False, True, True]),
        label_spec=spec, mode="delta_feedback")
    result["temporal_valid"][1:3] = False
    return result


def predictions(theta, n=5):
    return dict(mean_delta=theta[:3].expand(n, -1), logvar_delta=theta[3:6].expand(n, -1),
        absolute_before=None, absolute_after=None,
        event_logits=theta[6:15].reshape(1, 3, 3).expand(n, -1, -1),
        temporal_delta=theta[15:18].reshape(1, 1, 3).expand(n, 41, -1), repair_logit=theta[18].expand(n))


def test_g1_real_loss_composition_microbatch_matches_global_different_masks():
    first = torch.linspace(-.15, .15, 19, requires_grad=True)
    second = first.detach().clone().requires_grad_()
    t = targets(); p = predictions(first)
    full = reduce_contributions(effect_loss_contributions(p, **t), WEIGHTS)
    full.total.backward()
    pieces = []
    for lo, hi in ((0, 2), (2, 3), (3, 5)):
        subt = {k: v[lo:hi] if torch.is_tensor(v) else v for k, v in t.items()}
        subp = {k: v[lo:hi] if torch.is_tensor(v) else v for k, v in predictions(second).items()}
        pieces.append(effect_loss_contributions(subp, **subt))
    micro = reduce_contributions(combine_contributions(pieces), WEIGHTS)
    micro.total.backward()
    # Declared FP32 accumulation tolerance; not an assertion of bitwise CUDA/DDP equivalence.
    torch.testing.assert_close(micro.total, full.total, rtol=0, atol=1e-6)
    torch.testing.assert_close(second.grad, first.grad, rtol=0, atol=1e-6)
    for key in WEIGHTS:
        torch.testing.assert_close(micro.terms[key], full.terms[key], rtol=0, atol=1e-6)
    assert micro.contributions["ranking"].denominator == pytest.approx(1.37)


def source_model(commit):
    name = "starVLA.model.modules.action_effect._independent_review_model_" + commit
    module = types.ModuleType(name)
    sys.modules[name] = module
    source = subprocess.check_output(["git", "show", commit + ":starVLA/model/modules/action_effect/effect_model.py"], text=True)
    exec(compile(source, "effect_model.py@" + commit, "exec"), module.__dict__)
    return module


def test_absolute_repair_auxiliary_old_missing_modified_gradient_is_fixed():
    spec = FeedbackLabelSpec.from_label_arrays({"names": ["score"], "units": ["0..1"]})
    context = {"memory": torch.ones(1, 2, 4), "ego": torch.tensor([[1., 0.]]),
               "mask": torch.ones(1, 2, dtype=torch.bool)}
    base = torch.zeros(1, 8, 3); base[..., 0] = torch.arange(1, 9)
    for commit, should_observe_modified in (("9d6dddce", False), ("5ee472df", True)):
        module = source_model(commit)
        cfg = module.EffectModelConfig(mode="absolute_feedback", context_dim=4,
            world_tokens=1, action_tokens=1, ego_dim=2, hidden_dim=8, layers=1, heads=2)
        model = module.ActionEffectModel(cfg, spec)
        proposed = base.clone(); proposed[..., 1] = .3
        proposed.requires_grad_()
        value = model(context, base, proposed)["repair_logit"].sum()
        gradient, = torch.autograd.grad(value, proposed, allow_unused=True)
        observed = gradient is not None and bool(gradient.abs().sum() > 0)
        assert observed == should_observe_modified


def test_compact_content_gate_rejects_rehashed_labels_before_training(tmp_path, monkeypatch):
    """Synthetic checksum-gate test; native replay is checked separately on real data."""
    from types import SimpleNamespace
    from tools.action_effect.train_effect_model import PairData
    import tools.action_effect.build_effect_pairs as builder
    parent = {"binding_identity": "frozen-binding", "base_checkpoint": {"sha256": "base"}}
    parent["identity"] = digest(parent)
    write_json(tmp_path / "REGISTRATION.json", parent)
    write_json(tmp_path / "COMPLETE.json", {"registration_identity": parent["identity"]})
    row = {"modified_labels": {"values": [.2]}}
    row["identity"] = digest(row)
    manifest = {"schema": "action_effect_pair_dataset_v1", "status": "COMPLETE", "split": "train",
        "registration": artifact(tmp_path / "REGISTRATION.json"),
        "population_completion": artifact(tmp_path / "COMPLETE.json"),
        "registration_identity": parent["identity"], "entries": [row]}
    manifest["identity"] = digest(manifest)
    write_json(tmp_path / "original.json", manifest)
    altered = deepcopy(manifest)
    altered["entries"][0]["modified_labels"]["values"] = [.9]
    altered["entries"][0].pop("identity"); altered["entries"][0]["identity"] = digest(altered["entries"][0])
    altered.pop("identity"); altered["identity"] = digest(altered)
    write_json(tmp_path / "altered.json", altered)
    # Pure reconstruction is independently validated by the real 64-scene load.
    # Isolate here that matching self-hashes cannot bypass the expected source hash.
    monkeypatch.setattr(builder, "build", lambda _: {"pair_dataset": artifact(tmp_path / "original.json")})
    reader = SimpleNamespace(binding={"identity": "frozen-binding", "base_checkpoint": {"sha256": "base"}})
    with pytest.raises(ValueError, match="differ from their verified native"):
        PairData(tmp_path / "altered.json", reader)
