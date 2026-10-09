from copy import deepcopy
from dataclasses import replace
import pytest
import torch
from torch import nn
from iqe.model import IQEModel
from iqe.scorer import TrajectoryScorer, geometry_diagnostics
from iqe.router import SceneRouter, soft_targets
from iqe.selector import SelectionRule, RouterRule
from iqe.expert import assert_no_alias, AdapterExpert, ResidualExpert
from iqe.contracts import FeatureBundle, periodic_error
from iqe.losses import reduce_terms, scorer_terms
from iqe.evaluation.retention import frozen_snapshot, module_hash
from conftest import TensorAdapter


@pytest.mark.parametrize("k", [1,2,5])
def test_shared_once_and_k1_identity(k):
    adapter = TensorAdapter().eval()
    x = torch.randn(3,3,8)
    model = IQEModel(adapter).eval()
    for i in range(1,k): model.append_expert(f"expert_{i}")
    adapter.encodes = 0
    with torch.no_grad():
        out = model.forward_candidates(x)
    assert adapter.encodes == 1
    assert_no_alias(dict(model.experts))
    assert torch.equal(out.raw[:,0], adapter.action(out.features).raw)
    for i in range(1,k): assert torch.equal(out.raw[:,0], out.raw[:,i])
    if k == 1:
        trajectory, ids, _ = model.predict(x, lambda *args: pytest.fail("K1 selector called"))
        assert ids == ["expert_0"] * 3
        assert torch.equal(trajectory, out.trajectories[:,0])


def test_new_query_decoder_heads_update_old_buffers_frozen(features):
    model = IQEModel(TensorAdapter())
    model.append_expert("expert_1")
    model.set_trainable_stage("expert_train", "expert_1")
    model.train()
    assert not model.adapter.training and not model.experts["expert_0"].training
    assert not model.adapter.shared[1].training
    before = frozen_snapshot(model, "expert_1")
    new_before = module_hash(model.experts["expert_1"])
    optimizer = torch.optim.AdamW(model.optimizer_parameters(), lr=.01)
    model.audit_optimizer(optimizer)
    out = model.experts["expert_1"](features.detached())
    # Intermediate supervision enabled to exercise every necessary copied head.
    loss = sum(t.abs().mean() for t in out.intermediates)
    loss.backward()
    for part in (model.experts["expert_1"].query, model.experts["expert_1"].decoder, model.experts["expert_1"].heads):
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in part.parameters())
    assert all(p.grad is None for p in model.adapter.parameters())
    optimizer.step()
    assert module_hash(model.experts["expert_1"]) != new_before
    assert frozen_snapshot(model, "expert_1") == before
    bad = torch.optim.AdamW(list(model.optimizer_parameters()) + list(model.experts["expert_0"].parameters()))
    with pytest.raises(ValueError, match="optimizer parameter"): model.audit_optimizer(bad)


@pytest.mark.parametrize("variant", ["query_only","adapter","residual"])
def test_ablation_init_and_modes(features, variant):
    model = IQEModel(TensorAdapter()).eval()
    model.append_expert("expert_1", variant)
    assert torch.equal(model.experts["expert_0"](features).raw, model.experts["expert_1"](features).raw)
    model.set_trainable_stage("expert_train", "expert_1"); model.train()
    if variant in {"query_only","adapter"}:
        assert not model.experts["expert_1"].decoder.training
        assert not model.experts["expert_1"].heads.training
    if variant == "residual": assert not model.experts["expert_1"].initial.training


def test_storage_alias_rejected():
    a = TensorAdapter().action
    with pytest.raises(ValueError, match="storage alias"): assert_no_alias({"old": a,"new": a})


def test_inference_tensor_normalization_backward():
    with torch.inference_mode():
        f = FeatureBundle(torch.randn(2,3,8),torch.randn(2,1,8),torch.ones(2,3,dtype=torch.bool),"unit",("a","b"))
    normal = f.detached()
    assert not torch.is_inference(normal.scene)
    expert = TensorAdapter().action
    expert(normal).raw.sum().backward()
    assert expert.query.weight.grad is not None
    with pytest.raises(RuntimeError, match="Inference tensors"):
        nn.Linear(8,8)(f.scene).sum().backward()


def test_scorer_independent_candidates_and_detach(features):
    torch.manual_seed(1)
    features.scene.requires_grad_(True); features.ego.requires_grad_(True)
    scorer = TrajectoryScorer(8,8,["nc","dac"],16,2,2).eval()
    tau = torch.randn(3,2,4,3,requires_grad=True)
    original = scorer(features,tau)
    permuted = scorer(features,tau[:,[1,0]])
    duplicate = scorer(features,torch.cat([tau,tau[:,:1]],1))
    single = scorer(features,tau[:,:1])
    torch.testing.assert_close(original.values,permuted.values[:,[1,0]],atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(duplicate.values[:,:2],original.values,atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(duplicate.values[:,0],duplicate.values[:,2],atol=1e-7,rtol=1e-6)
    torch.testing.assert_close(single.values[:,0],original.values[:,0],atol=1e-7,rtol=1e-6)
    original.values.sum().backward()
    assert tau.grad is None and features.scene.grad is None and features.ego.grad is None
    assert scorer.trajectory_encoder[0].weight.grad.norm() > 0
    diagnostic = geometry_diagnostics(scorer, features, tau)
    assert set(diagnostic) == {"original","time_shuffle","scene_mismatch","candidate_spread"}


def test_scorer_padding_and_nan(features):
    features.valid_tokens[:,2] = False
    tau = torch.randn(3,2,4,3); tau[1,1] = float("nan")
    scorer = TrajectoryScorer(8,8,[],16,1,2)
    out = scorer(features,tau)
    assert torch.isfinite(out.values).all() and not out.valid[1,1]
    before = scorer(features,tau).values
    features.scene[:,2] = 99999
    torch.testing.assert_close(before,scorer(features,tau).values)


def test_router_refresh_expansion_and_preroute(features):
    ids = ("expert_0","expert_1")
    router = SceneRouter(8,8,ids,16)
    expanded = router.expanded((*ids,"expert_2"))
    assert torch.equal(router.head.weight,expanded.head.weight[:2])
    scores = torch.tensor([[.1,.2],[.3,.3],[float('nan'),.9]])
    valid = torch.tensor([[True,True],[True,True],[False,True]])
    p, good = soft_targets(scores,valid)
    assert torch.allclose(p.sum(-1),torch.ones(3)) and torch.equal(p[1],torch.tensor([.5,.5])) and p[2,0] == 0
    p2,_ = soft_targets(torch.cat([scores,torch.ones(3,1)],1),torch.cat([valid,torch.ones(3,1,dtype=torch.bool)],1))
    assert not torch.equal(p2[:,:2],p)
    model = IQEModel(TensorAdapter())
    model.append_expert("expert_1"); model.router = router
    calls = {eid:[] for eid in ids}
    hooks = [e.register_forward_pre_hook(lambda module,args,eid=eid: calls[eid].extend(args[0].scene_ids)) for eid,e in model.experts.items()]
    chosen = torch.tensor([1,0,1])
    tau, winners = model.forward_prerouted(None,lambda *_: chosen,features=features)
    assert calls == {"expert_0":["b"],"expert_1":["a","c"]} and winners == ["expert_1","expert_0","expert_1"]
    for h in hooks:h.remove()
    expected = model.forward_candidates(None,features=features).trajectories[torch.arange(3),chosen]
    torch.testing.assert_close(tau,expected)


def test_coordinates_periodic_roundtrip():
    c = TensorAdapter().contract
    physical = torch.randn(2,4,3); physical[0,0,2] = torch.pi - 1e-6; physical[0,1,2] = -torch.pi + 1e-6
    restored = c.physical(c.raw(physical))
    torch.testing.assert_close(physical,restored,atol=1e-6,rtol=1e-6)
    assert abs(periodic_error(physical[0,0,2],physical[0,1,2])) < 3e-6
    with pytest.raises(ValueError,match="shape"): c.raw(torch.randn(2,5,3))
