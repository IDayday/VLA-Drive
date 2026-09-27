import copy
from types import SimpleNamespace
import torch
from starVLA.model.modules.joint_world.local_planner import LocalPlanningBridge, predict_local
from tests.local_interaction_mask_v2.test_local_flow import setup


def fixture():
    _, graph, current, _ = setup()
    current['context_mask'] = torch.tensor([[True, True, True, False, False]])
    cfg = dict(dim=16, heads=2, layers=2, steps=3, scale_m=20., edge_feature_dim=8, sampling_steps=2)
    return graph, current, cfg, torch.randn(1,3,16).bfloat16()


def test_equal_trainable_capacity_initialization_gate_and_risk_context():
    graph, current, cfg, native = fixture()
    models = [LocalPlanningBridge(16, cfg, mode) for mode in ('current', 'all', 'mask')]
    states = [{n:p for n,p in m.named_parameters() if p.requires_grad} for m in models]
    assert states[0].keys() == states[1].keys() == states[2].keys()
    for name in states[0]:
        assert all(torch.equal(states[0][name], s[name]) for s in states[1:])
    for model in models:
        condition,_ = model(native, current, ['scene'])
        assert torch.equal(condition, native.float())
        condition.square().mean().backward()
        assert model.adapter.gate.grad.abs() > 0
        model.zero_grad();model.adapter.gate.data.fill_(.3)
        condition,_ = model(native, current, ['scene'])
        condition.square().mean().backward()
        assert model.current_projection.weight.grad.norm() > 0
        poisoned = copy.deepcopy(current)
        for k in ('actor_features','current_xy','existence'):
            poisoned[k][:,-1] = float('nan')
        poisoned['context'][:,3:] = float('nan')
        actual,_=model(native,poisoned,['scene'])
        torch.testing.assert_close(actual,condition,atol=0,rtol=0)
        # Relevant uncertain current context is available to CURRENT_MEMORY too.
        changed=copy.deepcopy(current);changed['context'][:,1] += torch.arange(16)*10
        assert not torch.allclose(model(native,changed,['scene'])[0], condition)
        restored=LocalPlanningBridge(16,cfg,model.mode);restored.load_state_dict(model.state_dict(),strict=True)
        assert torch.equal(restored(native,current,['scene'])[0], condition)


def test_gate_zero_final_action_uses_original_noise_values_as_fp32():
    _,current,cfg,native=fixture()
    class Head:
        config=SimpleNamespace(action_horizon=3,action_dim=4)
        def predict_action(self,condition,initial_noise):
            assert condition.dtype==initial_noise.dtype==torch.float32
            return initial_noise+condition.mean(-1,keepdim=True)
    head=Head();base=predict_local(head,None,native,current,['scene'])
    for mode in ('current','all','mask'):
        bridge=LocalPlanningBridge(16,cfg,mode)
        actual=predict_local(head,bridge,native,current,['scene'])
        assert torch.equal(actual['normalized_actions'],base['normalized_actions'])
