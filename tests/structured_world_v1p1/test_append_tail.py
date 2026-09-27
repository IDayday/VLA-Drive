import pytest
import torch
from starVLA.model.modules.structured_world.tokens import append_world_tokens
from starVLA.model.modules.structured_world.action_adapter import WorldToActionAdapter


def native():
    ids=torch.arange(7)[None].repeat(2,1)
    emb=torch.randn(2,7,16)
    mask=torch.tensor([[1,1,1,1,1,1,1],[1,1,1,1,1,0,0]])
    rope=torch.arange(7)[None,None].repeat(3,2,1)
    rope[:,0,1:3]=torch.tensor([[1,1],[2,3],[4,5]])
    return ids,emb,mask,rope,torch.tensor([[4,5],[3,4]]),torch.randn(2,3,16)


def test_native_prefix_mrope_and_action_unchanged():
    args=native();out=append_world_tokens(*args,99)
    for x,y in zip(args[:3],out[:3]):assert torch.equal(x,y[:,:7])
    assert torch.equal(args[3],out[3][:,:,:7])
    assert torch.equal(args[4],out[5])
    assert out[4].tolist()==[[7,8,9],[7,8,9]]
    assert out[3][0,:,-3:].tolist()==[[7,8,9],[5,6,7]]
    assert out[2][:,-3:].all()


def test_padding_actions_rejected():
    args=list(native());args[4][1,1]=5
    with pytest.raises(ValueError):append_world_tokens(*args,99)


def test_gate_zero_and_later_branch_gradient():
    torch.manual_seed(4)
    adapter=WorldToActionAdapter(16,4)
    action=torch.randn(2,3,16,requires_grad=True);world=torch.randn(2,5,16,requires_grad=True)
    out=adapter(action,world)
    assert torch.equal(out,action)
    out.square().sum().backward()
    assert adapter.gate.grad.abs()>0
    assert world.grad.abs().sum()==0
    with torch.no_grad():adapter.gate.add_(-.001*adapter.gate.grad)
    world.grad=None;adapter.zero_grad()
    adapter(action,world).square().sum().backward()
    assert world.grad.abs().sum()>0
