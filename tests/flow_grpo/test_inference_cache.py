"""Snapshot API invariants; toy CPU tests do not certify real CUDA equivalence."""
import copy
import pytest
import torch
from torch import nn
from starVLA.rl.flow_grpo.inference_cache import inference_velocity_snapshot


class Head(nn.Module):
    def __init__(self):
        super().__init__();self.linear=nn.Linear(4,4);self.norm=nn.LayerNorm(4)
        self.register_buffer('scale',torch.tensor(2.))
    def predict_velocity(self,x,bucket,condition):return self.norm(self.linear(x)+condition)*self.scale


def test_scoped_values_parameter_objects_and_expiration():
    torch.manual_seed(42);head=Head().eval();x=torch.randn(2,4);condition=torch.randn(2,4)
    before={n:(id(p),p.detach().clone()) for n,p in head.named_parameters()}
    with torch.no_grad():
        expected=head.predict_velocity(x,None,condition)
        with inference_velocity_snapshot(head) as call:
            assert torch.equal(expected,call(x,None,condition))
            assert torch.equal(expected,call(x,None,condition))
        with pytest.raises(RuntimeError,match='expired'):call(x,None,condition)
    for n,p in head.named_parameters():assert id(p)==before[n][0] and torch.equal(p,before[n][1])
    head.predict_velocity(x,None,condition).square().sum().backward()
    assert all(p.grad is not None for p in head.parameters())


def test_snapshot_cannot_detach_training_and_does_not_survive_next_update():
    head=Head().eval();x=torch.randn(2,4);c=torch.randn(2,4)
    with pytest.raises(RuntimeError,match='autograd'):
        with inference_velocity_snapshot(head):pass
    with torch.no_grad():
        with inference_velocity_snapshot(head) as call:
            old=call(x,None,c)
            with torch.enable_grad(),pytest.raises(RuntimeError,match='autograd'):call(x,None,c)
        head.norm.bias.add_(.5)
        with inference_velocity_snapshot(head) as call:new=call(x,None,c)
        assert not torch.equal(old,new)
        assert torch.equal(new,head.predict_velocity(x,None,c))


def test_restores_on_forward_exception_and_preserves_parameter_aliases():
    head=Head().eval();head.alias=head.linear
    before={n:id(p) for n,p in head.named_parameters(remove_duplicate=False)}
    with torch.no_grad(),pytest.raises(RuntimeError):
        with inference_velocity_snapshot(head) as call:call(torch.ones(1,3),None,torch.ones(1,4))
    assert before=={n:id(p) for n,p in head.named_parameters(remove_duplicate=False)}
    assert head.alias.weight is head.linear.weight


def test_fp64_and_train_mode_are_rejected():
    with torch.no_grad():
        with pytest.raises(ValueError,match='eval'):
            with inference_velocity_snapshot(Head()):pass
        with pytest.raises(ValueError,match='BF16/FP32'):
            with inference_velocity_snapshot(Head().double().eval()):pass
