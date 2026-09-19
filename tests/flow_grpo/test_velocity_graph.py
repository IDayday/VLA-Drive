import pytest
import torch
from torch import nn
from starVLA.rl.flow_grpo.velocity_graph import NoGradVelocityGraph,configure_velocity_graph
from starVLA.rl.flow_grpo.rollout import velocity


class Head(nn.Module):
    def __init__(self):super().__init__();self.linear=nn.Linear(3,3)
    def predict_velocity(self,x,bucket,condition):return self.linear(x)+condition+bucket[:,None,None]


class Policy(nn.Module):
    def __init__(self):super().__init__();self.action_model=Head()


def test_graph_option_never_detaches_trainable_path():
    policy=Policy().eval();configure_velocity_graph(policy,True)
    x=torch.ones(1,2,3);cond=torch.randn_like(x,requires_grad=True)
    result=velocity(policy,x,torch.zeros(1,dtype=torch.long),cond)
    result.sum().backward()
    assert cond.grad is not None and policy.action_model.linear.weight.grad is not None
    assert policy._flow_velocity_graph is None


def test_graph_rejects_autograd_before_cuda_calls():
    with pytest.raises(RuntimeError,match='eval/no-grad'):
        NoGradVelocityGraph(Head().eval(),None,None,None)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real CUDA graph required')
def test_cuda_graph_live_weights_inputs_and_output_lifetime():
    policy=Policy().cuda().eval();configure_velocity_graph(policy,True)
    x=torch.ones(1,2,3,device='cuda');cond=torch.randn_like(x);bucket=torch.zeros(1,device='cuda',dtype=torch.long)
    with torch.no_grad():
        eager=policy.action_model.predict_velocity(x,bucket,cond)
        first=velocity(policy,x,bucket,cond);saved=first.clone()
        assert torch.equal(first,eager)
        policy.action_model.linear.weight.add_(.1)
        second=velocity(policy,x*2,bucket+3,cond*2)
        assert torch.equal(second,policy.action_model.predict_velocity(x*2,bucket+3,cond*2))
        assert torch.equal(first,saved) and not torch.equal(first,second)
        with pytest.raises(ValueError,match='input profile'):
            velocity(policy,x.expand(2,-1,-1),bucket,cond)
    assert all(p.grad is None for p in policy.parameters())
