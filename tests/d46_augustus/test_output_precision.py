"""Numerical precision and state lineage, not a claim of improved PDMS."""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from starVLA.rl.d46_augustus.output_precision import install_fp32_action_output,bf16_output_spacing_physical


def test_fp32_projection_is_real_fp32_and_keeps_parameters_Adam_and_RNG():
    torch.manual_seed(42);layer=nn.Linear(16,3);decision=SimpleNamespace(action_output=layer)
    optimizer=torch.optim.AdamW(layer.parameters(),lr=1e-6)
    x=torch.randn(2,8,16);layer(x).square().mean().backward();optimizer.step();optimizer.zero_grad()
    old_params=tuple(layer.parameters());old_states={id(k):v['step'].clone()for k,v in optimizer.state.items()}
    before=torch.get_rng_state().clone();new=install_fp32_action_output(decision)
    assert torch.equal(before,torch.get_rng_state())and tuple(new.state_dict())==tuple(layer.state_dict())
    assert all(a is b for a,b in zip(old_params,new.parameters()))
    with torch.autocast('cpu',dtype=torch.bfloat16):
        hidden=x.to(torch.bfloat16);actual=new(hidden);old=layer(hidden).float()
    expected=F.linear(hidden.float(),new.weight,new.bias)
    assert actual.dtype==torch.float32 and torch.equal(actual,expected)and not torch.equal(actual,old)
    actual.square().mean().backward();optimizer.step()
    assert all(v['step']==old_states[id(k)]+1 for k,v in optimizer.state.items())


def test_bf16_return_cast_does_not_remove_physical_quantization():
    z=torch.tensor([-.8,-.06,-.05]).repeat(8,1)
    spacing=bf16_output_spacing_physical(z)
    assert .13<float(spacing[0,0])<.131
    assert .005<float(spacing[0,1])<.0052
    assert .0004<float(spacing[0,2])<.0005
