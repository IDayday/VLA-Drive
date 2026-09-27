import torch
from starVLA.model.modules.joint_world.public_adapters import LowRankDelta,AdaptedLinear


def test_zero_init_preservation_and_two_step_gradient_then_restore():
    torch.manual_seed(42);original=torch.nn.Linear(6,4).requires_grad_(False);delta=LowRankDelta(6,4,2,4)
    module=AdaptedLinear(original,delta);x=torch.randn(3,6);assert torch.equal(module(x),original(x))
    opt=torch.optim.SGD(delta.parameters(),lr=.1)
    module(x).square().sum().backward()
    assert delta.up.weight.grad.abs().sum()>0
    opt.step();opt.zero_grad();module(x).square().sum().backward()
    assert delta.down.weight.grad.abs().sum()>0
    other=LowRankDelta(6,4,2,4);other.load_state_dict(delta.state_dict(),strict=True)
    assert torch.equal(AdaptedLinear(original,other)(x),module(x))
    assert all(p.grad is None for p in original.parameters())
