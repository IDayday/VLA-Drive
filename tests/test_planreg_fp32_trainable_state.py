import torch
from torch import nn
from navsim.agents.EpisodeDrive.layers.planning_registers import InternViTQVLoRALinear
from navsim.agents.EpisodeDrive.precision import audit_precision


def test_bf16_base_fp32_adapters_and_adam_moments_really_update():
    torch.manual_seed(20)
    model = InternViTQVLoRALinear(nn.Linear(8, 24).bfloat16(), rank=4)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5)
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = model(torch.randn(2, 9, 8).bfloat16()).float().square().mean()
        loss.backward()
        optimizer.step()
    result = audit_precision(model, optimizer, require_initialized_moments=True)
    assert set(result['adam_moment_storage']) == {'torch.float32'}
    for name, p in model.named_parameters():
        assert torch.equal(p, before[name]) != p.requires_grad
    assert model.base_layer.weight.dtype == torch.bfloat16
