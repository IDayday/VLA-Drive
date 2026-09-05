import random
import numpy as np
import torch
from navsim.agents.EpisodeDrive.gradient_diagnostics import compare_loss_gradients, preserve_rng


def test_same_batch_analytic_gradients_no_grad_or_rng_contamination():
    parameter = torch.nn.Parameter(torch.tensor([1.,2.,3.]))
    parameter.grad = torch.tensor([8.,9.,10.])
    original_grad = parameter.grad.clone()
    plan = parameter.square().sum()
    wm = -parameter.square().sum()
    results = compare_loss_gradients(plan, wm, {'shared': [parameter]}, .1)['shared']
    assert abs(results['weighted_wm_to_plan_ratio'] - .1) < 1e-6
    assert abs(results['cosine'] + 1) < 1e-6
    assert torch.equal(parameter.grad, original_grad)
    torch_state = torch.random.get_rng_state().clone()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    with preserve_rng():
        torch.rand(3)
        random.random()
        np.random.rand(3)
    assert torch.equal(torch_state, torch.random.get_rng_state())
    assert python_state == random.getstate()
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])


def test_functional_checkpoint_diagnostic_does_not_touch_original_leaves():
    from torch import nn
    from torch.utils.checkpoint import checkpoint
    from navsim.agents.EpisodeDrive.gradient_diagnostics import isolated_same_batch_audit
    class Agent(nn.Module):
        def __init__(self):
            super().__init__()
            self.world_model_enabled = True
            self.gradient_checkpointing = True
            self.backbone = nn.Module()
            self.backbone.planning_register_adapter = nn.Module()
            self.backbone.planning_register_adapter.planning_registers = nn.Parameter(torch.ones(2,3))
        def forward(self, features):
            p = self.backbone.planning_register_adapter.planning_registers
            fn = lambda value: value.square()
            value = checkpoint(fn, p, use_reentrant=True) if self.gradient_checkpointing else fn(p)
            return {'planning_registers':value}
        def compute_loss(self, features, targets, predictions):
            value = predictions['planning_registers']
            return {'plan_loss':value.sum(), 'wm_loss':-value.sum(), 'wm_weight_current':value.new_tensor(.1)}
    agent = Agent().train()
    parameter = next(agent.parameters())
    parameter.grad = torch.ones_like(parameter)
    calls = []
    parameter.register_hook(lambda grad: calls.append(grad))
    output = isolated_same_batch_audit(agent, {}, {})
    assert output['groups']['planning_registers']['plan_norm'] > 0
    assert output['groups']['planning_registers']['cosine'] < -.99
    assert torch.equal(parameter.grad, torch.ones_like(parameter)) and not calls
    assert agent.gradient_checkpointing is True
