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
