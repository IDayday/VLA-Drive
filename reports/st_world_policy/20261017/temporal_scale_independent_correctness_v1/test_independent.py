"""Independent CPU algebra/call-chain tests; never a real-VLA qualification."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from starVLA.model.framework.st_world_model import SpaceTimeStudent
from starVLA.model.modules.st_world.losses import optimizer_batch_counts
from starVLA.model.modules.st_world.requests import sample_requests, gather_current, gather_future
from tools.st_world_policy.calibration_measurement import measure, preserved_rng, OBJECTIVES


class Action(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, condition, target, *, noise, times):
        self.calls.append((len(condition), tuple(target.shape), tuple(noise.shape), len(times)))
        # Deliberately use all real original-forward arguments. This is an
        # analytic action stub, not the production DiT implementation.
        prediction = condition.mean((1, 2))[:, None, None] + noise * .1 + times[:, None, None]
        return (prediction - target).square().mean()


class Readout(nn.Module):
    def forward(self, world, view, xy, seconds):
        return world[:, :1, :1] * (1 + seconds[:, :, None]) + xy[:, :, :1] * .1


class SmallActualForward(SpaceTimeStudent):
    def __init__(self):
        nn.Module.__init__(self)
        self.base = nn.Module()
        self.base.language = nn.Parameter(torch.tensor([.3]))
        self.base.foresight_queries = nn.Parameter(torch.tensor([.2]))
        self.base.action_model = Action()
        action = SimpleNamespace(noise_beta_beta=1., repeated_diffusion_steps=8,
                                 noise_s=.999, noise_beta_alpha=1.5)
        self.base.config = SimpleNamespace(framework=SimpleNamespace(action_model=action))
        self.base.amp = nullcontext
        self.base.build_planner_condition = lambda encoded: encoded['A']
        self.st_config = SimpleNamespace(current=True, future=True, delta=True,
            future_times_s=(.5, 1., 2., 4.), current_weight=1., future_weight=1., delta_weight=1.)
        self.sparse_readout = Readout()
        self._deployment_only = False
        self.calls = 0

    def encode_inputs(self, observations):
        self.calls += 1
        # One stochastic forward; no labels can enter this signature.
        draw = torch.rand(()) + .5
        a = (self.base.language * draw).expand(len(observations), 1, 1)
        w = (self.base.language * draw + self.base.foresight_queries).expand_as(a)
        return {'A': a, 'W': w}


def setup_case():
    model = SmallActualForward()
    request = sample_requests(4, 1, (1, 2), 2,
        patch_generator=torch.Generator().manual_seed(12), future=True,
        horizon_generator=torch.Generator().manual_seed(11))
    target = {'ego': torch.linspace(0., .1, 4*8*4).reshape(4, 8, 4),
        'current_dense_features': torch.zeros(4, 1, 1, 2, 1),
        'current_valid': torch.ones(4, 1, 1, 2, dtype=torch.bool),
        'future_dense_features': torch.zeros(4, 4, 1, 1, 2, 1),
        'future_valid': torch.ones(4, 4, 1, 1, 2, dtype=torch.bool),
        'future_times_s': torch.tensor([.5, 1., 2., 4.]).expand(4, -1)}
    target['current_valid'][1] = False
    target['future_valid'][2, 1, :, :, 0] = False
    _, cm = gather_current(target['current_dense_features'], target['current_valid'], request)
    _, fm = gather_future(target['future_dense_features'], target['future_valid'], request)
    counts = {k: int(v) for k,v in optimizer_batch_counts(1,current_valid=cm,future_valid=fm,
                                                        delta=True,distributed=False).items()}
    counts['ego_scenes'] = 4
    generators = {name: torch.Generator().manual_seed(seed) for name,seed in [('noise',9),('time',10)]}
    groups = {'last_language_block':['base.language'], 'world_queries':['base.foresight_queries']}
    return model,request,target,counts,generators,groups


def test_actual_space_time_forward_per_task_gradients_repeat8_and_fixed_requests(monkeypatch):
    # The actual forward applies this existing config validator; only replace
    # weighted total construction because the test intentionally uses a small
    # non-production config. Individual differentiable losses are untouched.
    import starVLA.model.framework.st_world_model as module
    monkeypatch.setattr(module,'weighted_temporal_loss',lambda losses,cfg:sum(losses.values()))
    model,request,target,counts,generators,groups = setup_case()
    closures = [lambda a=a: model([None],{k:v[a:a+1] for k,v in target.items()},
        requests=request.slice(a,a+1),global_counts=counts,
        noise_generator=generators['noise'],time_generator=generators['time']) for a in range(4)]
    params = [model.base.language,model.base.foresight_queries]
    expected_losses = dict.fromkeys(OBJECTIVES,0.)
    expected_grads = {task:[torch.zeros_like(p) for p in params] for task in OBJECTIVES}
    with preserved_rng(generators):
        for fn in closures:
            terms = fn()['losses']
            for i,task in enumerate(OBJECTIVES):
                expected_losses[task] += float(terms[task].detach())
                grad = torch.autograd.grad(terms[task],params,retain_graph=i<3,allow_unused=True)
                for dst,value in zip(expected_grads[task],grad):
                    if value is not None: dst.add_(value)
    model.calls=0; model.base.action_model.calls.clear()
    before_rng=torch.get_rng_state().clone()
    before_request=(request.patch_indices.clone(),request.future_pair.clone())
    measured=measure(model,closures,groups=groups,counts=counts,generators=generators,max_microbatches=4)
    assert measured['losses'] == expected_losses
    for task in OBJECTIVES:
        for at,group in enumerate(groups):
            assert measured['gradient_groups'][group]['norms'][task] == pytest.approx(float(expected_grads[task][at].norm()))
    assert model.calls == 4
    assert model.base.action_model.calls == [(8,(8,8,4),(8,8,4),8)]*4
    assert torch.equal(torch.get_rng_state(),before_rng)
    assert torch.equal(request.patch_indices,before_request[0])
    assert torch.equal(request.future_pair,before_request[1])
    assert all(p.grad is None for p in model.parameters())


def test_valid_loss_with_zero_shared_group_fails_even_when_union_nonzero():
    model,_,_,counts,generators,groups=setup_case()
    def forward():
        current=model.base.language.sum()+model.base.foresight_queries.sum()
        return {'losses':{'ego_fm':current,'current':current,'future':current,
                          'delta':model.base.language.sum()}}
    result=measure(model,[forward],groups=groups,counts=counts,generators=generators,max_microbatches=1)
    assert result['gradient_groups']['declared_shared_union']['norms']['delta'] > 0
    assert result['proposal']['status'] == 'UNCALIBRATABLE'
    assert 'zero_gradient_group:world_queries:delta' in result['proposal']['reasons']


def test_partial_forward_failure_restores_all_explicit_streams_and_gradients():
    model,_,_,counts,generators,groups=setup_case()
    before={k:g.get_state().clone() for k,g in generators.items()}
    default=torch.get_rng_state().clone()
    def broken():
        torch.rand(4)
        for g in generators.values(): torch.rand(4,generator=g)
        raise RuntimeError('injected-before-loss')
    with pytest.raises(RuntimeError,match='injected'):
        measure(model,[broken],groups=groups,counts=counts,generators=generators,max_microbatches=1)
    assert torch.equal(default,torch.get_rng_state())
    assert all(torch.equal(before[k],g.get_state()) for k,g in generators.items())
    assert all(p.grad is None for p in model.parameters())
