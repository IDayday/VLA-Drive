from types import SimpleNamespace

import pytest
import torch

from scripts.analysis.constrained_flow_noise.stable_saved_probability import StableSavedProbability, replace_transitions
from starVLA.rl.constrained_flow_noise.transition import DiagonalTransition, density_views


class Toy(torch.nn.Module):
    def __init__(self, perturb=0.):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(.2))
        self.policy=SimpleNamespace(action_model=SimpleNamespace(num_timestep_buckets=1000))
        self.perturb=perturb

    def condition(self,raw):return raw*self.weight

    def distribution(self,x,buckets,condition,steps):
        return DiagonalTransition(x+self.weight,torch.ones_like(x))

    def saved_statistics(self,raw,chain,chunk=None):
        mean=chain[:,:,:-1]+self.weight
        logp=DiagonalTransition(mean,torch.ones_like(mean)).logprob(chain[:,:,1:])
        delta=torch.zeros_like(logp);delta[:,15,9]=self.perturb
        return dict(mean=mean,std=torch.ones_like(mean),**density_views(logp+delta))


def fixture(perturb):
    actor=Toy(0.)
    pg=dict(raw=torch.zeros(1,8,4),chain=torch.zeros(1,16,11,8,4),scene_key='one')
    pg['old_elementwise_logp']=actor.saved_statistics(pg['raw'],pg['chain'])['elementwise_logp'].detach().clone()
    actor.perturb=perturb
    controller=StableSavedProbability();controller.bind({'collection':1},0)
    return actor,pg,controller


def test_no_fallback_preserves_values_and_gradient():
    actor,pg,controller=fixture(0.)
    expected=actor.saved_statistics(pg['raw'],pg['chain'])
    actual,entry=controller.statistics(actor,actor.saved_statistics,pg)
    assert entry['indices']==[]
    for key in actual:assert torch.equal(actual[key],expected[key])
    assert torch.equal(torch.autograd.grad(actual['elementwise_logp'].sum(),actor.weight)[0],
                       torch.autograd.grad(expected['elementwise_logp'].sum(),actor.weight)[0])


def test_fixed_selection_survives_update_and_inner_restore():
    actor,pg,controller=fixture(.001)
    actual,entry=controller.statistics(actor,actor.saved_statistics,pg)
    assert entry['indices']==[159]
    assert torch.equal(actual['elementwise_logp'],pg['old_elementwise_logp'])
    saved=controller.state_dict()
    with torch.no_grad():actor.weight.add_(.1)
    resumed=StableSavedProbability();resumed.load_state_dict(saved,pending_inner=True)
    resumed.bind({'collection':1},0)
    changed,second=resumed.statistics(actor,actor.saved_statistics,pg)
    assert second['indices']==[159] and second['calls']==2
    assert (changed['mean_transition_logp']-pg['old_elementwise_logp'].mean((-2,-1))).abs().max()>.001
    assert torch.equal(pg['old_elementwise_logp'],actual['elementwise_logp'])
    resumed.bind({'collection':2},0)
    assert resumed.entries=={}


def test_missing_resume_decisions_and_early_refresh_fail():
    actor,pg,controller=fixture(.001)
    controller.statistics(actor,actor.saved_statistics,pg)
    with pytest.raises(ValueError):controller.bind({'collection':2},0)
    with pytest.raises(ValueError):StableSavedProbability().load_state_dict(None,pending_inner=True)
    with pytest.raises(ValueError):StableSavedProbability().load_state_dict(dict(schema='cfn_saved_transition_layout_v1',collection='x',entries={}),pending_inner=True)


def test_true_probability_mismatch_is_not_hidden():
    actor,pg,controller=fixture(.001)
    pg['old_elementwise_logp'][:,15,9]-=.1
    with pytest.raises(ValueError,match='unchanged probability guard'):
        controller.statistics(actor,actor.saved_statistics,pg)


def test_nonfinite_or_unbounded_fallback_stops():
    actor,pg,controller=fixture(0.)
    pg['old_elementwise_logp'][0,0,0,0,0]=float('nan')
    with pytest.raises(FloatingPointError):controller.statistics(actor,actor.saved_statistics,pg)
    flat=actor.saved_statistics(pg['raw'],pg['chain'])
    with pytest.raises(ValueError,match='Too many'):
        replace_transitions(actor,pg['raw'],pg['chain'],flat,list(range(9)))
