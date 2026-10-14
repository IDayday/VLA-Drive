"""Small independent mathematical and source-reference acceptance tests."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Normal, Independent, kl_divergence

from starVLA.rl.constrained_flow_noise.transition import gaussian_logprob, gaussian_kl_parts, density_views, flow_noise_transition
from starVLA.rl.constrained_flow_noise.noise_policy import NoisePolicy, KernelSpec, legacy_std_range
from starVLA.rl.constrained_flow_noise.vendor.rlinf_explore_noise_net import ExploreNoiseNet
from starVLA.rl.constrained_flow_noise.vendor import rlinf_sampler
from starVLA.rl.constrained_flow_noise.component_advantages import component_advantages
from starVLA.rl.constrained_flow_noise.lagrange import CostMultipliers
from starVLA.rl.constrained_flow_noise.effective_group_collector import EffectiveGroupCollector
from starVLA.rl.constrained_flow_noise.safety_costs import bounded_margin_cost, body_margins

FLOORS = [.01, 1e-8, 1e-8, 1e-8, .01, .01]


def upstream_module():
    path = Path(os.environ.get('CFN_UPSTREAM_ROOT', '/mnt/project/constrained-flow-noise-20261014/upstream'))/'rlinf/rlinf/models/embodiment/modules/explore_noise_net.py'
    assert path.is_file(), 'Fetch pinned upstream with lock_upstream before acceptance'
    logger = types.ModuleType('rlinf.utils.logging'); logger.get_logger = types.SimpleNamespace(info=lambda *a: None)
    spec = importlib.util.spec_from_file_location('official_noise', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'rlinf.utils.logging': logger}): spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('batch,group', [(1,1),(2,1),(1,16),(2,16)])
def test_actual_upstream_noise_outputs_and_all_gradients(batch, group):
    torch.manual_seed(3)
    official = upstream_module().ExploreNoiseNet(12, 4, [128,64], 'tanh', [.01,.1], 'learn')
    local = ExploreNoiseNet(12, 4, [128,64], 'tanh', [.01,.1], 'learn')
    local.load_state_dict(official.state_dict())
    x = torch.randn(batch*group*10,8,12,requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a, b = official(x), local(y)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    weight=torch.randn_like(a);(a*weight).sum().backward();(b*weight).sum().backward()
    torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
    for p,q in zip(official.parameters(),local.parameters()): torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    assert a.min()>=.01 and a.max()<=.1
    assert not local.logvar_min.requires_grad and not local.logvar_max.requires_grad
    assert set(dict(local.named_buffers()))=={'logvar_min','logvar_max'}


def test_logvar_bounds_are_standard_deviations_not_logged_values():
    n=ExploreNoiseNet(2,4,[128,64],'tanh',[.02,.08],'learn')
    for x,expected in [(-100.,.02),(0.,math.sqrt(.02*.08)),(100.,.08)]:
        torch.testing.assert_close(n.post_process(torch.full((2,8,4),x)),torch.full((2,8,4),expected))
    for values in [(0.,.1),(-1.,1.),(.1,.1),(.1,float('nan'))]:
        with pytest.raises(ValueError): KernelSpec('N_FLOW_NOISE',*values)


@pytest.mark.parametrize('b,g',[(1,1),(2,1),(1,16),(2,16)])
def test_openpi_to_ddp_mean_and_no_dt_sigma_scaling(b,g):
    torch.manual_seed(4);n=b*g*10
    x=torch.randn(n,8,4);v=torch.randn_like(x);sigma=torch.rand_like(x)*.09+.01
    steps=torch.arange(10).repeat(b*g)
    mean,std=rlinf_sampler.sample_mean_var(x,-v,steps,noise_method='flow_noise',noise_level=.1,num_steps=10,noise_std=sigma)
    ours=flow_noise_transition(x,v,sigma)
    torch.testing.assert_close(ours.mean,mean,atol=8e-7,rtol=2e-6)
    torch.testing.assert_close(ours.std,std,atol=0,rtol=0)
    assert torch.all(ours.std[steps==9]>0)


def test_probability_joint_chain_and_gaussian_kl_independent_oracles():
    torch.manual_seed(8)
    mu=torch.randn(2,16,10,8,4,dtype=torch.float64,requires_grad=True)
    logstd=torch.randn_like(mu,requires_grad=True);std=logstd.exp()
    x=torch.randn_like(mu);ref_mu=torch.randn_like(mu);ref_std=torch.rand_like(mu)+.2
    ours=gaussian_logprob(x,mu,std);dist=Normal(mu,std)
    torch.testing.assert_close(ours,dist.log_prob(x),atol=2e-12,rtol=1e-12)
    views=density_views(ours)
    torch.testing.assert_close(views['joint_transition_logp'],Independent(dist,2).log_prob(x))
    torch.testing.assert_close(views['joint_chain_logp'],Independent(dist,3).log_prob(x))
    a,b=gaussian_kl_parts(mu,std,ref_mu,ref_std)
    torch.testing.assert_close(a+b,kl_divergence(dist,Normal(ref_mu,ref_std)),atol=2e-12,rtol=1e-12)
    ours.sum().backward();assert mu.grad.abs().sum()>0 and logstd.grad.abs().sum()>0
    with pytest.raises(FloatingPointError):gaussian_logprob(x,mu,torch.zeros_like(std))


def test_unclipped_gaussian_score_function_gradient_direction():
    gen=torch.Generator().manual_seed(91)
    mu=torch.tensor(0.,dtype=torch.float64,requires_grad=True);logstd=torch.tensor(math.log(1.5),dtype=torch.float64,requires_grad=True)
    x=(mu+logstd.exp()*torch.randn(250000,generator=gen,dtype=torch.float64)).detach()
    reward=-(x-2).square()
    objective=(reward.detach()*gaussian_logprob(x,mu,logstd.exp())).mean()
    a,b=torch.autograd.grad(objective,(mu,logstd))
    assert abs(a.item()-4)<.10 and abs(b.item()+4.5)<.16


def test_ratio_one_and_changes_in_both_mean_and_sigma():
    x=torch.randn(2,16,10,8,4);mu=torch.zeros_like(x);s=torch.ones_like(x)*.03
    old=gaussian_logprob(x,mu,s).detach().mean((-2,-1))
    assert torch.equal((gaussian_logprob(x,mu,s).mean((-2,-1))-old).exp(),torch.ones_like(old))
    assert not torch.equal(gaussian_logprob(x,mu,s*1.001).mean((-2,-1)),old)
    assert not torch.equal(gaussian_logprob(x,mu+.001,s).mean((-2,-1)),old)


@pytest.mark.parametrize('equal_reward',[.1,.7,0.])
def test_constant_decimal_components_exact_zero(equal_reward):
    r=torch.full((3,16),equal_reward);c=torch.full((3,16,5),.3)
    a,stats=component_advantages(r,c,torch.ones(5),FLOORS)
    assert torch.equal(a,torch.zeros_like(a)) and (stats['component_z']==0).all()


def test_multiplier_has_uncancelled_linear_effect_and_tiny_EP_stays_small():
    r=torch.ones(2,16)*.9;c=torch.zeros(2,16,5);c[0,:8,0]=1
    lam=torch.ones(5,requires_grad=True)
    a,_=component_advantages(r,c,lam,FLOORS);b,_=component_advantages(r,c,lam.detach()*2,FLOORS)
    torch.testing.assert_close(b,2*a);assert (b[0,:8]<a[0,:8]).all()
    assert not a.requires_grad and lam.grad is None
    zero,_=component_advantages(r,c,torch.zeros(5),FLOORS);assert (zero==0).all()
    r[1,0]+=.00001
    tiny,_=component_advantages(r,torch.zeros_like(c),lam,FLOORS);assert tiny.abs().max()<.002


def test_component_reference_and_scene_rank_partition_invariance():
    torch.manual_seed(8);r=torch.rand(5,16);c=torch.rand(5,16,5);lam=torch.tensor([1.,2.,.2,.3,.5])
    a,_=component_advantages(r,c,lam,FLOORS,mode='official_scadv')
    x=torch.cat([r[...,None],c],-1).double(); z=(x-x.mean(1,keepdim=True))/(x.std(1,keepdim=True)+1e-4)
    oracle=(z*torch.cat([torch.ones(1),-lam])).sum(-1).float()
    torch.testing.assert_close(a,oracle,atol=3e-7,rtol=1e-6)
    full,_=component_advantages(r,c,lam,FLOORS)
    pieces=torch.cat([component_advantages(r[s],c[s],lam,FLOORS)[0] for s in [slice(0,2),slice(2,5)]])
    torch.testing.assert_close(full,pieces,rtol=0,atol=0)


def test_dual_sign_separation_and_exact_resume():
    for mean,direction in [(1.,1),(0.,-1)]:
        d=CostMultipliers([.5]*5);before=d.snapshot()
        costs=torch.full((16,16,5),mean,requires_grad=True)
        d.update(costs,collection=0);assert ((d.snapshot()-before)*direction>0).all() and costs.grad is None
        restored=CostMultipliers([.5]*5);restored.load_state_dict(d.state_dict())
        for obj in (d,restored):obj.update(torch.full((16,16,5),.7),collection=1)
        assert torch.equal(d.snapshot(),restored.snapshot())
        with pytest.raises(ValueError):d.update(costs,collection=1)


def row(active,index):
    a=torch.zeros(16)
    if active:a[0],a[1]=1.,-1.
    return {'advantage':a,'costs':torch.full((16,5),index/32),'candidate_ids':list(range(16))}


def test_collector_order_cap_first16_surplus_and_restore():
    ident={'version':1};lam=torch.ones(5);c=EffectiveGroupCollector(ident,lam)
    # Receive suffix before prefix; neither completion order nor cost chooses scenes.
    c.add_returned([(i,row(i%2==0,i)) for i in range(8,16)],ident,lam);assert not c.rows
    c.add_returned([(i,row(i%2==0,i)) for i in reversed(range(8))],ident,lam)
    assert c.accepted==list(range(0,16,2));assert c.next_count(100)==8
    restored=EffectiveGroupCollector.restore(c.state_dict());first=c.first_costs().clone()
    for obj in (c,restored):
        obj.add_returned([(i,row(True,i)) for i in reversed(range(16,24))],ident,lam)
        assert obj.accepted==list(range(0,16,2))+list(range(16,24))
        assert obj.done and obj.next_count(100)==0 and torch.equal(first,obj.first_costs())
        assert all(len(obj.rows[i]['candidate_ids'])==16 for i in obj.accepted)
    with pytest.raises(ValueError):c.add_returned([],{'version':2},lam)
    with pytest.raises(ValueError):c.add_returned([],ident,lam*2)


@pytest.mark.parametrize('active_count',[0,3,16])
def test_collector_all_zero_partial_and_finite_cap(active_count):
    c=EffectiveGroupCollector({'version':1},torch.ones(5))
    while not c.done:
        start=len(c.rows);n=c.next_count(100)
        c.add_returned([(i,row(i<active_count,i)) for i in range(start,start+n)],{'version':1},torch.ones(5))
    assert len(c.accepted)==active_count
    assert len(c.rows)==(16 if active_count==16 else 32)
    if active_count==0:assert c.first_costs().mean()>0


def test_margins_seams_holes_same_time_and_empty_are_explicit():
    from shapely.geometry import box,Polygon
    from shapely.ops import unary_union
    road=unary_union([box(-5,-5,0,5),box(0,-5,5,5)])
    body=np.array([[-1,-1],[-1,1],[1,1],[1,-1],[0,0]],dtype=float)
    bodies=np.tile(body,(41,1,1));objects=[[] for _ in range(41)];objects[1]=[box(2,-1,3,1)];objects[2]=[box(0,0,1,1)]
    g=body_margins(bodies,road,objects)
    assert g['road_clearance_m'][1]==4 and g['object_distance_m'][1]==1 and g['object_distance_m'][2]==0
    assert g['no_objects'][0] and g['object_distance_m'][0] is None
    assert abs(bounded_margin_cost(g['object_distance_m'],2,no_objects=g['no_objects'])-1.5/40)<1e-12
    hole=Polygon(box(-5,-5,5,5).exterior.coords,[box(-.1,-.1,.1,.1).exterior.coords])
    g=body_margins(bodies,hole,[[] for _ in range(41)]);assert g['road_clearance_m'][1]==0 and g['body_outside_union_area_m2'][1]>0
    with pytest.raises(ValueError):bounded_margin_cost([None]*41,1)
    a=np.ones(41)*3;a[0]=0;assert bounded_margin_cost(a,1)==0
