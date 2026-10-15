from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from iqe.contracts import TrajectoryContract
from iqe.internal_baseline.scorer import initialization_rng
from iqe.decision_selection.models import SceneRouter,SetSelector,choose
from iqe.decision_selection.objectives import cost_weights,cost_ce,fixed_normalizer
from iqe.decision_selection.data import split_components

torch.set_num_threads(2)


def context(batch=3):
    return dict(world=torch.randn(batch,5,12,requires_grad=True),action=torch.randn(batch,3,16,requires_grad=True),
                current_ego=torch.randn(batch,4,requires_grad=True),world_valid=torch.ones(batch,5,dtype=torch.bool),
                action_valid=torch.ones(batch,3,dtype=torch.bool))


def contract():
    return TrajectoryContract(horizon=4,dt=.5,mean=(0.,0.,0.,0.),scale=(10.,3.,1.,1.),raw_representation='normalized_xy_sincos',
                              frame='ego_relative',origin='rear_axle',yaw_unit='radian',statistics_source='test')


def test_majority_winner_is_not_lowest_expected_cost():
    rewards=torch.tensor([[.90,.92]]*95+[[.90,0.]]*5)
    valid=torch.ones_like(rewards,dtype=torch.bool)
    w=cost_weights(rewards,valid); assert torch.allclose(w.mean(0),torch.tensor([.045,.019]),atol=1e-7)
    z=fixed_normalizer(rewards,valid)
    shared_logits=torch.zeros(2,requires_grad=True)
    term=cost_ce(shared_logits.expand(100,-1),rewards,valid,z)
    (term.numerator/term.denominator).backward()
    assert shared_logits.grad[0]<0 and shared_logits.grad[1]>0
    optimum=w.sum(0)/w.sum(); assert optimum.argmax()==0
    assert float(optimum[0])==pytest.approx(.045/.064,abs=1e-6)
    # The forbidden scene-normalized classifier reverses the decision.
    assert (w/w.sum(-1,keepdim=True)).mean(0).argmax()==1


def test_weights_offsets_ties_mask_k1_and_microbatch():
    r=torch.tensor([[.1,.9,.4],[.5,.5,.5],[.6,.3,float('nan')]])
    valid=torch.isfinite(r);w=cost_weights(r,valid)
    assert (w>=0).all() and w[2,2]==0 and w[0,0]==0
    shifted=r+.05;assert torch.allclose(w,cost_weights(shifted,valid))
    logits=torch.randn(3,3,requires_grad=True);z=fixed_normalizer(r,valid)
    full=cost_ce(logits,r,valid,z);g=torch.autograd.grad(full.numerator/full.denominator,logits,retain_graph=True)[0]
    loss=sum(cost_ce(logits[i:i+1],r[i:i+1],valid[i:i+1],z).numerator/3 for i in range(3))
    assert torch.allclose(torch.autograd.grad(loss,logits)[0],g)
    assert g[1].eq(0).all() and g[2,2]==0
    one=torch.randn(2,1,requires_grad=True);t=cost_ce(one,torch.ones_like(one),torch.ones_like(one,dtype=torch.bool),1.)
    t.numerator.backward();assert t.numerator==0 and one.grad.eq(0).all()
    with pytest.raises(ValueError):cost_ce(torch.full((1,2),float('nan')),torch.zeros(1,2),torch.ones(1,2,dtype=torch.bool),1.)
    with pytest.raises(ValueError):cost_weights(torch.zeros(1,2),torch.zeros(1,2,dtype=torch.bool))


def test_scratch_common_context_rng_restoration_and_no_label_input():
    old=torch.get_rng_state().clone()
    with initialization_rng(73):a=SceneRouter(12,16,['expert_0','expert_1','expert_2','expert_3'],hidden=32)
    assert torch.equal(old,torch.get_rng_state())
    with initialization_rng(73):b=SetSelector(12,16,contract(),hidden=32)
    assert all(torch.equal(a.context.state_dict()[k],v) for k,v in b.context.state_dict().items())
    with initialization_rng(74):c=SceneRouter(12,16,['expert_0','expert_1','expert_2','expert_3'],hidden=32)
    assert not torch.equal(a.query,c.query)
    x=context();result=a(x)
    with pytest.raises(TypeError):a(x,torch.randn(3,4,4,3))
    with pytest.raises(ValueError):a(x|{'scores':torch.rand(3,4)})
    result['decision_logits'].sum().backward()
    assert all(x[k].grad is None for k in ('world','action','current_ego'))
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters())


def test_set_permutation_padding_duplicates_and_trajectory_dependence():
    torch.manual_seed(91);m=SetSelector(12,16,contract(),hidden=32).eval();x=context()
    t=torch.randn(3,4,4,3,requires_grad=True);role=torch.tensor([[1,0,0,0]]*3,dtype=torch.bool)
    a=m(x,t,role);perm=torch.tensor([2,0,3,1]);b=m(x,t[:,perm],role[:,perm])
    assert torch.allclose(a['decision_logits'][:,perm],b['decision_logits'],atol=1e-6)
    ids=['expert_0','expert_1','expert_2','expert_3'];sid=choose(a['decision_logits'],ids)
    psid=choose(b['decision_logits'],[ids[i] for i in perm]);assert torch.equal(perm[psid],sid)
    duplicate=t.detach().clone();duplicate[:,2]=duplicate[:,1]
    d=m(x,duplicate,role)['decision_logits'];assert torch.allclose(d[:,1],d[:,2],atol=1e-6)
    changed=m(x,t.detach().flip(2),role)['decision_logits'];assert not torch.allclose(changed,a['decision_logits'])
    padded=t.detach().clone();padded[:,3]=float('nan');out=m(x,padded,role)
    assert out['decision_probabilities'][:,3].eq(0).all() and torch.isfinite(out['decision_probabilities']).all()
    a['decision_logits'].sum().backward();assert t.grad is None
    assert all(x[k].grad is None for k in ('world','action','current_ego'))


def test_stable_ties_and_logit_margin():
    ids=['expert_3','expert_0','expert_1','expert_2']
    logits=torch.tensor([[.3,.3,.3,.3],[1.,0.,1.,0.]])
    assert choose(logits,ids).tolist()==[1,2]
    assert choose(logits,ids,gamma=1.).tolist()==[1,1]
    assert choose(logits,ids,gamma=.5).tolist()==[1,2]


def test_group_parent_components_cannot_cross_inner_split():
    def s(i,log,group,parent=None):return SimpleNamespace(scene_id=i,source_log_id=log,source_group_id=group,parent_group_id=parent)
    scenes=[s('a','L1','G1'),s('b','L1','G2'),s('c','L2','G3','G1')]+[s(str(i),'L'+str(i),'G'+str(i)) for i in range(4,20)]
    fit,val=split_components(scenes,123)
    assert {'a','b','c'}<=fit or {'a','b','c'}<=val
    assert not fit&val and fit|val=={s.scene_id for s in scenes}
    assert (fit,val)==split_components(scenes[::-1],123)
