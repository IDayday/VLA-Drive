import copy
import numpy as np
import pytest
import torch
from iqe.specialist_study.objectives import pair_population,population_ranking,terms,denominators,StudySampler,SceneSampler
from iqe.specialist_study.metrics import select,metrics
from iqe.scoring.reference import COMPONENTS


def test_gap_cost_not_cancelled_and_q_not_squared():
    scores=torch.tensor([[.95,0.],[.02,0.]])
    pred=torch.tensor([[.5,.5],[.5,.5]],requires_grad=True)
    loss=population_ranking(pred,scores,torch.ones_like(scores,dtype=torch.bool))
    (loss.numerator/loss.denominator).backward()
    assert float(pred.grad[0,0]/pred.grad[1,0])==pytest.approx(.95/.02,rel=1e-5)
    pairs,weight=pair_population(scores.numpy(),np.ones((2,2),bool))
    assert weight[0]/weight[1]==pytest.approx(.95/.02,rel=1e-5)
    q=weight/weight.sum()
    # The sampled pair gradient has the same magnitude before q weighting.
    gradient=-5*np.ones(2)
    assert (q*gradient)[0]/(q*gradient)[1]==pytest.approx(.95/.02,rel=1e-5)


def test_variable_k_denominator_and_ties():
    truth=np.array([[0.,.95,.95],[0.,.02,0.]])
    valid=np.array([[1,1,1],[1,1,0]],bool)
    pair,weights=pair_population(truth,valid)
    assert pair.tolist()==[[0,0,1],[0,0,2],[1,0,1]]
    assert weights==pytest.approx([.95/3,.95/3,.02])
    pair,weight=pair_population(np.ones((4,1)),np.ones((4,1),bool))
    assert len(pair)==len(weight)==0


def batch(scores,natural,comparison):
    s=torch.tensor(scores,dtype=torch.float32);v=torch.isfinite(s)
    return {'scores':s,'valid':v,'components':torch.ones(*s.shape,6),'component_valid':v[...,None].expand(-1,-1,6),
            'natural':torch.tensor(natural),'comparison':torch.tensor(comparison)}


def test_microbatch_full_denominators_and_zero_pairs():
    b=batch([[.9,.1],[.8,.75],[.5,.5],[.2,float('nan')]],[True,False,True,True],[False,True,False,False])
    p=torch.tensor([[.6,.4]]*4,requires_grad=True);l=torch.zeros(4,2,6,requires_grad=True)
    d=denominators(b);t=terms({'values':p,'component_logits':l},b)
    reference=sum(x.numerator/d[k].clamp_min(1) for k,x in t.items());reference.backward();grad=p.grad.clone()
    p.grad=None;l.grad=None
    for start in (0,2):
        sub={k:v[start:start+2] for k,v in b.items()};part=terms({'values':p[start:start+2],'component_logits':l[start:start+2]},sub)
        sum(x.numerator/d[k].clamp_min(1) for k,x in part.items()).backward()
    assert torch.equal(p.grad,grad)
    one=batch([[.3],[float('nan')]],[True,True],[False,False]);out=terms({'values':torch.ones(2,1)*.4,'component_logits':torch.zeros(2,1,6)},one)
    assert out['ranking'].denominator==0
    assert torch.isfinite(out['value'].numerator)


def test_nonfinite_prediction_not_masked():
    b=batch([[.8,.4]],[True],[False])
    with pytest.raises(Exception,match='nonfinite'):
        terms({'values':torch.tensor([[float('nan'),.4]]),'component_logits':torch.zeros(1,2,6)},b)


@pytest.mark.parametrize('phase',[2999,3000])
def test_two_sampler_resume_at_stage_boundary(phase):
    a=StudySampler(['a','b'],np.array([[.9,.1],[.6,.7]]),np.ones((2,2),bool),seed=9,bank_hash='bank')
    a.step=phase;a.peek(128);state=copy.deepcopy(a.state_dict());next_ids=a.peek(128)
    b=StudySampler(['a','b'],np.array([[.9,.1],[.6,.7]]),np.ones((2,2),bool),seed=9,bank_hash='bank');b.load_state_dict(state)
    assert next_ids==b.peek(128)
    a.consume(128);b.consume(128)
    assert a.state_dict()==b.state_dict()
    assert sum(e['comparison'] for e in next_ids)==(0 if phase==2999 else 64)


def test_changed_q_rejected_and_pending_not_consumed():
    a=StudySampler(['a'],np.array([[.9,.1]]),np.ones((1,2),bool),seed=9,bank_hash='bank');a.peek(128)
    b=StudySampler(['a'],np.array([[.8,.1]]),np.ones((1,2),bool),seed=9,bank_hash='bank')
    with pytest.raises(Exception,match='identity'):b.load_state_dict(a.state_dict())
    assert a.state_dict()['natural_consumed']==0


def test_argmax_and_oracle_regret_identity():
    p=np.array([[.6,.6,.5],[.5,.55,.9],[.5,.51,.5]])
    assert select(p).tolist()==[0,2,1]
    assert select(p,margin=.02).tolist()==[0,2,0]
    scores=np.array([[1.,0.,.1],[.2,.3,.9],[.9,.8,.7]])
    result=metrics(scores,np.ones((3,3,len(COMPONENTS))),select(p),['a','b','c'],repetitions=10)
    assert result['selected_gain_01']==pytest.approx(result['oracle_gain_01']-result['regret_01'])


def test_exact_tie_follows_stable_id_after_candidate_permutation():
    ids=['expert_0','expert_1','expert_2'];p=np.array([[.2,.9,.9],[.8,.8,.8]])
    expected=[ids[i] for i in select(p,expert_ids=ids)]
    order=[2,0,1];permuted=[ids[i] for i in order]
    assert [permuted[i] for i in select(p[:,order],expert_ids=permuted)]==expected==['expert_1','expert_0']
    with pytest.raises(Exception,match='nonfinite'):
        select(np.array([[.2,np.nan]]),np.ones((1,2),bool))
    assert select(np.array([[.2,np.nan]]),np.array([[True,False]])).tolist()==[0]


def test_explicit_padding_nan_does_not_contaminate_loss_or_gradient():
    b=batch([[.8,float('nan')]],[True],[False])
    p=torch.tensor([[.5,float('nan')]],requires_grad=True)
    logits=torch.zeros(1,2,6,requires_grad=True)
    with torch.no_grad():logits[:,1]=float('nan')
    out=terms({'values':p,'component_logits':logits},b)
    loss=sum(v.numerator/v.denominator.clamp_min(1) for v in out.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(p.grad).all() and p.grad[0,1]==0
    assert torch.isfinite(logits.grad).all() and logits.grad[0,1].abs().sum()==0
