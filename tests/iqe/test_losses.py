import pytest
import torch
from iqe.scorer import ScorePrediction
from iqe.losses import scorer_terms, reduce_terms, scorer_denominators, router_terms


def pred(v, components=None):
    return ScorePrediction(v,components or {},torch.ones_like(v,dtype=torch.bool))


@pytest.mark.parametrize("k", [1,2,5])
def test_ties_rank_graph_zero(k):
    values = torch.full((2,k),.5,requires_grad=True)
    terms = scorer_terms(pred(values),values.detach(),torch.ones_like(values,dtype=torch.bool),{}, {})
    assert terms["ranking"].denominator == 0 and terms["ranking"].numerator == 0
    loss,logs = reduce_terms(terms); loss.backward()
    assert torch.isfinite(values.grad).all()


def test_ranking_direction_tie_filter_gap_weight():
    values = torch.tensor([[.5,.5,.5]],requires_grad=True)
    truth = torch.tensor([[.9,.1,.105]])
    terms = scorer_terms(pred(values),truth,torch.ones_like(truth,dtype=torch.bool),{}, {})
    rank = terms["ranking"].numerator
    assert terms["ranking"].denominator == 1
    torch.testing.assert_close(rank,torch.log(torch.tensor(2.)))
    rank.backward()
    assert values.grad[0,0] < 0 and values.grad[0,1] > 0 and values.grad[0,2] > 0
    assert values.grad[0,1] > values.grad[0,2]


def test_masks_soft_components_and_scenewise_rank():
    v = torch.tensor([[.4,.8,.2],[.5,.7,.1]],requires_grad=True)
    scores = torch.tensor([[.5,.9,float('nan')],[.8,.8,.8]])
    valid = torch.tensor([[True,True,False],[True,True,True]])
    components = {"nc":torch.tensor([[1.,.5,float('nan')],[1.,1.,1.]])}
    cm = {"nc":torch.tensor([[True,True,False],[False,False,False]])}
    output = pred(v,{"nc":torch.zeros_like(v,requires_grad=True),"missing":torch.zeros_like(v,requires_grad=True)})
    terms = scorer_terms(output,scores,valid,components,cm)
    assert terms["value"].denominator == 5 and terms["components"].denominator == 2 and terms["ranking"].denominator == 1
    data = {"scores":scores,"valid":valid,"trajectories":torch.zeros(2,3,4,3),"components":components,"component_valid":cm}
    expected = scorer_denominators(data)
    assert all(expected[k] == t.denominator for k,t in terms.items())
    loss,_=reduce_terms(terms);loss.backward()
    assert torch.isfinite(v.grad).all() and output.component_logits["missing"].grad is not None


def test_empty_rank_labels_backward_and_scale_reject():
    v=torch.tensor([[.5,.4]],requires_grad=True)
    terms=scorer_terms(pred(v),torch.tensor([[float('nan'),float('nan')]]),torch.zeros(1,2,dtype=torch.bool),{}, {})
    loss,_=reduce_terms(terms);loss.backward()
    assert loss == 0 and torch.equal(v.grad,torch.zeros_like(v))
    with pytest.raises(ValueError,match="\[0,1\]"):scorer_terms(pred(v),torch.tensor([[80.,90.]]),torch.ones(1,2,dtype=torch.bool),{}, {})


def test_router_no_valid_backward():
    logits=torch.randn(2,3,requires_grad=True)
    terms=router_terms(logits,torch.zeros(2,3),torch.zeros(2,3,dtype=torch.bool))
    loss,_=reduce_terms(terms);loss.backward()
    assert loss == 0 and torch.isfinite(logits.grad).all()
