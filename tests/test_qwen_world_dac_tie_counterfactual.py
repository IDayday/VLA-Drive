"""Directed CPU checks of eligibility, missing fields and normalization scope."""
import pytest
import torch

from starVLA.rl.qwen_world_research_algorithms.dac_tie_counterfactual import (
    raw_tie_counterfactual,global_normalization_effect,centered_zero_group_std_effect,
)
from starVLA.rl.qwen_world_research_algorithms.safety_efficiency import safety_efficiency_parts


def materials():
    x=torch.ones(8,8,4,dtype=torch.float64)
    x[:,:,3]=torch.linspace(.1,.8,8,dtype=torch.float64)
    x[0,:,1]=0  # exactly the eligible DAC-only tie
    x[1,4:,2]=0  # mixed feasible
    x[2,:,0]=0;x[2,4:,2]=0  # allfail non-tie
    x[3,:,2]=0  # allfail equal coarse cost, wrong failure component
    x[4,:,1]=0  # eligible but field unavailable
    return x,safety_efficiency_parts(x)['A_raw']


def test_only_exact_subtype_changes_and_missing_is_not_filled():
    x,a=materials();before=a.clone()
    flags=torch.zeros(8,41,dtype=torch.bool)
    for k in range(8):flags[k,1:2+5*k]=True
    out=raw_tie_counterfactual(x,a,offroad_by_group={0:flags})
    assert torch.equal(a,before)
    assert torch.equal(out['raw_advantages'][1:],a[1:])
    assert out['groups'][4]['status']=='UNAVAILABLE_REAL_41STEP_FIELD'
    dense=flags[:,1:].double().mean(1)
    torch.testing.assert_close(out['raw_advantages'][0],dense.mean()-dense,rtol=0,atol=1e-16)
    assert int(out['raw_advantages'][0].argmax())==0
    assert int(out['raw_advantages'][0].argmin())==7
    assert out['groups'][0]['t0_all_drivable']
    assert bool(out['eligible'][0]) and bool(out['eligible'][4])
    with pytest.raises(TypeError):raw_tie_counterfactual(x,a,offroad_by_group={0:flags},raw_progress=100.)


def test_rejects_guessed_continuous_margin_or_incomplete_time_axis():
    x,a=materials()
    with pytest.raises(ValueError):raw_tie_counterfactual(x,a,offroad_by_group={0:torch.full((8,41),.5)})
    with pytest.raises(ValueError):raw_tie_counterfactual(x,a,offroad_by_group={0:torch.zeros(8,40)})


def test_other_raw_unchanged_but_global_std_rescales_other_final_advantages():
    x,a=materials();flags=torch.zeros(8,41,dtype=torch.bool)
    for k in range(8):flags[k,1:1+4*k]=True
    out=raw_tie_counterfactual(x,a,offroad_by_group={0:flags})
    n=global_normalization_effect(a,out['raw_advantages'])
    analytical=centered_zero_group_std_effect(n['original_std'],out['raw_advantages'][0])
    assert analytical['counterfactual_std']==pytest.approx(n['counterfactual_std'],abs=1e-15)
    assert n['unchanged_raw_normalized_scale']<1
    expected=(a[1:]/(n['counterfactual_std']+1e-8)).float()
    assert torch.equal(n['counterfactual_advantages'][1:],expected)
    assert not torch.equal(n['original_advantages'][1],n['counterfactual_advantages'][1])
    # An equal dense cost is still a true tie and must not invent credit.
    same=raw_tie_counterfactual(x,a,offroad_by_group={0:torch.ones(8,41,dtype=torch.bool)})
    assert torch.equal(same['raw_advantages'],a)
    assert not same['groups'][0]['t0_all_drivable']  # reported, never silently relabeled safe
