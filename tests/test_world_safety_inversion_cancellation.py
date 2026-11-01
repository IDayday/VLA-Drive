"""Two directed CPU checks; no model, runtime, scoring or optimizer needed."""
import copy

import numpy as np
import pytest
import torch

from starVLA.rl.qwen_world_safety_exploration.inversion_cancellation import cancel_inversions


def feedback(raw, ep, comfort, *, g=8):
    raw = torch.tensor(raw, dtype=torch.float64).reshape(-1, g)
    c = torch.ones(*raw.shape, 4, dtype=torch.float64)
    c[:, :, 3] = torch.tensor(ep, dtype=torch.float64).reshape_as(raw)
    comfort = torch.tensor(comfort, dtype=torch.float64).reshape_as(raw)
    score = c[:, :, 0]*c[:, :, 1]*(5*c[:, :, 3]+5*c[:, :, 2]+2*comfort)/12
    return raw, score, c, comfort


def test_exact_gates_order_preservation_sum_and_positive_budget_exhaustion():
    # Two high negatives; the higher-score one receives the lower-score positive
    # mass first. Some negative advantage remains when that positive is exhausted.
    args = feedback([[-.25,-.125,.125,0,0,0,0,0]],
                    [[.90,.80,1,1,1,1,1,1]], [[1,1,0,1,1,1,1,1]])
    before = copy.deepcopy(args); out = cancel_inversions(*args)
    assert out['pairs'] == [dict(group=0, high_candidate=0, lower_score_positive_candidate=2, transfer=.125)]
    assert out['updated_rawA'].tolist() == [[-.125,-.125,0.,0.,0.,0.,0.,0.]]
    assert out['counts']['remaining_negative_high'] == 2
    assert out['counts']['remaining_qualifying_pairs'] == 0
    assert torch.equal(out['group_sum_before'], out['group_sum_after'])
    assert all(torch.equal(x, y) for x, y in zip(args, before))
    assert torch.equal(args[0]+out['delta'], out['updated_rawA'])
    assert out['delta'][0,3:].eq(0).all()
    # A_safe belongs to the caller and is retained; only the EP component changes.
    a_safe = torch.full_like(args[0], .125); a_ep = args[0]-a_safe
    torch.testing.assert_close(a_safe+(a_ep+out['delta']), out['updated_rawA'], atol=0, rtol=0)
    # Equal canonical ties do not qualify; strict4 and strict-feasible gates apply.
    for mode in ('equal_score','negative_not_strict4','positive_not_feasible','below_threshold','allzero'):
        raw, score, c, comfort = feedback([[-.25,.25,0,0,0,0,0,0]],
            [[.9,1,1,1,1,1,1,1]], [[1,0,1,1,1,1,1,1]])
        if mode == 'equal_score': c[0,1,3]=.9;comfort[0,1]=1
        if mode == 'negative_not_strict4': comfort[0,0]=0
        if mode == 'positive_not_feasible': c[0,1,2]=0
        if mode == 'below_threshold': c[0,0,3]=.7
        if mode == 'allzero': raw.zero_()
        score = c[:,:,0]*c[:,:,1]*(5*c[:,:,3]+5*c[:,:,2]+2*comfort)/12
        got = cancel_inversions(raw,score,c,comfort)
        assert not got['pairs'] and got['delta'].eq(0).all() and torch.equal(got['updated_rawA'],raw)
    with pytest.raises(ValueError): cancel_inversions(args[0].float(),*args[1:])
    with pytest.raises(ValueError): cancel_inversions(args[0],args[1]+.01,*args[2:])


def test_exact_previous_numpy_replay_order_and_transfer_values_G8_G16():
    for g in (8, 16):
        # Equal-scored endpoints exercise candidate-index tie breaking. Inputs can
        # have nonzero group sum; cancellation must preserve it without recentering.
        raw = [-.20,-.125,.0625,.125,-.25,.125,.0625,0]+[0.]*(g-8)
        ep = [.9,.9,1,1,.8,1,1,1]+[1.]*(g-8)
        comfort = [1,1,0,0,1,0,0,1]+[1.]*(g-8)
        args = feedback([raw,raw], [ep,ep], [comfort,comfort], g=g)
        got = cancel_inversions(*args); expected=args[0].numpy().copy(); pairs=[]
        score=args[1].numpy(); c=args[2].numpy(); com=args[3].numpy()
        for group in range(2):
            a=expected[group]
            good=np.all(c[group,:,:3]==1,axis=1)&(com[group]==1)
            neg=sorted(np.flatnonzero((a<0)&(score[group]>=.9)&good),key=lambda i:(-score[group,i],i))
            pos=sorted(np.flatnonzero((a>0)&np.all(c[group,:,:3]==1,axis=1)),key=lambda j:(score[group,j],j))
            for i in neg:
                for j in pos:
                    if not (a[i]<0 and a[j]>0 and score[group,j]<score[group,i]):continue
                    amount=min(-a[i],a[j]);a[i]+=amount;a[j]-=amount
                    pairs.append(dict(group=group,high_candidate=int(i),lower_score_positive_candidate=int(j),transfer=float(amount)))
        assert torch.equal(got['updated_rawA'],torch.from_numpy(expected)) and got['pairs']==pairs
        torch.testing.assert_close(got['group_sum_after'],got['group_sum_before'],atol=2e-16,rtol=0)
        assert got['counts']['remaining_qualifying_pairs']==0
        assert got['counts']['triggered_groups']==2
