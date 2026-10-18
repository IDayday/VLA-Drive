import numpy as np
import pytest
import torch
from scripts.analysis.fm_transfer.analyze_saved_reward_counterfactual import compare_rewards
from scripts.analysis.fm_transfer.probe_sampler_transfer import probe_tokens,means,probe_generators


def rewards(metrics):
    nc,dac,ttc,ep,comfort=np.moveaxis(metrics,-1,0)
    return (torch.tensor(nc*dac*(10*ep+5*ttc+2*comfort)/17),
            torch.tensor(nc*dac*(5*ep+5*ttc+2*comfort)/12))


def test_progress_only_ranking_survives_reward_change():
    m=np.ones((8,8,5));m[:,:,3]=np.arange(64).reshape(8,8)/64
    out=compare_rewards(*rewards(m),m)
    assert out['counts']['progress_only_groups']==8
    assert out['counts']['progress_only_robust_sign_flips']==0
    assert out['advantage_cosine']>0.999999


def test_TTC_progress_tradeoff_changes_relative_advantage():
    m=np.ones((8,8,5));m[:,:,3]=0
    m[:,0,2]=0;m[:,0,3]=.9
    out=compare_rewards(*rewards(m),m)
    assert out['counts']['source_positive_unsafe_with_safe_alternative']==8
    assert out['counts']['canonical_positive_unsafe_with_safe_alternative']==0
    assert out['counts']['source_unsafe_outranks_safe_pairs']==56
    assert out['counts']['canonical_unsafe_outranks_safe_pairs']==0


@pytest.mark.parametrize('nc',[0,1])
def test_tied_groups_have_no_false_signal(nc):
    m=np.ones((8,8,5));m[:,:,0]=nc
    out=compare_rewards(*rewards(m),m)
    assert out['counts']['source_effective_groups']==0
    assert out['counts']['canonical_effective_groups']==0
    assert out['advantage_cosine'] is None


def test_probe_population_independent_of_input_order_and_excludes_dev():
    tokens=[f'{i:016x}' for i in range(512)]
    p=dict(train_tokens=tokens,dev_tokens=[],navtest_tokens=[])
    a=probe_tokens(p,dict(explicit_tokens=tokens))
    assert len(set(a))==64
    assert a==probe_tokens(p,dict(explicit_tokens=list(reversed(tokens))))
    p['dev_tokens']=[tokens[0]]
    with pytest.raises(ValueError):probe_tokens(p,dict(explicit_tokens=tokens))


def test_probe_rejects_partial_or_invalid_candidate_groups():
    with pytest.raises(ValueError):means([])
    m=dict(score=0,no_at_fault_collisions=0,drivable_area_compliance=1,
        time_to_collision_within_bound=0,ego_progress=0,comfort=1,driving_direction_compliance=1)
    out=means([dict(status='OK',metrics=m)]*8)
    assert out['all_zero'] and not out['feasible'] and out['finite_best_safe'] is None
    with pytest.raises(ValueError):means([dict(status='CACHE_MISSING',metrics=m)]*8)


def test_probe_RNG_matches_source_CPU_generator_contract():
    from starVLA.rl.simwam_recipe.vendor.action_math import ActionGRPOMixin
    root,innovation=probe_generators('0123456789abcdef')
    assert root.device==innovation.device==torch.device('cpu')
    state=innovation.get_state().clone()
    torch.randn((8,8,4),generator=root)
    assert torch.equal(innovation.get_state(),state)
    value=ActionGRPOMixin._randn(None,(8,8,4),torch.device('cpu'),torch.float32,innovation)
    _,same=probe_generators('0123456789abcdef')
    assert torch.equal(value,torch.randn((8,8,4),generator=same))


def test_SFT_cohort_membership_does_not_follow_RL_outcomes():
    from scripts.analysis.fm_transfer.analyze_fixed_sft_cohorts import cohort_statistics
    # Official submetrics and their exact canonical readout.
    base=np.array([[1,1,1,1,1,1,1],[.8,1,1,1,.52,1,1],[0,0,1,1,.5,1,1]],dtype=float)
    cur=base.copy();cur[0,0]=0;cur[0,1]=0;cur[2,0]=1;cur[2,1]=1;cur[2,4]=1
    out=cohort_statistics(base,cur,['a','b','c'])
    assert [v['scenes'] for v in out.values()]==[1,1,1]
    assert out['SFT_safe_score_ge09']['new_joint_violations']==1
    assert out['SFT_not_fully_safe']['joint_recoveries']==1
    assert out['SFT_safe_score_lt09']['negative_degradation_sum']==0
