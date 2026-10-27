import copy

import pytest
import torch

from starVLA.rl.simwam_recipe.recovery_state import validate_boundary, validate_ranks, rollout_digest


def fixture():
    cfg=dict(max_steps=50000,action_shape=[8,3],variant='R_CANONICAL',
             model_family='QwenWorldPostCoT',prediction_type='x0')
    identity=dict(schema='world_postcot_cleanx_canonical_RL_v1',recipe=cfg,
                  world=8,sha256='run',projected_condition_identity='observation')
    state=dict(updates=0,collections=1,next_inner=0,phase='ROLLOUT_COMMITTED',
               scene_cursor=8,raw_scene_exposures=8,new_candidates=64)
    reward=torch.arange(8,dtype=torch.float64)[None]/10
    std=reward.std(unbiased=False)
    advantage=((reward-reward.mean(1,keepdim=True))/(std+1e-8)).float()[0]
    raw_logp=torch.zeros(8,3,8,3)
    rollout=dict(cond_bg=dict(schema='world_postcot_frozen_observation_v1',token='scene',
        identity='observation',repeats=8),token='scene',log='log',chain=torch.zeros(8,11,8,3),
        timesteps=torch.arange(10),deltas=torch.ones(10),stoch_idx=torch.tensor([7,8,9]),
        old_logp=raw_logp.mean((-1,-2)),old_probability=dict(source_clamped_logprob=raw_logp),
        records=[],rewards=reward.float(),advantages=advantage,adv_mask=torch.ones(8),
        global_std=float(reward.float().double().std(unbiased=False)),raw_group_std=std,
        advantage_statistics={},behavior_version=dict(run='run',update=0))
    # Match the actual fresh FP32-reward -> FP64-statistic conversion.
    offsets=rollout['rewards'].double()-rollout['rewards'].double()[:,:1]
    rollout['advantages']=((offsets-offsets.mean(1,keepdim=True))/(rollout['global_std']+1e-8)).float()[0]
    return state,rollout,identity


def test_world_3d_committed_chain_and_condition_are_validated_and_hashed():
    state,rollout,identity=fixture()
    summary=validate_boundary(state,rollout,identity)
    assert validate_ranks([summary]*8)[0]['pending']
    before=rollout_digest(rollout)
    rollout['cond_bg']['identity']='other'
    assert rollout_digest(rollout)['cond_bg']!=before['cond_bg']
    with pytest.raises(ValueError,match='ancestry'):validate_boundary(state,rollout,identity)


@pytest.mark.parametrize('change',['shape','prediction','condition_schema','token','repeat','descriptor_extra'])
def test_world_recovery_rejects_wrong_actor_or_condition(change):
    state,r,i=fixture()
    if change=='shape':r['chain']=torch.zeros(8,11,8,4)
    if change=='prediction':i['recipe']['prediction_type']='velocity'
    if change=='condition_schema':r['cond_bg']['schema']='native_frozen_observation_v1'
    if change=='token':r['cond_bg']['token']='other'
    if change=='repeat':r['cond_bg']['repeats']=16
    if change=='descriptor_extra':r['cond_bg']['reward']=1
    with pytest.raises(ValueError):validate_boundary(state,r,i)


def test_original_native_and_ddp_contracts_are_preserved():
    state,r,i=fixture()
    i['schema']='native_qwen_sourceflow_RL_v1'
    r['cond_bg']['schema']='native_frozen_observation_v1'
    assert validate_boundary(state,r,i)['pending']
    i['schema']='simwam_recipe_DDP_v1'
    with pytest.raises(ValueError,match='chain'):validate_boundary(state,r,i)
    r['chain']=torch.zeros(8,11,8,4)
    assert validate_boundary(state,r,i)['pending']
