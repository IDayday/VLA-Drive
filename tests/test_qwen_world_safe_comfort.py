"""Directed CPU checks for the one-factor PPO_SAFE_COMFORT candidate."""
import copy
import pytest
import torch

from starVLA.rl.qwen_world_research_algorithms.safe_comfort import (
    COMFORT_SCHEMA, COMFORT_PROTOCOL, comfort_aligned_rewards,
    apply_comfort_alignment, finalize_safe_comfort_advantages,
    validate_comfort_boundary, validate_comfort_ranks,
)


def records(rank=0):
    out=[]
    for i in range(8):
        nc=0. if i==5 else .5 if i==6 else 1.
        dac=0. if i==7 else 1.
        ep=.2+.06*i+.005*rank;ttc=0. if i in (3,4) else 1.;c=float(i%2)
        score=nc*dac*(5*ep+5*ttc+2*c)/12
        out.append(dict(status='OK',metrics=dict(no_at_fault_collisions=nc,
            drivable_area_compliance=dac,ego_progress=ep,
            time_to_collision_within_bound=ttc,comfort=c,score=score),
            readouts=dict(canonical_PDMS=score)))
    return out


def test_safe_comfort_preserves_canonical_feedback_and_all_unsafe_rewards():
    rec=records();saved=copy.deepcopy(rec);values=comfort_aligned_rewards(rec)
    raw=values['canonical_rewards'];new=values['rewards']
    assert rec==saved and raw.shape==(1,8) and new.dtype==torch.float32
    assert bool(((0<=new)&(new<=1)).all())
    assert new[0,0]>raw[0,0] and new[0,2]>raw[0,2]
    assert torch.equal(new[0,3:],raw[0,3:])  # includes TTC, NC and DAC failures
    assert new[0,1]==raw[0,1]  # already comfortable
    # An exact pass gate must not grant a bonus to partial NC=.5.
    assert new[0,6]==raw[0,6]
    r=dict(records=rec,rewards=raw.clone(),old_logp=torch.zeros(8,3),chain=torch.zeros(8,11,8,3))
    old_id=id(r['old_logp']);chain_id=id(r['chain']);record_id=id(r['records'])
    apply_comfort_alignment(r)
    assert id(r['records'])==record_id and id(r['old_logp'])==old_id and id(r['chain'])==chain_id
    assert r['reward_transform']==COMFORT_PROTOCOL
    torch.testing.assert_close(r['canonical_rewards'],raw,rtol=0,atol=0)
    with pytest.raises(ValueError,match='exactly once'):
        apply_comfort_alignment(r)


def test_safe_comfort_native_boundary_keeps_full64_raw_std_and_rejects_false_reward(monkeypatch):
    import starVLA.rl.simwam_recipe.source_advantages as source
    from starVLA.rl.simwam_recipe.recovery_state import tensor_digest
    recs=[records(rank) for rank in range(8)]
    all_rewards=torch.cat([comfort_aligned_rewards(rec)['rewards'].double() for rec in recs])
    calls=[]
    monkeypatch.setattr(source.dist,'is_available',lambda:True)
    monkeypatch.setattr(source.dist,'is_initialized',lambda:True)
    monkeypatch.setattr(source.dist,'get_world_size',lambda:8)
    def gather(out,local):
        calls.append(1)
        for rank in range(8):out[rank].copy_(all_rewards[rank:rank+1])
    monkeypatch.setattr(source.dist,'all_gather',gather)
    identity=dict(world=8,sha256='new-arm',schema='world_postcot_cleanx_scope_RL_v1',
        projected_condition_identity='fixed-condition',recipe=dict(max_steps=50000,
            action_shape=[8,3],model_family='QwenWorldPostCoT',prediction_type='x0',
            adaptation='ATTN_LORA_MEMORY',variant='R_CANONICAL',
            research=dict(policy_objective='ppo',reward_transform=COMFORT_SCHEMA)))
    state=dict(updates=0,collections=1,next_inner=0,phase='ROLLOUT_COMMITTED',
               scene_cursor=8,raw_scene_exposures=8,new_candidates=64)
    summaries=[]
    for rank in range(8):
        raw=comfort_aligned_rewards(recs[rank])['canonical_rewards']
        r=dict(rewards=raw,records=recs[rank],chain=torch.zeros(8,11,8,3),
            old_logp=torch.zeros(8,3),timesteps=torch.arange(10),deltas=-torch.ones(10)/10,
            stoch_idx=torch.tensor([7,8,9]),
            old_probability={'source_clamped_logprob':torch.zeros(8,3,8,3)},
            token=f'scene{rank}',log=f'log{rank}',behavior_version=dict(run='new-arm',update=0,adapter_sha256='fixed'),
            cond_bg=dict(schema='world_postcot_trainable_memory_observation_v1',
                         token=f'scene{rank}',identity='fixed-condition',repeats=8))
        finalize_safe_comfort_advantages(r)
        assert len(calls)==rank+1
        assert r['global_std']==all_rewards.std(unbiased=False)
        offset=all_rewards-all_rewards[:,:1]
        expected=((offset-offset.mean(1,keepdims=True))/(all_rewards.std(unbiased=False)+1e-8)).float()
        torch.testing.assert_close(r['advantages'],expected[rank],rtol=0,atol=0)
        before=tensor_digest(r['advantages'])
        summaries.append(validate_comfort_boundary(state,r,identity))
        assert tensor_digest(r['advantages'])==before
    validate_comfort_ranks(summaries)
    bad=copy.deepcopy(r);bad['canonical_rewards'][0,0]+=.01
    with pytest.raises(ValueError,match='reward material differs'):
        validate_comfort_boundary(state,bad,identity)
    bad_id=copy.deepcopy(identity);bad_id['recipe']['research']['advantage_normalization']='anything'
    with pytest.raises(ValueError,match='raw-batchstd'):
        validate_comfort_boundary(state,r,bad_id)


def test_safe_comfort_rejects_scoring_failures_and_noncanonical_input():
    rec=records();rec[0]['status']='FAIL'
    with pytest.raises(ValueError,match='Scoring failure'):
        comfort_aligned_rewards(rec)
    rec=records();raw=comfort_aligned_rewards(rec)['canonical_rewards']
    with pytest.raises(ValueError,match='original canonical'):
        apply_comfort_alignment(dict(records=rec,rewards=raw+.01))
