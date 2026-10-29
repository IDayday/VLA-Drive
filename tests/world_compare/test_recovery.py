"""Four-framework recovery: native descriptors, transactions and exact CPU resume."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from starVLA.model.modules.qwen_world_compare.contracts import CurrentContext
from starVLA.model.modules.qwen_world_compare.qwen_flow_adapter import DCurrentEncoding
from starVLA.rl.qwen_world_compare import checkpoint as cp
from starVLA.rl.qwen_world_compare.contracts import fingerprint
from starVLA.rl.qwen_world_compare.recovery_state import (
    RL_SCHEMA, OBSERVATION_SCHEMA, boundary_phase, rollout_digest,
    tensor_digest, validate_boundary, validate_ranks,
)


def identity(arch='B',world=1):
    x=dict(schema=RL_SCHEMA,architecture_id=arch,world=world,SFT_sha256='own_IL_parent',
        projected_condition_identity='frozen_current_own_IL_and_data_identity',
        recipe=dict(max_steps=8,action_shape=[8,3],prediction_type='x0',variant='R_CANONICAL'))
    x['sha256']=fingerprint(x)
    return x


def state():
    return dict(updates=0,collections=0,next_inner=0,phase='READY',scene_cursor=0,
                raw_scene_exposures=0,new_candidates=0)


class TinyActor(nn.Module):
    def __init__(self):
        super().__init__();self.lora_A=nn.Parameter(torch.tensor([.3,-.4,.2]));self.device=torch.device('cpu')


def bundle():
    a=TinyActor();o=torch.optim.AdamW(a.parameters(),lr=.01,betas=(.9,.95))
    s=torch.optim.lr_scheduler.LambdaLR(o,lambda step:min(1.,(step+1)/4))
    g=torch.Generator().manual_seed(71)
    return a,o,s,g


def collection(i,*,rank=0,generator=None,version=0):
    world=i['world'];reward=torch.stack([torch.arange(8).float()/10+.01*r for r in range(world)])
    std=reward.double().std(unbiased=False)
    offsets=reward[rank:rank+1].double()-reward[rank:rank+1,:1].double()
    advantages=((offsets-offsets.mean(1,keepdim=True))/(std+1e-8)).float()[0]
    raw=torch.randn(8,3,8,3,generator=generator)
    return dict(cond_bg=dict(schema=OBSERVATION_SCHEMA,token='scene'+str(version),
        identity=i['projected_condition_identity'],repeats=8),
        token='scene'+str(version),log='TRAIN_log',chain=torch.randn(8,11,8,3,generator=generator),
        timesteps=torch.linspace(1000,100,10),deltas=torch.full((10,),-.1),stoch_idx=torch.tensor([7,8,9]),
        old_probability=dict(source_clamped_logprob=raw),old_logp=raw.mean((-1,-2)),
        records=[],rewards=reward[rank:rank+1],advantages=advantages,adv_mask=torch.ones(8),
        global_std=float(std),raw_group_std=offsets.std(unbiased=False),advantage_statistics={},
        behavior_version=dict(run=i['sha256'],update=version,adapter_sha256='fixed_behavior_'+str(version)))


def advance(a,o,s,g,t,rollout,i,until):
    while t['updates']<until:
        if rollout is None:
            rollout=collection(i,generator=g,version=t['updates'])
            t['collections']+=1;t['scene_cursor']+=i['world'];t['raw_scene_exposures']+=i['world'];t['new_candidates']+=i['world']*8
            t['phase']='ROLLOUT_COMMITTED'
        o.zero_grad(set_to_none=True)
        target=rollout['chain'][:,-1].mean((0,1))
        (a.lora_A-target).square().sum().backward();o.step();s.step()
        t['updates']+=1;t['next_inner']+=1
        if t['next_inner']==4:
            t['next_inner']=0;rollout=None
        t['phase']=boundary_phase(t,i['recipe']['max_steps'])
        validate_boundary(t,rollout,i,o,s)
    return t,rollout


@pytest.mark.parametrize('arch',list('ABCD'))
def test_four_architecture_committed_3d_descriptor_and_digest(arch):
    i=identity(arch);t=state();r=collection(i);t.update(collections=1,scene_cursor=1,raw_scene_exposures=1,new_candidates=8,phase='ROLLOUT_COMMITTED')
    summary=validate_boundary(t,r,i);assert validate_ranks([summary])[0]['architecture_id']==arch
    before=rollout_digest(r);r['cond_bg']['identity']='wrong_frozen_parent'
    assert rollout_digest(r)['cond_bg']!=before['cond_bg']
    with pytest.raises(ValueError,match='ancestry'):validate_boundary(t,r,i)


@pytest.mark.parametrize('bad',['shape','rich_condition','old_schema','wrong_architecture','bad_identity','wrong_behavior','unfinalized'])
def test_foreign_or_incomplete_rollout_is_not_recoverable(bad):
    i=identity();t=state();r=collection(i);t.update(collections=1,scene_cursor=1,raw_scene_exposures=1,new_candidates=8,phase='ROLLOUT_COMMITTED')
    if bad=='shape':r['chain']=torch.zeros(8,11,8,4)
    if bad=='rich_condition':r['cond_bg']['current']='a_prefix_or_current_dataclass'
    if bad=='old_schema':i['schema']='world_postcot_cleanx_scope_RL_v1'
    if bad=='wrong_architecture':i['architecture_id']='old_DDP'
    if bad=='bad_identity':i['SFT_sha256']='different_parent'
    if bad=='wrong_behavior':r['behavior_version']['update']=4
    if bad=='unfinalized':del r['advantages']
    with pytest.raises(ValueError):validate_boundary(t,r,i)


def test_dataclass_cpu_device_and_digest_preserve_native_current_encoding():
    e=DCurrentEncoding(inputs_embeds=torch.randn(1,6,4,requires_grad=True),valid=torch.ones(1,6,dtype=torch.bool),
        positions=torch.arange(6).reshape(1,1,6).expand(3,1,6),visual_mask=torch.tensor([[True,False,True,False,False,False]]),
        deepstack=(torch.randn(2,4,requires_grad=True),),ego=torch.ones(1,8),scene_identity=('scene',))
    value=CurrentContext(scene=None,valid=None,ego=torch.ones(1,8),encoded=e,
        scene_identity=('scene',),policy_version='current',execution_identity='FP32/BF16',position_mask_identity='native')
    cpu=cp.cpu_tree({'current':value})['current'];assert isinstance(cpu,CurrentContext)and isinstance(cpu.encoded,DCurrentEncoding)
    assert not cpu.encoded.inputs_embeds.requires_grad and not cpu.encoded.deepstack[0].requires_grad
    back=cp.device_tree(cpu,torch.device('cpu'));assert tensor_digest(back)==tensor_digest(cpu)
    altered=copy.deepcopy(cpu);altered.encoded.inputs_embeds[0,0,0]+=1
    assert tensor_digest(altered)!=tensor_digest(cpu)


def test_ready_resume_restores_all_RNG_and_empty_optimizer(tmp_path):
    random.seed(41);np.random.seed(42);torch.manual_seed(43)
    a,o,s,g=bundle();i=identity();t=state();path=cp.save(tmp_path,a,o,s,g,t,None,i)
    expected=(random.random(),np.random.rand(),torch.rand(3),torch.rand(3,generator=g))
    random.seed(100);np.random.seed(101);torch.manual_seed(102);g.manual_seed(103)
    with torch.no_grad():a.lora_A.fill_(99)
    recovered,pending=cp.restore(path,a,o,s,g,i)
    actual=(random.random(),np.random.rand(),torch.rand(3),torch.rand(3,generator=g))
    assert recovered==t and pending is None
    assert actual[0:2]==expected[0:2]
    torch.testing.assert_close(actual[2],expected[2],rtol=0,atol=0);torch.testing.assert_close(actual[3],expected[3],rtol=0,atol=0)
    torch.testing.assert_close(a.lora_A,torch.tensor([.3,-.4,.2]),rtol=0,atol=0)


@pytest.mark.parametrize('inner',[2,3])
def test_mid_inner_resume_completes_remaining_old_chain_exactly(tmp_path,inner):
    i=identity();a,o,s,g=bundle();t,r=advance(a,o,s,g,state(),None,i,inner)
    saved_chain=rollout_digest(r);path=cp.save(tmp_path,a,o,s,g,t,r,i)
    expected_state,expected_rollout=advance(a,o,s,g,copy.deepcopy(t),copy.deepcopy(r),i,8)
    expected_weights=a.lora_A.detach().clone();expected_adam=tensor_digest(o.state_dict());expected_rng=g.get_state().clone()
    a2,o2,s2,g2=bundle();t2,r2=cp.restore(path,a2,o2,s2,g2,i)
    assert t2['next_inner']==inner and rollout_digest(r2)==saved_chain
    actual_state,actual_rollout=advance(a2,o2,s2,g2,t2,r2,i,8)
    assert actual_state==expected_state and actual_rollout is expected_rollout is None
    torch.testing.assert_close(a2.lora_A,expected_weights,rtol=0,atol=0)
    assert tensor_digest(o2.state_dict())==expected_adam and s2.state_dict()==s.state_dict()
    assert torch.equal(g2.get_state(),expected_rng)


@pytest.mark.parametrize('point',['rank_write','before_complete'])
def test_failed_publication_retains_latest_and_old_incomplete_does_not_block_retry(tmp_path,point):
    a,o,s,g=bundle();i=identity();t=state();cp.save(tmp_path,a,o,s,g,t,None,i)
    before=(tmp_path/'latest.json').read_bytes()
    def fault(where,rank):
        if where==point:raise RuntimeError('injected '+point)
    with pytest.raises(RuntimeError,match='Uncommitted'):cp.save(tmp_path,a,o,s,g,t,None,i,tag='new_boundary',fault=fault)
    assert (tmp_path/'latest.json').read_bytes()==before and not(tmp_path/'checkpoints/new_boundary').exists()
    leftovers=list((tmp_path/'checkpoints').glob('new_boundary.incomplete.*'));assert len(leftovers)==1
    with pytest.raises(RuntimeError):cp.restore(leftovers[0],a,o,s,g,i)
    path=cp.save(tmp_path,a,o,s,g,t,None,i,tag='new_boundary')
    assert Path(path).is_dir() and json.loads((tmp_path/'latest.json').read_text())['checkpoint']==path


def test_collection_failure_is_evidence_only_and_rejects_save(tmp_path):
    a,o,s,g=bundle();i=identity();t=state();cp.save(tmp_path,a,o,s,g,t,None,i)
    before=(tmp_path/'latest.json').read_bytes();t['phase']='COLLECTING';partial=collection(i);del partial['advantages']
    with pytest.raises(RuntimeError,match='COLLECTING'):cp.save(tmp_path,a,o,s,g,t,partial,i,tag='bad_collection')
    target=cp.failure_evidence(tmp_path,t,'nonrank0 mock scorer failure','attempt2',rollout=partial)
    assert json.loads((Path(target)/'evidence.json').read_text())['recoverable']is False
    assert not(Path(target)/'COMPLETE').exists()and(tmp_path/'latest.json').read_bytes()==before
    bad=cp.failure_evidence(tmp_path,t,'CUDA error: illegal memory access','cuda_failure',rollout=partial)
    assert not(Path(bad)/'partial_rollout.pt').exists()


def test_rank_consensus_keeps_original_global_reward_std_semantics():
    i=identity(world=2);t=state();t.update(collections=1,scene_cursor=2,raw_scene_exposures=2,new_candidates=16,phase='ROLLOUT_COMMITTED')
    summaries=[validate_boundary(t,collection(i,rank=r),i)for r in range(2)]
    assert len(validate_ranks(summaries))==2
    bad=copy.deepcopy(summaries);bad[1]['global_std']+=.01
    with pytest.raises(ValueError,match='Advantage'):validate_ranks(bad)
    bad=copy.deepcopy(summaries);bad[1]['next_inner']=1
    with pytest.raises(ValueError,match='Ranks disagree'):validate_ranks(bad)


@pytest.mark.parametrize('all_tied',[False,True])
def test_tied_ranks_preserve_raw_population_global_std_and_zero_mask(all_tied):
    i=identity(world=2);t=state();t.update(collections=1,scene_cursor=2,raw_scene_exposures=2,new_candidates=16,phase='ROLLOUT_COMMITTED')
    rewards=torch.ones(2,8)
    if not all_tied:rewards[1]=torch.arange(8).float()/10
    std=rewards.double().std(unbiased=False);summaries=[]
    for rank in range(2):
        r=collection(i,rank=rank);r['rewards']=rewards[rank:rank+1]
        offsets=r['rewards'].double()-r['rewards'].double()[:,:1]
        r['advantages']=((offsets-offsets.mean(1,keepdim=True))/(std+1e-8)).float()[0]
        r['adv_mask']=torch.full((8,),float(offsets.std(unbiased=False)>1e-8));r['global_std']=float(std)
        summaries.append(validate_boundary(t,r,i))
    assert validate_ranks(summaries)[0]['mask']==[0.]*8
    if all_tied:assert summaries[1]['advantages']==[0.]*8


def test_changed_rank_payload_refuses_restore_before_any_parameter_load(tmp_path):
    a,o,s,g=bundle();i=identity();path=Path(cp.save(tmp_path,a,o,s,g,state(),None,i))
    with torch.no_grad():a.lora_A.fill_(99)
    (path/'rank0.pt').write_bytes(b'injected corruption')
    with pytest.raises(RuntimeError,match='content changed'):cp.restore(path,a,o,s,g,i)
    assert torch.equal(a.lora_A,torch.full((3,),99.))and not o.state
