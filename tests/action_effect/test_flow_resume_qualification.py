"""CPU qualification-controller negatives; CUDA arrays here are explicit fixtures."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import random

import numpy as np
import pytest
import torch

from tools.action_effect import qualify_flow_resume as q
from tools.action_effect.common import artifact,digest,read_json,write_json


def small_state(step=16):
    names=['adapter.output.weight','adapter.effect_head.weight']
    identity={'private_names':sorted(names),'recipe':{'batch_scenes':32}}
    state={'schema':'action_effect_private_checkpoint_v1','run_identity':digest(identity),
        'model':{name:torch.ones(2,2) for name in names},'optimizer_step':step,
        'optimizer':{'param_groups':[{'params':[0,1]}], 'state':{0:{'exp_avg':torch.ones(2,2),
            'exp_avg_sq':torch.ones(2,2),'step':torch.tensor(float(step))}}},
        'scheduler':{'last_epoch':step},'sampler':torch.Generator().manual_seed(143).get_state(),
        'stateful':{'flow_noise':torch.tensor([3],dtype=torch.uint8),'flow_time':torch.tensor([4],dtype=torch.uint8)},
        'rng':{'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),
               'cuda':[torch.tensor([5],dtype=torch.uint8)]},
        'consumed_scenes':step*32,'terminal':False,'termination_reason':'OPERATOR_PAUSED',
        'history':[{'optimizer_step':i,'losses':{'velocity':.1},'gradient_norm':.2,'lr':[1e-4],
            'consumed_scenes':i*32} for i in range(1,step+1)]}
    return state,identity,names


def test_dormant_B0_aux_parameters_need_no_Adam_state():
    state,identity,names=small_state()
    result=q.validate_state(state,identity,16,names)
    assert result['dormant_aux_without_Adam_names']==['adapter.effect_head.weight']
    assert result['active_Adam_parameter_names']==['adapter.output.weight']


@pytest.mark.parametrize('kind',['missing_active','FP16_moment','step','optimizer_order','private_order','missing_stream','missing_CUDA','cursor','terminal'])
def test_real_state_contract_negative_cases(kind):
    state,identity,names=small_state()
    if kind=='missing_active':state['optimizer']['state']={}
    elif kind=='FP16_moment':state['optimizer']['state'][0]['exp_avg']=torch.ones(2,2,dtype=torch.float16)
    elif kind=='step':state['optimizer']['state'][0]['step']=torch.tensor(15.)
    elif kind=='optimizer_order':state['optimizer']['param_groups'][0]['params']=[1,0]
    elif kind=='private_order':names=names[::-1]
    elif kind=='missing_stream':del state['stateful']['flow_time']
    elif kind=='missing_CUDA':state['rng']['cuda']=[]
    elif kind=='cursor':state['consumed_scenes']+=32
    elif kind=='terminal':state['terminal']=True
    with pytest.raises(ValueError):q.validate_state(state,identity,16,names)


@pytest.mark.parametrize('field',['model','optimizer','scheduler','flow_noise','flow_time','rng','loss','gradient','lr'])
def test_complete_byte_compare_detects_state_and_update_drift(field):
    left,_,_=small_state();right=deepcopy(left)
    assert q.compare_states(left,right,16)['exact']
    if field=='model':right['model']['adapter.output.weight'][0,0]+=.01
    elif field=='optimizer':right['optimizer']['state'][0]['exp_avg'][0,0]+=.01
    elif field=='scheduler':right['scheduler']['last_epoch']-=1
    elif field in ('flow_noise','flow_time'):right['stateful'][field][0]+=1
    elif field=='rng':right['rng']['cpu'][0]^=1
    elif field=='loss':right['history'][8]['losses']['velocity']=.2
    elif field=='gradient':right['history'][8]['gradient_norm']=.3
    elif field=='lr':right['history'][8]['lr']=[2e-4]
    assert not q.compare_states(left,right,16)['exact']


def test_next_scene_ids_preserve_actual_balanced_reference_pool():
    pool=[{'scene_id':'repair','target_kind':'verified_local_repair'}]*3+[
        {'scene_id':f'keep{i}','target_kind':'identity_preservation'} for i in range(3)]
    sampler=torch.Generator().manual_seed(143)
    for _ in range(8):torch.randint(len(pool),(32,),generator=sampler)
    state8={'sampler':sampler.get_state()};torch.randint(len(pool),(32,),generator=sampler)
    state9={'sampler':sampler.get_state()}
    result=q.next_scene_proof(pool,state8,state9,state9,seed=42,batch_size=32)
    assert result['exact'] and result['virtual_reference_count']==6 and result['pool_unique_scenes']==4
    assert len(result['next9_scene_ids'])==32
    assert {row['reference_index'] for row in result['next9_scene_ids']}<=set(range(6))
    state9['sampler']=torch.Generator().manual_seed(7).get_state()
    assert not q.next_scene_proof(pool,state8,state9,state9,seed=42,batch_size=32)['exact']


@pytest.mark.parametrize('key,value',[('max_updates',16),('warmup_updates',8),('batch_scenes',16),('microbatch_scenes',4)])
def test_no_scheduler_or_batch_shortcut(key,value):
    cfg={'training':{'max_updates':1000,'warmup_updates':50,'batch_scenes':32,'microbatch_scenes':8,'validation_every':250}}
    before=deepcopy(cfg);changed=q.validate_recipe(cfg)
    assert cfg==before and changed['training']['validation_every']==9
    changed['training']['validation_every']=250;assert changed==cfg
    cfg['training'][key]=value
    with pytest.raises(ValueError):q.validate_recipe(cfg)


def test_segment_reserves_before_child_and_keeps_1000_axis(tmp_path,monkeypatch):
    run=tmp_path/'continuous16';cfg=tmp_path/'QUALIFICATION_CONFIG.json';cfg.write_text('{}')
    args=SimpleNamespace(context_manifest=str(tmp_path/'CONTEXT.json'),repair_dataset=str(tmp_path/'REPAIR.json'),
        device='cuda:0',seed=42,resume=False,output=str(tmp_path),max_seconds_per_segment=1200)
    calls=[];checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'fixture')
    monkeypatch.setattr(q,'validate_run',lambda *a,**kw:None)
    monkeypatch.setattr(q,'checkpoints',lambda run:[])
    monkeypatch.setattr(q,'boundary',lambda *a:(16,0,checkpoint,None,None))
    def fake_run(cmd,**kw):
        ledger=read_json(tmp_path/'DISPATCH_BUDGET.json')
        assert ledger['scheduled_updates']==16 and ledger['reservations'][0]['command']==cmd
        calls.append(cmd);return SimpleNamespace(returncode=0)
    monkeypatch.setattr(q.subprocess,'run',fake_run)
    result=q.segment(cfg,args,run,16,expected={},head={});cmd=calls[0]
    assert result['new_updates']==16 and cmd[cmd.index('--max-updates')+1]=='1000'
    assert cmd[cmd.index('--stop-after')+1]=='16' and '--resume' not in cmd
    assert cmd[cmd.index('--arm')+1]=='B0' and 'tools.action_effect.train_effect_flow' in cmd
    assert '--effect-checkpoint' not in cmd


def test_failed_dispatch_budget_cannot_be_silently_spent_again(tmp_path,monkeypatch):
    args=SimpleNamespace(context_manifest='C',repair_dataset='R',device='cuda:0',seed=42,resume=True,
        output=str(tmp_path),max_seconds_per_segment=1200)
    monkeypatch.setattr(q,'validate_run',lambda *a,**kw:None);monkeypatch.setattr(q,'checkpoints',lambda run:[])
    monkeypatch.setattr(q.subprocess,'run',lambda *a,**kw:pytest.fail('must not exceed reserved32'))
    write_json(tmp_path/'DISPATCH_BUDGET.json',{'scheduled_updates':32,'reservations':[{'new_update_ceiling':32}]})
    with pytest.raises(ValueError,match='budget exhausted'):q.segment(tmp_path/'cfg',args,tmp_path/'full',16,expected={},head={})
    write_json(tmp_path/'DISPATCH_BUDGET.json',{'scheduled_updates':-1,'reservations':[]},immutable=False)
    with pytest.raises(ValueError,match='Malformed'):q.segment(tmp_path/'cfg',args,tmp_path/'full',16,expected={},head={})


def test_training_source_and_runtime_mismatch_rejected(tmp_path):
    expected={'source_commit':'fixed','arm':'B0'};head={'strict_load':True}
    identity={**expected,'head':head,'runtime':{'device':'cuda:0','trainable_dtype':'float32','autocast':'disabled',
        'deterministic':True,'matmul_tf32':False,'cudnn_tf32':False,'cudnn_benchmark':False}}
    write_json(tmp_path/'IDENTITY.json',identity);q.validate_run(tmp_path,expected,head,'cuda:0')
    identity['source_commit']='foreign';write_json(tmp_path/'IDENTITY.json',identity,immutable=False)
    with pytest.raises(ValueError):q.validate_run(tmp_path,expected,head,'cuda:0')
    identity['source_commit']='fixed';identity['runtime']['device']='cpu';write_json(tmp_path/'IDENTITY.json',identity,immutable=False)
    with pytest.raises(ValueError,match='runtime'):q.validate_run(tmp_path,expected,head,'cuda:0')


def test_foreign_or_lifecycle_mismatched_receipt_rejected(tmp_path,monkeypatch):
    local=tmp_path/'local';local.mkdir();foreign=tmp_path/'foreign';foreign.mkdir()
    path=foreign/'step_000009.pt';path.write_bytes(b'fixture')
    receipt=path.with_suffix('.COMPLETE.json');write_json(receipt,{'terminal':False,'termination_reason':'RUNNING_CHECKPOINT'})
    row=(9,0,path,{'terminal':False,'termination_reason':'RUNNING_CHECKPOINT'},receipt)
    monkeypatch.setattr(q,'at_step',lambda *a:row)
    with pytest.raises(ValueError,match='namespace'):q.boundary(local,9)
    write_json(receipt,{'terminal':True,'termination_reason':'MAX_UPDATES'},immutable=False)
    with pytest.raises(ValueError,match='lifecycle'):q.boundary(foreign,9)


def test_dry_run_never_checks_GPU_or_loads_real_head(tmp_path,monkeypatch):
    args=SimpleNamespace(config='configs/action_effect/main.yaml',context_manifest=str(tmp_path/'C.json'),
        repair_dataset=str(tmp_path/'R.json'),output=str(tmp_path/'out'),device='cuda:0',arm='B0',seed=42,
        max_seconds_per_segment=1200,resume=False,dry_run=True)
    for p in (Path(args.context_manifest),Path(args.repair_dataset)):p.write_text('{}')
    monkeypatch.setattr(torch.cuda,'device_count',lambda:pytest.fail('dry-run touched GPU'))
    result=q.qualify(args)
    assert result['status']=='NOT_RUN' and result['identity']['total_new_update_ceiling']==32
    cfg=read_json(Path(args.output)/'QUALIFICATION_CONFIG.json')
    assert cfg['artifact_root']=='/mnt/project/action-effect-repair-20261020'
    assert cfg['training']['max_updates']==1000 and cfg['training']['validation_every']==9
    assert not (Path(args.output)/'DISPATCH_BUDGET.json').exists()
