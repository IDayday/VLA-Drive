"""Exact-source helper tests; all scene names, receipts and tensors are synthetic."""
import ast
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

ROOT=Path('/mnt/project/VLA-Drive-st-rollout-scale-runtime-v2-20261018')


def extract(relative, name, namespace):
    tree=ast.parse((ROOT/relative).read_text())
    node=next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/relative),'exec'),namespace)
    return namespace[name]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_hash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


immutable=extract('tools/st_world_policy/base_binding.py','immutable_json',
                  dict(Path=Path,json=json,os=os,uuid=uuid))
read_export=extract('tools/st_world_policy/score_world.py','read_complete_export',
                    dict(Path=Path,json=json,fingerprint=fingerprint,file_sha256=file_hash))
scene_noise=extract('tools/foresight/checkpoints.py','scene_noise',dict(hashlib=hashlib))


def assigned(n,rank,world):
    tree=ast.parse((ROOT/'tools/st_world_policy/export_world.py').read_text())
    node=next(x for x in ast.walk(tree) if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='assigned' for t in x.targets))
    return eval(compile(ast.Expression(node.value),'<actual-export-assignment>','eval'),{'args':SimpleNamespace(rank=rank,world_size=world),'data':range(n)})


@pytest.mark.parametrize('count',[1696,7,2])
def test_actual_assignment_unique_coverage_and_tail(count):
    parts=[assigned(count,rank,4) for rank in range(4)]
    assert sorted(x for part in parts for x in part)==list(range(count))
    assert sum(map(len,parts))==count
    assert max(map(len,parts))-min(map(len,parts))<=1
    if count==1696:assert list(map(len,parts))==[424]*4


def test_actual_scene_noise_is_independent_of_rank_order_and_global_rng():
    before=torch.random.get_rng_state().clone()
    tokens=['scene_'+str(i) for i in range(13)]
    serial={t:scene_noise(t,42,'cpu') for t in tokens}
    assert torch.equal(before,torch.random.get_rng_state())
    torch.manual_seed(952);changed=torch.random.get_rng_state().clone()
    for rank in [3,1,0,2]:
        for i in assigned(len(tokens),rank,4):
            t=tokens[i];assert torch.equal(serial[t],scene_noise(t,42,'cpu'))
            assert serial[t].dtype==torch.float32 and tuple(serial[t].shape)==(1,8,4)
    assert torch.equal(changed,torch.random.get_rng_state())
    assert not torch.equal(serial[tokens[0]],scene_noise(tokens[0],43,'cpu'))


def test_concurrent_equal_immutable_publication_and_conflict(tmp_path):
    p=tmp_path/'IDENTITY.json';value={'same':'fixed','world_size':4}
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _:immutable(p,value),range(4)))
    assert json.loads(p.read_text())==value
    assert not list(tmp_path.glob('*.tmp.*'))
    before=p.read_bytes()
    with pytest.raises(ValueError,match='Immutable artifact conflict'):
        immutable(p,{'same':'foreign','world_size':1})
    assert p.read_bytes()==before


def fixture_export(root,world=4,count=11):
    scenes=[{'token':'scene_'+str(i),'log':'log_'+str(i%2)} for i in range(count)]
    signature={'schema':'st_single_trajectory_export_v1','observations':{'split':'dev','identity':'observations'},
               'candidates_per_scene':1,'scorer':None,'targets_read':False,'teacher_cache_read':False,
               'world_size':world,'requested_scenes':count,'selected_scenes_sha256':fingerprint(scenes),
               'environment':{'torch':'same'},'seed':42,'solver':'original_Euler_10','precision':'FP32_master_eval_TF32_off','scene_noise':'foresight-action-v1'}
    signature['identity']=fingerprint(signature)
    immutable(root/'IDENTITY.json',signature)
    immutable(root/'OBSERVATIONS.json',{'scenes':scenes,'identity':'observations'})
    for rank in range(world):
        receipts=[]
        for row in scenes[rank::world]:
            meta=root/'predictions'/(row['token']+'.json')
            immutable(meta,{'identity':signature['identity'],'token':row['token'],'fixture':True})
            receipts.append({'token':row['token'],'receipt_sha256':file_hash(meta)})
        immutable(root/f'COMPLETE_rank{rank}.json',{'rank':rank,'identity':signature['identity'],'requested':len(receipts),'receipts':receipts})
    return signature,scenes


def test_complete_four_shard_merge_and_world1_pairing(tmp_path):
    a,rows=fixture_export(tmp_path/'old',1)
    b,_=fixture_export(tmp_path/'new',4)
    assert read_export(tmp_path/'old')==(a,rows)
    assert read_export(tmp_path/'new')==(b,rows)
    assert a['identity']!=b['identity']
    runtime=lambda e:fingerprint({k:e[k] for k in ('environment','seed','solver','precision','scene_noise')})
    assert runtime(a)==runtime(b)
    assert a['observations']==b['observations']


@pytest.mark.parametrize('kind',['missing_rank','foreign_identity','rank_number','wrong_count','duplicate_or_omitted','receipt_hash','population_hash'])
def test_merge_rejects_partial_or_foreign_shards(tmp_path,kind):
    fixture_export(tmp_path)
    p=tmp_path/'COMPLETE_rank2.json';value=json.loads(p.read_text())
    if kind=='missing_rank':p.unlink()
    elif kind=='population_hash':
        q=tmp_path/'OBSERVATIONS.json';v=json.loads(q.read_text());v['scenes'][0]['log']='foreign';q.write_text(json.dumps(v))
    else:
        if kind=='foreign_identity':value['identity']='foreign'
        if kind=='rank_number':value['rank']=3
        if kind=='wrong_count':value['requested']+=1
        if kind=='duplicate_or_omitted':value['receipts'][0]=value['receipts'][1]
        if kind=='receipt_hash':value['receipts'][0]['receipt_sha256']='0'*64
        p.write_text(json.dumps(value))
    with pytest.raises((ValueError,FileNotFoundError)):read_export(tmp_path)


def ledger_class():
    def atomic(p,v,immutable=False):globals()['immutable'](Path(p),v)
    def art(p):return {'path':str(p),'sha256':file_hash(p)}
    def verify(a):assert file_hash(a['path'])==a['sha256']
    def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);torch.save(v,p)
    def validate(result,request):assert result['valid'] and result['identity']==request.identity()
    return extract('tools/st_world_policy/train_flow_policy.py','OfficialFeedbackLedger',
        dict(Path=Path,atomic_json=atomic,read_json=lambda p:json.loads(Path(p).read_text()),identity_hash=fingerprint,
             file_sha256=file_hash,torch=torch,_verified_artifact=verify,_save_torch=save,artifact=art,
             _validate_scored=validate,json=json,time=time))


class Request:
    def __init__(self,i):self.i=i
    def identity(self):return {'fixture_scene':self.i}


class Service:
    def __init__(self,crash=False):self.calls=0;self.crash=crash
    def score(self,requests,**kwargs):
        self.calls+=len(requests)
        if self.crash:raise RuntimeError('fixture interrupted after admission')
        return [{'identity':r.identity(),'valid':True} for r in requests]


def test_single_ledger_reuses_and_caps_across_coordinator_restart(tmp_path):
    cls=ledger_class();requests=[Request(i) for i in range(4)];service=Service()
    first=cls(tmp_path,identity='fixed',max_calls=5,retry_attempts=1)
    first.score(requests,policy_version='ckpt',service=service)
    assert service.calls==4 and first.admitted_calls==4
    second=cls(tmp_path,identity='fixed',max_calls=5,retry_attempts=1)
    second.score(requests,policy_version='ckpt',service=service)
    assert service.calls==4 and second.admitted_calls==4
    with pytest.raises(ValueError,match='budget exhausted'):
        second.score([Request(5),Request(6)],policy_version='ckpt',service=service)
    assert service.calls==4


def test_interrupted_unknown_admission_remains_charged(tmp_path):
    cls=ledger_class();requests=[Request(i) for i in range(4)]
    with pytest.raises(RuntimeError,match='fixture interrupted'):
        cls(tmp_path,identity='fixed',max_calls=5,retry_attempts=1).score(requests,policy_version='ckpt',service=Service(True))
    restarted=cls(tmp_path,identity='fixed',max_calls=5,retry_attempts=1)
    assert restarted.admitted_calls==4
    service=Service()
    with pytest.raises(ValueError,match='budget exhausted'):
        restarted.score(requests,policy_version='ckpt',service=service)
    assert service.calls==0


def test_actual_inference_and_scoring_topology_source_path():
    export=(ROOT/'tools/st_world_policy/export_world.py').read_text()
    assert 'model.predict_raw([observation], initial_noise=noise)[0].float()' in export
    assert 'noise = scene_noise(token, args.seed, args.device)' in export
    assert 'with (out / f"rank_{args.rank}.lock").open("a+") as lock:' in export
    assert 'fcntl.LOCK_EX | fcntl.LOCK_NB' in export
    score=(ROOT/'tools/st_world_policy/score_world.py').read_text()
    assert 'with (root / "EVALUATION.lock").open("a+") as lock:' in score
    assert 'for at in range(0, len(requests), 4):' in score
    assert 'max_calls=args.max_calls' in score
    analyzer=ast.parse((ROOT/'tools/st_world_policy/evaluate_campaign.py').read_text())
    load_run=next(n for n in analyzer.body if isinstance(n,ast.FunctionDef) and n.name=='load_run')
    # Per-arm contract checks stay strict; execution shard-count equality is not a paired scientific condition.
    assert not any(isinstance(n,ast.Constant) and n.value=='world_size' for n in ast.walk(load_run))
    flow=ast.parse((ROOT/'tools/st_world_policy/flow_checkpoints.py').read_text())
    wrapper=next(n for n in flow.body if isinstance(n,ast.ClassDef) and n.name=='FlowStandalone')
    prediction=next(n for n in wrapper.body if isinstance(n,ast.FunctionDef) and n.name=='predict_raw')
    assert not any(isinstance(n,ast.Name) and n.id in ('rank','world_size') for n in ast.walk(prediction))
