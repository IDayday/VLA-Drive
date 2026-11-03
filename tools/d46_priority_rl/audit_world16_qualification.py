"""CPU-only audit of the actual two-node full-model pending-inner recovery."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from pathlib import Path
import sys
import time

SCIENCE=Path('/mnt/project/d46-priority-rl-20261103/runtime/world16_science_v3')
sys.path.insert(0,str(SCIENCE))
import numpy as np
import torch
from starVLA.rl.d46_world16.native import World16Boundary,validate_virtual
from starVLA.rl.qwen_world_compare.contracts import file_sha,fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json

ROOT=Path('/mnt/project/d46-priority-rl-20261103')


def read(p):return json.loads(Path(p).read_text())


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor)and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray)and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict)and a.keys()==b.keys()and all(same(a[k],b[k])for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b)and len(a)==len(b)and all(same(x,y)for x,y in zip(a,b))
    return a==b


def audit():
    torch.set_num_threads(1)
    run=ROOT/'qualification/WORLD16_D0_u1400_native_v3'
    before=run/'checkpoints/update001403_inner1';after=run/'checkpoints/update001404_inner0'
    checkpoints={};payloads={}
    for label,cp in [('pending',before),('resumed',after)]:
        seal=read(cp/'COMPLETE');inventory=read(cp/'files.json');identity=read(cp/'identity.json')
        assert seal['world']==identity['world']==16 and seal['files_sha256']==file_sha(cp/'files.json')
        assert identity['qualification']is True and identity['fresh_Adam']is False
        for name,digest in inventory.items():assert file_sha(cp/name)==digest,name
        common=torch.load(cp/'optimizer.pt',map_location='cpu',weights_only=False)
        tx=read(cp/'transaction.json');summaries=[];owns=[]
        for i in range(16):
            own=torch.load(cp/f'rank{i}.pt',map_location='cpu',weights_only=False)
            summary=World16Boundary.validate_boundary(own['state'],own['rollout'],identity,common['optimizer'],common['scheduler'])
            assert summary==tx['ranks'][i]
            summaries.append(summary);owns.append(own)
        World16Boundary.validate_ranks(summaries)
        checkpoints[label]=dict(path=str(cp),seal=seal,inventory_sha256=file_sha(cp/'files.json'),identity_sha256=identity['sha256'],
                                optimizer_sha256=inventory['optimizer.pt'])
        payloads[label]=(common,owns)
    old_common,old=payloads['pending'];new_common,new=payloads['resumed']
    ranks=[]
    clocks=('valid_scene_exposures','raw_scene_exposures','new_candidates','valid_PG_candidates','collections','scene_cursor')
    for i in range(16):
        a=old[i];b=new[i];assert a['state']['updates']==1403 and a['state']['next_inner']==1 and a['rollout']is not None
        assert b['state']['updates']==1404 and b['state']['next_inner']==0 and b['rollout']is None
        assert all(a['state'][k]==b['state'][k]for k in clocks)
        assert same(a['rng'],b['rng']),'All native RNG fields must survive pending-inner reuse'
        assert validate_virtual(a['state'])==validate_virtual(b['state'])
        report=read(run/f'result_rank{i}.json')
        assert report['status']=='QUALIFICATION_STOPPED'and report['updates']==1404
        assert report['frozen_before']==report['frozen_after']
        parity=read(run/f'native_B1_restore_parity_rank{i}_u001403.json')
        assert parity['status']=='PASS'and all(v==0 for v in parity['max_abs'].values())
        assert parity['source_optimizer_sha256']==checkpoints['pending']['optimizer_sha256']
        ranks.append(dict(rank=i,RNG_exact_after_pending_reuse=True,virtual_lanes_exact=True,
                          clocks_unchanged={k:b['state'][k]for k in clocks},frozen_parameters_unchanged=True,
                          B1_native_restore_max_abs=parity['max_abs']))
    steps=[float(v['step'])for v in new_common['optimizer']['state'].values()if 'step'in v]
    assert steps and min(steps)==max(steps)==1404 and new_common['scheduler']['last_epoch']==1404
    jobs=[]
    for stage in ('third','resume2'):
        for node in (0,1):
            p=ROOT/f'jobs/WORLD16_QUALIFICATION/D0_u1400/{stage}/node{node}/RESULT.json'
            result=read(p);assert result['returncode']==0
            jobs.append(dict(path=str(p),file_sha256=file_sha(p),returncode=0))
    journal=[]
    for i in range(16):
        rows=[json.loads(x)for x in(run/f'metrics_rank{i}.jsonl').read_text().splitlines()]
        assert [x['updates']for x in rows]==[1401,1402,1403,1404]
        journal.append(rows)
    result=dict(schema='D46_actual_full_model_world16_pending_resume_qualification_v1',status='PASS',at=time.time(),
                scientific_source='87bfecc87d5dda6397c195122b21c4eadc3a90a0',worker_source='a41857c',world=16,B=64,G=16,M=2,
                checkpoints=checkpoints,nodes_exited0=jobs,ranks=ranks,Adam_steps=1404,scheduler_steps=1404,
                new_Adam_updates=4,initial_three_update_seconds_critical_rank=[max(journal[r][j]['update_seconds']for r in range(16))for j in range(3)],
                extra_PDM_queries_during_pending_resume=0,full_RNG_fields_preserved=True,
                native_midinner_restored=True,qualification_weights_never_formal=True,
                actual_formal_world16_training_started=False,
                limits=['Short shared-GPU qualification; not a long-training speed benchmark',
                        'Actual RNG/native state parity verified; full-model uninterrupted-vs-resumed parameter bitexact comparison was not run',
                        'Earlier startup-control timeout and final-JSON failure retained; third/resume2 alone are the successful pair'])
    result['sha256']=fingerprint(result)
    return result


if __name__=='__main__':
    value=audit();path=ROOT/'qualification/ACTUAL_WORLD16_PENDING_RESUME_PASS.json';atomic_json(path,value)
    print(json.dumps(dict(status=value['status'],output=str(path),world=value['world'],Adam_steps=value['Adam_steps'],new_Adam_updates=value['new_Adam_updates'])))
