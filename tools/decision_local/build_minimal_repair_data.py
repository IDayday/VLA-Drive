"""Versioned real-plan pools, polygon labels, and bounded offline target solve.

This is CPU work over frozen q0, existing annotations, and map APIs. It never
re-encodes a VLM or changes an old cache. No valid/failed scene is silently dropped.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import numpy as np
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,digest,identity,read,source_sha,check_host
from starVLA.dataloader.foresight_dataset import encode_ego,decode_ego
from starVLA.dataloader.minimal_repair_scene import load_annotated_scene
from starVLA.model.modules.decision_local.recovery_plans import RecoveryConfig,build_plan_pool
from starVLA.model.modules.decision_local.trajectory_queries import QueryConfig,trajectory_queries
from starVLA.model.modules.decision_local.local_constraint_checker import ConstraintConfig,LocalConstraintChecker
from starVLA.model.modules.decision_local.minimal_repair_targets import SolverConfig,solve_minimal_repair
from starVLA.model.modules.decision_local.minimal_repair_contract import SCHEMA


def build_scene(task):
    torch.set_num_threads(1)
    row,source,offset,anchor,split,outroot,contract,ident=task
    begin=time.monotonic();cpu=time.process_time();root=Path(outroot)
    scene=load_annotated_scene(anchor,split,row)
    checker=LocalConstraintChecker(scene,ConstraintConfig(**contract['constraints']))
    with safe_open(source,framework='pt',device='cpu') as f:
        q0=f.get_slice('q0')[offset].numpy().copy()
        q0_encoded=f.get_slice('q0_encoded')[offset].numpy().copy()
        state=f.get_slice('state')[offset].clone()
    velocity=decode_ego(state[None])[0,:2].numpy()/.5
    data={k:[] for k in ('plan_in_physical','plan_in_encoded','poses','in_grid','kinematic_valid',
                         'relations','valid','common_valid','repair_target','target_eligible','case_status')}
    cases=[]
    for seed_index,proposal in enumerate(q0):
        pool=build_plan_pool(proposal,velocity,row['token'],RecoveryConfig(**contract['recovery']))
        per={k:[] for k in data}
        for plan_index,plan in enumerate(pool['plans']):
            solved=solve_minimal_repair(plan,velocity,checker,SolverConfig(**contract['solver']))
            target=solved.pop('target')
            if solved['status']==0 and not np.array_equal(target,plan):raise AssertionError('KEEP changed actual input')
            query=trajectory_queries(plan,velocity,QueryConfig(**contract['query_config']))
            values=[];valid=[]
            for source_kind in ('CURRENT','CV','FUTURE'):
                vv=[];mm=[]
                for poses in query['poses']:
                    v,m,s=checker.relations(poses,query['times'],source_kind);vv.append(v);mm.append(m)
                values.append(np.stack(vv));valid.append(np.stack(mm))
            _,_,support=checker.relations(query['poses'].reshape(-1,3),np.tile(query['times'],7),'CURRENT')
            per['plan_in_physical'].append(plan)
            per['plan_in_encoded'].append(q0_encoded[seed_index] if plan_index==0 else encode_ego(plan))
            per['poses'].append(query['poses']);per['kinematic_valid'].append(query['kinematic_valid'])
            per['in_grid'].append(support['in_grid'].reshape(7,len(plan)))
            per['relations'].append(np.stack(values).astype(np.float32));per['valid'].append(np.stack(valid))
            per['common_valid'].append(np.logical_and.reduce(valid))
            per['repair_target'].append(encode_ego(target) if target is not None else np.zeros((len(plan),4),np.float32))
            per['target_eligible'].append(solved['target_valid']);per['case_status'].append(solved['status'])
            cases.append({'anchor_index':seed_index,'plan_index':plan_index,**solved})
        for k in data:data[k].append(np.stack(per[k]))
    # GT ego is loaded only after the complete input/solver calculations.
    gt=torch.load(Path(anchor['data'][split]['current_root'])/'ego'/(row['token']+'.pt'),map_location='cpu',weights_only=True)
    if gt['token']!=row['token']:raise ValueError('GT row mismatch')
    arrays={k:np.stack(v) for k,v in data.items()}
    arrays.update(ego=gt['ego'].float().numpy(),future_valid=np.ones(8,bool),
                  body=np.array([scene.body[k] for k in ('length','width','rear_axle_to_center')],np.float32))
    if not np.isfinite(arrays['ego']).all():arrays['target_eligible'][:]=False
    path=root/'scenes'/(row['token']+'.npz');temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:np.savez_compressed(f,**arrays)
    temporary.replace(path)
    receipt={'token':row['token'],'log':row['log'],'identity':ident,'sha256':digest(path),
        'wall_seconds':time.monotonic()-begin,'cpu_seconds':time.process_time()-cpu,
        'geometry_quality':scene.quality,'cases':cases,'source_feature_chunk':source,
        'source_feature_offset':offset,'bytes':path.stat().st_size}
    atomic_json(root/'records'/(row['token']+'.json'),receipt)
    return {k:receipt[k] for k in ('token','wall_seconds','cpu_seconds','bytes')}|{'case_counts':np.bincount(arrays['case_status'].ravel(),minlength=7).tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('anchor','features','index','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--max-seconds',type=float,required=True)
    p.add_argument('--solver-seconds',type=float,default=3.)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--finalize',action='store_true',help='Verify every existing scene and close a distributed build; no workers')
    a=p.parse_args()
    if not 0<=a.rank<a.shards or a.workers<1:raise ValueError('Invalid finite shard/worker assignment')
    resources=check_host();anchor=read(a.anchor);features=read(a.features/'identity.json');index=read(a.index)
    if features['anchor_identity']!=anchor['identity'] or features['split'] not in ('train','dev'):raise ValueError('Wrong frozen anchor/split')
    full={r['token']:r for r in features['index']}
    if len({r['token'] for r in index})!=len(index) or any(full.get(r['token'])!=r for r in index):raise ValueError('Foreign/duplicate row')
    files=[Path(__file__),ROOT/'starVLA/dataloader/minimal_repair_scene.py']+sorted((ROOT/'starVLA/model/modules/decision_local').glob('*repair*.py'))+[ROOT/'starVLA/model/modules/decision_local/recovery_plans.py',ROOT/'starVLA/model/modules/decision_local/local_constraint_checker.py']
    contract={'schema':SCHEMA,'feature_identity':features['identity'],'anchor_identity':anchor['identity'],
        'split':features['split'],'index':index,'index_sha256':digest(a.index),'source_sha':source_sha(ROOT),
        'source_files':{str(f.relative_to(ROOT)):digest(f) for f in files},'recovery':RecoveryConfig().to_dict(),
        'constraints':ConstraintConfig().to_dict(),'solver':SolverConfig(max_seconds=a.solver_seconds).to_dict(),
        'query_config':QueryConfig().to_dict(),'target':'same TRUE_FUTURE local constraints for all LOG/MIN input sources',
        'observed_free_support':'separate from annotated object geometry; empty object set not proof of observed free',
        'resource_scope':'CPU immutable labels; no model calls or old task restart'}
    ident=identity(contract);a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'identity.json').exists() and read(a.output/'identity.json')['identity']!=ident:raise ValueError('Preserve old label version')
    atomic_json(a.output/'identity.json',{'identity':ident,**contract})
    for x in ('records','scenes'):(a.output/x).mkdir(exist_ok=True)
    suffix='' if a.shards==1 else '_rank%03d'%a.rank
    atomic_json(a.output/('COMMAND'+('_finalize' if a.finalize else suffix)+'.json'),{'argv':sys.argv,'cwd':str(ROOT),'resource_check':resources})
    wanted={r['token'] for r in index};locations={}
    for path in sorted((a.features/'records').glob('*.json')):
        rec=read(path)
        for offset,token in enumerate(rec['tokens']):
            if token in wanted:locations[token]=(str(a.features/'chunks'/(path.stem+'.safetensors')),offset)
    if set(locations)!=wanted:raise ValueError('Missing source features')
    tasks=[];done=[];start=time.monotonic();errors=[]
    selected=index if a.finalize else index[a.rank::a.shards]
    for row in selected:
        saved=a.output/'records'/(row['token']+'.json')
        if saved.exists():
            rec=read(saved)
            if rec['identity']!=ident or digest(a.output/'scenes'/(row['token']+'.npz'))!=rec['sha256']:raise ValueError('Changed existing label')
            done.append(rec);continue
        worker_contract={k:v for k,v in contract.items() if k not in ('index','source_files')}
        tasks.append((row,*locations[row['token']],anchor,features['split'],str(a.output),worker_contract,ident))
    if a.finalize:
        if tasks:raise ValueError('Cannot close incomplete labels: %d missing scenes'%len(tasks))
        atomic_json(a.output/'COMPLETE.json',{'identity':ident,'scenes':len(done),'expected':len(index),'errors':[],
            'CPU_process_seconds':sum(x['cpu_seconds'] for x in done),'worker_wall_seconds':sum(x['wall_seconds'] for x in done),
            'bytes':sum(x['bytes'] for x in done),'no_frozen_VLA_encoding':True,
            'closure':'all per-scene identities and SHA256 verified; individual shard wall times retained'})
        print(json.dumps({'finalized_scenes':len(done),'identity':ident}),flush=True);return
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        pending={};it=iter(tasks)
        def fill():
            while len(pending)<a.workers and time.monotonic()-start<a.max_seconds and not (a.output/'STOP_REQUESTED').exists():
                task=next(it,None)
                if task is None:break
                pending[pool.submit(build_scene,task)]=task[0]['token']
        fill()
        while pending:
            completed,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
            for future in completed:
                token=pending.pop(future)
                try:done.append(future.result())
                except Exception as e:errors.append({'token':token,'error':repr(e)})
            atomic_json(a.output/('STATUS'+suffix+'.json'),{'complete':len(done),'expected':len(selected),'errors':errors[-10:],'seconds':time.monotonic()-start})
            if len(done)%20==0 or not pending:print(json.dumps({'complete':len(done),'expected':len(selected),'errors':errors[-2:],'seconds':time.monotonic()-start}),flush=True)
            fill()
    summary={'identity':ident,'scenes':len(done),'expected':len(selected),'errors':errors,'seconds':time.monotonic()-start,
        'CPU_process_seconds':sum(x['cpu_seconds'] for x in done),'worker_wall_seconds':sum(x['wall_seconds'] for x in done),
        'bytes':sum(x['bytes'] for x in done),'max_simultaneous_workers':a.workers,'no_frozen_VLA_encoding':True}
    complete=len(done)==len(selected) and not errors
    atomic_json(a.output/(('COMPLETE' if complete else 'INCOMPLETE')+suffix+'.json'),summary)
    print(json.dumps(summary),flush=True)
    if not complete:raise SystemExit(1)


if __name__=='__main__':main()
