"""Canonical D_S deployment columns; exact same endpoints, batches and records.

Two selected raw endpoints already occur in the fixed rich precompute. Reuse
their verified bytes rather than running those frozen heads a second time.
The primary bank coordinator alone requests/validates official labels.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import os
import time

import numpy as np
import torch

from iqe.contracts import require
from iqe.io import atomic_json, digest, file_hash, lock, read_json
from iqe.internal_baseline.bank import save_candidate
from .rich_precompute import checked_candidate


def plan(config):
    from .bank import pool
    from .binding import scenes_by_role
    p=pool(config);scenes=scenes_by_role(config)['scorer_fit']
    rich=read_json(config.root/'runtime/rich_precompute/PLAN.json')
    require(rich['hash']==digest({k:v for k,v in rich.items() if k!='hash'})
            and rich['pool_hash']==p['hash'] and rich['scene_ids']==[s.scene_id for s in scenes],
            'rich/deployment observation or pool mismatch')
    columns=[]
    for item in p['experts']:
        value={k:item[k] for k in ('cell','step','endpoint','checkpoint','checksum','model_hash','private_reader','expert_id')}
        matching=[r for r in rich['columns'] if r['model_hash']==item['model_hash'] and r['checksum']==item['checksum']]
        value['reuse_rich_source']=matching[0] if matching else None
        columns.append(value)
    value={'schema':'canonical_deployment_endpoint_records_v1','config_hash':config.identity,
           'pool_hash':p['hash'],'Base_binding_hash':file_hash(config.root/'BASE_BINDING.json'),
           'scenes_hash':digest([asdict(s) for s in scenes]),'scene_ids':[s.scene_id for s in scenes],
           'batch_size':32,'columns':columns,'rich_plan_hash':rich['hash'],'labels_generated':False}
    value['hash']=digest(value)
    atomic_json(config.root/'runtime/deploy_precompute/PLAN.json',value,immutable=True)
    return value,scenes


def validate_plan(config,p,scenes):
    require(p['hash']==digest({k:v for k,v in p.items() if k!='hash'}) and p['config_hash']==config.identity,
            'deployment precompute plan changed')
    require(p['scene_ids']==[s.scene_id for s in scenes] and p['scenes_hash']==digest([asdict(s) for s in scenes]),
            'deployment precompute observation/order changed')
    require(p['pool_hash']==read_json(config.root/'SERVING_CANDIDATE_POOL.json')['hash']
            and p['Base_binding_hash']==file_hash(config.root/'BASE_BINDING.json'),'deployment Base/pool changed')


def receipt_path(config,item,index):
    return config.root/'runtime/deploy_precompute'/item['expert_id']/item['model_hash']/f'batch_{index:06d}.json'


def batch_binding(p,item,scenes,index):
    require([s.scene_id for s in scenes]==p['scene_ids'][index*32:(index+1)*32],'deployment batch layout changed')
    return {'plan_hash':p['hash'],'column':item['expert_id'],'model_hash':item['model_hash'],
            'batch_index':index,'scene_hash':digest([asdict(s) for s in scenes])}


def read_batch(config,p,item,scenes,index):
    receipt=read_json(receipt_path(config,item,index))
    require(receipt['binding']==batch_binding(p,item,scenes,index),'completed deployment batch identity changed')
    rows=[checked_candidate(config,s,item) for s in scenes]
    require([c.key for c in rows]==receipt['candidate_keys']
            and [c.trajectory_hash for c in rows]==receipt['trajectory_hashes'],'deployment record checksum/key changed')
    return rows


def verified_rich_values(config,p,item,scenes,index):
    source=item['reuse_rich_source']
    require(source and source['model_hash']==item['model_hash'] and source['checksum']==item['checksum'],
            'rich reuse requires exactly the same checkpoint and model')
    receipt=config.root/'runtime/rich_precompute'/source['expert_id']/source['model_hash']/f'batch_{index:06d}.json'
    r=read_json(receipt)
    expected={'plan_hash':p['rich_plan_hash'],'column':source['expert_id'],'model_hash':source['model_hash'],
              'batch_index':index,'scene_hash':digest([asdict(s) for s in scenes])}
    require(r['binding']==expected,'rich source original batch/observation changed')
    records=[checked_candidate(config,s,source) for s in scenes]
    require([x.key for x in records]==r['candidate_keys'] and [x.trajectory_hash for x in records]==r['trajectory_hashes'],
            'rich source records changed')
    values=[]
    for candidate in records:
        with np.load(candidate.trajectory_ref) as payload:values.append(payload['trajectory'].copy())
    return values


@torch.no_grad()
def write_batch(config,p,item,scenes,index,model,device,cache):
    out=receipt_path(config,item,index);binding=batch_binding(p,item,scenes,index)
    require(file_hash(item['checkpoint'])==item['checksum'],'deployment endpoint changed')
    with lock(str(out)+'.CLAIM.lock'):
        if out.exists():return read_batch(config,p,item,scenes,index),True
        if item['reuse_rich_source']:
            values=verified_rich_values(config,p,item,scenes,index)
        else:
            require(model is not None,'missing deployment head')
            model.eval();values=model(cache.batch(scenes,device)).physical.cpu().numpy()
        rows=[]
        for scene,value in zip(scenes,values):
            old=config.root/'candidates'/item['expert_id']/item['model_hash']/(scene.scene_id+'.json')
            rows.append(checked_candidate(config,scene,item) if old.exists() else
                        save_candidate(config,scene,item['expert_id'],item['model_hash'],value))
        atomic_json(out,{'binding':binding,'candidate_keys':[c.key for c in rows],
                        'trajectory_hashes':[c.trajectory_hash for c in rows],
                        'reused_rich_endpoint':bool(item['reuse_rich_source'])},immutable=True)
        return rows,False


def collect(config,scenes,item):
    path=config.root/'runtime/deploy_precompute/PLAN.json'
    if not path.exists():return None
    p=read_json(path);validate_plan(config,p,scenes)
    found=[x for x in p['columns'] if x['expert_id']==item['expert_id'] and x['model_hash']==item['model_hash']]
    require(len(found)==1,'deployment endpoint outside immutable plan')
    rows=[];deadline=time.monotonic()+7200
    for index in range((len(scenes)+31)//32):
        while not receipt_path(config,found[0],index).exists():
            require(time.monotonic()<deadline,'bounded deployment precompute wait; preserve partial records')
            time.sleep(5)
        rows+=read_batch(config,p,found[0],scenes[index*32:(index+1)*32],index)
    return rows


def run(config_path,shard,shards,read_mib_per_second):
    from .config import load_config
    from .resource_guard import check
    from .temporary_lease import active_lease,can_admit,record_boundary
    from .experiments import load_endpoint
    from .data import ReusedContext
    require(0<=shard<shards<=24 and 0<read_mib_per_second<=64,'bounded deployment resources')
    config=load_config(config_path);check(config,'precompute-deployment-candidates')
    lease=active_lease('precompute-deployment-candidates');p,scenes=plan(config)
    folder=config.root/'runtime/deploy_precompute/workers';name=f'worker_{shard:02d}_of_{shards:02d}'
    done=reused=source_reused=0;start=time.time();cache=ReusedContext(config,64)
    with lock(folder/(name+'.lock')):
        for item in p['columns']:
            if lease and not can_admit(lease):break
            model=None if item['reuse_rich_source'] else load_endpoint(config,item,'cuda:0')
            for index in range(shard,(len(scenes)+31)//32,shards):
                if not record_boundary(lease):
                    atomic_json(folder/(name+'.json'),{'status':'LEASE_DRAINED','completed_batches':done,'plan_hash':p['hash']});return
                sub=scenes[index*32:(index+1)*32];before=time.monotonic()
                _,was_reused=write_batch(config,p,item,sub,index,model,'cuda:0',cache)
                done+=1;reused+=int(was_reused);source_reused+=int(bool(item['reuse_rich_source']))
                if not was_reused and not item['reuse_rich_source']:
                    size=sum(t.numel()*t.element_size() for t in cache.read(sub[0]).values())
                    time.sleep(max(0.,len(sub)*size/(read_mib_per_second*1024**2)-(time.monotonic()-before)))
                atomic_json(folder/(name+'.json'),{'status':'RUNNING','completed_batches':done,'reused_batches':reused,
                    'rich_source_batches_reused':source_reused,'column':item['expert_id'],'last_batch':index,
                    'plan_hash':p['hash'],'pid':os.getpid(),'host':os.uname().nodename,'updated_unix':time.time()})
            del model
        atomic_json(folder/(name+'.json'),{'status':'COMPLETE' if not lease or can_admit(lease) else 'LEASE_DRAINED',
                    'completed_batches':done,'reused_batches':reused,'rich_source_batches_reused':source_reused,
                    'plan_hash':p['hash'],'pid':os.getpid(),'host':os.uname().nodename,'seconds':time.time()-start})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,required=True)
    p.add_argument('--read-mib-per-second',type=float,default=32.)
    a=p.parse_args();run(a.config,a.shard,a.shards,a.read_mib_per_second)


if __name__=='__main__':main()
