"""Cooperate on the existing six rich-bank endpoint columns, in original batches.

Only trajectories are precomputed. The permanent coordinator still screens
duplicates, applies the 12-candidate budget, and creates official score jobs.
"""
import argparse
from dataclasses import asdict
from pathlib import Path
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from iqe.contracts import CandidateRecord, require, strict_record
from iqe.io import atomic_json, digest, file_hash, lock, read_json


def plan(config):
    from .bank import pool
    from .binding import scenes_by_role
    from .experiments import endpoint
    p=pool(config);scenes=scenes_by_role(config)['scorer_fit']
    columns=[]
    for item in p['experts']:
        for step in (1000,3000):
            e=endpoint(config,item['cell'],step,'raw')
            e['expert_id']=item['cell']+f'_raw_{step}'
            columns.append(e)
    value={'schema':'canonical_rich_endpoint_records_v1','config_hash':config.identity,
           'pool_hash':p['hash'],'Base_binding_hash':file_hash(config.root/'BASE_BINDING.json'),
           'scenes_hash':digest([asdict(s) for s in scenes]),'scene_ids':[s.scene_id for s in scenes],
           'batch_size':32,'columns':columns,'labels_generated':False,'additional_scientific_candidates':False}
    value['hash']=digest(value)
    atomic_json(config.root/'runtime/rich_precompute/PLAN.json',value,immutable=True)
    return value,scenes


def column(config,expert_id,model_hash,scenes):
    path=config.root/'runtime/rich_precompute/PLAN.json'
    if not path.exists():return None
    p=read_json(path)
    require(p['hash']==digest({k:v for k,v in p.items() if k!='hash'})
            and p['config_hash']==config.identity and p['scenes_hash']==digest([asdict(s) for s in scenes]),
            'rich precompute plan or observations changed')
    require(p['Base_binding_hash']==file_hash(config.root/'BASE_BINDING.json')
            and p['pool_hash']==read_json(config.root/'SERVING_CANDIDATE_POOL.json')['hash'],
            'rich precompute Base/pool changed')
    found=[x for x in p['columns'] if x['expert_id']==expert_id and x['model_hash']==model_hash]
    require(len(found)==1,'rich precompute endpoint outside fixed columns')
    return p,found[0]


def collect_completed(config, scenes, expert_id, model_hash, workers=8):
    """Read complete canonical records without loading a model or its features.

    The checkpoint, original batch receipt and every candidate byte are still
    verified. Threads only overlap immutable file reads; returned scene order
    and the numerical inference batch that originally created each row stay fixed.
    """
    require(1 <= workers <= 8, 'bounded rich record readers')
    found = column(config, expert_id, model_hash, scenes)
    if found is None:
        return None
    p, item = found
    paths = [config.root/'runtime/rich_precompute'/expert_id/model_hash/f'batch_{i:06d}.json'
             for i in range((len(scenes)+31)//32)]
    if not all(path.exists() for path in paths):
        return None
    require(file_hash(item['checkpoint']) == item['checksum'], 'rich source checkpoint changed')
    def read(index):
        sub = scenes[index*32:(index+1)*32]
        receipt = read_json(paths[index])
        expected = {'plan_hash': p['hash'], 'column': expert_id, 'model_hash': model_hash,
                    'batch_index': index, 'scene_hash': digest([asdict(s) for s in sub])}
        require(receipt['binding'] == expected, 'rich completed batch identity mismatch')
        rows = [checked_candidate(config, scene, item) for scene in sub]
        require([c.key for c in rows] == receipt['candidate_keys'] and
                [c.trajectory_hash for c in rows] == receipt['trajectory_hashes'],
                'rich completed candidate bytes changed')
        return rows
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = [c for batch_rows in executor.map(read, range(len(paths))) for c in batch_rows]
    require(file_hash(item['checkpoint']) == item['checksum'], 'rich source checkpoint changed during read')
    return rows


def checked_candidate(config,scene,item):
    path=config.root/'candidates'/item['expert_id']/item['model_hash']/(scene.scene_id+'.json')
    require(path.exists(),'rich precompute candidate missing')
    c=strict_record(CandidateRecord,read_json(path))
    binding=read_json(config.root/'BASE_BINDING.json')
    tr=read_json(config.root/'OFFLINE_BASE_CONTRACT.json')['trajectory']
    require(c.scene_id==scene.scene_id and c.expert_id==item['expert_id']
            and c.expert_checkpoint_hash==item['model_hash'] and c.feature_contract_hash==binding['contract_hash']
            and c.generation_config_hash==config.identity and c.finite_valid,'rich candidate identity mismatch')
    require(Path(c.trajectory_ref)==path.with_suffix('.npz') and file_hash(c.trajectory_ref)==c.trajectory_hash,
            'rich trajectory corrupt')
    require(c.horizon==tr['horizon'] and c.dt==tr['dt'] and c.physical_frame==tr['frame']
            and c.raw_action_representation==tr['raw_representation'],'rich trajectory contract changed')
    with np.load(c.trajectory_ref) as f:
        x=f['trajectory'];require(x.dtype==np.float32 and x.shape==(tr['horizon'],3) and np.isfinite(x).all(),
                                 'rich trajectory invalid')
    return c


@torch.no_grad()
def batch(config,p,item,scenes,batch_index,model,device,cache):
    from iqe.internal_baseline.bank import save_candidate
    require([s.scene_id for s in scenes]==p['scene_ids'][batch_index*32:(batch_index+1)*32],
            'rich inference batch layout changed')
    require(file_hash(item['checkpoint'])==item['checksum'],'rich source checkpoint changed')
    out=config.root/'runtime/rich_precompute'/item['expert_id']/item['model_hash']/f'batch_{batch_index:06d}.json'
    expected={'plan_hash':p['hash'],'column':item['expert_id'],'model_hash':item['model_hash'],
              'batch_index':batch_index,'scene_hash':digest([asdict(s) for s in scenes])}
    with lock(str(out)+'.CLAIM.lock'):
        if out.exists():
            receipt=read_json(out);require(receipt['binding']==expected,'rich completed batch identity mismatch')
            rows=[checked_candidate(config,s,item) for s in scenes]
            require([c.key for c in rows]==receipt['candidate_keys'],'rich completed candidate key changed')
            return rows,True
        model.eval();physical=model(cache.batch(scenes,device)).physical.cpu().numpy();rows=[]
        for scene,value in zip(scenes,physical):
            old=config.root/'candidates'/item['expert_id']/item['model_hash']/(scene.scene_id+'.json')
            # Preserve a completed record from a partially committed batch.
            # The surrounding immutable plan fixes original observation identity.
            rows.append(checked_candidate(config,scene,item) if old.exists() else
                        save_candidate(config,scene,item['expert_id'],item['model_hash'],value))
        atomic_json(out,{'binding':expected,'candidate_keys':[c.key for c in rows],
                        'trajectory_hashes':[c.trajectory_hash for c in rows]},immutable=True)
        return rows,False


def run(config_path,shard,shards,read_mib_per_second):
    from .config import load_config
    from .resource_guard import check
    from .temporary_lease import active_lease,can_admit,record_boundary
    from .experiments import load_endpoint
    from .data import ReusedContext
    require(0<=shard<shards<=24 and 0<read_mib_per_second<=64,'bounded rich precompute resources')
    config=load_config(config_path);check(config,'precompute-rich-candidates')
    lease=active_lease('precompute-rich-candidates');p,scenes=plan(config)
    folder=config.root/'runtime/rich_precompute/workers';name=f'worker_{shard:02d}_of_{shards:02d}'
    done=reused=0;started=time.time();cache=ReusedContext(config,64)
    with lock(folder/(name+'.lock')):
        for item in p['columns']:
            if lease and not can_admit(lease):break
            model=load_endpoint(config,item,'cuda:0')
            for index in range(shard,(len(scenes)+31)//32,shards):
                if not record_boundary(lease):
                    atomic_json(folder/(name+'.json'),{'status':'LEASE_DRAINED','completed_batches':done,
                                'reused_batches':reused,'plan_hash':p['hash'],'pid':os.getpid()})
                    return
                sub=scenes[index*32:(index+1)*32];before=time.monotonic()
                rows,was_reused=batch(config,p,item,sub,index,model,'cuda:0',cache)
                done+=1;reused+=int(was_reused)
                # Conservative logical feature bytes, measured from actual cached tensors.
                bytes_per_scene=sum(t.numel()*t.element_size() for t in cache.read(sub[0]).values())
                if not was_reused:
                    target=len(sub)*bytes_per_scene/(read_mib_per_second*1024**2)
                    time.sleep(max(0.,target-(time.monotonic()-before)))
                atomic_json(folder/(name+'.json'),{'status':'RUNNING','completed_batches':done,'reused_batches':reused,
                            'column':item['expert_id'],'last_batch':index,'pid':os.getpid(),'host':os.uname().nodename,
                            'plan_hash':p['hash'],'read_mib_per_second_limit':read_mib_per_second,
                            'seconds':time.time()-started,'updated_unix':time.time()})
            del model
        atomic_json(folder/(name+'.json'),{'status':'COMPLETE' if not lease or can_admit(lease) else 'LEASE_DRAINED',
                    'completed_batches':done,'reused_batches':reused,'pid':os.getpid(),'host':os.uname().nodename,
                    'plan_hash':p['hash'],'seconds':time.time()-started})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,required=True)
    p.add_argument('--read-mib-per-second',type=float,default=32.)
    a=p.parse_args();run(a.config,a.shard,a.shards,a.read_mib_per_second)


if __name__=='__main__':main()
