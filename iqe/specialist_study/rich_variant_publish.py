"""Prepare the already specified rich construction rows with bounded ordered IO.

The canonical bank owner still assembles and validates its original ordered job.
Source-specific jobs here only make the same immutable label requests available
earlier to record-locked CPU helpers. No new construction or score rule exists.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import socket
import threading
import time
import numpy as np
import pandas as pd
from iqe.contracts import require
from iqe.io import atomic_json, digest, file_hash, lock, read_json


def construct(config, scene, base, proposal, hashes, count, asset, dt):
    from .bank import trajectory_variants
    from iqe.internal_baseline.bank import save_candidate
    observation=read_json(scene.observation_ref)
    require(observation['token']==scene.scene_id and np.isfinite(observation['ego_speed']),
            'invalid current state for variant constructor')
    require(digest({'record':observation,'images':{path:file_hash(path) for path in observation['image_paths']}})==scene.observation_hash,
            'current observation changed before rich construction')
    checks=[]
    proposal=list(proposal)+trajectory_variants(base,dt,float(observation['ego_speed']),checks)
    requests=[];seen=set(hashes)
    for source,trajectory in proposal:
        if count+len(requests)>=config.max_rich_candidates:break
        candidate=save_candidate(config,scene,'rich_'+source,digest([asset,source]),trajectory)
        if candidate.trajectory_hash not in seen:
            requests.append((source,candidate));seen.add(candidate.trajectory_hash)
    return requests,checks


def publish(config_path,workers=8):
    from .config import load_config
    from .resource_guard import check
    from .bank import load_bank,pool,geometry,duplicates_original_gt
    from .binding import scenes_by_role
    from .experiments import endpoint
    from .rich_precompute import collect_completed
    from .rich_labels import pending_indices
    from .data import ReusedContext,admissible_optimized_source
    from .target_rules import verify_target_assets
    from iqe.internal_baseline.bank import prepare_score_job
    from iqe.internal_baseline import score_worker
    require(1<=workers<=8,'bounded rich construction IO workers')
    config=load_config(config_path);check(config,'precompute-rich-official-labels')
    out=config.root/'runtime/rich_variant_precompute'
    with lock(out/'RUN.lock'):
        if (out/'COMPLETE.json').exists():return read_json(out/'COMPLETE.json')
        p=pool(config);meta,data=load_bank(config,'deploy','scorer_fit');scenes=scenes_by_role(config)['scorer_fit']
        require(meta['scene_ids']==[s.scene_id for s in scenes],'rich construction D_S scene order')
        plan={'schema':'fixed_rich_variant_publication_v1','config_identity':config.identity,
              'Base_binding_hash':file_hash(config.root/'BASE_BINDING.json'),'pool_hash':p['hash'],
              'deployment_bank_hash':meta['hash'],'scene_ids':meta['scene_ids'],
              'no_new_candidates':True,'original_max_candidates':config.max_rich_candidates}
        atomic_json(out/'PLAN.json',plan,immutable=True)
        hashes=[{row['candidate']['trajectory_hash'] for row in rows} for rows in meta['records']]
        counts=[len(rows) for rows in meta['records']]
        for item in p['experts']:
            for step in (1000,3000):
                e=endpoint(config,item['cell'],step,'raw')
                candidates=collect_completed(config,scenes,item['cell']+f'_raw_{step}',e['model_hash'])
                require(candidates is not None,'rich endpoint incomplete; no forward fallback')
                pending_indices(candidates,hashes,counts,config.max_rich_candidates)
        verify_target_assets(config)
        info=pd.read_parquet(Path(config.optimized_root)/'optimized_labels.parquet').set_index('token')
        with np.load(Path(config.optimized_root)/'optimized_trajectories.npz') as f:
            optimized={str(i):t for i,t in zip(f['tokens'],f['trajectories'])}
        asset=file_hash(Path(config.optimized_root)/'optimized_trajectories.npz')
        dt=read_json(config.root/'OFFLINE_BASE_CONTRACT.json')['trajectory']['dt']
        local=threading.local()
        def one(index):
            scene=scenes[index]
            if counts[index]>=config.max_rich_candidates:return [],[]
            proposal=[]
            if scene.scene_id in info.index and scene.scene_id in optimized and admissible_optimized_source(str(info.loc[scene.scene_id,'source'])):
                trajectory=optimized[scene.scene_id]
                if geometry(trajectory,dt,horizon=trajectory.shape[0])['valid']:
                    if not hasattr(local,'cache'):local.cache=ReusedContext(config,capacity=1)
                    if not duplicates_original_gt(trajectory,local.cache.read(scene)['target_physical'].numpy()):
                        proposal.append(('offline_optimized',trajectory))
            return construct(config,scene,data['trajectories'][index,0],proposal,hashes[index],counts[index],asset,dt)
        rows=[];sources={};counts_geometry=Counter();start=time.monotonic()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Bound submitted jobs and preserve the original scene/source order.
            for at in range(0,len(scenes),256):
                for index,(requests,checks) in zip(range(at,min(at+256,len(scenes))),executor.map(one,range(at,min(at+256,len(scenes))))):
                    for source,candidate in requests:
                        rows.append((index,source,candidate))
                        sources.setdefault(source,[]).append((scenes[index],candidate))
                    for item in checks:counts_geometry['accepted' if item['valid'] else 'rejected']+=1
                atomic_json(out/'STATUS.json',{'status':'CONSTRUCTING','pid':os.getpid(),'host':socket.gethostname(),
                            'scenes_completed':min(at+256,len(scenes)),'scenes_total':len(scenes),
                            'requests':len(rows),'geometry':dict(counts_geometry),'seconds':time.monotonic()-start})
        jobs=[]
        for source,pairs in sources.items():
            path,job=prepare_score_job(config,[s for s,_ in pairs],[c for _,c in pairs],'fixed_rich_source_'+source)
            atomic_json(path.parent/'COOPERATIVE_PROTOCOL.json',{
                'schema':'record_locked_official_score_v1','job_hash':file_hash(path),
                'worker_code_hash':file_hash(score_worker.__file__),'lock_suffix':'.SCORING.lock',
                'official_numeric_function_changed':False},immutable=True)
            jobs.append({'source':source,'job':str(path),'records':len(job['items']),'job_hash':file_hash(path)})
        # Also publish exactly the canonical all-source job, making direct hash
        # comparison with the original bank owner's eventual request possible.
        path,job=prepare_score_job(config,[scenes[i] for i,_,_ in rows],[c for _,_,c in rows],'study_rich_variants')
        result={'status':'COMPLETE','jobs':jobs,'canonical_job':str(path),'canonical_job_hash':file_hash(path),
                'requests':len(rows),'geometry':dict(counts_geometry),'plan_hash':file_hash(out/'PLAN.json'),
                'seconds':time.monotonic()-start,'primary_workers_started':0,'new_candidates':False}
        atomic_json(out/'COMPLETE.json',result,immutable=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();print(publish(a.config,a.workers),flush=True)


if __name__=='__main__':main()
