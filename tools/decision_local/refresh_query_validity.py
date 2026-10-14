"""Versioned current-only kinematic repair, preserving all geometric labels.

Do not rebuild fields: assert every query pose bit-exact, then update only its
kinematic statistics/mask. Original labels and running producers stay untouched.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys
import time
import numpy as np
import torch
from safetensors.torch import load_file,save_file
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,digest,identity,read,source_sha
from starVLA.model.modules.decision_local.trajectory_queries import QueryConfig,trajectory_queries
from starVLA.dataloader.foresight_dataset import decode_ego


def update_scene(task):
    q0,state,old_poses,config=task
    torch.set_num_threads(1)
    velocity=decode_ego(torch.from_numpy(state)[None])[0,:2].numpy()/.5
    masks=[];stats=[]
    for j,q in enumerate(q0):
        result=trajectory_queries(q,velocity,QueryConfig(**config))
        np.testing.assert_array_equal(result['poses'],old_poses[j],err_msg='Query geometry changed; must regenerate physical labels')
        masks.append(result['kinematic_valid'])
        stats.append([[s[k] for k in ('speed_max_mps','acceleration_max_mps2','yaw_rate_max_rps')] for s in result['kinematics']])
    return torch.tensor(np.array(masks)),torch.tensor(np.array(stats),dtype=torch.float32)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('features','labels','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--max-seconds',type=float,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--follow-labels',action='store_true');a=p.parse_args();started=time.monotonic()
    fm,lm=read(a.features/'identity.json'),read(a.labels/'identity.json')
    if lm['anchor_cache_identity']!=fm['identity']:raise ValueError('Wrong feature/label pair')
    config=QueryConfig(**lm['query_config']).to_dict()
    meta={k:v for k,v in lm.items() if k!='identity'}
    meta.update(query_config=config,parent_geometry_label_identity=lm['identity'],kinematic_source_sha=source_sha(ROOT),
        kinematic_revision_sha256=digest(ROOT/'starVLA/model/modules/decision_local/trajectory_queries.py'),
        kinematic_refresh_tool_sha256=digest(__file__),
        kinematic_repair='current-only heading transitions included in yaw-rate validity; query positions/heading and relation labels exactly preserved')
    ident=identity(meta);a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'identity.json').exists() and read(a.output/'identity.json')['identity']!=ident:raise ValueError('Foreign output')
    atomic_json(a.output/'identity.json',{'identity':ident,**meta})
    for folder in ('chunks','records'):(a.output/folder).mkdir(exist_ok=True)
    seen=set();done=changed=0
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        while time.monotonic()-started<a.max_seconds:
            complete=(a.labels/'COMPLETE.json').exists()
            for receipt in sorted((a.labels/'records').glob('*.json')):
                if receipt.name in seen:continue
                if time.monotonic()-started>=a.max_seconds:break
                r=read(receipt);fr=read(a.features/'records'/receipt.name)
                if r['tokens']!=fr['tokens']:raise ValueError('Reordered current features')
                output=a.output/'records'/receipt.name
                if output.exists():
                    saved=read(output)
                    if saved['identity']!=ident or saved['sha256']!=digest(a.output/'chunks'/(receipt.stem+'.safetensors')):raise ValueError('Changed output')
                    seen.add(receipt.name);done+=len(r['tokens']);changed+=saved['kinematic_variants_changed'];continue
                label_path=a.labels/'chunks'/(receipt.stem+'.safetensors');feature_path=a.features/'chunks'/(receipt.stem+'.safetensors')
                if digest(label_path)!=r['sha256'] or digest(feature_path)!=fr['sha256']:raise ValueError('Parent chunk changed')
                l,f=load_file(str(label_path)),load_file(str(feature_path))
                tasks=[(f['q0'][i].numpy(),f['state'][i].numpy(),l['poses'][i].numpy(),config) for i in range(len(r['tokens']))]
                results=list(pool.map(update_scene,tasks));new_mask=torch.stack([x[0] for x in results]);n=int((l['kinematic_valid']!=new_mask).sum())
                l['kinematic_valid']=new_mask;l['kinematics']=torch.stack([x[1] for x in results])
                dest=a.output/'chunks'/(receipt.stem+'.safetensors');tmp=dest.with_suffix('.tmp')
                save_file(l,str(tmp),metadata={'identity':ident});tmp.replace(dest)
                atomic_json(output,{**r,'identity':ident,'sha256':digest(dest),'bytes':dest.stat().st_size,
                    'parent_label_sha256':r['sha256'],'kinematic_variants_changed':n,'query_poses_bit_exact':True})
                seen.add(receipt.name);done+=len(r['tokens']);changed+=n
                print(f'scenes={done} changed_variants={changed} seconds={time.monotonic()-started:.1f}',flush=True)
            if complete or not a.follow_labels:break
            time.sleep(5.)
    status={'identity':ident,'scenes':done,'kinematic_variants_changed':changed,'seconds':time.monotonic()-started,
        'CPU_worker_wall_hours_including_wait':a.workers*(time.monotonic()-started)/3600.,'original_files_modified':False}
    atomic_json(a.output/'STATUS.json',status)
    if done==len(fm['index']):atomic_json(a.output/'COMPLETE.json',status)


if __name__=='__main__':main()
