"""Current-only query cache for deployment/export; opens no scene GT files."""
import argparse
import os
from pathlib import Path
import sys
import time
import torch
from safetensors.torch import load_file,save_file
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,check_host,digest,identity,read,source_sha
from starVLA.model.modules.decision_local.trajectory_queries import QueryConfig,trajectory_queries
from starVLA.model.modules.structured_world.grid import GridSpec
from starVLA.model.modules.structured_world.queries import body_points
from starVLA.dataloader.foresight_dataset import decode_ego


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--query-config',type=Path);p.add_argument('--max-seconds',type=float,required=True)
    a=p.parse_args();resources=check_host();started=time.monotonic();torch.set_num_threads(1)
    fmeta=read(a.features/'identity.json')
    if not (a.features/'COMPLETE.json').exists():raise ValueError('Complete frozen feature population required')
    config=QueryConfig(**read(a.query_config)) if a.query_config else QueryConfig()
    body={'length':5.176,'width':2.297,'rear_axle_to_center':1.461};grid=GridSpec()
    declaration={'schema':'decision_local_current_only_queries_v1','features':fmeta['identity'],
        'source_sha':source_sha(ROOT),'writer_sha256':digest(__file__),'query_config':config.to_dict(),'body':body,
        'grid':grid.to_dict() if hasattr(grid,'to_dict') else vars(grid),'contains_GT':False,'split':fmeta['split']}
    ident=identity(declaration);a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'identity.json').exists() and read(a.output/'identity.json')['identity']!=ident:raise ValueError('Foreign queries')
    atomic_json(a.output/'identity.json',{'identity':ident,**declaration})
    for k in ('chunks','records'):(a.output/k).mkdir(exist_ok=True)
    done=0
    for receipt in sorted((a.features/'records').glob('*.json')):
        if time.monotonic()-started>=a.max_seconds:raise TimeoutError('Query budget exhausted')
        meta=read(receipt);path=a.features/'chunks'/(receipt.stem+'.safetensors')
        if digest(path)!=meta['sha256']:raise ValueError('Feature chunk changed')
        f=load_file(str(path));rows=[]
        for i in range(len(meta['tokens'])):
            v=decode_ego(f['state'][i:i+1])[0,:2].numpy()/.5;anchors=[]
            for plan in f['q0'][i]:
                q=trajectory_queries(plan.numpy(),v,config);poses=torch.from_numpy(q['poses'])
                _,invalid=grid.normalized_reference(body_points(poses,body))
                anchors.append({'poses':poses,'in_grid':~invalid.any(-1),'kinematic_valid':torch.from_numpy(q['kinematic_valid'])})
            rows.append({k:torch.stack([q[k] for q in anchors]) for k in anchors[0]})
        tensors={k:torch.stack([q[k] for q in rows]) for k in rows[0]}
        tensors['body']=torch.tensor([[body[k] for k in ('length','width','rear_axle_to_center')]]).expand(len(rows),-1).contiguous()
        dest=a.output/'chunks'/(receipt.stem+'.safetensors');tmp=dest.with_suffix('.tmp');save_file(tensors,str(tmp),metadata={'identity':ident});tmp.replace(dest)
        atomic_json(a.output/'records'/receipt.name,{'identity':ident,'tokens':meta['tokens'],'sha256':digest(dest)})
        done+=len(rows)
    if done!=len(fmeta['index']):raise ValueError('Incomplete query population')
    atomic_json(a.output/'COMPLETE.json',{'identity':ident,'scenes':done,'CPU_seconds':time.monotonic()-started,'resources':resources,'contains_GT':False})


if __name__=='__main__':main()
