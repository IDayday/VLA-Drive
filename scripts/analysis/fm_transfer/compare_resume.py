"""Verify full parameters, Adam, scheduler, per-rank RNG and sample continuation."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def equal(a,b,path='root'):
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):raise ValueError(path)
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise ValueError(path)
    elif isinstance(a,dict):
        if set(a)!=set(b):raise ValueError(path+' keys')
        for key in a:equal(a[key],b[key],path+'/'+str(key))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise ValueError(path+' length')
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
    elif a!=b:raise ValueError(path)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    if (root/'validation.json').exists():raise FileExistsError(root/'validation.json')
    inv=json.loads((root/'invocation.json').read_text())
    if inv['status']!='EXECUTION_COMPLETE_REQUIRES_COMPARISON':raise ValueError('Both exact processes must finish')
    dirs=[root/x/'checkpoints/update000008' for x in ['continuous','resumed']]
    seals=[]
    for d in dirs:
        seal=json.loads((d/'COMPLETE').read_text());files=json.loads((d/'files.json').read_text())
        if file_sha(d/'files.json')!=seal['files_sha256']:raise ValueError('Unsealed checkpoint')
        for name,h in files.items():
            if file_sha(d/name)!=h:raise ValueError('Changed checkpoint file '+name)
        seals.append(seal)
    for name in ['model.pt','optimizer.pt',*[f'rank{i}.pt' for i in range(8)]]:
        left,right=[torch.load(d/name,map_location='cpu',weights_only=False,mmap=True) for d in dirs]
        equal(left,right,name)
        del left,right
    results=[]
    for rank in range(8):
        base=json.loads((root/'continuous'/f'result_rank{rank}.json').read_text())
        res=json.loads((root/'resumed'/f'result_rank{rank}.json').read_text())
        if base['end_update']!=8 or res['start_update']!=4 or res['end_update']!=8:raise ValueError('Wrong update windows')
        if not all(base['actual_parameter_sample_changes'].values()):
            raise ValueError('Some full-head parameters did not change in their sampled coordinates')
        for norms in base['gradient_norms'].values():
            if not all(np.isfinite(v) and v>0 for v in norms.values()):raise ValueError('Missing effective gradient')
        def journal(which):
            paths=list((root/which/'invocations').glob(f'*/metrics_rank{rank}.jsonl'))
            return sorted([json.loads(line) for p in paths for line in p.read_text().splitlines()],key=lambda r:r['updates'])
        b=journal('continuous');r=journal('resumed')
        for x,y in zip(b[4:],r):
            for key in ['updates','cursor','scene_tokens','loss','gradient_norm_before_clip','learning_rate','unweighted_mse','time_weight']:
                equal(x[key],y[key],key)
        results.append(dict(rank=rank,parameter_tensors=len(base['actual_parameter_sample_changes']),
            all_gradient_norms_positive=True,all_sampled_parameter_deltas_nonzero=True,
            peak_GPU_bytes=max(base['peak_GPU_bytes'],res['peak_GPU_bytes']),
            steady_update_seconds=[x['seconds_update'] for x in b[1:]],data_seconds=[x['seconds_data'] for x in b[1:]]))
    result=dict(status='PASS',scope='Full model eight-GPU SFT: continuous8 versus saved4/resumed8',
        model_Adam_scheduler_RNG_cursor_exact=True,loss_and_data_continuation_exact=True,
        qualification_only=True,source=inv['source'],ranks=results,seals=seals,
        limits=['This is optimizer/numerical acceptance, not driving performance.','Original-observation deployment and final SFT planning evaluation are separate gates.'])
    atomic_json(root/'validation.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('ranks','seals')}))


if __name__=='__main__':main()
