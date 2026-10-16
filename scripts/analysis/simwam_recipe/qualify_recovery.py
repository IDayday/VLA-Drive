"""Exact native recovery comparison, including the next fresh behavior chain."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from starVLA.rl.flow_grpo.transactions import atomic_json


def equal(a,b,path='root'):
    if torch.is_tensor(a):
        if not torch.equal(a,b):raise ValueError('Tensor resume mismatch: '+path)
    elif isinstance(a,np.ndarray):
        if not np.array_equal(a,b):raise ValueError('Array resume mismatch: '+path)
    elif isinstance(a,dict):
        if set(a)!=set(b):raise ValueError('Keys differ: '+path)
        for k in a:equal(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise ValueError(path)
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'.'+str(i))
    elif a!=b:raise ValueError('Value resume mismatch: '+path)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--continuous',required=True);p.add_argument('--resumed',required=True);p.add_argument('--output',required=True)
    p.add_argument('--interrupted-after',type=int,nargs='*',default=[1]);a=p.parse_args()
    roots=[Path(a.continuous),Path(a.resumed)]
    def read(root,rel):return torch.load(root/rel,map_location='cpu',weights_only=False)
    ck='checkpoints/update000008_inner0/'
    equal(*[read(r,ck+'optimizer.pt') for r in roots])
    world=json.loads((roots[0]/ck/'COMPLETE').read_text())['world']
    if json.loads((roots[1]/ck/'COMPLETE').read_text())['world']!=world:raise ValueError('World mismatch')
    for rank in range(world):
        ranks=[read(r,ck+f'rank{rank}.pt') for r in roots]
        equal(ranks[0]['rng'],ranks[1]['rng'],f'rank{rank}.RNG')
        for k in ranks[0]['state']:
            if not k.startswith('seconds_'):equal(ranks[0]['state'][k],ranks[1]['state'][k],k)
        for collection in (1,2):
            rows=[read(r,f'behavior/collection{collection:06}/rank{rank}.pt') for r in roots]
            for key in ('chain','timesteps','deltas','stoch_idx','old_logp','advantages','adv_mask','global_std','rewards','old_probability','behavior_version','token','log','stream_position'):
                equal(rows[0][key],rows[1][key],f'rank{rank}.collection{collection}.{key}')
        metrics=[[json.loads(l) for l in (r/f'metrics_rank{rank}.jsonl').read_text().splitlines()] for r in roots]
        for i in range(8):
            for k in ('loss','gradient_norm','ratio_max_abs_from_one','metrics','lr'):
                equal(metrics[0][i][k],metrics[1][i][k],f'rank{rank}.update{i+1}.{k}')
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='PASS',world=world,all_ranks_checked=True,interrupted_after_inner=a.interrupted_after,resumed_remaining_inner_updates=True,final_updates=8,
                optimizer_adapters_scheduler_RNG_exact=True,next_fresh_chain_old_probability_advantages_exact=True,
                losses_gradnorm_ratios_learning_rate_exact=True,continuous=str(roots[0]),resumed=str(roots[1]))
    atomic_json(out,report);print(json.dumps(report))


if __name__=='__main__':main()
