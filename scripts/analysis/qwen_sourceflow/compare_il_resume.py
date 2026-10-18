"""Compare native READY boundaries after continuous versus resumed updates."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from starVLA.rl.qwen_sourceflow.loading import inspect_il
from starVLA.rl.flow_grpo.transactions import atomic_json


def compare(left,right,path,stats):
    if isinstance(left,torch.Tensor):
        stats['tensors']+=1;stats['elements']+=left.numel()
        same=left.shape==right.shape and left.dtype==right.dtype and torch.equal(left,right)
        if not same:
            item=dict(path=path,shape=list(left.shape))
            if left.shape==right.shape:
                item['max_abs']=float((left.double()-right.double()).abs().max())
            stats['differences'].append(item)
    elif isinstance(left,np.ndarray):
        if not np.array_equal(left,right):stats['differences'].append(dict(path=path,kind='numpy'))
    elif isinstance(left,dict):
        if left.keys()!=right.keys():stats['differences'].append(dict(path=path,kind='keys'));return
        for key in left:compare(left[key],right[key],path+'/'+str(key),stats)
    elif isinstance(left,(tuple,list)):
        if len(left)!=len(right):stats['differences'].append(dict(path=path,kind='length'));return
        for i,(l,r) in enumerate(zip(left,right)):compare(l,r,path+'/'+str(i),stats)
    elif left!=right:stats['differences'].append(dict(path=path,left=str(left),right=str(right)))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--continuous',required=True);p.add_argument('--resumed',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    torch.set_num_threads(2);paths=[Path(a.continuous),Path(a.resumed)]
    info=[inspect_il(p,qualification=True) for p in paths]
    if info[0]['identity']!=info[1]['identity'] or info[0]['seal']['updates']!=info[1]['seal']['updates']:
        raise ValueError('Unmatched source/config or optimizer boundaries')
    report=dict(schema='native_IL_actual_resume_v1',continuous=str(paths[0]),resumed=str(paths[1]),updates=info[0]['seal']['updates'],files={})
    for name in ['model.pt','optimizer.pt']+[f'rank{i}.pt' for i in range(info[0]['seal']['world'])]:
        stats=dict(tensors=0,elements=0,differences=[])
        left,right=[torch.load(p/name,map_location='cpu',mmap=True,weights_only=False) for p in paths]
        compare(left,right,name,stats);report['files'][name]=stats;del left,right
    report['status']='PASS' if not any(s['differences'] for s in report['files'].values()) else 'FAIL'
    atomic_json(a.output,report)
    print(json.dumps(dict(status=report['status'],updates=report['updates'],differences={n:len(s['differences']) for n,s in report['files'].items()})))
    if report['status']!='PASS':raise RuntimeError('Native restart differs; inspect actual tensor/RNG evidence')


if __name__=='__main__':main()
