"""Compare every tensor in equivalent online/rolling checkpoints and all rank logs.

This deliberately ignores storage identity, not training state. It does not
accept relative error or a subset of parameter tensors as an exact continuation.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from iqe.io import atomic_json, read_json, file_hash


def equal(a,b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        return isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict) or isinstance(b,dict):
        return isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)) or isinstance(b,(tuple,list)):
        return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('online',type=Path);p.add_argument('cached',type=Path);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(2)
    states=[];paths=[]
    for root in (a.online,a.cached):
        result=read_json(root/'RESULT.json');path=root/f'step_{result["final_step"]:06d}.pt'
        receipt=read_json(str(path)+'.COMPLETE.json')
        assert file_hash(path)==receipt['checksum'],'checkpoint receipt mismatch'
        states.append(torch.load(path,map_location='cpu',mmap=True,weights_only=False));paths.append(str(path))
    keys=['model','optimizer','scheduler','rng_by_rank','optimizer_step','epoch','offset','fit_scene_ids','seed','dependencies']
    assert all(k in states[0] and k in states[1] for k in keys),'missing training state'
    checks={k:equal(states[0][k],states[1][k]) for k in keys}
    ranks=[]
    for rank in range(8):
        lhs=read_json(a.online/f'rank_{rank}.json');rhs=read_json(a.cached/f'rank_{rank}.json')
        rows={r['optimizer_step']:r for r in lhs['rows']}
        fields=['sample_ids','global_batch','losses','gradient_norm','lr','valid_denominators','used_parameter_names_hash']
        ranks.append({'rank':rank,'steps':len(rhs['rows']),
            'all_updates_equal':all(x['optimizer_step'] in rows and all(x[k]==rows[x['optimizer_step']][k] for k in fields) for x in rhs['rows']),
            'no_image_or_visual_on_hit':rhs['production_cache']['calls']['visual_calls_on_hit']==0 and
                rhs['production_cache']['calls']['online_encode']==0 and
                rhs['production_cache']['cache_reader']['image_decodes']==0})
    result={'status':'PASS' if all(checks.values()) and all(r['all_updates_equal'] and r['no_image_or_visual_on_hit'] for r in ranks) else 'FAIL',
        'checkpoints':paths,'full_training_state_checks':checks,'ranks':ranks,'exact':True}
    atomic_json(a.output,result);print(result)
    return 0 if result['status']=='PASS' else 2

if __name__=='__main__':raise SystemExit(main())
