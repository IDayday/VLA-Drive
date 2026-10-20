"""Compare trusted task snapshots: model, optimizer, RNG, and update identity."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from asset_adapter import digest


def fingerprint(value):
    h=hashlib.sha256()
    def visit(x):
        h.update(type(x).__name__.encode()+b':')
        if isinstance(x,torch.Tensor):
            h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode())
            for chunk in x.detach().reshape(-1).split(1024*1024):
                h.update(chunk.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray):
            h.update(str(x.dtype).encode());h.update(str(x.shape).encode());h.update(x.tobytes())
        elif isinstance(x,dict):
            for key in sorted(x,key=lambda k:(type(k).__name__,repr(k))):visit(key);visit(x[key])
        elif isinstance(x,(tuple,list)):
            h.update(str(len(x)).encode())
            for item in x:visit(item)
        elif x is None or isinstance(x,(str,int,float,bool,bytes)):
            h.update(repr(x).encode())
        else:raise TypeError('Unsupported checkpoint state type: '+type(x).__name__)
        h.update(b';')
    visit(value)
    return h.hexdigest()


def checkpoint_hashes(path):
    path=Path(path)
    side=json.loads(path.with_suffix('.json').read_text())
    if digest(path)!=side['sha256']:raise ValueError('Checkpoint SHA256 mismatch')
    # Only trusted locally generated snapshots. mmap bounds peak resident memory.
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    keys=['schema','identity','arm','seed','completed_updates','model','optimizer',
          'python_rng','numpy_rng','torch_rng','cuda_rng']
    return {k:fingerprint(state[k]) for k in keys}


def compare(left,right):
    a=checkpoint_hashes(left);b=checkpoint_hashes(right)
    different=[k for k in a if a[k]!=b[k]]
    return {'status':'PASS' if not different else 'DIFFERENT','different_fields':different,
            'left':str(left),'right':str(right),'field_sha256':{'continuous':a,'resumed':b},
            'comparison':'Exact continuation state; elapsed time and parent paths intentionally excluded'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--continuous',required=True);p.add_argument('--resumed',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    result=compare(a.continuous,a.resumed)
    with Path(a.out).open('x') as f:json.dump(result,f,indent=2)
    raise SystemExit(0 if result['status']=='PASS' else 2)
