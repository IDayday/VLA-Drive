"""Compare D's credit adapter against executable functions from pinned AUG.

Extract only the two pure NumPy functions and their coordinate helper; this
avoids importing the upstream Ray/verl controller into our training process.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Optional
import numpy as np
import torch
from starVLA.rl.d46_augustus.split_credit import normalize_centered,reverse_travel
from starVLA.rl.flow_grpo.transactions import atomic_json

UPSTREAM_SHA='edf3422304e4b2987e839a9212cf1176782e7380'


def released_functions(root):
    root=Path(root)
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=UPSTREAM_SHA:
        raise ValueError('Unregistered AUG source revision')
    namespace=dict(np=np,Optional=Optional)
    hashes={}
    for relative,names in (
        ('AUGUstus_GRPO/recipe/transfusion/transfusion_reward.py',('_as_xy','reverse_travel')),
        ('AUGUstus_GRPO/recipe/transfusion/sync_driver.py',('_group_zscore',))):
        path=root/relative;raw=path.read_bytes()
        expected=subprocess.check_output(['git','show',UPSTREAM_SHA+':'+relative],cwd=root)
        if raw!=expected:raise ValueError('Modified official source '+relative)
        hashes[relative]=hashlib.sha256(raw).hexdigest()
        nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)and n.name in names]
        if {n.name for n in nodes}!=set(names):raise ValueError('Missing official pure function')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['_group_zscore'],namespace['reverse_travel'],hashes


def audit(root):
    zscore,reverse,hashes=released_functions(root)
    rng=np.random.default_rng(42);normalization=[];reversal=[]
    for group in (8,16,32):
        values=np.concatenate([rng.uniform(0,5,(512,group)),np.ones((1,group)),
            rng.uniform(0,1e-8,(1,group)),np.arange(group)[None]*1e-6],axis=0)
        raw=values-values.mean(1,keepdims=True)
        expected=np.stack([zscore(v)for v in values]).astype(np.float32)
        actual=normalize_centered(torch.from_numpy(raw)).numpy()
        np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-6)
        normalization.append(dict(group=group,groups=len(values),maximum_abs_error=float(np.max(abs(actual-expected)))))
    paths=rng.normal(size=(1024,8,3)).astype(np.float32)
    paths[:3,:,0]=np.array([np.arange(1,9),np.full(8,-1.),np.full(8,-.005)])
    expected=np.array([reverse(p)for p in paths]);actual=reverse_travel(torch.from_numpy(paths)).numpy()
    np.testing.assert_array_equal(actual,expected)
    return dict(status='PASS',upstream_sha=UPSTREAM_SHA,source_hashes=hashes,
        normalization=normalization,reverse_paths=len(paths),reverse_max_abs_error=float(np.max(abs(actual-expected))),
        scope='Pure formula equivalence, not training or PDMS benefit',NAVTEST_used=False)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--upstream',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=audit(a.upstream);atomic_json(a.output,result);print(json.dumps(result))


if __name__=='__main__':main()
