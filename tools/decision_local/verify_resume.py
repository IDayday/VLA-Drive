"""Real single-GPU uninterrupted/interrupted optimizer-boundary comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,check_host


def equal(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        if not torch.equal(a,b):raise AssertionError(path+': tensor differs')
    elif isinstance(a,dict):
        if set(a)!=set(b):raise AssertionError(path+': keys differ')
        for k in a:equal(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise AssertionError(path+': length differs')
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'.'+str(i))
    elif hasattr(a,'dtype'):
        import numpy as np
        np.testing.assert_array_equal(a,b,err_msg=path)
    elif a!=b:raise AssertionError(path+': value differs')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('features','labels','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();resources=check_host();a.output.mkdir(parents=True,exist_ok=True)
    records={}
    for kind in ('E_NONE','P_DIFF'):
        base=[sys.executable,'-u',str(ROOT/'tools/decision_local/train_small_model.py'),'--features',str(a.features),
              '--labels',str(a.labels),'--kind',kind,'--scope','smoke','--epochs','40','--batch','64','--microbatch','16','--max-seconds','300']
        for name,n,resume in [('uninterrupted',12,False),('interrupted',5,False),('interrupted',12,True)]:
            dest=a.output/kind/name;command=base+['--output',str(dest),'--max-updates',str(n)]+(['--resume'] if resume else [])
            log=a.output/(kind+'_'+name+'_'+str(n)+'.log')
            with log.open('w') as stream:subprocess.run(command,cwd=ROOT,stdout=stream,stderr=stream,check=True,timeout=360)
        x=torch.load(a.output/kind/'uninterrupted/latest.pt',map_location='cpu',weights_only=False)
        y=torch.load(a.output/kind/'interrupted/latest.pt',map_location='cpu',weights_only=False)
        keys=['identity','model','optimizer','scheduler','rng','completed','epoch','offset','exposure','order']
        for key in keys:equal(x[key],y[key],key)
        records[kind]={'passed':True,'exact_keys':keys,'completed':x['completed'],'interrupted_boundary':5,'exposure':x['exposure']}
    atomic_json(a.output/'COMPLETE.json',{'passed':True,'comparisons':records,'resources':resources,
        'scope':'single-GPU small heads, original S0 always frozen; not a new VLA training seed or large-model resume qualification'})


if __name__=='__main__':main()
