"""Compare actual model, FP32 Adam states and each rank's RNG after real resume."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tools.ddpolicy_vehicle.prepare_data import atomic_json


def differences(a,b,path='root',result=None):
    if result is None:result=[]
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype:result.append({'path':path,'kind':'tensor_contract'})
        else:
            x,y=a.reshape(-1),b.reshape(-1)
            maximum=0.;changed=0
            for start in range(0,x.numel(),1<<20):
                left,right=x[start:start+(1<<20)],y[start:start+(1<<20)]
                changed+=int((left!=right).sum())
                if left.is_floating_point() and left.numel():maximum=max(maximum,float((left.float()-right.float()).abs().max()))
            if changed:result.append({'path':path,'changed_elements':changed,'max_abs':maximum})
    elif isinstance(a,np.ndarray):
        if not isinstance(b,np.ndarray) or not np.array_equal(a,b):result.append({'path':path,'kind':'numpy'})
    elif isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys():result.append({'path':path,'kind':'keys'})
        else:
            for key in a:differences(a[key],b[key],path+'.'+str(key),result)
    elif isinstance(a,(list,tuple)):
        if type(a)!=type(b) or len(a)!=len(b):result.append({'path':path,'kind':'sequence'})
        else:
            for i,(left,right) in enumerate(zip(a,b)):differences(left,right,path+'.'+str(i),result)
    elif type(a).__module__=='deepspeed.runtime.fp16.loss_scaler':
        if type(a)!=type(b):result.append({'path':path,'kind':'loss_scaler_type'})
        else:differences(vars(a),vars(b),path+'.state',result)
    elif a!=b:result.append({'path':path,'kind':'value','left':str(a),'right':str(b)})
    return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--continuous',required=True);p.add_argument('--resumed',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();torch.set_num_threads(4);left,right=Path(a.continuous),Path(a.resumed)
    report={'scope':'same source, device count, deterministic setting, real camera updates; no cross-device exactness claim',
            'continuous':str(left),'resumed':str(right),'files':{}}
    ls={p.name for p in left.glob('*.pt')};rs={p.name for p in right.glob('*.pt')}
    if ls!=rs:raise ValueError('Checkpoint file inventories differ')
    for name in sorted(ls):
        # Sequential loading avoids page-sized mmap traffic on distributed FS.
        # One pair of files at a time; this check needs ~32GB host RAM per pair.
        x=torch.load(left/name,map_location='cpu',weights_only=False,mmap=False)
        y=torch.load(right/name,map_location='cpu',weights_only=False,mmap=False)
        diff=differences(x,y);report['files'][name]={'exact':not diff,'differences':diff};del x,y
        print(name,'EXACT' if not diff else 'DIFFERENT',flush=True)
    report['passed']=all(r['exact'] for r in report['files'].values())
    atomic_json(a.output,report)
    if not report['passed']:raise AssertionError('Resume differs; detailed differences retained')


if __name__=='__main__':main()
