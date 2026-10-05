"""Complete tensor comparison of deterministic real continuous4 versus2+2 states."""
import argparse,json
from pathlib import Path
import torch
from tools.ddpolicy_vehicle.prepare_data import atomic_json


def compare(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or a.shape!=b.shape or a.dtype!=b.dtype:raise AssertionError('Tensor metadata mismatch '+path)
        if not torch.equal(a,b):raise AssertionError('Tensor differs '+path)
        return a.numel()
    if isinstance(a,dict):
        if not isinstance(b,dict) or a.keys()!=b.keys():raise AssertionError('Mapping differs '+path)
        return sum(compare(v,b[k],path+'.'+str(k)) for k,v in a.items())
    if isinstance(a,(list,tuple)):
        if not isinstance(b,type(a)) or len(a)!=len(b):raise AssertionError('Sequence differs '+path)
        return sum(compare(x,y,path+f'[{i}]') for i,(x,y) in enumerate(zip(a,b)))
    if hasattr(a,'shape'):
        import numpy as np
        if not np.array_equal(a,b):raise AssertionError('Array differs '+path)
        return int(a.size)
    if hasattr(a,'__dict__'):
        if type(a)!=type(b):raise AssertionError('State object type differs '+path)
        return compare(vars(a),vars(b),path+'.__dict__')
    if a!=b:raise AssertionError('Value differs '+path+': '+str(a)+' / '+str(b))
    return 0


def main():
    # Full ZeRO states live on a distributed filesystem. Sequential reads avoid
    # thousands of tiny mmap page faults without changing exact comparisons.
    torch.set_num_threads(4)
    p=argparse.ArgumentParser(__doc__)
    for k in ('continuous','resumed','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();first=Path(a.continuous);second=Path(a.resumed);elements=0;files=[]
    for root in (first,second):
        done=json.loads((root/'COMPLETE.json').read_text())
        if done['completed']!=4 or done['exposure']!=128:raise ValueError('Actual4-update128-exposure checkpoints required')
    names={p.name for p in first.glob('*.pt')}
    if names!={p.name for p in second.glob('*.pt')}:raise AssertionError('Checkpoint file populations differ')
    for name in sorted(names):
        x=torch.load(first/name,map_location='cpu',weights_only=False,mmap=False);y=torch.load(second/name,map_location='cpu',weights_only=False,mmap=False)
        # DeepSpeed client checkpoint tag/attempt ID is administrative, not model state.
        if 'model_states' in name:
            # Run identity differs by run ID; checkpoint tags are administrative.
            # Everything else, including scheduler and DeepSpeed progress, is compared.
            x={k:v for k,v in x.items() if k not in ('identity','tag')}
            y={k:v for k,v in y.items() if k not in ('identity','tag')}
        count=compare(x,y,name);elements+=count;files.append({'file':name,'compared_elements':count})
        del x,y
    # Also compare precise scheduler/data/task exposure and all per-rank RNGs.
    c=json.loads((first/'COMPLETE.json').read_text());r=json.loads((second/'COMPLETE.json').read_text())
    for k in ('completed','epoch','offset','exposure','counters','scheduler'):compare(c[k],r[k],k)
    atomic_json(a.output,{'passed':True,'files':files,'compared_elements':elements,'continuous_updates':4,'resumed_updates':4,
        'exposure_each':128,'model_optimizer_rng_exact':True,'scheduler_data_task_progress_exact':True,
        'boundary':'same source/world4/hardware/deterministic settings; no claim across devices/world sizes/non-deterministic operators'})

if __name__=='__main__':main()
