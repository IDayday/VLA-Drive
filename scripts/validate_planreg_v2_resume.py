"""Bounded real-data 4-step reference versus 2+2-step complete resume."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import torch


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--accumulate',type=int,default=1)
    args=p.parse_args()
    root=Path(args.output)
    if root.exists():raise FileExistsError('New resume audit directory required')
    root.mkdir(parents=True)
    common=[sys.executable,'-u','scripts/train_planreg_v2.py','--config',args.config,'--manifest',args.manifest,
            '--smoke-steps','4','--microbatch','1','--workers','2','--accumulate',str(args.accumulate)]
    cases=[('reference',[]),('interrupted',['--stop-after','2']),
           ('resumed',['--resume',str(root/'interrupted'/'last.ckpt')])]
    for label,extra in cases:
        destination=root/('interrupted' if label=='resumed' else label)
        print('RUNNING',label,flush=True)
        with (root/(label+'.log')).open('w') as log:
            subprocess.run(common+['--output',str(destination)]+extra,stdout=log,stderr=subprocess.STDOUT,check=True)
    reference=torch.load(root/'reference'/'last.ckpt',map_location='cpu',weights_only=False)
    resumed=torch.load(root/'interrupted'/'last.ckpt',map_location='cpu',weights_only=False)
    differences={}
    for name,value in reference['model'].items():
        other=resumed['model'][name]
        if torch.is_tensor(value) and not torch.equal(value,other):
            differences[name]=float((value.double()-other.double()).abs().max())
        elif not torch.is_tensor(value) and value!=other:
            differences[name]='metadata mismatch'
    moments_equal=True
    for index,state in reference['optimizer']['state'].items():
        for name,value in state.items():
            if torch.is_tensor(value) and not torch.equal(value,resumed['optimizer']['state'][index][name]):moments_equal=False
    result=dict(model_equal=not differences,model_differences=differences,optimizer_moments_equal=moments_equal,
        scheduler_equal=reference['scheduler']==resumed['scheduler'],
        sampler_progress_equal=(reference['epoch'],reference['step_in_epoch'])==(resumed['epoch'],resumed['step_in_epoch']),
        rng_equal=all(torch.equal(a['torch'],b['torch']) and all(torch.equal(x,y) for x,y in zip(a['cuda'],b['cuda']))
                      for a,b in zip(reference['rng_states'],resumed['rng_states'])))
    (root/'resume_parity.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    if not all(result[k] for k in ('model_equal','optimizer_moments_equal','scheduler_equal','sampler_progress_equal','rng_equal')):
        raise AssertionError('Exact short resume parity failed; inspect report')


if __name__=='__main__':main()
