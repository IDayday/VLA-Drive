"""Bounded real eight-GPU continuation vs inner-boundary recovery acceptance."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from starVLA.rl.simwam_recipe.contracts import ROOT,require_dispatch,load_recipe


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--name',required=True)
    p.add_argument('--root',default=ROOT)
    p.add_argument('--config',default='configs/flow_rl/simwam_source_ddp.yaml')
    p.add_argument('--manifest',default=ROOT+'/qualification/real_TRAIN_tokens.json')
    p.add_argument('--interrupted-after',type=int,choices=[1,2,3],default=1)
    p.add_argument('--reuse-frozen-reference',action='store_true',
                   help='Exercise the same explicit execution option in both continuation paths')
    a=p.parse_args();require_dispatch();load_recipe(a.config)
    root=Path(a.root)/'qualification'/a.name;root.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',NO_ALBUMENTATIONS_UPDATE='1')
    common=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','--module','scripts.analysis.simwam_recipe.train',
            '--config',a.config,'--manifest',a.manifest,'--qualification','--save-every','100']
    if a.reuse_frozen_reference:common.append('--reuse-frozen-reference')
    boundary=a.interrupted_after
    tasks=[('continuous',8,None),('resumed',boundary,None),
           ('resumed',8,root/f'resumed/checkpoints/update{boundary:06}_inner{boundary}')]
    for index,(name,end,resume) in enumerate(tasks):
        cmd=common+['--output',str(root/name),'--stop-after-update',str(end)]
        if resume:cmd+=['--resume',str(resume)]
        with (root/f'{index}_{name}.log').open('xb') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,'-m','scripts.analysis.simwam_recipe.qualify_recovery','--continuous',str(root/'continuous'),
                    '--resumed',str(root/'resumed'),'--interrupted-after',str(boundary),
                    '--output',str(root/'result.json')],env=env,check=True)


if __name__=='__main__':main()
