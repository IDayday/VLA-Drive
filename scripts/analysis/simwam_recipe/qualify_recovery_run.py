"""Bounded real eight-GPU continuation vs inner-boundary recovery acceptance."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from starVLA.rl.simwam_recipe.contracts import ROOT,require_dispatch


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--name',required=True);a=p.parse_args();require_dispatch()
    root=Path(ROOT)/'qualification'/a.name;root.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',NO_ALBUMENTATIONS_UPDATE='1')
    common=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','--module','scripts.analysis.simwam_recipe.train',
            '--config','configs/flow_rl/simwam_source_ddp.yaml','--manifest',ROOT+'/qualification/real_TRAIN_tokens.json','--qualification','--save-every','100']
    tasks=[('continuous',8,None),('resumed',1,None),('resumed',8,root/'resumed/checkpoints/update000001_inner1')]
    for index,(name,end,resume) in enumerate(tasks):
        cmd=common+['--output',str(root/name),'--stop-after-update',str(end)]
        if resume:cmd+=['--resume',str(resume)]
        with (root/f'{index}_{name}.log').open('xb') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,'-m','scripts.analysis.simwam_recipe.qualify_recovery','--continuous',str(root/'continuous'),
                    '--resumed',str(root/'resumed'),'--output',str(root/'result.json')],env=env,check=True)


if __name__=='__main__':main()
