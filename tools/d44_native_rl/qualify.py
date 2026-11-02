"""Execute a bounded real-model qualification via the same training entry."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
import yaml
from starVLA.rl.d44_native_rl.config import defaults


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--arm',default='D0');p.add_argument('--output',required=True)
    p.add_argument('--gpus',type=int,default=8);p.add_argument('--updates',type=int,default=2)
    p.add_argument('--group',type=int,default=8);p.add_argument('--scenes-per-rank',type=int,default=1)
    p.add_argument('--resume');a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    cfg=defaults(a.arm);cfg['group']=a.group;cfg['save_every']=1
    config=root/'qualification_recipe.yaml'
    if config.exists()and yaml.safe_load(config.read_text())!=cfg:raise ValueError('Qualification recipe changed')
    config.write_text(yaml.safe_dump(cfg,sort_keys=False))
    cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={a.gpus}',
        '-m','tools.d44_native_rl.train','--config',str(config),'--output',str(root/'run'),
        '--qualification','--stop-after-update',str(a.updates),'--scenes-per-rank',str(a.scenes_per_rank)]
    if a.resume:cmd+=['--resume',a.resume]
    raise SystemExit(subprocess.call(cmd,env=os.environ.copy()))


if __name__=='__main__':main()
