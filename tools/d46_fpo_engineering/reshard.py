"""Typed native 16/32-rank execution transition; no optimizer reset."""
import argparse
import json
from starVLA.rl.d46_fpo_engineering.reshard import prepare

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for field in('checkpoint','source','output'):p.add_argument('--'+field,required=True)
    p.add_argument('--world',type=int,choices=(16,32),required=True)
    a=p.parse_args()
    print(json.dumps(prepare(a.checkpoint,a.source,a.output,target_world=a.world)))
