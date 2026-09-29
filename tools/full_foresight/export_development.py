"""Use allocated host-local GPUs for independent current-only FP32 export shards."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('training-run','checkpoint-tag','current-root','output','campaign-root','run-id'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--gpus',type=int,required=True);p.add_argument('--sampling-seed',type=int,default=42)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--campaign-gpu-hours',type=float,required=True)
    a=p.parse_args();cards=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(cards)!=a.gpus:raise ValueError('Use only explicitly allocated GPUs')
    logs=Path(a.campaign_root)/'logs';children=[]
    for rank,card in enumerate(cards):
        cmd=[sys.executable,'-m','tools.foresight.export_predictions']
        for k in ('training_run','checkpoint_tag','current_root','output','campaign_root','sampling_seed','limit','campaign_gpu_hours'):
            cmd.extend(['--'+k.replace('_','-'),str(getattr(a,k))])
        cmd+=['--run-id',a.run_id+f'_rank{rank}','--rank',str(rank),'--world-size',str(a.gpus),'--max-seconds','21600']
        with (logs/(a.run_id+f'_rank{rank}.log')).open('a') as stream:
            children.append(subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=card),stdout=stream,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in children]
    if any(codes):raise RuntimeError('Failed export shards retained: '+str(codes))


if __name__=='__main__':main()
