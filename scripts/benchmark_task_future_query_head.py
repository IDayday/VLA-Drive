#!/usr/bin/env python3
"""GPU micro-cost only: fixed production shapes, not a real-data accuracy smoke."""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from navsim.agents.EpisodeDrive.layers.world_model.physical_query_decoder import PhysicalQueryDecoder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.manual_seed(0)
    head=PhysicalQueryDecoder().cuda().eval()
    b,k=4,8
    r=torch.randn(b,16,256,device='cuda')
    candidate=torch.randn(b,k,8,3,device='cuda')
    status=torch.randn(b,8,device='cuda')
    pose=torch.randn(b,3,device='cuda')
    future=torch.randn(b,3,16,256,device='cuda')
    def current():return head(r,candidate,status)
    def hindsight():
        return [head.forward_hindsight(r,future[:,i],candidate,status,pose,h)
                for i,h in enumerate((.5,1.5,3.))]
    result=dict(scope='Synthetic production-shape GPU forward microbenchmark, not a real-model/data smoke or full-step throughput',
                batch_size=b,candidates=k,parameters=sum(p.numel() for p in head.parameters()),
                device=torch.cuda.get_device_name(),dtype='FP32 storage / BF16 autocast',timed_iterations=100)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for name,fn in [('current_head',current),('three_hindsight_heads',hindsight)]:
            for _ in range(20):fn()
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(100):fn()
            torch.cuda.synchronize()
            result[name+'_seconds']=(time.perf_counter()-start)/100
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
