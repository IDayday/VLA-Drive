"""Probe mixed parameter storage with the existing ZeRO-2 configuration."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from starVLA.rl.constrained_flow_noise.contracts import require_dispatch
from starVLA.rl.flow_grpo.config import resolve_config
from starVLA.rl.flow_grpo.trainer import make_accelerator
from starVLA.rl.flow_grpo.zero2_precision import install_fp32_partitions
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__); p.add_argument('--output',required=True); a=p.parse_args()
    host=require_dispatch(); torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)))
    cfg,_=resolve_config('configs/flow_grpo/frozen_action_head_batched_single8_epoch1.yaml',overrides=['checkpoint_contract.source_config=/mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-source/checkpoint_code/action-only-checkpoints-v1/configs/frozen-visual-step100000-config.yaml'])
    accelerator=make_accelerator(cfg)
    class Probe(nn.Module):
        def __init__(self):
            super().__init__(); self.actor=nn.Linear(4,8).bfloat16(); self.noise=nn.Linear(8,4).float()
        def forward(self,x):
            y=self.actor(x)
            return torch.func.functional_call(self.noise,{n:v.float() for n,v in self.noise.named_parameters()},(y.float(),)).square().mean()
    model=Probe().cuda()
    before={n:str(p.dtype) for n,p in model.named_parameters()}
    opt=torch.optim.AdamW([{'params':model.actor.parameters()},{'params':model.noise.parameters(),'weight_decay':0}],lr=1e-6)
    model,opt=accelerator.prepare(model,opt)
    precision=install_fp32_partitions(model)
    after={n:str(p.dtype) for n,p in model.module.named_parameters()}
    for _ in range(2):
        with accelerator.accumulate(model):
            accelerator.backward(model(torch.ones(1,4,device='cuda',dtype=torch.bfloat16)))
    result=dict(host=host,world=accelerator.num_processes,before=before,after=after,precision=precision,
                mixed_storage_preserved=all(after[n]==dtype for n,dtype in before.items()),
                chosen='uniform_BF16_storage_FP32_noise_functional_arithmetic',
                optimizer_master_dtypes=[str(p.dtype) for p in model.optimizer.single_partition_of_fp32_groups])
    if accelerator.is_main_process:
        path=Path(a.output)
        if path.exists(): raise FileExistsError(path)
        atomic_json(path,result);print(json.dumps(result),flush=True)


if __name__=='__main__': main()
