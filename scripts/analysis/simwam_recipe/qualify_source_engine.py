"""Measure the source JSON's actual gradient-clipping value in a real engine."""
import argparse
import json
import os
import math
from pathlib import Path
import socket
import torch
import torch.distributed as dist
from accelerate import DeepSpeedPlugin
import deepspeed
from starVLA.rl.simwam_recipe.contracts import ROOT,require_dispatch,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source',default=ROOT+'/upstream/SimWAM');a=p.parse_args()
    require_dispatch();torch.cuda.set_device(0);os.environ['LOCAL_RANK']='0'
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    dist.init_process_group('nccl',init_method=f'tcp://127.0.0.1:{port}',world_size=1,rank=0)
    path=Path(a.source)/'scripts/ds_configs/ds_zero1_config.json'
    plugin=DeepSpeedPlugin(hf_ds_config=json.loads(path.read_text()))
    # These are the actual source Accelerate prepare substitutions. It only
    # resolves existing keys; a passed gradient_clipping=1 does not add a key.
    plugin.deepspeed_config_process(must_match=False,train_batch_size=1,train_micro_batch_size_per_gpu=1,
                                   gradient_accumulation_steps=1,gradient_clipping=1.0)
    model=torch.nn.Linear(4,4).cuda();opt=torch.optim.AdamW(model.parameters(),lr=5e-5,betas=(.9,.95),weight_decay=.01)
    engine,_,_,_=deepspeed.initialize(model=model,optimizer=opt,config=plugin.deepspeed_config)
    value=float(engine.gradient_clipping())
    if value!=0.:raise ValueError('Actual engine differs; resolve recipe before training')
    loss=engine(torch.ones(1,4,device='cuda')).sum()*100
    engine.backward(loss);engine.step()
    out=Path(ROOT)/'qualification/source_engine.json';out.parent.mkdir(parents=True,exist_ok=True)
    def json_safe(v):
        if isinstance(v,dict):return {k:json_safe(x) for k,x in v.items()}
        if isinstance(v,list):return [json_safe(x) for x in v]
        if isinstance(v,float) and not math.isfinite(v):return str(v)
        return v
    result=dict(status='PASS',source_json_sha256=file_sha(path),YAML_max_grad_norm=1.,actual_engine_gradient_clipping=value,
                engine_optimizer_steps=engine.global_steps,resolved_DS_config=json_safe(plugin.deepspeed_config),
                scope='Source DeepSpeed clipping behavior only; formal DDP LoRA equivalence is tested separately')
    atomic_json(out,result);print(json.dumps(result));dist.destroy_process_group()


if __name__=='__main__':main()
