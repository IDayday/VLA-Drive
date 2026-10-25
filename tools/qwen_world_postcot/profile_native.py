"""Bounded actual native baseline updates; isolated outputs, original100k LR.

Uses the full trained native model and Adam state, original micro1/accum4,
global32, and the same fixed legal TRAIN sequence as the new full-model profile.
No state is published to the parent run and no Navtest is executed.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from starVLA.rl.qwen_world_postcot.contracts import require_dispatch,boundary_stop_due,file_sha
from starVLA.rl.qwen_world_postcot.data import WorldSceneData
from starVLA.rl.qwen_sourceflow.contracts import configure_native_numerics
from starVLA.rl.qwen_sourceflow.il_optimization import parameter_groups,lr_scale
from starVLA.model.modules.qwen_sourceflow.model import QwenSourceFlowVLA
from starVLA.model.modules.qwen_sourceflow.source_initialization import load_action_initialization
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--data',required=True);p.add_argument('--updates',type=int,default=32);p.add_argument('--warmup',type=int,default=4)
    p.add_argument('--trace-steps',type=int,default=0);a=p.parse_args();require_dispatch(admit_long_job=True)
    if not 8<=a.updates<=64 or not 2<=a.warmup<a.updates-3 or not 0<=a.trace_steps<=3:raise ValueError('Explicit bounded meaningful timing budget required')
    configure_native_numerics();torch.set_num_threads(4);local=int(os.environ['LOCAL_RANK']);rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=8:raise ValueError('Baseline full global32 measurement requires8GPUs')
    torch.cuda.set_device(local);device=torch.device('cuda',local);dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    parent=Path(a.checkpoint);identity=json.loads((parent/'identity.json').read_text());cfg=identity['config'];seal=json.loads((parent/'COMPLETE').read_text());files=json.loads((parent/'files.json').read_text())
    if file_sha(parent/'files.json')!=seal['files_sha256'] or file_sha(parent/'model.pt')!=files['model.pt']:raise ValueError('Baseline checkpoint changed')
    data=WorldSceneData(a.data);processor=QwenObservationProcessor('/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct')
    action=load_action_initialization('/mnt/project/qwen-sourceflow-native-20261018/assets/action_generic.pt',device='cpu',verify=rank==0)
    model=QwenSourceFlowVLA('/mnt/project/DriveDreamer-Policy/models/Qwen3-VL-2B-Instruct',action).to(device)
    model.load_state_dict(torch.load(parent/'model.pt',map_location='cpu',weights_only=False),strict=True)
    model.qwen.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
    optimizer=torch.optim.AdamW(parameter_groups(model,cfg),betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'],foreach=False,fused=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:lr_scale(s,cfg));state=torch.load(parent/'optimizer.pt',map_location='cpu',weights_only=False)
    optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler']);del state
    ddp=DDP(model,device_ids=[local],find_unused_parameters=True,gradient_as_bucket_view=True,bucket_cap_mb=25,static_graph=False)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);profiler=None;start=time.monotonic()
    with (out/f'metrics_rank{rank}.jsonl').open('x',buffering=1) as journal:
        for step in range(a.updates):
            if boundary_stop_due():break
            torch.cuda.synchronize();tick=time.monotonic();optimizer.zero_grad(set_to_none=True);tokens=[];loss_sum=0.;data_seconds=0.
            if a.trace_steps and step==a.warmup:
                profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True);profiler.__enter__()
            for m in range(4):
                ds=time.monotonic();token=data.tokens[data.index(step*32+m*8+rank)];tokens.append(token);obs=data.observation(token);batch=processor([obs]);ego=torch.tensor([obs['proprio']],device=device)
                target=ActionCodec.norm_odo(data.target(token)[None].to(device));data_seconds+=time.monotonic()-ds
                with ddp.no_sync() if m<3 else nullcontext():
                    with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=ddp(batch,ego,target)
                    (loss/4).backward();loss_sum+=float(loss)/4
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True,foreach=False);optimizer.step();scheduler.step();torch.cuda.synchronize()
            elapsed=time.monotonic()-tick;slow=torch.tensor(elapsed,device=device);dist.all_reduce(slow,op=dist.ReduceOp.MAX)
            row=dict(update=step+1,parent_updates=seal['updates'],seconds=elapsed,slowest_rank_seconds=float(slow),data_seconds=data_seconds,
                loss=loss_sum,tokens=tokens,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),warmup=step<a.warmup,
                profiler_active=profiler is not None)
            journal.write(json.dumps(row)+'\n')
            if rank==0:print(json.dumps(row),flush=True)
            if profiler is not None and step+1==a.warmup+a.trace_steps:
                profiler.__exit__(None,None,None);profiler.export_chrome_trace(str(out/f'trace_rank{rank}.json'));profiler=None
    if profiler is not None:profiler.__exit__(None,None,None)
    (out/f'result_rank{rank}.json').write_text(json.dumps(dict(status='COMPLETE',bounded_profile=True,updates=step+1,
        parent_model_sha256=files['model.pt'],parent_updates=seal['updates'],global_batch=32,micro_batch=1,accumulation=4,
        wall_seconds=time.monotonic()-start,source_sha256=file_sha(__file__),data_sha256=data.sha)))
    dist.destroy_process_group()
if __name__=='__main__':main()
