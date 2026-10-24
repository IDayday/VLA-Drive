"""Finite full-model DDP IL timings on copies; never saves training weights."""
import argparse,json,os,random,time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from starVLA.rl.qwen_sourceflow.contracts import ROOT,QWEN_PATH,require_dispatch,configure_native_numerics,file_sha
from starVLA.rl.qwen_sourceflow.loading import load_il
from starVLA.rl.qwen_sourceflow.data import RawSceneData
from starVLA.rl.qwen_sourceflow.il_optimization import set_stage,parameter_groups
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.rl.flow_grpo.transactions import atomic_json


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--warmup',type=int,default=2);p.add_argument('--updates',type=int,default=5)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--trace',action='store_true');a=p.parse_args()
    if a.updates not in range(1,11) or a.warmup not in range(1,4) or a.threads not in [1,2,4,8]:raise ValueError('Finite profile only')
    host=require_dispatch();world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0))
    if world not in [1,2,4,8,16,32] or 32%world:raise ValueError('Keep global32')
    configure_native_numerics();torch.set_num_threads(a.threads);torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(seconds=300))
    started=time.perf_counter();random.seed(42);np.random.seed(42);torch.manual_seed(42)
    cfg=yaml.safe_load(Path('configs/qwen_sourceflow/il.yaml').read_text());model,parent=load_il(a.checkpoint,qualification=True,device=device,verify=rank==0)
    model.qwen.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});set_stage(model,cfg['interface_updates'],cfg)
    ddp=DDP(model,device_ids=[local],find_unused_parameters=True,static_graph=False,bucket_cap_mb=25,gradient_as_bucket_view=True)
    optimizer=torch.optim.AdamW(parameter_groups(model,cfg),betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'],foreach=False,fused=False)
    data=RawSceneData(ROOT/'data/TRAIN_v3',cohort='TRAIN');processor=QwenObservationProcessor(QWEN_PATH)
    random.seed(42+rank);np.random.seed(42+rank);torch.manual_seed(42+rank)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);path=out/f'rank{rank}.json'
    if path.exists():raise FileExistsError(path)
    accumulation=32//world;rows=[];cursor=0
    dist.barrier();setup=time.perf_counter()-started
    def step(index,traced=False):
        nonlocal cursor
        dist.barrier();torch.cuda.synchronize();begin=time.perf_counter();phase={};losses=[];tokens=[]
        tick=time.perf_counter();optimizer.zero_grad(set_to_none=True);phase['zero_grad']=time.perf_counter()-tick
        for micro in range(accumulation):
            tick=time.perf_counter();token=data.tokens[data.index(cursor+micro*world+rank)];obs=data.observation(token);batch=processor([obs])
            state=torch.tensor([obs['proprio']],device=device,dtype=torch.float32);target=ActionCodec.norm_odo(data.target(token)[None].to(device))
            torch.cuda.synchronize();phase['data']=phase.get('data',0)+time.perf_counter()-tick;tokens.append(token)
            sync=ddp.no_sync() if micro<accumulation-1 else nullcontext()
            with sync:
                tick=time.perf_counter()
                with torch.autocast('cuda',dtype=torch.bfloat16):loss,_=ddp(batch,state,target)
                torch.cuda.synchronize();phase['forward']=phase.get('forward',0)+time.perf_counter()-tick
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite copied IL loss')
                tick=time.perf_counter();(loss/accumulation).backward();torch.cuda.synchronize()
                phase['backward']=phase.get('backward',0)+time.perf_counter()-tick;losses.append(float(loss.detach()))
            del batch,state,target,loss
        tick=time.perf_counter();grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['gradient_clip'],error_if_nonfinite=True,foreach=False)
        torch.cuda.synchronize();phase['grad_clip']=time.perf_counter()-tick
        tick=time.perf_counter();optimizer.step();torch.cuda.synchronize();phase['AdamW']=time.perf_counter()-tick;cursor+=32
        return dict(index=index,warmup=index<a.warmup,traced=traced,wall_seconds=time.perf_counter()-begin,phases_seconds=phase,
            loss=float(np.mean(losses)),grad_norm=float(grad),tokens=tokens,peak_GPU_bytes=torch.cuda.max_memory_allocated())
    try:
        for i in range(a.warmup+a.updates):
            rows.append(step(i));atomic_json(path,dict(status='RUNNING',host=host,rank=rank,world=world,rows=rows))
            if rank==0:print(json.dumps(rows[-1]),flush=True)
        if a.trace:
            if rank==0:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False) as prof:
                    traced=step(a.warmup+a.updates,True)
                prof.export_chrome_trace(str(out/'trace_rank0.json'))
                operators=[dict(name=e.key,count=e.count,self_CPU_us=e.self_cpu_time_total,self_device_us=e.self_device_time_total) for e in prof.key_averages()]
                atomic_json(out/'operators_rank0.json',dict(traced=traced,operators=sorted(operators,key=lambda e:e['self_device_us'],reverse=True)))
            else:step(a.warmup+a.updates,True)
        atomic_json(path,dict(status='PASS',qualification_only=True,host=host,rank=rank,world=world,global_batch=32,accumulation=accumulation,
            CPU_threads=a.threads,rows=rows,setup_seconds=setup,total_wall_seconds=time.perf_counter()-started,
            checkpoint=str(Path(a.checkpoint).resolve()),parent_model_sha256=parent['files']['model.pt'],script_sha256=file_sha(__file__),
            optimizer='Fresh throwaway FP32 Adam; no weights published, no formal state changed',timing='Synchronized host regions; explicit sync overhead is included, trace update excluded from throughput'))
        dist.barrier()
        if rank==0:atomic_json(out/'COMPLETE.json',dict(status='PASS',world=world,qualification_only=True))
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
