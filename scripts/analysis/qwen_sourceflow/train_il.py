"""Native Qwen/source-action IL. SIGUSR1 saves a completed optimizer boundary."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import fcntl
import gc
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback
import uuid
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from starVLA.rl.qwen_sourceflow.contracts import ROOT,QWEN_PATH,require_dispatch,file_sha,fingerprint,source_files
from starVLA.rl.qwen_sourceflow.data import RawSceneData
from starVLA.rl.qwen_sourceflow.il_optimization import parameter_groups,set_stage,lr_scale
from starVLA.rl.qwen_sourceflow import il_checkpoint
from starVLA.rl.simwam_recipe import checkpoint as common
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.model.framework.qwen_sourceflow import QwenSourceFlowVLA
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.model.modules.qwen_sourceflow.source_initialization import load_action_initialization

STOP=False
def stop(*_):
    global STOP
    STOP=True


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',default='configs/qwen_sourceflow/il.yaml')
    p.add_argument('--initialization',required=True);p.add_argument('--output',required=True)
    p.add_argument('--data',default=str(ROOT/'data/TRAIN_v3'));p.add_argument('--resume')
    p.add_argument('--qualification',action='store_true');p.add_argument('--stop-after',type=int)
    p.add_argument('--qualification-interface-updates',type=int)
    a=p.parse_args();host=require_dispatch();cfg=yaml.safe_load(Path(a.config).read_text())
    world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0))
    if not a.qualification and world!=8:raise ValueError('Formal native IL requires one authorized eight-GPU host')
    if a.qualification_interface_updates is not None:
        if not a.qualification:raise ValueError('No formal stage override')
        cfg['interface_updates']=a.qualification_interface_updates
    if a.qualification:cfg['global_batch']=world*cfg['qualification_accumulation']
    if cfg['global_batch']%world:raise ValueError('Nonintegral gradient accumulation')
    accumulation=cfg['global_batch']//world
    torch.set_num_threads(8);torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if world>1:
        dist.init_process_group('nccl',timeout=timedelta(minutes=30));common.init_control_group()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    lk=(out/f'rank{rank}.lock').open('a');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sig in (signal.SIGUSR1,signal.SIGTERM):signal.signal(sig,stop)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    data=RawSceneData(a.data,cohort='TRAIN');processor=QwenObservationProcessor(QWEN_PATH)
    action=load_action_initialization(a.initialization,device='cpu',verify=rank==0)
    model=QwenSourceFlowVLA(QWEN_PATH,action).to(device)
    model.qwen.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer=torch.optim.AdamW(parameter_groups(model,cfg),betas=tuple(cfg['betas']),eps=cfg['eps'],
        weight_decay=cfg['weight_decay'],foreach=False,fused=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:lr_scale(step,cfg))
    identity=dict(schema='native_qwen_sourceflow_IL_v1',config=cfg,qualification=a.qualification,world=world,
        data_identity=data.sha,initialization_identity=json.loads(Path(a.initialization).with_suffix('.identity.json').read_text()),
        qwen_identity=json.loads((ROOT/'assets/qwen_identity.json').read_text()),source_files=source_files())
    identity['sha256']=fingerprint(identity)
    if (out/'identity.json').exists():
        if not a.resume or json.loads((out/'identity.json').read_text())!=identity:raise ValueError('Existing run requires exact resume')
    elif rank==0:atomic_json(out/'identity.json',identity)
    state=dict(updates=0,cursor=0,phase='READY')
    random.seed(42+rank);np.random.seed(42+rank);torch.manual_seed(42+rank)
    if a.resume:state=il_checkpoint.restore(a.resume,model,optimizer,scheduler,identity)
    stage=None;ddp=None;start_update=state['updates'];started=time.monotonic();invocation=uuid.uuid4().hex
    invdir=out/'invocations'/invocation;invdir.mkdir(parents=True,exist_ok=True)
    record=dict(invocation=invocation,host=host,rank=rank,pid=os.getpid(),start_update=start_update,start_unix=time.time(),resume=a.resume,status='RUNNING')
    atomic_json(invdir/f'rank{rank}.json',record);last_checkpoint=a.resume
    stop_at=min(a.stop_after or cfg['max_updates'],cfg['max_updates'])
    try:
        with (invdir/f'metrics_rank{rank}.jsonl').open('x',buffering=1) as journal:
            while state['updates']<stop_at:
                next_stage='full' if state['updates']>=cfg['interface_updates'] else 'interface'
                if next_stage!=stage:
                    if ddp is not None:del ddp;gc.collect();torch.cuda.empty_cache()
                    stage=set_stage(model,state['updates'],cfg)
                    ddp=DDP(model,device_ids=[local],find_unused_parameters=True,static_graph=False,
                        bucket_cap_mb=25,gradient_as_bucket_view=True) if world>1 else model
                begin=time.monotonic();optimizer.zero_grad(set_to_none=True);loss_sum=0.;tokens=[];data_seconds=0.
                for micro in range(accumulation):
                    tick=time.monotonic();index=data.index(state['cursor']+micro*world+rank);token=data.tokens[index];tokens.append(token)
                    observation=data.observation(token);batch=processor([observation])
                    state_input=torch.tensor([observation['proprio']],device=device,dtype=torch.float32)
                    target=ActionCodec.norm_odo(data.target(token)[None].to(device))
                    data_seconds+=time.monotonic()-tick
                    sync=ddp.no_sync() if world>1 and micro<accumulation-1 else nullcontext()
                    with sync:
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            loss,terms=ddp(batch,state_input,target)
                        if not torch.isfinite(loss):raise FloatingPointError('Invalid IL loss; no checkpoint publication')
                        (loss/accumulation).backward()
                    loss_sum+=float(loss.detach())/accumulation
                gradients=[p.grad for p in model.parameters() if p.requires_grad]
                if any(v is None for v in gradients):raise ValueError('Missing IL parameter gradient')
                grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],cfg['gradient_clip'],error_if_nonfinite=True,foreach=False)
                del gradients
                lrs={g['name']:g['lr'] for g in optimizer.param_groups}
                optimizer.step();scheduler.step();state['updates']+=1;state['cursor']+=cfg['global_batch']
                torch.cuda.synchronize();elapsed=time.monotonic()-begin
                row=dict(updates=state['updates'],cursor=state['cursor'],stage=stage,rank=rank,tokens=tokens,
                    loss=loss_sum,grad_norm=float(grad),learning_rates=lrs,seconds=elapsed,seconds_data=data_seconds,
                    invocation=invocation,peak_GPU_bytes=torch.cuda.max_memory_allocated())
                journal.write(json.dumps(row,allow_nan=False)+'\n')
                if rank==0 and (state['updates']<=5 or state['updates']%20==0):print(json.dumps(row),flush=True)
                flag=torch.tensor(int(STOP),device=device)
                if world>1:dist.all_reduce(flag,op=dist.ReduceOp.MAX)
                end=bool(flag) or state['updates']==stop_at
                if end or state['updates']%cfg['save_every']==0 or state['updates'] in cfg['milestones']:
                    tick=time.monotonic();last_checkpoint=il_checkpoint.save(out,model,optimizer,scheduler,state,identity)
                    if rank==0:atomic_json(out/'progress.json',dict(state,checkpoint=last_checkpoint,stage=stage,
                        status='COMPLETE' if state['updates']==cfg['max_updates'] else 'STOPPED' if end else 'RUNNING',
                        invocation_start_update=start_update,invocation_wall_seconds=time.monotonic()-started,save_seconds=time.monotonic()-tick))
                if end:break
        record.update(status='COMPLETE' if state['updates']==cfg['max_updates'] else 'STOPPED_AT_SAVED_BOUNDARY',
            end_update=state['updates'],checkpoint=last_checkpoint,wall_seconds=time.monotonic()-started,
            updates_this_invocation=state['updates']-start_update,peak_GPU_bytes=torch.cuda.max_memory_allocated())
    except BaseException:
        record.update(status='FAILED',error=traceback.format_exc(),last_legal_checkpoint=last_checkpoint,wall_seconds=time.monotonic()-started)
        raise
    finally:
        atomic_json(invdir/f'rank{rank}.json',record);atomic_json(out/f'result_rank{rank}.json',record)
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
