"""Actual full-action-head SFT with native optimizer/RNG/cursor recovery.

No scorer, RL candidates, teacher predictions, or privileged actor features.
Qualification runs cannot become formal initializations.
"""
import argparse
from datetime import timedelta
import fcntl
import json
import math
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
from torch.nn.parallel import DistributedDataParallel
from starVLA.rl.fm_transfer.contracts import config,require_dispatch,source_files,ROOT,fingerprint,CONDITION_SHA,SFT_SHA256
from starVLA.rl.fm_transfer.data import TrainingData
from starVLA.rl.fm_transfer.model import make_policy
from starVLA.rl.fm_transfer import checkpoint
from starVLA.rl.simwam_recipe import checkpoint as common
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.flow_grpo.reproducibility import configure_numerics

STOP=False
def request_stop(*_):
    global STOP
    STOP=True


def scale(step,cfg):
    warm=max(1,math.ceil(cfg['max_updates']*cfg['warmup_fraction']))
    if step<warm:return (step+1)/warm
    x=min(1.,(step-warm)/max(1,cfg['max_updates']-warm))
    return .1+.9*.5*(1+math.cos(math.pi*x))


def sample_parameters(model):
    return {n:p.detach().flatten()[:64].cpu().clone() for n,p in model.named_parameters()}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--variant',choices=['A_DDP_TIME','B_SOURCE_ACTION_DIT'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--data',default=str(ROOT/'supervised_data'))
    p.add_argument('--resume');p.add_argument('--qualification',action='store_true')
    p.add_argument('--stop-after',type=int);p.add_argument('--save-every',type=int)
    a=p.parse_args();host=require_dispatch();cfg=config(a.variant)
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK',0)));device=torch.device('cuda',torch.cuda.current_device())
    world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0))
    if not a.qualification and world!=8:raise ValueError('Formal SFT is single-host eight-GPU')
    if a.qualification:cfg.update(world=world,global_batch=world*cfg['per_rank_batch'])
    configure_numerics()
    if world>1:
        dist.init_process_group('nccl',timeout=timedelta(minutes=15))
        common.CONTROL_GROUP=dist.new_group(backend='gloo',timeout=timedelta(minutes=15))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    lock=(out/f'rank{rank}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sig in (signal.SIGUSR1,signal.SIGTERM):signal.signal(sig,request_stop)
    data=TrainingData(a.data)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model=make_policy(cfg['architecture'],device=device,verify_sft=(rank==0)).requires_grad_(True).train()
    manifest={n:dict(shape=list(v.shape),dtype=str(v.dtype),numel=v.numel()) for n,v in model.named_parameters()}
    numel=sum(v['numel'] for v in manifest.values())
    if cfg['architecture']=='ddp' and (numel!=819503620 or len(manifest)!=359):
        raise ValueError('Original complete action-model parameter closure changed')
    identity=dict(schema='fm_transfer_SFT_identity_v1',config=cfg,qualification=a.qualification,
        parameters=manifest,trainable_numel=numel,data_identity=data.sha,condition_identity=CONDITION_SHA,
        original_SFT=SFT_SHA256 if cfg['architecture']=='ddp' else None,source_files=source_files())
    identity['sha256']=fingerprint(identity)
    error=None
    if rank==0:
        try:
            if (out/'identity.json').exists():
                if not a.resume or json.loads((out/'identity.json').read_text())!=identity:raise ValueError('Existing run/identity requires exact resume')
            elif a.resume:
                # A new qualification output may resume the same scientific
                # identity; formal runs stay in their original output root.
                if not a.qualification:raise ValueError('Formal resume must preserve run directory')
                atomic_json(out/'identity.json',identity)
            else:atomic_json(out/'identity.json',identity)
        except Exception as exc:error=repr(exc)
    common.check_errors(error)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],betas=tuple(cfg['betas']),
        eps=cfg['eps'],weight_decay=cfg['weight_decay'],foreach=False,fused=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:scale(step,cfg))
    ddp=DistributedDataParallel(model,device_ids=[device.index],**cfg['ddp']) if world>1 else model
    # Independent data/dropout/FM RNGs across ranks, identical initialization.
    random.seed(42+rank);np.random.seed(42+rank);torch.manual_seed(42+rank);torch.cuda.manual_seed_all(42+rank)
    state=dict(updates=0,cursor=0,phase='READY')
    if a.resume:state=checkpoint.restore(a.resume,model,optimizer,scheduler,identity)
    start_update=state['updates'];invocation=uuid.uuid4().hex
    invdir=out/'invocations'/invocation;invdir.mkdir(parents=True,exist_ok=True)
    record=dict(status='RUNNING',rank=rank,pid=os.getpid(),host=host,start_unix=time.time(),
        start_update=start_update,resume=a.resume,identity=identity['sha256'])
    atomic_json(invdir/f'rank{rank}.json',record)
    before=sample_parameters(model) if a.qualification else None
    gradients={};stop_at=min(a.stop_after or cfg['max_updates'],cfg['max_updates'])
    if stop_at<=start_update:raise ValueError('No updates requested after resume')
    save_every=a.save_every or cfg['save_every']
    if a.save_every and not a.qualification:raise ValueError('Formal save cadence is registered')
    start=time.monotonic();last_checkpoint=None
    try:
        # Preserve the actual initialization for the pre-SFT planning readout.
        # Saving does not draw training RNGs; resume never overwrites this file.
        if not a.resume:
            last_checkpoint=checkpoint.save(out,model,optimizer,scheduler,state,identity)
        with (invdir/f'metrics_rank{rank}.jsonl').open('x',buffering=1) as journal:
            while state['updates']<stop_at:
                begin=time.monotonic();raw,target,tokens=data.batch(state['cursor'],rank,world,cfg['per_rank_batch'],device)
                loaded=time.monotonic();optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss,terms=ddp(raw,target,repeat=cfg['FM_repeats'])
                if not torch.isfinite(loss):raise ValueError('Nonfinite SFT loss; no optimizer update')
                loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True,foreach=False)
                if a.qualification and state['updates']<start_update+2:
                    names=[];values=[]
                    for n,v in model.named_parameters():
                        if v.grad is None:raise ValueError('Missing gradient: '+n)
                        names.append(n);values.append(v.grad.float().norm())
                    gradients[str(state['updates']+1)]=dict(zip(names,torch.stack(values).detach().cpu().tolist()))
                lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
                state['updates']+=1;state['cursor']+=cfg['global_batch']
                torch.cuda.synchronize();now=time.monotonic()
                row=dict(updates=state['updates'],cursor=state['cursor'],rank=rank,invocation=invocation,
                    scene_tokens=tokens,loss=float(loss.detach()),gradient_norm_before_clip=float(grad),
                    learning_rate=lr,seconds_data=loaded-begin,seconds_update=now-loaded,
                    wall_seconds_this_invocation=now-start,**{k:float(v) for k,v in terms.items()})
                journal.write(json.dumps(row,allow_nan=False)+'\n')
                if rank==0 and (state['updates']<=5 or state['updates']%20==0):print(json.dumps(row),flush=True)
                flag=torch.tensor(int(STOP),device=device)
                if world>1:dist.all_reduce(flag,op=dist.ReduceOp.MAX)
                stop=bool(flag.item()) or state['updates']==stop_at
                if stop or state['updates']%save_every==0 or state['updates'] in cfg['milestones']:
                    save_start=time.monotonic();last_checkpoint=checkpoint.save(out,model,optimizer,scheduler,state,identity)
                    if rank==0:atomic_json(out/'progress.json',dict(state,status='COMPLETE' if state['updates']==cfg['max_updates'] else 'STOPPED' if stop else 'RUNNING',
                        checkpoint=last_checkpoint,save_seconds=time.monotonic()-save_start,
                        invocation=invocation,invocation_updates=state['updates']-start_update,invocation_wall=time.monotonic()-start))
                if stop:break
        record.update(status='COMPLETE' if state['updates']==cfg['max_updates'] else 'STOPPED_AT_SAVED_BOUNDARY',
            end_update=state['updates'],cursor=state['cursor'],checkpoint=last_checkpoint,
            updates_this_invocation=state['updates']-start_update,wall_seconds=time.monotonic()-start,
            peak_GPU_bytes=torch.cuda.max_memory_allocated(),qualification=a.qualification)
        if a.qualification:
            after=sample_parameters(model)
            record.update(actual_parameter_sample_changes={n:bool(torch.count_nonzero(after[n]-before[n])) for n in before},
                gradient_norms=gradients)
        atomic_json(out/f'result_rank{rank}.json',record)
        atomic_json(invdir/f'rank{rank}.json',record)
        if rank==0:print(json.dumps({k:v for k,v in record.items() if k not in ('gradient_norms','actual_parameter_sample_changes')}),flush=True)
    except BaseException:
        record.update(status='FAILED',error=traceback.format_exc(),last_legal_checkpoint=last_checkpoint,
            wall_seconds=time.monotonic()-start)
        atomic_json(invdir/f'rank{rank}.json',record)
        raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
