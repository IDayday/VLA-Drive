"""New temporal/current-DINO run, retaining native parent Adam and cosine phase."""
import argparse,json,os,random,signal,time,traceback,uuid,gc
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from starVLA.model.modules.qwen_sourceflow.temporal_dino import TemporalDINOFlowVLA,TemporalQwenProcessor
from starVLA.model.modules.qwen_sourceflow.vendor.action_dit import ActionDiT
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.qwen_sourceflow.contracts import QWEN_PATH,ACTION_CONFIG,file_sha,fingerprint,runtime_identity,configure_native_numerics
from starVLA.rl.qwen_sourceflow.temporal_contracts import ROOT,PARENT,require_temporal_dispatch,temporal_sources
from starVLA.rl.qwen_sourceflow.loading import inspect_il_checkpoint
from starVLA.rl.qwen_sourceflow.temporal_data import TemporalSceneData
from starVLA.rl.qwen_sourceflow.current_dino_targets import CurrentDINOTargets
from starVLA.rl.qwen_sourceflow.temporal_optimizer import build_optimizer,inherit_parent_adam,local_parameter_names,validate_local_adam
from starVLA.rl.qwen_sourceflow.il_optimization import lr_scale
from starVLA.rl.qwen_sourceflow.il_checkpoint import publish_identity
from starVLA.rl.qwen_sourceflow import temporal_checkpoint
from starVLA.rl.simwam_recipe import checkpoint as common
from starVLA.rl.flow_grpo.transactions import atomic_json

STOP=False
def stop(*_):
    global STOP
    STOP=True


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',default='configs/qwen_sourceflow/temporal_dino_il.yaml')
    p.add_argument('--output',required=True);p.add_argument('--parent',default=str(PARENT));p.add_argument('--resume')
    p.add_argument('--teacher',default='/var/tmp/qwen-sourceflow-temporal-dino-20261024/current_C1')
    p.add_argument('--teacher-manifest',default=str(ROOT/'current_teacher_manifest.json'))
    p.add_argument('--qualification',action='store_true');p.add_argument('--stop-after',type=int)
    p.add_argument('--qualification-batch',type=int,default=2);p.add_argument('--qualification-save-every',type=int,default=2)
    a=p.parse_args();host=require_temporal_dispatch();configure_native_numerics()
    world=int(os.environ['WORLD_SIZE']);rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    cfg=yaml.safe_load(Path(a.config).read_text())
    if not a.qualification and (world!=8 or cfg['max_updates']!=40000 or cfg['global_batch']!=32):raise ValueError('Fixed formal eight-GPU/40k/global32 contract')
    if a.qualification:cfg=dict(cfg,global_batch=a.qualification_batch,save_every=a.qualification_save_every)
    if cfg['global_batch']%world:raise ValueError('Nonintegral accumulation')
    accumulation=cfg['global_batch']//world
    torch.set_num_threads(8);torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30));common.init_control_group()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lk=(out/f'rank{rank}.lock').open('a');fcntl.flock(lk,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sig in [signal.SIGUSR1,signal.SIGTERM]:signal.signal(sig,stop)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    info=inspect_il_checkpoint(a.parent);parent_cfg=info['identity']['config']
    if info['seal']['updates']!=60000 or info['seal']['world']!=8 or info['identity']['qualification']:raise ValueError('Exact original native60k parent required')
    data=TemporalSceneData(ROOT/'data/TRAIN_v3',cohort='TRAIN');teacher=CurrentDINOTargets(a.teacher)
    manifest=json.loads(Path(a.teacher_manifest).read_text());body=dict(manifest);sha=body.pop('sha256')
    if fingerprint(body)!=sha or manifest['data_sha256']!=data.sha or manifest['teacher_identity']!=teacher.identity['identity'] or manifest['scenes']!=len(data.tokens):
        raise ValueError('Teacher/current-data preflight identity changed')
    if rank==0:
        for name in ['model.pt','optimizer.pt']:
            if file_sha(Path(a.parent)/name)!=info['files'][name]:raise ValueError('Changed full parent asset')
    dist.barrier()
    model=TemporalDINOFlowVLA(QWEN_PATH,ActionDiT(**ACTION_CONFIG))
    parent=torch.load(Path(a.parent)/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    initialized=model.state_dict();head={k:v for k,v in initialized.items() if k.startswith('current_dino_head.')}
    if set(parent)!=set(initialized)-set(head) or set(head)!={'current_dino_head.projection.weight','current_dino_head.projection.bias'}:
        raise ValueError('Warm-start may add only the explicit current-DINO projection')
    model.load_state_dict(dict(parent,**head),strict=True);del parent,initialized,head;gc.collect()
    model=model.to(device).train();model.qwen.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    optimizer=build_optimizer(model,cfg)
    payload=torch.load(Path(a.parent)/'optimizer.pt',map_location='cpu',weights_only=False,mmap=True)
    name_by_id={id(p):n for n,p in model.named_parameters()};parent_steps={};offset=0
    for group in optimizer.param_groups[:3]:
        for param in group['params']:
            parent_steps[name_by_id[id(param)]]=int(payload['optimizer']['state'][offset]['step']);offset+=1
    inherit_parent_adam(model,optimizer,payload);del payload;gc.collect();torch.cuda.empty_cache()
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda delta:lr_scale(60000+delta,parent_cfg))
    identity=dict(schema='native_temporal_DINO_IL_v1',config=cfg,qualification=a.qualification,world=world,
        parent=dict(path=str(Path(a.parent).resolve()),model_sha256=info['files']['model.pt'],optimizer_sha256=info['files']['optimizer.pt'],update=60000),
        parent_steps=parent_steps,runtime=runtime_identity(),data_sha256=data.sha,teacher_manifest_sha256=file_sha(a.teacher_manifest),
        teacher_identity=teacher.identity,source_files=temporal_sources(),optimizer='FP32 AdamW exact arithmetic; torch2.5 ZeRO state partition',
        schedule='original native100k cosine evaluated at60000+new_updates',seed=42,teacher_loss='previous C1 global element MSE, weight1')
    identity['sha256']=fingerprint(identity);publish_identity(out,identity,resume=bool(a.resume))
    own=torch.load(Path(a.parent)/f'rank{rank}.pt',map_location='cpu',weights_only=False)
    state=dict(updates=0,cursor=own['state']['cursor'],phase='READY')
    rng=own['rng'];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'])
    if a.resume:state=temporal_checkpoint.restore(a.resume,model,optimizer,scheduler,identity)
    ddp=DDP(model,device_ids=[local],find_unused_parameters=True,static_graph=False,bucket_cap_mb=25,gradient_as_bucket_view=True)
    processor=TemporalQwenProcessor(QWEN_PATH);start=time.monotonic();start_update=state['updates'];inv=uuid.uuid4().hex
    invdir=out/'invocations'/inv;invdir.mkdir(parents=True,exist_ok=True);last_checkpoint=a.resume
    record=dict(invocation=inv,host=host,rank=rank,pid=os.getpid(),start_update=start_update,start_unix=time.time(),resume=a.resume,status='RUNNING')
    atomic_json(invdir/f'rank{rank}.json',record);stop_at=min(a.stop_after or cfg['max_updates'],cfg['max_updates'])
    try:
        if not a.qualification and not a.resume:
            last_checkpoint=temporal_checkpoint.save(out,model,optimizer,scheduler,state,identity)
            if rank==0:atomic_json(out/'progress.json',dict(state,checkpoint=last_checkpoint,status='RUNNING'))
        with (invdir/f'metrics_rank{rank}.jsonl').open('x',buffering=1) as journal:
            while state['updates']<stop_at:
                begin=time.monotonic();optimizer.zero_grad(set_to_none=True);fm_sum=dino_sum=0.;tokens=[];input_tokens=[]
                for micro in range(accumulation):
                    token=data.tokens[data.index(state['cursor']+micro*world+rank)];tokens.append(token)
                    observation=data.observation(token);batch=processor([observation]);input_tokens.append(int((batch['input_ids']==model.qwen.image_token_id).sum()))
                    y,valid=teacher.read(token,manifest['targets'][token])
                    state_input=torch.tensor([observation['proprio']],device=device,dtype=torch.float32)
                    target=ActionCodec.norm_odo(data.target(token)[None].to(device))
                    sync=ddp.no_sync() if micro<accumulation-1 else nullcontext()
                    with sync:
                        with torch.autocast('cuda',dtype=torch.bfloat16):loss,terms=ddp(batch,state_input,target,dino_target=y[None].to(device),dino_valid=valid[None].to(device))
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite FM/DINO loss')
                        (loss/accumulation).backward()
                    fm_sum+=float(terms['FM'])/accumulation;dino_sum+=float(terms['current_DINO'])/accumulation
                if any(p.grad is None for p in model.parameters()):raise ValueError('Missing full-model/auxiliary gradient')
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True,foreach=False)
                lrs={g['name']:g['lr'] for g in optimizer.param_groups};optimizer.step();scheduler.step()
                state['updates']+=1;state['cursor']+=cfg['global_batch'];torch.cuda.synchronize()
                row=dict(updates=state['updates'],parent_plus_updates=60000+state['updates'],cursor=state['cursor'],rank=rank,tokens=tokens,
                    FM=fm_sum,current_DINO=dino_sum,loss=fm_sum+dino_sum,grad_norm=float(grad),learning_rates=lrs,
                    visual_tokens=input_tokens,seconds=time.monotonic()-begin,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
                    new_scene_exposures=state['updates']*cfg['global_batch'],invocation=inv)
                journal.write(json.dumps(row,allow_nan=False)+'\n')
                if rank==0 and (state['updates']<=5 or state['updates']%20==0):print(json.dumps(row),flush=True)
                flag=torch.tensor(int(STOP),device=device);dist.all_reduce(flag,op=dist.ReduceOp.MAX)
                end=bool(flag) or state['updates']==stop_at
                if end or state['updates']%cfg['save_every']==0 or state['updates'] in cfg['milestones']:
                    tick=time.monotonic();last_checkpoint=temporal_checkpoint.save(out,model,optimizer,scheduler,state,identity)
                    if rank==0:atomic_json(out/'progress.json',dict(state,checkpoint=last_checkpoint,
                        status='COMPLETE' if state['updates']==cfg['max_updates'] else 'STOPPED' if end else 'RUNNING',
                        invocation_start_update=start_update,invocation_wall_seconds=time.monotonic()-start,save_seconds=time.monotonic()-tick))
                    error=None
                    try:
                        if rank==0:temporal_checkpoint.retain_rolling(out,identity)
                    except Exception:error=traceback.format_exc()
                    common.check_errors(error)
                if end:break
        record.update(status='COMPLETE' if state['updates']==cfg['max_updates'] else 'STOPPED_AT_SAVED_BOUNDARY',end_update=state['updates'],
            checkpoint=last_checkpoint,wall_seconds=time.monotonic()-start,updates_this_invocation=state['updates']-start_update)
    except BaseException:
        record.update(status='FAILED',error=traceback.format_exc(),last_legal_checkpoint=last_checkpoint,wall_seconds=time.monotonic()-start);raise
    finally:
        atomic_json(invdir/f'rank{rank}.json',record);atomic_json(out/f'result_rank{rank}.json',record);dist.destroy_process_group()


if __name__=='__main__':main()
