"""Current-camera DDP student; no teacher/target encoder is instantiated here.

Use torchrun, including --nproc_per_node=1. Complete checkpoints contain model,
FP32-master optimizer, fixed scheduler identity, all task RNG and exact data
progress. Auxiliary denominators span the global optimizer batch, not microbatches.
"""
import argparse
from contextlib import nullcontext
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run
from tools.ddpolicy_vehicle.training_state import epoch_batches
from tools.ddpolicy_vehicle.campaign import charged_gpu_hours
from tools.ddpolicy_vehicle.optimizer_safety import bounded_parameter_groups,capture_master_samples,master_update_evidence,MAX_GROUP_ELEMENTS
from starVLA.model.modules.vehicle_joint.initialization import identity_hash,module_manifest,tensor_hash
from starVLA.dataloader.foresight_dataset import ForesightTrainingDataset,collate_training
from .student_state import capture_rng,restore_rng,optimizer_batch_counts,learning_rate,validate_rank_batches


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('config','data','campaign-root','run-id'):p.add_argument('--'+key,required=True)
    p.add_argument('--future-root');p.add_argument('--future-identity')
    p.add_argument('--dino-root');p.add_argument('--dino-index');p.add_argument('--dino-identity')
    p.add_argument('--clip-root');p.add_argument('--clip-identity')
    p.add_argument('--local-image-root')
    p.add_argument('--loader-workers',type=int,default=0)
    p.add_argument('--interaction-root');p.add_argument('--interaction-identity')
    p.add_argument('--global-batch',type=int,default=32);p.add_argument('--micro-batch',type=int,default=1)
    p.add_argument('--updates',type=int,required=True);p.add_argument('--schedule-updates',type=int,required=True)
    p.add_argument('--warmup',type=int,required=True);p.add_argument('--save-every',type=int,default=200)
    p.add_argument('--milestones',default='0');p.add_argument('--stop-after',type=int,default=0)
    p.add_argument('--max-seconds',type=float,required=True);p.add_argument('--campaign-gpu-hours',type=float,required=True)
    p.add_argument('--scope',choices=('startup','small_fit','profile','formal'),required=True)
    p.add_argument('--limit',type=int,default=0,help='Diagnostic prefix only; formal requires0')
    p.add_argument('--resume',action='store_true');p.add_argument('--acknowledge-stop',action='store_true')
    p.add_argument('--deterministic',action='store_true');p.add_argument('--offload-optimizer',action='store_true')
    p.add_argument('--registration')
    a=p.parse_args()
    if min(a.global_batch,a.micro_batch,a.updates,a.save_every,a.max_seconds,a.campaign_gpu_hours)<=0:raise ValueError('Positive training/budget settings required')
    if not 0<=a.warmup<a.schedule_updates or not a.updates<=a.schedule_updates or not 0<=a.stop_after<=a.updates or a.limit<0:raise ValueError('Invalid schedule/progress bounds')
    if a.scope=='startup' and (a.updates>4 or a.max_seconds>1800):raise ValueError('Startup <=4 updates/1800seconds')
    if a.scope=='small_fit' and (not 0<a.limit<=64 or a.updates>512):raise ValueError('Small fit <=64scenes/512updates')
    if a.scope=='formal' and a.limit:raise ValueError('Formal must use full manifest')
    if not 0<=a.loader_workers<=8:raise ValueError('Bounded per-rank I/O workers required')
    if a.scope=='profile' and (a.updates>120 or not a.limit or a.max_seconds>14400):raise ValueError('Profile <=120 updates, explicit prefix and <=4h')
    milestones={int(x) for x in a.milestones.split(',') if x}
    if any(x<0 or x>a.updates for x in milestones):raise ValueError('Milestone out of range')
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);dist.init_process_group('nccl')
    if a.global_batch%world:raise ValueError('Global batch must divide by world size')
    attempt=[f'{a.run_id}_attempt_{time.time_ns()}' if rank==0 else None];dist.broadcast_object_list(attempt,0)
    context=metered_run(a.campaign_root,attempt[0],world,{'kind':'foresight_student_'+a.scope,'run_id_parent':a.run_id}) if rank==0 else nullcontext(({},None,lambda:None))
    try:
        with context as (meter,_,save_meter):run(a,milestones,rank,world,attempt[0],meter,save_meter)
    finally:
        dist.destroy_process_group()


def run(a,milestones,rank,world,attempt,meter,save_meter):
    begin=time.time();root=Path(a.campaign_root);out=root/'students'/a.run_id
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze a clean source before training')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    cfg=OmegaConf.load(a.config);arm=cfg.foresight.arm
    from starVLA.model.modules.foresight.tradeoff import CANDIDATES
    full_method=bool(cfg.foresight.get('full_algorithm',False))
    clip_method=cfg.foresight.get('future_target_type','legacy_single_frame') != 'legacy_single_frame'
    tradeoff=arm in CANDIDATES and not full_method
    dino_campaign=full_method or arm.startswith('W_') or arm=='R_NATIVE' or tradeoff
    need_cur=bool(cfg.foresight.get('enable_current_dino',False));need_fut=bool(cfg.foresight.get('enable_future_dino',False))
    need_vis=arm in ('B','D');need_int=bool(cfg.foresight.get('enable_interaction',False)) if dino_campaign else arm in ('C','D')
    if bool(a.future_root)!=need_vis or bool(a.interaction_root)!=need_int:raise ValueError('Auxiliary cache inventory does not match arm')
    if need_vis and float(cfg.foresight.lambda_vis)<=0 or need_int and float(cfg.foresight.lambda_int)<=0:raise ValueError('Active auxiliary loss requires calibrated nonzero weight')
    if bool(a.dino_root)!=(need_cur or need_fut):raise ValueError('DINO label inventory does not match arm')
    if need_cur and float(cfg.foresight.lambda_cur)<=0 or need_fut and float(cfg.foresight.lambda_fut)<=0:raise ValueError('Active DINO weights must be positive')
    data_kwargs=dict(future_root=a.future_root,expected_future=a.future_identity,interaction_root=a.interaction_root,expected_interaction=a.interaction_identity)
    if clip_method:
        from starVLA.dataloader.action_video_foresight_dataset import ActionVideoForesightDataset
        if bool(a.clip_root) != need_fut:
            raise ValueError('Configured clip task requires exactly its identified target cache')
        data=ActionVideoForesightDataset(a.data,dino_root=a.dino_root,dino_index=a.dino_index,expected_dino=a.dino_identity,
            candidate=cfg.foresight.candidate,current=need_cur,future=need_fut,
            clip_root=a.clip_root,expected_clip=a.clip_identity,future_type=cfg.foresight.future_target_type,
            allow_partial=a.scope!='formal',image_root=a.local_image_root,**data_kwargs)
    elif a.clip_root or a.clip_identity:
        raise ValueError('Legacy training must not silently consume a video cache')
    elif full_method:
        from starVLA.dataloader.full_foresight_dataset import FullForesightDataset
        data=FullForesightDataset(a.data,dino_root=a.dino_root,dino_index=a.dino_index,expected_dino=a.dino_identity,
            candidate=cfg.foresight.candidate,current=need_cur,future=need_fut,
            allow_partial=a.scope!='formal',image_root=a.local_image_root,**data_kwargs)
    elif tradeoff:
        from starVLA.dataloader.tradeoff_dataset import TradeoffDataset
        data=TradeoffDataset(a.data,dino_root=a.dino_root,dino_index=a.dino_index,expected_dino=a.dino_identity,
            candidate=arm,allow_partial=a.scope!='formal',image_root=a.local_image_root,**data_kwargs)
    elif dino_campaign:
        from starVLA.dataloader.dino_foresight_dataset import DINOTrainingDataset
        data=DINOTrainingDataset(a.data,dino_root=a.dino_root,dino_index=a.dino_index,expected_dino=a.dino_identity,
            current=need_cur,future=need_fut,allow_partial=a.scope!='formal',**data_kwargs)
    else:data=ForesightTrainingDataset(a.data,**data_kwargs)
    if data.identity['split']!='train' or not (Path(a.data)/'COMPLETE.json').exists():raise ValueError('Only completed training split can update student')
    size=min(len(data),a.limit) if a.limit else len(data)
    validate_rank_batches(size,a.global_batch,world,a.micro_batch)
    # Global ranks are not host-local CUDA ordinals in a two-server run.
    device_names=[None]*world
    dist.all_gather_object(device_names,torch.cuda.get_device_name(torch.cuda.current_device()))
    schedule={'horizon':a.schedule_updates,'warmup':a.warmup,'base':float(cfg.trainer.learning_rate.base),
              'minimum':float(cfg.trainer.scheduler_specific_kwargs.min_lr),'type':'fixed_cosine'}
    # Explicit CLI schedule is authoritative over inherited example trainer limits.
    identity={'schema':'foresight_student_training_v1','source_sha':source,'config':OmegaConf.to_container(cfg,resolve=True),
              'schedule':schedule,'data':data.identity,'ego':data.ego_identity,'future':data.future_identity,'interaction':data.interaction_identity,
              'selected_index_hash':identity_hash(data.index[:size]),'scene_count':size,'scope':a.scope,
              'world_size':world,'global_batch':a.global_batch,'micro_batch':a.micro_batch,'updates':a.updates,
              'loader_workers':a.loader_workers,
              'precision':'BF16 model, FP32 AdamW master/moments','deterministic':a.deterministic,
              'offload_optimizer':a.offload_optimizer,'optimizer_max_group_elements':MAX_GROUP_ELEMENTS,
              'normalization':'global valid elements per optimizer batch; every scene ego loss',
              'device_names':device_names}
    if dino_campaign:identity.update(schema='foresight_dino_student_training_v1',dino=data.dino_identity,
                                    future_sampling='one independent request from fixed1/2/4; no valid-label resampling')
    if tradeoff:identity.update(schema='dino_tradeoff_student_v1',future_sampling='DISABLED',local_images=bool(a.local_image_root),
        candidate=CANDIDATES[arm].record(),attention='native causal; state→3view reasoning blocks→action; reasoning retained at deployment')
    if full_method:identity.update(schema='ddp_full_foresight_student_v1',local_images=bool(a.local_image_root),
        candidate=CANDIDATES[cfg.foresight.candidate].record(),
        attention='native causal; current→view,row,column W→action; W retained at deployment',
        auxiliary_frequency='one current and one requested future per original scene; valid MAE once; Qwen once')
    if clip_method:
        identity.update(schema='ddp_action_video_student_v1',future_clip=data.clip_identity,
            future_sampling='one complete eight-real-frame request per scene; whole-view mask on any missing frame',
            auxiliary_frequency='ego repeat8, current once, full native clip once, valid MAE once; Qwen once')
    if full_method and a.scope=='formal':
        from starVLA.model.modules.vehicle_joint.initialization import file_sha256
        if not a.registration:raise ValueError('Full formal registration required')
        registered=json.loads(Path(a.registration).read_text())
        if registered['training_source_sha']!=source or registered['scene_count']!=size:raise ValueError('Registration source/population changed')
        if clip_method:
            expected={'run_id':a.run_id,'config_sha256':identity_hash(OmegaConf.to_container(cfg,resolve=True)),
                'global_batch':a.global_batch,'micro_batch':a.micro_batch,'world_size':world,
                'updates':a.updates,'schedule_updates':a.schedule_updates,'milestones':sorted(milestones)}
            if any(registered.get(k)!=v for k,v in expected.items()):
                raise ValueError('Action/video formal registration configuration/schedule mismatch')
        identity['registration_sha256']=file_sha256(a.registration)
    signature=identity_hash(identity)
    lock=None
    if rank==0:
        if out.exists() and not a.resume:raise FileExistsError('Run already exists')
        out.mkdir(parents=True,exist_ok=True);lock=(out/'RUN.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.resume:
            if json.loads((out/'identity.json').read_text())!={'identity':signature,**identity}:raise ValueError('Resume source/config/data/targets/device-count mismatch')
            if json.loads((out/'status.json').read_text())['status']=='COMPLETE':raise ValueError('Completed run cannot be extended')
            if not a.acknowledge_stop:raise ValueError('Resume requires explicit stop acknowledgment')
            if (out/'STOP_REQUESTED').exists():(out/'STOP_REQUESTED').rename(out/('STOP_ACKNOWLEDGED_'+str(time.time_ns())))
        else:atomic_json(out/'identity.json',{'identity':signature,**identity})
    dist.barrier()
    if charged_gpu_hours(root)>=a.campaign_gpu_hours:raise RuntimeError('Campaign budget exhausted before model loading')
    random.seed(int(cfg.seed)+rank);np.random.seed(int(cfg.seed)+rank);torch.manual_seed(int(cfg.seed)+rank)
    torch.set_num_threads(2)
    if a.deterministic:
        torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
    from starVLA.model.framework import build_framework
    model=build_framework(cfg)
    if rank==0 and not a.resume:
        init={'action':module_manifest(model.action_model),'state':module_manifest(model.action_input_model),
              'driving_tokens':model.qwen_vl_interface.driving_token_initialization,
              'W':tensor_hash(model.foresight_queries) if hasattr(model,'foresight_queries') else None,
              'generic_source_manifest':json.loads(Path(cfg.from_scratch.source_manifest).read_text()),
              'driving_weights_loaded':False,'future_teacher_in_model':False}
        atomic_json(out/'initialization.json',init)
        if full_method:
            atomic_json(out/'auxiliary_initialization.json',{
                name:module_manifest(getattr(model,name)) for name in ('query_geometry','dino_head','interaction_head','spatiotemporal_head')
                if hasattr(model,name)})
        model.qwen_vl_interface.processor.save_pretrained(str(out/'tokenizer'))
        atomic_json(out/'parameters.json',{'total':sum(p.numel() for p in model.parameters()),
            'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'groups':{key:sum(p.numel() for name,p in model.named_parameters() if name.startswith(key) and p.requires_grad)
                      for key in ('qwen_vl_interface','action_model','action_input_model','foresight_queries','query_geometry','future_head','dino_head','interaction_head','spatiotemporal_head')}})
    import deepspeed
    ds_config={'train_micro_batch_size_per_gpu':a.micro_batch,'gradient_accumulation_steps':1,
               'train_batch_size':a.micro_batch*world,'bf16':{'enabled':True},'gradient_clipping':1.,'steps_per_print':1000000,
               'zero_optimization':{'stage':2,'overlap_comm':True,'contiguous_gradients':True,'reduce_bucket_size':50000000,'allgather_bucket_size':50000000},
               'optimizer':{'type':'AdamW','params':{'lr':schedule['base'],'betas':list(cfg.trainer.optimizer.betas),
                    'eps':float(cfg.trainer.optimizer.eps),'weight_decay':float(cfg.trainer.optimizer.weight_decay)}}}
    if a.offload_optimizer:
        ds_config['zero_optimization']['offload_optimizer']={'device':'cpu','pin_memory':True};ds_config['zero_force_ds_cpu_optimizer']=True
    groups,group_records=bounded_parameter_groups(model.named_parameters())
    if rank==0 and not a.resume:atomic_json(out/'optimizer_groups.json',group_records)
    engine,_,_,_=deepspeed.initialize(model=model,model_parameters=groups,config=ds_config)
    initial_memory=torch.tensor([torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()],device='cuda',dtype=torch.int64)
    initial_memory_rows=[torch.empty_like(initial_memory) for _ in range(world)]
    dist.all_gather(initial_memory_rows,initial_memory)
    if rank==0 and not a.resume:atomic_json(out/'startup_memory.json',{'per_rank_allocated_reserved_bytes':[v.cpu().tolist() for v in initial_memory_rows]})
    generators={key:torch.Generator(device='cpu' if key=='horizon' else torch.device('cuda',torch.cuda.current_device())).manual_seed(int(cfg.seed)+offset+rank)
                for key,offset in (('noise',9000),('time',10000),('horizon',11000))}
    completed=epoch=offset=exposure=0;counters={'visual_elements':0,'interaction_elements':0,'horizon_scenes':[0,0,0],'valid_views':[0,0,0]}
    if dino_campaign:counters.update(current_dino_elements=0,future_dino_elements=0,current_requests=0,future_requests=0)
    if a.resume:
        tag=(out/'checkpoints/latest').read_text().strip()
        if Path(tag).name!=tag:raise ValueError('Unsafe checkpoint pointer')
        saved=json.loads((out/'checkpoints'/tag/'COMPLETE.json').read_text())
        if saved['identity']!=signature:raise ValueError('Incomplete/foreign checkpoint')
        location,state=engine.load_checkpoint(str(out/'checkpoints'),tag=tag,load_module_strict=True,load_optimizer_states=True)
        if not location or any(state.get(k)!=v for k,v in saved.items()):raise ValueError('Checkpoint progress mismatch')
        completed,epoch,offset,exposure=[state[k] for k in ('completed','epoch','offset','exposure')];counters=state['counters']
        restore_rng(torch.load(out/'checkpoints'/tag/f'rng_rank{rank}.pt',map_location='cpu',weights_only=False),generators)
    initial_updates=completed;initial_exposure=exposure
    stopping=[False]
    def handler(*_):stopping[0]=True
    signal.signal(signal.SIGINT,handler);signal.signal(signal.SIGTERM,handler)
    status={'status':'RUNNING','identity':signature,'source_sha':source,'attempt':attempt,'host':socket.gethostname(),'pid':os.getpid()}
    def write_status():
        if rank!=0:return
        status.update(completed=completed,epoch=epoch,offset=offset,exposure=exposure,counters=counters,updated_unix=time.time())
        atomic_json(out/'status.json',status)
        meter.update(real_optimizer_updates=completed-initial_updates,total_run_updates=completed,
                     sample_presentations=exposure-initial_exposure,total_run_exposure=exposure);save_meter()
    def checkpoint(tag):
        checkpoint_start=time.time()
        destination=out/'checkpoints'/tag
        state={'identity':signature,'completed':completed,'epoch':epoch,'offset':offset,'exposure':exposure,
               'counters':counters,'tag':tag,'scheduler':{**schedule,'completed':completed}}
        latest=out/'checkpoints/latest'
        if tag.startswith(('paused_','final_')) and latest.exists():
            previous=out/'checkpoints'/latest.read_text().strip()/'COMPLETE.json'
            saved=json.loads(previous.read_text())
            if {k:v for k,v in saved.items() if k!='tag'}=={k:v for k,v in state.items() if k!='tag'}:
                # A just-saved immutable milestone is already a complete recovery
                # point. Do not duplicate every model/optimizer shard for the pause.
                dist.barrier();write_status();return
        if destination.exists():raise FileExistsError('Checkpoint tag is immutable')
        engine.save_checkpoint(str(out/'checkpoints'),tag=tag,client_state=state,save_latest=False)
        torch.save(capture_rng(generators),destination/f'rng_rank{rank}.pt');dist.barrier()
        if rank==0:
            atomic_json(destination/'COMPLETE.json',state)
            tmp=out/'checkpoints/latest.tmp';tmp.write_text(tag+'\n');tmp.replace(out/'checkpoints/latest')
            if tag.startswith('periodic_'):
                import shutil
                for old in (out/'checkpoints').glob('periodic_*'):
                    if old==destination or not (old/'COMPLETE.json').exists():continue
                    if json.loads((old/'COMPLETE.json').read_text())['identity']!=signature:raise ValueError('Foreign checkpoint in rolling-save directory')
                    shutil.rmtree(old)
        dist.barrier();write_status()
        if rank==0:
            with (out/'checkpoint_costs.jsonl').open('a') as stream:stream.write(json.dumps({'tag':tag,'seconds':time.time()-checkpoint_start})+'\n')
    write_status()
    io_pool=ThreadPoolExecutor(max_workers=a.loader_workers) if a.loader_workers else None
    head_events=[];head_hooks=[]
    if a.scope=='profile':
        # CUDA events measure readout FORWARD work only. Total optimizer-step
        # wall time above still includes both backward paths and synchronization.
        def attach_head(name,module):
            pending=[]
            def before(*_):
                event=torch.cuda.Event(enable_timing=True);event.record();pending.append(event)
            def after(*_):
                end=torch.cuda.Event(enable_timing=True);end.record()
                head_events.append((name,pending.pop(),end))
            head_hooks.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
        for name in ('dino_head','interaction_head','spatiotemporal_head'):
            if hasattr(model,name):attach_head(name,getattr(model,name))
    try:
        if not a.resume and 0 in milestones:checkpoint('milestone_000000')
        model.train()
        while completed<a.updates:
            batches=epoch_batches(size,a.global_batch,int(cfg.seed),epoch)
            if offset==len(batches):epoch+=1;offset=0;continue
            stop=stopping[0] or (out/'STOP_REQUESTED').exists() or time.time()-begin>=a.max_seconds or bool(a.stop_after and completed>=a.stop_after)
            if rank==0:stop=stop or charged_gpu_hours(root)>=a.campaign_gpu_hours
            stop_tensor=torch.tensor(int(stop),device='cuda');dist.all_reduce(stop_tensor,op=dist.ReduceOp.MAX)
            if stop_tensor:
                status['status']='PAUSED';checkpoint(f'paused_{completed:06d}_{attempt}');meter['status']='PAUSED';break
            started=time.time();head_events.clear();indices=batches[offset][rank::world]
            # map preserves scene order. No augmentation/task RNG runs in workers.
            samples=list(io_pool.map(data.__getitem__,indices)) if io_pool else [data[i] for i in indices]
            observations,targets=collate_training(samples)
            data_seconds=time.time()-started
            torch.cuda.reset_peak_memory_stats()
            counts=optimizer_batch_counts(targets,generators['horizon'],'cuda')
            lr=learning_rate(completed,schedule['base'],schedule['minimum'],schedule['warmup'],schedule['horizon'])
            for group in engine.optimizer.param_groups:group['lr']=lr
            logs={};exposures=torch.zeros(6,device='cuda',dtype=torch.int64)
            if need_vis:
                h=targets['visual_horizon'];valid=targets['future_valid'][torch.arange(len(h)),h].sum(-1)
                for k in range(3):exposures[k]=(h==k).sum();exposures[k+3]=valid[h==k].sum()
            if need_fut and not clip_method:
                h=targets['dino_horizon'];valid=targets['future_dino_valid'][torch.arange(len(h)),h].flatten(2).any(-1).sum(-1)
                for k in range(3):exposures[k]=(h==k).sum();exposures[k+3]=valid[h==k].sum()
            dist.all_reduce(exposures)
            for at in range(0,len(indices),a.micro_batch):
                end=min(len(indices),at+a.micro_batch);boundary=end==len(indices)
                engine.set_gradient_accumulation_boundary(boundary)
                output=engine(observations[at:end],{k:v[at:end] for k,v in targets.items()},completed_updates=completed,
                    noise_generator=generators['noise'],time_generator=generators['time'],horizon_generator=generators['horizon'],global_counts=counts)
                if not torch.isfinite(output['loss']):raise FloatingPointError('Nonfinite student loss')
                engine.backward(output['loss'])  # already normalized across ALL microbatches/ranks
                for key,value in output['losses'].items():logs[key]=logs.get(key,0.)+float(value.detach())
                for key in ('visual_raw','interaction_raw','current_dino_raw','future_dino_raw','future_clip_raw'):
                    if key in output['metrics']:logs[key]=logs.get(key,0.)+float(output['metrics'][key])
                if boundary:before=capture_master_samples(engine.optimizer)
                observe = full_method and a.scope=='formal' and boundary and (completed+1 in {1,100,500,1000,2000} | milestones)
                if observe:
                    from tools.full_foresight.training_observation import before_step,after_step
                    observation_before=before_step(model)
                engine.step()
                if boundary:evidence=master_update_evidence(engine.optimizer,before,torch.device('cuda',torch.cuda.current_device()))
                if observe:evidence['shared_parameter_observation']=after_step(model,observation_before)
            if observe:
                # Read-only training-domain trajectory diagnostic; preserve every
                # training RNG and mode. This BF16 diagnostic is never official PDMS.
                saved_rng=capture_rng(generators);model.eval()
                try:
                    from tools.foresight.checkpoints import scene_noise
                    from starVLA.dataloader.foresight_dataset import decode_ego
                    with torch.inference_mode():
                        prediction=model.predict_action(observations[:1],initial_noise=scene_noise(observations[0]['token'],42,'cuda'))
                        truth=decode_ego(targets['ego'][:1].cuda().float())
                        distance=(prediction[...,:2]-truth[...,:2]).norm(dim=-1)
                        yaw=prediction[...,2]-truth[...,2]
                        yaw=torch.atan2(yaw.sin(),yaw.cos()).abs()
                        quality=torch.stack((distance.mean(),distance[:,-1].mean(),yaw.mean()))
                        if not torch.isfinite(quality).all():raise FloatingPointError('Invalid in-run ego diagnostic')
                        dist.all_reduce(quality);quality/=world
                    evidence['ego_training_diagnostic']={'ADE_m':float(quality[0]),'FDE_m':float(quality[1]),
                        'yaw_abs_rad':float(quality[2]),'scenes':world,'sampling_seed':42,
                        'protocol':'live BF16, first current scene per rank in this fixed training batch; not PDMS'}
                finally:
                    model.train();restore_rng(saved_rng,generators)
            for key in sorted(logs):
                value=torch.tensor(logs[key],device='cuda',dtype=torch.float64);dist.all_reduce(value);logs[key]=float(value/world)
            completed+=1;offset+=1;exposure+=len(batches[offset-1])
            counters['visual_elements']+=int(counts.get('visual',0));counters['interaction_elements']+=int(counts.get('interaction',0))
            if dino_campaign:
                for task in ('current_dino','future_dino'):counters[task+'_elements']+=int(counts.get(task,0))
                counters['current_requests']+=len(batches[offset-1])*int(need_cur)
                counters['future_requests']+=len(batches[offset-1])*int(need_fut)
            if clip_method:
                counters['future_clip_scenes']=counters.get('future_clip_scenes',0)+int(counts.get('future_clip',0))
                counters['physical_future_frame_requests']=counters.get('physical_future_frame_requests',0)+8*len(batches[offset-1])*int(need_fut)
            for k in range(3):counters['horizon_scenes'][k]+=int(exposures[k]);counters['valid_views'][k]+=int(exposures[k+3])
            torch.cuda.synchronize()
            readout_times={'current_dino':0.,'future_dino':0.,'interaction':0.};dino_call=0
            for name,start_event,end_event in head_events:
                if name=='interaction_head':key='interaction'
                elif name=='spatiotemporal_head':key='future_dino'
                else:
                    key='current_dino' if need_cur and (not need_fut or dino_call%2==0) else 'future_dino'
                    dino_call+=1
                readout_times[key]+=start_event.elapsed_time(end_event)
            perf=torch.tensor([time.time()-started,data_seconds,torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved(),
                               float(model.last_sequence_lengths.max())],device='cuda',dtype=torch.float64)
            all_perf=[torch.empty_like(perf) for _ in range(world)];dist.all_gather(all_perf,perf)
            if rank==0:
                row={'update':completed,'epoch':epoch,'offset':offset,'exposure':exposure,'lr':lr,'losses':logs,'counts':counts,'ended_unix':time.time(),
                     'seconds':time.time()-started,'peak_memory_bytes':torch.cuda.max_memory_allocated(),
                     'grad_norm':float(engine.get_global_grad_norm()),'per_rank_profile':[v.cpu().tolist() for v in all_perf],
                     'profile_fields':['total_step_seconds','data_seconds','peak_allocated_bytes','peak_reserved_bytes','max_sequence_length'],**evidence}
                if a.scope=='profile':row['rank0_readout_forward_gpu_ms']=readout_times
                if full_method:
                    fcfg=model.foresight_config;warm=min(1.,completed/fcfg.auxiliary_warmup)
                    row['effective_weights']={'ego_fm':1.,'current_dino':fcfg.lambda_cur if need_cur else 0.,
                        'future_dino':fcfg.lambda_fut*warm if need_fut else 0.,'interaction':fcfg.lambda_int*warm if need_int else 0.}
                    row['raw_losses']={'ego_fm':logs['ego_fm'],**{k:logs.get(k+'_raw') for k in ('current_dino','future_dino','interaction')}}
                    row['horizon_scene_requests']=exposures[:3].cpu().tolist();row['horizon_valid_views']=exposures[3:].cpu().tolist()
                    row['valid_patches']={k:int(counts.get(k,0))//fcfg.dino_feature_dim for k in ('current_dino','future_dino')}
                    row['valid_interaction_scenes']=int(counts.get('interaction',0))//int(np.prod(targets['interaction_latent'].shape[1:])) if need_int else 0
                    if clip_method:
                        row['effective_weights']['future_clip']=row['effective_weights'].pop('future_dino')
                        row['raw_losses']['future_clip']=logs.get('future_clip_raw')
                        row['raw_losses'].pop('future_dino',None)
                        row['valid_clip_scenes']=int(counts.get('future_clip',0))
                        row['planner_condition_mode']=cfg.foresight.planner_condition_mode
                with (out/'steps.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            write_status()
            if completed in milestones:checkpoint(f'milestone_{completed:06d}')
            elif completed%a.save_every==0:checkpoint(f'periodic_{completed:06d}')
        else:
            if offset==len(epoch_batches(size,a.global_batch,int(cfg.seed),epoch)):epoch+=1;offset=0
            status['status']='COMPLETE';checkpoint(f'final_{completed:06d}');meter['status']='COMPLETE'
            if a.scope=='profile':
                for hook in head_hooks:hook.remove()
                head_hooks.clear()
                model.eval();model.strip_auxiliary_heads();latencies=[]
                from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset
                current_data=ForesightCurrentDataset(a.data)
                current_data.local_image_root=a.local_image_root
                observation=current_data[0]  # deployment profiling has no target-file dependency
                for i in range(25):
                    torch.cuda.synchronize();start=time.perf_counter()
                    prediction=model.predict_action([observation],sampling_seed=123)
                    torch.cuda.synchronize()
                    if not torch.isfinite(prediction).all():raise FloatingPointError('Invalid profile prediction')
                    if i>=5:latencies.append(time.perf_counter()-start)
                gathered=[None]*world;dist.all_gather_object(gathered,latencies)
                if rank==0:
                    from tools.foresight.deployment_precision import describe
                    atomic_json(out/'inference_profile.json',{'batch':1,'warmup':5,'measured':20,'per_rank_seconds':gathered,
                        'includes_teacher':False,'reasoning_retained':True,'includes_rgb_loading':False,
                        'precision_protocol':describe(model,'live BF16 training parameters; not reconstructed FP32 masters','in-memory DeepSpeed model after training'),
                        'quality_pairing':'BF16 timing only; must not pair with FP32 PDMS'})
    except BaseException:
        status['status']='FAILED';write_status();raise
    finally:
        for hook in head_hooks:hook.remove()
        if io_pool:io_pool.shutdown(wait=True)
        write_status()


if __name__=='__main__':main()
