"""Executable V2 DDP training, 32-step smoke, and exact same-run resume."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import subprocess
import inspect
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from navsim.agents.EpisodeDrive.planreg_v2.agent import PlanRegV2Agent,file_sha256
from navsim.agents.EpisodeDrive.planreg_v2.data import InputOnlyV2Dataset,v2_collate
from navsim.agents.EpisodeDrive.planreg_v2.optimizer import audit_adam_state
from navsim.agents.EpisodeDrive.planreg_v2.runtime import ExactExposureSampler,rng_state,restore_rng,validate_formal,same_batch_gradient_audit,load_config,accumulated_batches,prepare_run_directory


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ('config','manifest','output'): parser.add_argument('--'+name,required=True)
    parser.add_argument('--resume'); parser.add_argument('--warm-start',help='Explicit migrated V1 artifact; never a formal VLM-only main result')
    parser.add_argument('--layout-lock'); parser.add_argument('--smoke-steps',type=int,default=0)
    parser.add_argument('--microbatch',type=int,default=1); parser.add_argument('--accumulate',type=int,default=1)
    parser.add_argument('--workers',type=int,default=2); parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--stop-after',type=int,help='Explicit same-schedule interruption for resume testing')
    parser.add_argument('--replay-export',action='store_true')
    args = parser.parse_args()
    if args.resume and args.warm_start: raise ValueError('Warm start and complete resume are different operations')
    if args.warm_start and not args.smoke_steps:
        raise ValueError('V1 warm start is allowed for bounded replay, not the VLM-only formal main result')
    if args.microbatch<1 or args.accumulate<1: raise ValueError('Positive microbatch and accumulation required')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    rank,world,local = int(os.getenv('RANK',0)),int(os.getenv('WORLD_SIZE',1)),int(os.getenv('LOCAL_RANK',0))
    torch.cuda.set_device(local)
    # Existing stress scripts yield to an actual CUDA context; allow them to release memory.
    reservation = torch.empty(1,device='cuda'); time.sleep(3); del reservation
    if world > 1: dist.init_process_group('nccl')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    cfg = load_config(args.config)
    dataset = InputOnlyV2Dataset(args.manifest,cfg['vlm_path'],cfg['world_model_enabled'])
    if dataset.manifest['split'] not in ('train','trainval_final_fit'):
        raise ValueError('Held-out/development/Navtest data cannot enter optimizer training')
    actual_batch = args.microbatch*world*args.accumulate
    cfg['global_batch'] = actual_batch
    steps_per_epoch = math.ceil(len(dataset)/actual_batch)
    if args.smoke_steps:
        if not 1 <= args.smoke_steps <= 64: raise ValueError('Smoke is bounded to at most 64 optimizer steps')
        cfg['total_steps'] = args.smoke_steps
    else:
        if os.getenv('LAUNCH_FORMAL') != '1': raise ValueError('Set LAUNCH_FORMAL=1 explicitly for one formal run')
        if not args.layout_lock: raise ValueError('New V2 memory/throughput layout lock required')
        layout = json.loads(Path(args.layout_lock).read_text())
        validate_formal(cfg,dataset.manifest,layout)
        if (args.microbatch,args.accumulate,world) != (layout['microbatch'],layout['accumulate'],layout['world_size']):
            raise ValueError('Launch does not match profiled layout')
        cfg['total_steps'] = steps_per_epoch*27
    output = Path(args.output)
    prepare_run_directory(output,resume=bool(args.resume))
    manifest_sha = file_sha256(args.manifest)
    run_contract=dict(seed=args.seed,microbatch=args.microbatch,accumulate=args.accumulate,workers=args.workers)
    checkpoint = torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    if checkpoint:
        if (checkpoint['config'] != cfg or checkpoint['manifest_sha256'] != manifest_sha or
                checkpoint['world_size'] != world or checkpoint.get('run_contract') != run_contract):
            raise ValueError('Complete resume requires identical config, manifest, layout and schedule')
    agent = PlanRegV2Agent(cfg,device='cuda')
    if args.warm_start:
        from navsim.agents.EpisodeDrive.planreg_v2.checkpoint import load_declared_warm_start
        migration=load_declared_warm_start(agent,args.warm_start)
        if rank==0:(output/'warm_start_audit.json').write_text(json.dumps(migration,indent=2))
    if checkpoint: agent.load_state_dict(checkpoint['model'],strict=True)
    agent.train()
    from navsim.agents.EpisodeDrive.planreg_v2.diagnostics import AttentionReadoutAudit
    readout_audit=AttentionReadoutAudit(agent)
    model = DistributedDataParallel(agent,device_ids=[local],broadcast_buffers=False) if world > 1 else agent
    optimizer,scheduler = agent.get_optimizers()
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer']); scheduler.load_state_dict(checkpoint['scheduler'])
    epoch = checkpoint['epoch'] if checkpoint else 0
    start_step = checkpoint['step_in_epoch'] if checkpoint else 0
    saved_rng = checkpoint['rng_states'][rank] if checkpoint else None
    commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    records=[]; start=time.perf_counter()
    initial = {n:p.detach().clone() for n,p in agent.named_parameters() if p.requires_grad} if args.smoke_steps else None
    if rank == 0:
        (output/'resolved_config.json').write_text(json.dumps(cfg,indent=2))
        metadata = dict(git_commit=commit,manifest_sha256=manifest_sha,steps_per_epoch=steps_per_epoch,total_steps=cfg['total_steps'],
            global_batch=actual_batch,world_size=world,optimizer_groups=agent.optimizer_summary,
            microbatch=args.microbatch,accumulate=args.accumulate,num_workers=args.workers,
            initialization_kind='declared_v1_warm_start' if args.warm_start else 'complete_v2_resume' if args.resume else 'vlm_only_new_planning',
            torch=torch.__version__,trainable_parameters=sum(p.numel() for p in agent.parameters() if p.requires_grad),
            frozen_parameters=sum(p.numel() for p in agent.parameters() if not p.requires_grad),
            python=subprocess.check_output(['python','--version'],text=True).strip(),
            environment={k:v for k,v in os.environ.items() if k.startswith(('CUDA_','NCCL_','OMP_','MKL_','OPENBLAS_','RANK','WORLD_SIZE','LOCAL_RANK'))})
        from navsim.agents.EpisodeDrive.formal_initialization import discover_weight_files
        from navsim.agents.EpisodeDrive.planreg_v2.backbone import V2_SYSTEM_PROMPT
        import hashlib
        metadata.update(shared_init_sha256=file_sha256(cfg['shared_init_path']) if cfg.get('shared_init_path') else None,
            normalizer_sha256=file_sha256(cfg['normalizer_path']),
            vlm_weight_sha256={str(p):file_sha256(p) for p in discover_weight_files(Path(cfg['vlm_path']))},
            prompt_sha256=hashlib.sha256(V2_SYSTEM_PROMPT.encode()).hexdigest(),run_contract=run_contract)
        (output/'run_metadata.json').write_text(json.dumps(metadata,indent=2))
        from navsim.agents.EpisodeDrive.score_module import compute_navsim_score as scoring
        from navsim.planning.metric_caching.metric_cache import MetricCache
        provenance={}
        for label,value in dict(vision=agent.backbone.model.vision_model,language=agent.backbone.model.language_model,
                               simulator=scoring.simulator,scorer=scoring.scorer,metric_cache=MetricCache).items():
            cls=value if isinstance(value,type) else type(value)
            path=inspect.getfile(cls)
            provenance[label]=dict(python_class=cls.__module__+'.'+cls.__name__,source_path=path,sha256=file_sha256(path))
        (output/'runtime_provenance.json').write_text(json.dumps(provenance,indent=2))
    last_features=None
    done=False
    while not done:
        sampler=ExactExposureSampler(len(dataset),actual_batch,world,rank,args.seed,epoch,start_step)
        generator=torch.Generator().manual_seed(args.seed+epoch+rank*1000)
        loader=DataLoader(dataset,batch_size=args.microbatch,sampler=sampler,collate_fn=v2_collate,
            num_workers=args.workers,pin_memory=True,persistent_workers=args.workers>0,generator=generator,
            **({'prefetch_factor':2} if args.workers else {}))
        iterator=iter(loader)
        if saved_rng is not None: restore_rng(saved_rng); saved_rng=None
        optimizer.zero_grad(set_to_none=True)
        accumulated_values={}
        optimizer_tick=time.perf_counter()
        for batch_idx,(features,targets,count_context) in enumerate(accumulated_batches(iterator,args.accumulate)):
            agent.valid_count_context=count_context
            from contextlib import nullcontext
            boundary=(batch_idx+1)%args.accumulate==0
            context=model.no_sync() if world>1 and not boundary else nullcontext()
            with context:
                readout_audit.enabled=int(agent.optimizer_updates)%500==0
                predictions=model(features)
                losses=agent.compute_loss(features,targets,predictions)
                if readout_audit.enabled and rank==0:
                    diagnostic=dict(agent.last_diagnostics,**readout_audit.collect())
                    (output/('task_diagnostics_step%06d.json'%int(agent.optimizer_updates))).write_text(json.dumps(diagnostic,indent=2))
                if args.smoke_steps and agent.world_model_enabled:
                    if not targets['future_valid_mask'].all():
                        raise ValueError('Representative full-WM smoke requires all three real future horizons valid')
                    if float(losses['wm_loss'].detach()) <= 0:
                        raise ValueError('Full-WM smoke produced no valid TF/RO supervision')
                finite=torch.isfinite(losses['loss']).to(dtype=torch.int32)
                if world>1:dist.all_reduce(finite,op=dist.ReduceOp.MIN)
                if not finite: raise FloatingPointError('Non-finite full loss; optimizer and EMA not updated on any rank')
                if boundary and int(agent.optimizer_updates)%500==0 and 'wm_loss' in losses:
                    audit=same_batch_gradient_audit(losses['trajectory_loss']+losses['scorer_loss'],losses['wm_loss'],agent.named_parameters(),float(losses['wm_weight']))
                    if rank==0: print('SAME_BATCH_GRADIENT',json.dumps(audit),flush=True)
                    if rank==0:(output/('gradient_audit_step%06d.json'%int(agent.optimizer_updates))).write_text(json.dumps(audit,indent=2))
                (losses['loss']/args.accumulate).backward()
            for name,value in losses.items():
                accumulated_values[name]=accumulated_values.get(name,0.)+float(value.detach())/args.accumulate
            if not boundary: continue
            norm=torch.nn.utils.clip_grad_norm_([p for p in agent.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            audit_update=(int(agent.optimizer_updates)+1)%500==0 or int(agent.optimizer_updates)<2
            before_update={n:p.detach().clone() for n,p in agent.named_parameters() if p.requires_grad} if audit_update else None
            applied_lrs={group['name']:group['lr'] for group in optimizer.param_groups}
            optimizer.step(); scheduler.step(); audit_adam_state(optimizer)
            optimizer.zero_grad(set_to_none=True)
            step=int(agent.optimizer_updates)
            last_features=features
            torch.cuda.synchronize()
            values=accumulated_values
            accumulated_values={}
            if world>1:
                ordered=sorted(values)
                packed=torch.tensor([values[k] for k in ordered],device='cuda',dtype=torch.float64)
                dist.all_reduce(packed);packed/=world
                values.update(zip(ordered,packed.cpu().tolist()))
            peak_memory=torch.tensor([torch.cuda.max_memory_allocated()/2**30,torch.cuda.max_memory_reserved()/2**30],device='cuda')
            if world>1:dist.all_reduce(peak_memory,op=dist.ReduceOp.MAX)
            values.update(step=step,epoch=epoch,step_seconds=time.perf_counter()-optimizer_tick,grad_norm=float(norm),
                peak_allocated_gib=float(peak_memory[0]),peak_reserved_gib=float(peak_memory[1]))
            if audit_update:
                update={n:dict(update_norm=float((p.detach()-before_update[n]).norm()),
                    update_weight_ratio=float((p.detach()-before_update[n]).norm()/before_update[n].norm().clamp_min(1e-12)))
                    for n,p in agent.named_parameters() if p.requires_grad}
                if rank==0:(output/('updates_step%06d.json'%step)).write_text(json.dumps(update,indent=2))
                del before_update
            values['learning_rates_applied']=applied_lrs
            values['learning_rates_next']={group['name']:group['lr'] for group in optimizer.param_groups}
            if agent.ema_teacher is not None:values['ema']=agent.ema_teacher.last_diagnostics
            if rank==0:
                print(json.dumps(values),flush=True)
                records.append(values)
            end_epoch=start_step+(batch_idx+1)//args.accumulate
            done=step>=cfg['total_steps'] or (args.stop_after is not None and step>=args.stop_after)
            save=done or end_epoch==steps_per_epoch
            if save:
                states=[None]*world
                if world>1: dist.all_gather_object(states,rng_state())
                else: states=[rng_state()]
                if rank==0:
                    payload=dict(schema='planreg_v2_training_v1',model=agent.state_dict(),optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),rng_states=states,epoch=epoch+int(end_epoch==steps_per_epoch),
                        step_in_epoch=0 if end_epoch==steps_per_epoch else end_epoch,config=cfg,world_size=world,
                        manifest_sha256=manifest_sha,git_commit=commit,run_contract=run_contract)
                    temp=output/'last.pending.ckpt'; torch.save(payload,temp); temp.replace(output/'last.ckpt')
                    if end_epoch==steps_per_epoch and epoch+1 in (5,10,15,20,25,27):
                        torch.save(payload,output/('epoch_%02d%s.ckpt'%(epoch+1,'_final' if epoch+1==27 else '')))
                if world>1: dist.barrier()
            if done: break
            optimizer_tick=time.perf_counter()
        epoch+=1; start_step=0
    if rank==0:
        changed = {n:float((p.detach()-initial[n]).norm()) for n,p in agent.named_parameters() if p.requires_grad} if initial else {}
        report=dict(status='executed',optimizer_steps=int(agent.optimizer_updates),elapsed_seconds=time.perf_counter()-start,
            records=records,trainable_updates=changed,fp32_trainable=all(p.dtype==torch.float32 for p in agent.parameters() if p.requires_grad),
            peak_allocated_gib=max(r['peak_allocated_gib'] for r in records),peak_reserved_gib=max(r['peak_reserved_gib'] for r in records),
            samples_per_second=len(records)*actual_batch/max(.001,time.perf_counter()-start),
            steady_median_step_seconds=float(np.median([r['step_seconds'] for r in records[1:]])) if len(records)>1 else None,
            steady_p90_step_seconds=float(np.quantile([r['step_seconds'] for r in records[1:]],.9)) if len(records)>1 else None,
            horizons_valid=[bool(t) for t in targets['future_valid_mask'].all(0)],performance_claim='NOT_EVALUATED')
        (output/'validation.json').write_text(json.dumps(report,indent=2))
        if args.replay_export:
            from navsim.agents.EpisodeDrive.planreg_v2.checkpoint import export_student,load_student
            agent.eval()
            current={k:v for k,v in last_features.items() if not k.startswith('future_')}
            with torch.no_grad(): expected=agent(current)['trajectory'].cpu()
            student_path=output/'student.ckpt'
            export_student(output/'last.ckpt',student_path)
            # Avoid a second simultaneous 2B CUDA model allocation.
            agent.to('cpu'); torch.cuda.empty_cache()
            student=load_student(student_path,device='cuda')
            with torch.no_grad(): actual=student(current)['trajectory'].cpu()
            diff=float((actual-expected).abs().max())
            (output/'export_replay.json').write_text(json.dumps(dict(max_abs_diff=diff,no_teacher=student.ema_teacher is None,
                no_predictor=student.wm_predictor is None,current_only=True),indent=2))
            if diff>1e-5: raise AssertionError('Student export current-policy mismatch')
    if agent._score_pool is not None: agent._score_pool.shutdown()
    if world>1: dist.destroy_process_group()


if __name__=='__main__': main()
