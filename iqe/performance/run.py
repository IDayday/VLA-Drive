"""Bounded real Query training, profiler and checkpoint runner. Schedule is inherited intact."""
import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import copy,fcntl,hashlib,inspect,json,os,platform,random,signal,time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from ..config import load_config
from ..contracts import require
from ..io import read_json,atomic_json,atomic_torch,file_hash,digest
from ..query_base import build_query_framework,bind_source
from ..resources import qualify_distributed
from ..training.base_trainer import original_dataset
from ..training.trainer import init_distributed,scheduler_factory
from ..training.checkpoint import capture_rng,restore_rng
from ..training.runtime_identity import configure_runtime,backend_flags
from ..training.local_checkpoint import stage_checkpoint
from ..training.execution import BatchAddress,EpochPlan,TrainingPrefetch,FrozenVisionAhead,collective_check,finite_gradients
from .options import PerformanceOptions
from .variants import configure_model,adam,load_adam
from .instrumentation import Instrumentation


def tensor_hash(t):
    t=t.detach().cpu().contiguous()
    return hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def state_hash(model,optimizer):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(tensor_hash(v).encode())
    weights=h.hexdigest();h=hashlib.sha256()
    for m in getattr(optimizer,'masters',[]):h.update(tensor_hash(m).encode())
    for i,s in optimizer.state_dict()['state'].items():
        h.update(str(i).encode())
        for k,v in s.items():h.update(k.encode());h.update((tensor_hash(v) if torch.is_tensor(v) else repr(v)).encode())
    return {'model':weights,'optimizer_tensors':h.hexdigest()}


def main(argv=None, *, production=None):
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True);p.add_argument('--runtime',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['bounded','continuation'],default='bounded')
    p.add_argument('--qualification',help='verified same-runtime throughput/numerics/resume certificate for explicit continuation')
    p.add_argument('--checkpoint-every',type=int,default=1000)
    p.add_argument('--updates',required=True,type=int);p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--trace',action='store_true');p.add_argument('--fingerprint',action='store_true')
    p.add_argument('--save',action='store_true');p.add_argument('--probe',action='store_true')
    p.add_argument('--build-vision-cache',action='store_true')
    p.add_argument('--mask-probe',action='store_true',help='two-update real-input missing-label/tail diagnostic, never a throughput or training checkpoint')
    p.add_argument('--local-checkpoint-root',default='/var/tmp/iqe-perf-v2/checkpoints')
    a=p.parse_args(argv);require(1<=a.updates<=(200 if a.mode=='bounded' else 100000),'explicit update budget, independent of LR schedule')
    require(a.checkpoint_every>0,'checkpoint interval must be positive')
    require(a.mode=='bounded' or (a.qualification and not any((a.trace,a.probe,a.mask_probe,a.build_vision_cache,a.fingerprint))), 'continuation requires qualification and excludes diagnostic modes')
    require(not a.mask_probe or (a.updates==2 and not a.save and not a.build_vision_cache),'mask probe is a two-step unsaved diagnostic')
    require(0<=a.warmup<a.updates,'invalid warmup');require(not a.trace or a.updates<=8,'profiler at most8 updates')
    opts=PerformanceOptions.load(a.runtime)
    implementation_hashes={str(p):file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    if production is not None:
        production.validate_options(opts, a)
    if a.mode=='continuation' and production is None:
        from .qualification import validate_continuation
        certificate=validate_continuation(a.qualification,opts,implementation_hashes)
        require(opts.vision_cache is None,'pilot cache is not qualified for full continuation')
        a.save=True
    require(not a.build_vision_cache or (opts.vision_cache and a.updates*32<=64),'explicit64-scene pilot budget')
    configure_runtime({'deterministic_backward':opts.deterministic})
    resources=qualify_distributed('cuda');rank,world=init_distributed('cuda')
    require(world==8,'performance comparisons require eight ranks, global32/micro4')
    device=f'cuda:{os.environ["LOCAL_RANK"]}';torch.cuda.set_device(device)
    if production is not None: production.resolve_checkpoint(a, rank, device)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    lock=None
    error=None
    if rank==0:
        try:
            lock=(root/'run.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            require(not (root/'LAUNCH.json').exists() or (production is not None and production.resume_from_run),
                    'output identity already exists: use explicit resume-from-run')
            if production is not None: production.lock_lineage(a)
        except Exception as ex:error=repr(ex)
    collective_check(error,device=device)
    config=load_config(a.config);contract=read_json(config['s0']['contract'])
    receipt=read_json(a.checkpoint+'.COMPLETE.json')
    staged=None;error=None
    if rank==0:
        try: staged=str(stage_checkpoint(a.checkpoint,receipt['checksum'],a.local_checkpoint_root))
        except Exception as ex:error=repr(ex)
    collective_check(error,device=device)
    value=[staged];dist.broadcast_object_list(value,src=0)
    state=torch.load(value[0],map_location='cpu',mmap=True,weights_only=False)
    require(state['optimizer_step']==receipt['optimizer_step'],'checkpoint step receipt mismatch')
    require(state['dependencies']['world_size']==world,'resume topology mismatch')
    require(state['seed']==config['seed'],'resume/data shuffle seed changed')
    require(state['dependencies'].get('original_targets')==config['data'],'original target/data identity changed')
    require(state['framework_contract_hash']==contract['framework_contract_hash'],'foreign framework')
    original_dependencies=copy.deepcopy(state['dependencies'])
    cfg=copy.deepcopy(state['dependencies']['config'])
    declared=copy.deepcopy(config['query_base']); inherited=copy.deepcopy(cfg)
    declared.pop('validation_every_steps');inherited.pop('validation_every_steps')
    require(declared==inherited,'performance cannot silently override optimization recipe')
    require(cfg['global_batch_size']==32 and cfg['micro_batch_size']==4 and cfg['max_optimizer_steps']==100000,'fixed batch and original100k schedule required')
    require(state['optimizer_step']+a.updates<=cfg['max_optimizer_steps'],'update budget exceeds schedule')
    import transformers
    execution_environment={'python':platform.python_version(),'host':platform.node(),'torch':torch.__version__,
        'cuda':torch.version.cuda,'nccl':list(torch.cuda.nccl.version()),'transformers':transformers.__version__,
        'device_name':torch.cuda.get_device_name(device),'device_capability':list(torch.cuda.get_device_capability(device)),
        'qwen_implementation_hash':file_hash(Path(transformers.__file__).parent/'models/qwen3_vl/modeling_qwen3_vl.py')}
    identity={'execution_environment':execution_environment,'options':opts.identity(),'backend_flags':backend_flags(),'framework_contract_hash':contract['framework_contract_hash'],
        'optimizer_config':cfg,'execution_code_hashes':implementation_hashes,'original_dependencies_hash':digest(original_dependencies),'fit_hash':digest(state['fit_scene_ids']),'world_size':world,'torch':torch.__version__}
    if a.mode=='continuation' and production is None:
        require(certificate['execution_environment']==execution_environment,'qualified execution environment changed; explicit new qualification required')
        require(certificate['framework_contract_hash']==identity['framework_contract_hash'] and certificate['fit_hash']==identity['fit_hash'],'qualification Base/data framework mismatch')
    if production is not None:
        identity['production_cache_runtime'] = production.identity()
        production.validate_state(state, identity, a)
    if state.get('performance_identity'):
        require(state['performance_identity']==identity,'performance resume execution identity changed')
    if rank==0:
        launch={'checkpoint':a.checkpoint,'checkpoint_hash':receipt['checksum'],'identity':identity,
            'resources':resources,'updates':a.updates,'trace':a.trace,'probe':a.probe,'mask_probe':a.mask_probe,'original_base_modified':False,'config':config,
            'mode':a.mode,'implementation_hashes':implementation_hashes}
        if not (root/'LAUNCH.json').exists(): atomic_json(root/'LAUNCH.json',launch,immutable=True)
        elif production is not None: atomic_json(root/f'RESUME_{receipt["optimizer_step"]:06d}.json',launch,immutable=True)
    bind_source(contract['source_root'],contract['source_commit'])
    for name,expected in contract['query_model_code_hashes'].items():require(file_hash(name)==expected,'protected Query source changed: '+name)
    random.seed(config['seed']+rank);np.random.seed(config['seed']+rank);torch.manual_seed(config['seed'])
    model=build_query_framework(OmegaConf.create(contract['source_config']),contract['query_architecture'],contract['source_root'],contract['source_commit']).float().to(device)
    model.load_state_dict(state['model'],strict=True);model.train();configure_model(model,opts)
    optimizer=adam(model,cfg,contract,opts)
    if opts.execution_weights=='bf16_master' and not state['optimizer'].get('master_names'):
        optimizer.initialize_masters(state['model'])
    scheduler=scheduler_factory(optimizer,cfg)
    load_adam(optimizer,state['optimizer'],opts);scheduler.load_state_dict(state['scheduler'])
    params=[p for p in model.parameters() if p.requires_grad]
    wrapped=DDP(model,device_ids=[int(os.environ['LOCAL_RANK'])],find_unused_parameters=True,
                gradient_as_bucket_view=opts.gradient_as_bucket_view,bucket_cap_mb=opts.bucket_cap_mb)
    dataset=original_dataset(config,contract,allow_partial=False)
    by_id={x['token']:i for i,x in enumerate(dataset.index)};fit=state['fit_scene_ids']
    require(all(t in by_id for t in fit),'fit manifest/source mismatch')
    if production is not None:
        error=None
        try: dataset=production.attach_dataset(dataset, model, contract, fit)
        except Exception as ex: error=repr(ex)
        collective_check(error,device=device)
    from tools.ddpolicy_vehicle.training_state import epoch_batches
    from tools.foresight.student_state import optimizer_batch_counts
    from starVLA.dataloader.foresight_dataset import collate_training
    step=state['optimizer_step'];initial_step=step;epoch,offset=state['epoch'],state['offset']
    # Preserve provenance from the real parent; absence is unknown, never False.
    provenance={k:copy.deepcopy(state[k]) for k in ('learned_driving_weights_loaded','initialization_provenance','origin','base_provenance') if k in state}
    if production is not None: provenance.update(production.checkpoint_metadata(state, a.checkpoint, receipt['checksum']))
    restore_rng(state['rng_by_rank'][rank]);del state
    instrument=Instrumentation(a.trace,device);instrument.attach(model)
    from .numerics import NumericalProbe
    checkpoint_seconds=0.;last_saved_step=None
    def save_boundary():
        nonlocal checkpoint_seconds,last_saved_step
        began=time.perf_counter();rngs=[None]*world;dist.all_gather_object(rngs,capture_rng())
        error=None
        if rank==0:
            try:
                path=root/f'step_{step:06d}.pt'
                require(not path.exists(),'refusing checkpoint overwrite')
                atomic_torch(path,{**provenance,'kind':'iqe_query_base','framework_contract_hash':contract['framework_contract_hash'],
                    'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'optimizer_step':step,'epoch':epoch,'offset':offset,'dependencies':original_dependencies,
                    'rng_by_rank':rngs,'resume_boundary':'optimizer_step','fit_scene_ids':fit,'seed':config['seed'],
                    'mode':'full' if a.mode=='continuation' else 'performance_v2','performance_identity':identity,'parent_checkpoint':a.checkpoint,
                    'terminal':step==cfg['max_optimizer_steps'],'termination_reason':('MAX_STEPS' if step==cfg['max_optimizer_steps'] else ('OPERATOR_PAUSED' if stop[0] or step==initial_step+a.updates else 'RUNNING_CHECKPOINT'))})
                atomic_json(str(path)+'.COMPLETE.json',{'checksum':file_hash(path),'optimizer_step':step,'dependencies':original_dependencies})
            except Exception as ex:error=repr(ex)
        collective_check(error,device=device);checkpoint_seconds+=time.perf_counter()-began;last_saved_step=step
    if a.mode=='continuation':
        from ..storage import capacity_plan,require_capacity
        plan_storage=capacity_plan(root,Path(a.checkpoint).stat().st_size,a.checkpoint_every,step+a.updates,step)
        require_capacity(plan_storage)
        if rank==0:atomic_json(root/'CAPACITY_PLAN.json',plan_storage)
    cache=None
    if opts.vision_cache:
        from .vision_cache import FrozenVisionCache
        h=hashlib.sha256()
        for n,v in model.qwen_vl_interface.model.model.visual.state_dict().items():h.update(n.encode());h.update(tensor_hash(v).encode())
        cache_identity={'schema':1,'visual_weights_hash':h.hexdigest(),'tokenizer_hash':contract['tokenizer_hash'],'prompt_hash':contract['prompt_hash'],
            'transform_hash':contract['transform_hash'],'visual_precision':opts.visual_precision,'backend':backend_flags(),
            'views':contract['input']['views'],'augmentation':'original_no_augmentation','framework_source_hashes':contract['source_hashes'],
            'payload_source_hash':file_hash(Path(__file__).with_name('vision_cache.py')),
            'torch_version':torch.__version__,'transformers_version':__import__('transformers').__version__,
            'processor_source_hash':file_hash(inspect.getfile(type(model.qwen_vl_interface.processor))),
            'visual_source_hash':file_hash(inspect.getfile(type(model.qwen_vl_interface.model.model.visual))),
            'deepstack_count':3}
        cache=FrozenVisionCache(model,opts.vision_cache,cache_identity,rank,build=a.build_vision_cache,max_scenes=8)
    vision=production.vision(model) if production is not None else FrozenVisionAhead(model)
    probe=NumericalProbe(model,a.probe,root,rank)
    rows=[];stop=[False]
    previous=signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,with_stack=False,profile_memory=False) if a.trace else nullcontext()
    dist.barrier();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();begin=time.perf_counter()
    with profiler as prof,ThreadPoolExecutor(max_workers=config['execution']['loader_workers'] or 1) as workers:
        plan=EpochPlan(len(fit),32,config['seed'],rank,world,epoch_batches)
        def prepare(address):
            start=time.perf_counter()
            with instrument.region('CPU_read_decode_resize_collate'):
                samples=list(workers.map(dataset.__getitem__,[by_id[fit[i]] for i in address.mine]));obs,targets=collate_training(samples)
                valid=torch.arange(len(address.mine))<address.valid_count
                if a.mask_probe and rank==0 and len(address.entries)==32:valid[:]=False
                require(bool(torch.isfinite(targets['ego'][valid]).all()),'nonfinite valid GT')
                for k,v in targets.items():
                    if k.endswith('_valid') or k=='future_valid':v[~valid]=False
                targets['iqe_sample_valid']=valid
            return obs,targets,None,time.perf_counter()-start
        def prepare_vision(batch):
            obs,targets,_,seconds=batch
            return obs,targets,vision.prepare(obs,4),seconds
        addresses=plan.iter_from(epoch,offset,a.updates)
        if a.mask_probe:
            first=next(addresses);second=next(addresses);entry=second.entries[0]
            addresses=iter([first,BatchAddress(second.epoch,second.offset,(entry,),(entry,),int(rank==0))])
        with TrainingPrefetch(addresses,a.updates,prepare,prepare_vision,depth=1) as ahead:
            for index in range(a.updates):
                t=time.perf_counter();error=None
                try:address,batch=ahead.pop()
                except Exception as ex:error=repr(ex)
                collective_check(error,device=device)
                obs,targets,prepared,load_seconds=batch
                with instrument.region('H2D_and_global_denominators'):
                    targets['iqe_sample_valid']=targets['iqe_sample_valid'].to(device)
                    counts=optimizer_batch_counts(targets,torch.Generator().manual_seed(config['seed']+step*world+rank),device)
                    counts['ego_scenes']=len(address.entries)
                    if a.mask_probe:
                        valid_count=targets['iqe_sample_valid'].sum();dist.all_reduce(valid_count)
                        counts['ego_scenes']=int(valid_count)
                wait=time.perf_counter()-t
                optimizer.zero_grad(set_to_none=True)
                probe.capture_prepared(prepared)
                with vision.activate(prepared),instrument.region('all_forward_losses'):
                    result=wrapped(obs,targets,completed_updates=step,global_counts=counts)
                    collective_check(None if bool(torch.isfinite(result['loss'])) else 'nonfinite loss',device=device)
                with instrument.region('all_backward_including_recompute_DDP'):result['loss'].backward()
                with instrument.region('gradient_check_clip'):
                    collective_check(None if finite_gradients(params) else 'nonfinite gradient',device=device)
                    norm=torch.nn.utils.clip_grad_norm_(params,cfg['grad_clip_norm'])
                probe.before_step(optimizer,result)
                if production is not None and a.fingerprint and not rows:
                    production.capture_gradient(model)
                with instrument.region('optimizer_step'):optimizer.step()
                probe.after_step(optimizer)
                with instrument.region('scheduler_logging'):
                    scheduler.step();step+=1;epoch,offset=address.epoch,address.offset+1
                    logs={k:float(v.detach()) for k,v in result['losses'].items()}
                    flag=torch.tensor(int(stop[0]),device=device);dist.all_reduce(flag,op=dist.ReduceOp.MAX)
                    stop[0]=bool(flag)
                # Sync consumer only: prefetch on the producer stream remains overlapped.
                torch.cuda.current_stream().synchronize()
                elapsed=time.perf_counter()-t
                row={'optimizer_step':step,'seconds':elapsed,'wait_seconds':wait,'loader_seconds':load_seconds,
                    'losses':logs,'gradient_norm':float(norm),'global_batch':len(address.entries),'sample_ids':[fit[i] for i in address.mine],
                    'lr':scheduler.get_last_lr(),'used_parameter_names_hash':digest([n for n,p in model.named_parameters() if p.requires_grad and p.grad is not None]),
                    'valid_denominators':{k:float(v) if torch.is_tensor(v) else v for k,v in counts.items()},'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
                rows.append(row)
                if rank==0:print(json.dumps({'event':'update',**{k:v for k,v in row.items() if k!='sample_ids'}}),flush=True)
                if a.trace:prof.step()
                if a.mode=='continuation' and step % a.checkpoint_every==0:save_boundary()
                if stop[0]:break
    vision.close();loop_wall=time.perf_counter()-begin
    if cache:
        cache.finish();dist.barrier()
        if rank==0 and a.build_vision_cache:
            indexes={f'rank_{i}.json':file_hash(Path(opts.vision_cache)/f'rank_{i}.json') for i in range(world)}
            atomic_json(Path(opts.vision_cache)/'COMPLETE.json',{'identity':cache_identity,'rank_indexes':indexes},immutable=True)
    if a.trace:
        prof.export_chrome_trace(str(root/f'trace_rank{rank}.json'))
        (root/f'operators_rank{rank}.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=50))
        instrument.finish(root/f'modules_rank{rank}.json',model,optimizer)
    identities=state_hash(model,optimizer) if a.fingerprint else None
    rng=capture_rng();rngs=[None]*world;dist.all_gather_object(rngs,rng)
    if a.save and last_saved_step!=step:save_boundary()
    save_seconds=checkpoint_seconds
    from collections import Counter
    actual_dtypes=Counter()
    for n,p in model.named_parameters():
        actual_dtypes[f"{'trainable' if p.requires_grad else 'frozen'}_parameter/{p.dtype}"]+=p.numel()
        if p.grad is not None:actual_dtypes[f'gradient/{p.grad.dtype}']+=p.grad.numel()
    for m in getattr(optimizer,'masters',[]):actual_dtypes[f'master/{m.dtype}']+=m.numel()
    for values in optimizer.state.values():
        for n,v in values.items():
            if torch.is_tensor(v):actual_dtypes[f'adam/{n}/{v.dtype}']+=v.numel()
    report={'rank':rank,'actual_dtype_elements':dict(actual_dtypes),'rows':rows,'loop_wall_seconds':loop_wall,'checkpoint_seconds':save_seconds,'fingerprint':identities,
        'rng_python_hash':hashlib.sha256(repr(rng['python']).encode()).hexdigest(),'rng_numpy_hash':hashlib.sha256(repr(rng['numpy']).encode()).hexdigest(),
        'scheduler_state':scheduler.state_dict(),'data_cursor':{'epoch':epoch,'offset':offset},'rng_cpu_hash':tensor_hash(rng['cpu']),'rng_cuda_hashes':[tensor_hash(t) for t in rng['cuda']],
        'cache_timings':cache.timings if cache else None,'DDP':wrapped._get_ddp_logging_data(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
    if production is not None: report['production_cache'] = production.report(vision)
    atomic_json(root/f'rank_{rank}.json',report)
    dist.barrier()
    if rank==0:
        ranks=[read_json(root/f'rank_{r}.json') for r in range(world)]
        stop_at=len(rows)-1 # fixed final producer-drain exclusion, shared by every candidate
        values=np.array([[r['seconds'] for r in x['rows'][a.warmup:stop_at]] for x in ranks])
        slowest=values.max(0) if values.size else np.array([])
        stats=lambda v:{'n':len(v),'mean':float(np.mean(v)),'p50':float(np.quantile(v,.5)),'p95':float(np.quantile(v,.95))} if len(v) else None
        atomic_json(root/'RESULT.json',{'status':'COMPLETE' if len(rows)==a.updates else 'OPERATOR_PAUSED','training_terminal':step==cfg['max_optimizer_steps'],'identity':identity,'initial_step':initial_step,'final_step':step,'updates':len(rows),
            'warmup_excluded':a.warmup,'drain_excluded':1,'rank0':stats(values[0]) if values.size else None,'slowest_rank_per_step':stats(slowest),
            'slowest_mean_rank':int(values.mean(1).argmax()) if values.size else None,'samples_per_second':32/slowest.mean() if values.size else None,
            'peak_allocated_bytes':max(x['peak_allocated_bytes'] for x in ranks),'peak_reserved_bytes':max(x['peak_reserved_bytes'] for x in ranks),
            'loop_allocated_gpu_seconds':max(x['loop_wall_seconds'] for x in ranks)*world,'checkpoint_seconds':max(x['checkpoint_seconds'] for x in ranks),
            'trace_run_not_throughput':a.trace,'probe_enabled':a.probe,'mask_probe_not_throughput':a.mask_probe,'scientific_gain':'UNTESTED','new_full_training_started':a.mode=='continuation'})
    signal.signal(signal.SIGTERM,previous);dist.destroy_process_group()

if __name__=='__main__':main()
