"""Bounded, resumable small-module optimization shared by G1/EGR/IEG.

Only private trainable tensors are saved. Frozen Base/teacher dependencies remain
immutable, externally checksummed artifacts in identity. A complete checkpoint
precedes RESULT; terminal recovery repairs this interruption window with no step.
"""
from __future__ import annotations
from pathlib import Path
import io, math, os, random, signal, time, hashlib
import numpy as np
import torch
from .common import digest, artifact, verify_artifact, read_json, write_json, atomic_bytes, lock


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu'])
    if state['cuda']:
        if len(state['cuda'])!=torch.cuda.device_count():raise ValueError('Exact resume requires same CUDA topology')
        torch.cuda.set_rng_state_all(state['cuda'])


def private_state(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}


def load_private(model,state):
    expected={n:p for n,p in model.named_parameters() if p.requires_grad}
    if set(expected)!=set(state):raise ValueError('Private trainable parameter names changed')
    with torch.no_grad():
        for n,p in expected.items():
            x=state[n]
            if x.shape!=p.shape or x.dtype!=p.dtype or not torch.isfinite(x).all():
                raise ValueError('Private checkpoint tensor mismatch: '+n)
            p.copy_(x.to(p.device))


def _state(obj):
    return obj.get_state() if isinstance(obj,torch.Generator) else obj.state_dict()


def _restore(obj,state):
    if isinstance(obj,torch.Generator):obj.set_state(state)
    else:obj.load_state_dict(state)


def resolve_checkpoint(output,identity_hash):
    valid=[];rejected=[]
    for receipt in sorted(Path(output).glob('step_*.COMPLETE.json')):
        try:
            r=read_json(receipt);p=verify_artifact(r['checkpoint'])
            if r['run_identity']!=identity_hash:raise ValueError('foreign run')
            c=torch.load(p,map_location='cpu',weights_only=False)
            if c['run_identity']!=identity_hash or c['optimizer_step']!=r['optimizer_step']:
                raise ValueError('checkpoint/receipt identity or step')
            valid.append((c['optimizer_step'],bool(c['terminal']),r.get('created_unix_ns',0),p,c))
        except Exception as e:rejected.append({'receipt':str(receipt),'reason':str(e)})
    if not valid:raise ValueError('No complete compatible checkpoint; refusing automatic restart')
    valid.sort(key=lambda x:(x[0],x[1],x[2],str(x[3])))
    step,_,_,path,state=valid[-1]
    write_json(Path(output)/'RESUME_RESOLUTION.json',{'selected':str(path),'optimizer_step':step,
        'rejected':rejected,'run_identity':identity_hash,'exact_resume':'same-runtime required'},immutable=False)
    state['_resolved_artifact']=artifact(path)
    return state


def run_training(model, *, output, identity, dataset_size, step_fn, validation_fn=None,
                 cfg, device, seed, max_updates, stop_after=None, resume=False, stateful=None):
    if type(dataset_size) is not int or dataset_size<1:raise ValueError('No valid training scenes')
    if not 1<=max_updates<=cfg['training']['max_updates']:raise ValueError('Explicit bounded training updates required')
    if stop_after is not None and not 1<=stop_after<=max_updates:raise ValueError('Invalid bounded stop_after')
    # This campaign uses one declared FP32 research backend. Keep input/output
    # precision and small-head resume comparisons independent of process defaults.
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    output=Path(output);output.mkdir(parents=True,exist_ok=True);t=cfg['training'];stateful=stateful or {}
    if 'source_commit' not in identity:raise ValueError('Source snapshot identity required')
    identity={**identity,'schema':'action_effect_small_training_v1','seed':seed,'dataset_size':dataset_size,
              'max_updates':max_updates,'recipe':t,'private_names':sorted(private_state(model)),
              'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'device':str(device),
                 'gpu_name':torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
                 'deterministic':torch.are_deterministic_algorithms_enabled(),
                 'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,
                 'cudnn_benchmark':torch.backends.cudnn.benchmark,
                 'trainable_dtype':'float32','autocast':'disabled'}}
    ih=digest(identity);write_json(output/'IDENTITY.json',identity)
    params=[p for p in model.parameters() if p.requires_grad]
    if not params or any(p.dtype!=torch.float32 for p in params):raise ValueError('Private parameters must be FP32')
    optimizer=torch.optim.AdamW(params,lr=t['learning_rate'],weight_decay=t['weight_decay'])
    warmup=min(t['warmup_updates'],max_updates)
    def rate(n):
        if n<warmup:return float(n+1)/warmup
        return .1+.9*.5*(1+math.cos(math.pi*min(1.,(n-warmup)/max(1,max_updates-warmup))))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,rate)
    sampler=torch.Generator().manual_seed(seed+101);step=0;consumed=0;history=[]
    terminal=None;paused=[False];started=time.perf_counter();saved=[]
    def save(reason):
        state={'schema':'action_effect_private_checkpoint_v1','run_identity':ih,'optimizer_step':step,
            'model':private_state(model),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'sampler':sampler.get_state(),'consumed_scenes':consumed,'rng':rng_state(),
            'stateful':{k:_state(v) for k,v in stateful.items()},'termination_reason':reason,
            'terminal':reason in ('MAX_UPDATES','EARLY_STOPPED'),'history':history}
        buf=io.BytesIO();torch.save(state,buf);payload=buf.getvalue();path=output/f'step_{step:06d}.pt'
        # Never overwrite an orphan/partial save or a completed periodic state
        # when a pause is requested at the same optimizer boundary.
        if path.exists() and path.read_bytes()!=payload:
            path=output/f'step_{step:06d}.{hashlib.sha256(payload).hexdigest()[:16]}.pt'
        atomic_bytes(path,payload)
        receipt_path=path.with_suffix('.COMPLETE.json')
        if receipt_path.exists():
            r=read_json(receipt_path);verify_artifact(r['checkpoint']);saved.append(r);return r
        r={'checkpoint':artifact(path),'optimizer_step':step,'run_identity':ih,
            'termination_reason':reason,'terminal':state['terminal'],'created_unix_ns':time.time_ns()}
        write_json(receipt_path,r);saved.append(r)
        return r
    with lock(output/'RUN.lock'):
        if resume:
            c=resolve_checkpoint(output,ih);load_private(model,c['model']);optimizer.load_state_dict(c['optimizer'])
            scheduler.load_state_dict(c['scheduler']);sampler.set_state(c['sampler']);step=c['optimizer_step']
            consumed=c['consumed_scenes'];history=c['history']
            if set(c['stateful'])!=set(stateful):raise ValueError('Stateful RNG/schema changed')
            for k,v in stateful.items():_restore(v,c['stateful'][k])
            restore_rng(c['rng']) # Last initialization operation; never seed again after resume.
            if c['terminal']:terminal=c['termination_reason']
        else:
            if list(output.glob('step_*.pt')):raise ValueError('Existing progress requires --resume')
            random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed)
            if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        oldhandler=signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM,lambda *_:paused.__setitem__(0,True))
        segment_updates=0
        try:
            if terminal is None:
                model.train()
                while step<max_updates:
                    if paused[0]:break
                    indices=torch.randint(dataset_size,(t['batch_scenes'],),generator=sampler).tolist()
                    begin=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                    stats=step_fn(model,indices,step)
                    if not isinstance(stats,dict):raise TypeError('step_fn must return numerical statistics after backward')
                    grads=[p.grad for p in params if p.grad is not None]
                    if not grads or any(not torch.isfinite(g).all() for g in grads):
                        raise FloatingPointError('Missing/nonfinite private gradients')
                    norm=torch.nn.utils.clip_grad_norm_(params,t['gradient_clip'],error_if_nonfinite=True)
                    optimizer.step();scheduler.step();step+=1;segment_updates+=1;consumed+=len(indices)
                    if str(device).startswith('cuda'):torch.cuda.synchronize(device)
                    row={'optimizer_step':step,'consumed_scenes':consumed,'losses':stats,
                         'gradient_norm':float(norm),'seconds':time.perf_counter()-begin,'lr':scheduler.get_last_lr()[0]}
                    history.append(row)
                    if step==1 or step%25==0:print(row,flush=True)
                    if validation_fn is not None and (step%t['validation_every']==0 or step==max_updates):
                        rng=rng_state();model.eval()
                        try:
                            with torch.no_grad():row['validation']=validation_fn(model,step)
                        finally:model.train();restore_rng(rng)
                    reason='MAX_UPDATES' if step==max_updates else 'RUNNING_CHECKPOINT'
                    if step==max_updates or paused[0] or (stop_after is not None and segment_updates>=stop_after):
                        reason=reason if step==max_updates else 'OPERATOR_PAUSED'
                        save(reason);terminal=reason;break
                    if step%t['validation_every']==0:save(reason)
                if terminal is None:
                    terminal='MAX_UPDATES' if step>=max_updates else 'OPERATOR_PAUSED'
                    save(terminal)
        finally:signal.signal(signal.SIGTERM,oldhandler)
        result={'schema':'action_effect_training_result_v1','run_identity':ih,'optimizer_step':step,
            'status':'COMPLETE' if terminal in ('MAX_UPDATES','EARLY_STOPPED') else 'PAUSED',
            'termination_reason':terminal,'segment_updates':segment_updates,'consumed_scenes':consumed,
            'elapsed_seconds':time.perf_counter()-started,'private_parameters':sum(p.numel() for p in params),
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else 0,
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(device) if str(device).startswith('cuda') else 0,
            'checkpoint':saved[-1] if saved else c['_resolved_artifact']}
        write_json(output/'RESULT.json',result,immutable=False);return result
