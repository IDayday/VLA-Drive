"""Deterministic padded exposure, complete RNG resume, explicit deployment/benchmark contracts."""
import math
import random
import numpy as np
import torch
from torch.utils.data import Sampler


def load_config(path):
    from pathlib import Path
    from omegaconf import OmegaConf
    visited=set()
    def unresolved(location):
        location=Path(location).resolve()
        if location in visited:raise ValueError('Cyclic V2 config inheritance')
        visited.add(location)
        child=OmegaConf.load(location)
        parent=child.pop('inherits',None)
        return OmegaConf.merge(unresolved(location.parent/parent),child) if parent else child
    return OmegaConf.to_container(unresolved(path),resolve=True)


def source_fingerprint():
    """Report-only commits do not alter this production/config/test identity."""
    import hashlib,subprocess
    from pathlib import Path
    root=Path(__file__).resolve().parents[4]
    files=subprocess.check_output(['git','ls-files','-z','navsim/agents/EpisodeDrive',
        'navsim/planning/script/config/common/agent/planreg_wm_v2*','scripts','tests','local_planreg_wm_v2'],cwd=root).decode().split('\0')
    hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in sorted(files) if p and (root/p).is_file() and Path(p).suffix in ('.py','.yaml','.sh')}
    identity=hashlib.sha256('\n'.join(n+':'+h for n,h in hashes.items()).encode()).hexdigest()
    return dict(sha256=identity,files=hashes)


class ExactExposureSampler(Sampler):
    def __init__(self,size,global_batch,world,rank,seed,epoch=0,start_step=0):
        if size <= 0 or global_batch % world: raise ValueError('Invalid exact exposure layout')
        self.steps = math.ceil(size/global_batch)
        generator = torch.Generator().manual_seed(seed+epoch)
        order = torch.randperm(size,generator=generator).tolist()
        padded = (order*math.ceil(self.steps*global_batch/size))[:self.steps*global_batch]
        self.indices = padded[start_step*global_batch:][rank::world]
        self.padding_count = self.steps*global_batch-size

    def __iter__(self): return iter(self.indices)
    def __len__(self): return len(self.indices)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'])
    if state['cuda']: torch.cuda.set_rng_state_all(state['cuda'])


def prepare_run_directory(path,resume=False):
    """Rank zero owns creation; all ranks receive the same failure instead of racing mkdir."""
    from pathlib import Path
    import torch.distributed as dist
    distributed=dist.is_available() and dist.is_initialized()
    rank=dist.get_rank() if distributed else 0
    error=[None]
    if rank==0:
        try:
            if Path(path).exists() and not resume:
                raise FileExistsError('New run output directory required; no automatic resume')
            Path(path).mkdir(parents=True,exist_ok=resume)
        except OSError as exc:error[0]=str(exc)
    if distributed:dist.broadcast_object_list(error,src=0)
    if error[0] is not None:raise FileExistsError(error[0])
    if distributed:dist.barrier()


def accumulated_batches(iterator,accumulate):
    """Count the whole optimizer batch before micro-forward; missing targets must not bias accumulation."""
    from itertools import islice
    import torch.distributed as dist
    while True:
        group=list(islice(iterator,accumulate))
        if not group:return
        if len(group)!=accumulate:raise ValueError('Sampler did not pad to a complete accumulated optimizer batch')
        counts=torch.zeros(8,dtype=torch.float32);error=None
        try:
            for _,target in group:
                times,valid=target['motion_timestamps'],target['motion_valid']
                from .motion import interval_selection,validate_motion_inputs
                validate_motion_inputs(times,valid,target.get('motion_sequence'))
                _,cover=interval_selection(times,valid)
                actions=cover.long().cumprod(-1).bool()
                future=target['future_valid_mask']
                ro=future&actions
                tf=ro.clone();tf[:,1:]&=future[:,:-1].long().cumprod(-1).bool()
                counts[:3]+=tf.sum(0);counts[3:6]+=ro.sum(0)
                counts[6]+=target['trajectory_valid'].any(-1).sum()
                counts[7]+=target['trajectory_long_valid'].sum()
        except ValueError as exc:error=str(exc)
        if dist.is_available() and dist.is_initialized():
            errors=[None]*dist.get_world_size();dist.all_gather_object(errors,error)
            if any(e is not None for e in errors):raise ValueError('Synchronized optimizer-batch data validation: '+str(errors))
            if dist.get_backend()=='nccl':counts=counts.cuda()
            dist.all_reduce(counts)
        elif error is not None:raise ValueError(error)
        context=dict(tf=counts[:3],ro=counts[3:6],trajectory=counts[6],long=counts[7],accumulate=accumulate)
        for features,target in group:yield features,target,context


def validate_formal(config,manifest,layout):
    import json
    from pathlib import Path
    from . import ARCHITECTURE_VERSION,RECIPE_VERSION,SCHEDULE_VERSION,LONG_TARGET_VERSION,CACHE_SCHEMA,NORMALIZER_SCHEMA
    if (config.get('architecture_version'),config.get('recipe_version'),config.get('schedule_version'))!=(ARCHITECTURE_VERSION,RECIPE_VERSION,SCHEDULE_VERSION):
        raise ValueError('Old recipe requires explicit warm start, never silent full resume')
    if (manifest.get('schema'),manifest.get('long_target_version'))!=(CACHE_SCHEMA,LONG_TARGET_VERSION):
        raise ValueError('Recompute progressive-long labels in a new versioned cache')
    if config.get('world_model_enabled') is not True and config.get('variant') != 'no_wm':
        raise ValueError('Formal Base/VQA both require world model; no-WM must be an explicitly labeled control')
    if manifest['split'] != 'trainval_final_fit' or len(manifest['records']) != 103288 or manifest.get('smoke'):
        raise ValueError('Formal run requires the complete 103,288-scene final-fit manifest')
    if config['epochs'] != 27 or config['checkpoint_path'] is not None:
        raise ValueError('Formal endpoint is 27 epochs with no M0 initialization')
    if layout.get('architecture_version') != config['architecture_version'] or not layout.get('passed'):
        raise ValueError('V2 must be profiled; V1 throughput evidence cannot authorize this layout')
    if layout['global_batch'] != config['global_batch']:
        raise ValueError('Layout global batch mismatch')
    if (layout.get('recipe_version')!=RECIPE_VERSION or layout.get('source_fingerprint_sha256')!=source_fingerprint()['sha256'] or
            not layout.get('profile_only') or not layout.get('hardware') or layout.get('profile_no_external_gpu_work') is not True):
        raise ValueError('Layout must bind actual V2.2 code/recipe/GB128/hardware and uncontended profiling evidence')
    if not config.get('shared_init_path'): raise ValueError('Shared V2 trainable initialization required')
    stats=json.loads(Path(config['normalizer_path']).read_text())['metadata']
    if (stats['count'],stats['split'],stats['token_sha256']) != (
            103288,'trainval_final_fit',manifest['token_sha256']):
        raise ValueError('Formal normalizer must be measured on this exact unique final-fit token manifest, not smoke/held-out data')
    if (stats.get('schema')!=NORMALIZER_SCHEMA or stats.get('raw_gt_sha256')!=manifest.get('raw_gt_sha256') or
            not stats.get('raw_gt_sha256') or stats.get('statistics_contract')!=manifest.get('statistics_contract')):
        raise ValueError('Statistics reuse requires matching raw GT content and normalization contract, not just token count')


def validate_profile_artifact(report,metadata,config):
    from . import ARCHITECTURE_VERSION,RECIPE_VERSION
    if (not report.get('profile_only') or not metadata.get('profile_only') or
            metadata.get('global_batch')!=128 or report.get('profile',{}).get('timed_optimizer_steps')!=8 or
            report.get('profile',{}).get('warmup_optimizer_steps')!=4):
        raise ValueError('Need actual profile_only GB128 with four warmup and eight timed optimizer steps')
    if config.get('architecture_version')!=ARCHITECTURE_VERSION or config.get('recipe_version')!=RECIPE_VERSION:
        raise ValueError('Profile recipe/architecture mismatch')
    if not config.get('world_model_enabled') or config.get('scene_memory_mode')!='per_tile_register_memory' or not report.get('fp32_trainable'):
        raise ValueError('Profile may not disable rich memory or world model or FP32 trainable storage')
    if metadata.get('profile_no_external_gpu_work') is not True:
        raise ValueError('Contended GPUs cannot authorize a throughput layout')
    if report['peak_allocated_gib']>=72 or not all(report['horizons_valid']):raise ValueError('Memory/future horizon gate failed')
    if any(not math.isfinite(r['loss']) or not math.isfinite(r['grad_norm']) for r in report['records']):
        raise ValueError('Nonfinite profile loss/gradient')
    if metadata.get('source_fingerprint',{}).get('sha256')!=source_fingerprint()['sha256']:
        raise ValueError('Profile belongs to a different source snapshot')


def validate_ttc_reduction(scores,accumulate=1):
    """Exact legacy loss is safe for this formal all-valid label protocol.

    Sentinel masking remains untouched in the scorer core. Before introducing
    sentinel-bearing distributed/accumulated labels, implement optimizer-batch
    TTC valid denominators; until then fail collectively, never bias or deadlock.
    """
    import torch.distributed as dist
    distributed=dist.is_available() and dist.is_initialized()
    bad=((scores[...,3]==2).any() & torch.tensor(distributed or accumulate>1,device=scores.device))
    bad=bad.to(torch.int32)
    if distributed:dist.all_reduce(bad,op=dist.ReduceOp.MAX)
    if bad.item():
        raise ValueError('TTC sentinel2 requires whole-optimizer-batch valid reduction; synchronized rejection on every rank')


def same_batch_gradient_audit(plan_loss,wm_loss,named_parameters,weight):
    selected = [(n,p) for n,p in named_parameters if p.requires_grad and
                any(part in n for part in ('.vision_model.','.planning_register_adapter.','.language_model.','.task_queries.'))]
    parameters = [p for _,p in selected]
    plan = torch.autograd.grad(plan_loss,parameters,retain_graph=True,allow_unused=True)
    wm = torch.autograd.grad(wm_loss,parameters,retain_graph=True,allow_unused=True)
    results = {}
    for group,match in (('vision_lora',lambda n:'.vision_model.' in n),
                        ('registers',lambda n:'.planning_register_adapter.' in n),
                        ('language_lora',lambda n:'.language_model.' in n),
                        ('semantic_queries',lambda n:'.task_queries.' in n)):
        aa=bb=ab=0.
        for (name,p),g,h in zip(selected,plan,wm):
            if not match(name): continue
            if g is not None: aa += float(g.detach().float().square().sum())
            if h is not None: bb += float(h.detach().float().square().sum())*weight*weight
            if g is not None and h is not None: ab += float((g.detach().float()*h.detach().float()).sum())*weight
        results[group] = dict(plan_norm=math.sqrt(aa),weighted_wm_norm=math.sqrt(bb),
                              cosine=ab/max(1e-30,math.sqrt(aa*bb)))
    return results


def component_gradient_audit(losses,named_parameters):
    """One graph/batch, three losses, before clipping; autograd.grad never writes .grad."""
    selected=[(n,p) for n,p in named_parameters if p.requires_grad and any(
        k in n for k in ('.vision_model.','.planning_register_adapter.','.language_model.','.task_queries.'))]
    if not selected:raise ValueError('No shared V2 parameters for component gradient audit')
    objectives={'trajectory':losses['trajectory_loss'],'scorer':losses['scorer_loss']}
    if 'wm_loss' in losses:objectives['weighted_wm']=losses['wm_loss']*losses['wm_weight']
    gradients={key:torch.autograd.grad(value,[p for _,p in selected],retain_graph=True,allow_unused=True)
               for key,value in objectives.items()}
    report={}
    for group,part in [('vision_lora','.vision_model.'),('registers_readout','.planning_register_adapter.'),
                       ('language_lora','.language_model.'),('semantic_queries','.task_queries.')]:
        dots={}
        for first,g in gradients.items():
            for second,h in gradients.items():
                dots[(first,second)]=sum(float((a.detach().float()*b.detach().float()).sum())
                    for (name,_),a,b in zip(selected,g,h) if part in name and a is not None and b is not None)
        norms={name:math.sqrt(max(0.,dots[name,name])) for name in gradients}
        report[group]=dict(norms=norms,cosines={a+'__'+b:dots[a,b]/max(1e-30,norms[a]*norms[b])
            for i,a in enumerate(gradients) for b in list(gradients)[i+1:]})
    return report
