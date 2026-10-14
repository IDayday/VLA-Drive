"""Specialist/Scorer cells on the existing IQE optimizer-boundary trainer."""
from dataclasses import asdict
from pathlib import Path
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import os,time
import numpy as np
import torch
from iqe.contracts import require
from iqe.io import read_json, atomic_json, atomic_torch, file_hash, digest, lock
from iqe.evaluation.retention import module_hash, module_hash_parameters, module_hash_buffers
from iqe.training.trainer import train
from iqe.losses import expert_il_terms
from iqe.internal_baseline.bank import save_candidate, score_candidates, COMPONENTS
from iqe.internal_baseline.scorer import initialization_rng
from .data import ReusedContext, verified_base_scores
from .binding import scenes_by_role
from .models import make_specialist, EMAExpert, StudyScorer
from .objectives import SceneSampler, StudySampler
from .metrics import metrics, select


def runtime(device):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    require(os.environ['CUBLAS_WORKSPACE_CONFIG']==':4096:8','deterministic head CUBLAS identity')
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.set_num_threads(2)
    if str(device).startswith('cuda'):
        index=torch.device(device).index or 0;os.environ['LOCAL_RANK']=str(index);torch.cuda.set_device(index)
    return str(device)


def preload(config,scenes):
    first=ReusedContext(config,capacity=0).read(scenes[0])
    size=sum(t.numel()*t.element_size() for t in first.values())+16384
    capacity=max(1,min(len(scenes),config.feature_memory_gib*1024**3//size))
    cache=ReusedContext(config,capacity=capacity)
    if capacity==len(scenes):
        def one(s):return ReusedContext(config,capacity=0).read(s)
        # One verified reader per thread; no shared mutable LRU.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for s,x in zip(scenes,pool.map(one,scenes)):cache.cache.loaded[s.scene_id]=x
    return cache


def trainer_config(config,kind):
    from .recipes import load
    expert=kind=='expert';recipe=load(config,'pilot_expert_2x2' if expert else 'scorer_2x2')
    require(recipe['max_optimizer_steps']==(config.expert_updates if expert else config.scorer_updates)
            and recipe['global_batch_size']==config.global_batch,'recipe exceeds fixed study budget')
    keys=('lr','weight_decay','global_batch_size','max_optimizer_steps','warmup_steps','lr_min_ratio',
          'grad_clip_norm','schedule','validation_steps','checkpoint_every_steps')
    return {k:recipe[k] for k in keys}|{'micro_batch_size':config.micro_batch,'validation_every_steps':1000,'early_stop_patience':10**9}


@torch.no_grad()
def candidate_labels(config,module,scenes,expert_id,device,cache=None):
    cache=cache or ReusedContext(config)
    module.eval();identity=module_hash(module);candidates=[]
    for start in range(0,len(scenes),32):
        sub=scenes[start:start+32];values=cache.batch(sub,device)
        physical=module(values).physical.cpu().numpy()
        with ThreadPoolExecutor(max_workers=4) as pool:
            candidates+=list(pool.map(lambda pair:save_candidate(config,pair[0],expert_id,identity,pair[1]),zip(sub,physical)))
    scores=score_candidates(config,scenes,candidates,expert_id+'_'+identity[:12])
    return candidates,scores


def matrix_manifest(config,cell):
    matrix=read_json(config.root/'EXPERT_MATRIX_DATA.json')
    if cell in matrix['cells']:
        spec=matrix['cells'][cell]
        require(spec.get('status','READY')=='READY',spec.get('status'))
        return matrix,spec
    extension=config.root/'extension_data'/(cell+'.json')
    require(extension.is_file(),'unknown expert cell')
    manifest=read_json(extension);return manifest,manifest['cell']


def train_expert(config,cell,device='cuda:0',resume=None,stop_after=None):
    require(cell in ('E00','E10','E01','E11','EXT_interaction','EXT_road_direction','EXT_safe_progress'),'unknown specialist cell')
    device=runtime(device);folder=config.root/'experts'/cell
    with lock(folder/'RUN.lock'):
        existing=[p for p in (config.root/'experts').glob('*/RUN_IDENTITY.json')]
        require(len(existing)<config.max_expert_runs or (folder/'RUN_IDENTITY.json').exists(),'six expert run budget exhausted')
        manifest,spec=matrix_manifest(config,cell)
        roles=scenes_by_role(config);lookup={s.scene_id:s for rows in roles.values() for s in rows}
        ids=manifest['scene_ids'];allowed={s.scene_id for s in roles['expert_fit']}
        require(set(ids)<=allowed and ids,'D_S/evaluation leaked into specialist fit')
        train_scenes=[lookup[i] for i in ids];cache=preload(config,train_scenes)
        raw=make_specialist(config,spec['private_reader']).to(device)
        base=make_specialist(config,False).to(device).eval().requires_grad_(False)
        old={'params':module_hash_parameters(base),'buffers':module_hash_buffers(base)}
        probe=cache.batch(train_scenes[:4],device)
        with torch.no_grad():
            raw.eval();before=base(probe);clone=raw(probe)
        tolerance={'raw_max_abs':1e-5,'physical_max_abs':1e-4}
        error={'raw_max_abs':float((before.raw-clone.raw).abs().max()),'physical_max_abs':float((before.physical-clone.physical).abs().max())}
        require(all(error[k]<=tolerance[k] for k in error),'Base-copy private-reader/output alignment failed')
        require(all(a.data_ptr()!=b.data_ptr() for a,b in zip(base.expert.parameters(),raw.expert.parameters())),'Base/private storage alias')
        from .recipes import load
        recipe=load(config,'pilot_expert_2x2');require(recipe['anchor_ratio']==0,'pure specialist supervision required')
        wrapper=EMAExpert(raw,recipe['ema_decay'])
        sampler=SceneSampler(ids,config.seed,manifest['hash'])
        target=None
        if spec['target']=='offline_optimized':
            with np.load(Path(config.optimized_root)/'optimized_trajectories.npz') as f:
                target={str(i):torch.from_numpy(t.copy()).float() for i,t in zip(f['tokens'],f['trajectories']) if str(i) in set(ids)}
            require(set(target)==set(ids),'optimized target missing from paired manifest')
        contract=read_json(config.root/'OFFLINE_BASE_CONTRACT.json')
        def fetch(entries,dev):
            rows=[lookup[e['scene_id']] for e in entries];x=cache.batch(rows,dev)
            y=x['target_raw'] if target is None else raw.trajectory_contract.raw(torch.stack([target[s.scene_id] for s in rows]).to(dev))
            return {'x':x,'target':y,'mask':torch.tensor([not e.get('_padding',False) for e in entries],device=dev)[:,None].expand(-1,y.shape[1])}
        def loss(m,b):return expert_il_terms(m(b['x']),b['target'],masks=b['mask'],prev_weight=contract['query_architecture']['prev_weight'])
        val=roles['stage_val'];base_labels=verified_base_scores(config,val);hv=set(manifest['hard_val'])
        def validate(m,step):
            reports={}
            for endpoint in ('raw','ema'):
                module=getattr(m,endpoint)
                candidates,scores=candidate_labels(config,module,val,cell+'_'+endpoint,device)
                truth=np.asarray([[base_labels[s.scene_id]['total_score_01'],r.total_score_01] for s,r in zip(val,scores)])
                components=np.asarray([[[base_labels[s.scene_id]['named_components'][c]
                    if base_labels[s.scene_id]['component_valid_masks'].get(c,False) else np.nan for c in COMPONENTS],
                    [r.named_components[c] if r.component_valid_masks.get(c,False) else np.nan for c in COMPONENTS]]
                    for s,r in zip(val,scores)])
                stats=metrics(truth,components,truth.argmax(1),[s.source_log_id for s in val],seed=config.bootstrap_seed,repetitions=config.bootstrap_repetitions)
                hard=np.array([s.scene_id in hv for s in val]);delta=truth[:,1]-truth[:,0]
                stats.update(hard_val_count=int(hard.sum()),hard_val_direct_gain=float(delta[hard].mean()) if hard.any() else None,
                             ordinary_direct_gain=float(delta[~hard].mean()) if (~hard).any() else None,
                             candidate_hash=module_hash(module),checkpoint_endpoint=endpoint)
                # Strict comparison retains the earlier step and raw on an exact raw/EMA tie.
                stats['selection_key']=[stats['oracle_gain_01'],stats['hard_val_direct_gain'] if hard.any() else -1.]
                atomic_torch(folder/f'{endpoint}_{step:06d}.pt',module.state_dict(),immutable=True)
                atomic_json(folder/f'{endpoint}_{step:06d}.pt.COMPLETE.json',{'checksum':file_hash(folder/f'{endpoint}_{step:06d}.pt'),
                        'step':step,'endpoint':endpoint,'model_hash':stats['candidate_hash']},immutable=True)
                reports[endpoint]=stats
            winner=max(('raw','ema'),key=lambda e:reports[e]['selection_key'])
            return {'selection_key':reports[winner]['selection_key'],'selected_endpoint':winner,'endpoints':reports,
                    'ema_updates':int(m.ema_updates),'ema_initial_Base_weight':m.decay**int(m.ema_updates)}
        deps={'campaign':file_hash(config.root/'CAMPAIGN_BINDING.json'),'data':manifest['hash'],'cell':cell,
              'cell_spec':digest(spec),'kind':'specialist_raw_ema','source_code':file_hash(__file__)}
        atomic_json(folder/'RUN_IDENTITY.json',{'dependencies':deps,'training':trainer_config(config,'expert'),
                  'private_parameters':sum(p.numel() for p in raw.parameters()),'initialization_error':error,
                  'initialization_tolerance_predeclared':tolerance,'target':spec['target'],'Base_unchanged':old},immutable=True)
        train(wrapper,loss,fetch,sampler,trainer_config(config,'expert'),deps,folder,device=device,resume=resume,stop_after=stop_after,
              validate=validate,seed=config.seed,denominator_function=lambda b:{'il':b['mask'].any(-1).sum().float()},
              after_optimizer_step=lambda m,step:m.update())
        with torch.no_grad():after=base(probe)
        audit={'before':old,'after':{'params':module_hash_parameters(base),'buffers':module_hash_buffers(base)},
               'raw_drift':float((after.raw-before.raw).abs().max()),'physical_drift':float((after.physical-before.physical).abs().max()),
               'VLM_calls':0,'shared_condition_updated':False,'private_reader_updated':spec['private_reader']}
        require(audit['before']==audit['after'] and audit['raw_drift']==audit['physical_drift']==0,'frozen Base changed')
        atomic_json(folder/'BASE_RETENTION_AUDIT.json',audit)
        return read_json(folder/'result.json')
