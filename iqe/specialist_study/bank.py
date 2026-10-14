"""Immutable official-score banks for one fixed deployment pool and its rich fit data."""
from dataclasses import asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import io,time
import numpy as np
import torch
from iqe.contracts import require,CandidateRecord,ScoreRecord,strict_record
from iqe.io import read_json,atomic_json,atomic_bytes,file_hash,digest
from iqe.internal_baseline.bank import save_candidate,score_candidates,COMPONENTS,protocol
from iqe.evaluation.retention import module_hash
from .binding import scenes_by_role
from .data import ReusedContext,admissible_optimized_source,geometry
from .experiments import load_endpoint,endpoint
from .training import candidate_labels


def pool(config):
    p=read_json(config.root/'SERVING_CANDIDATE_POOL.json')
    require(p['hash']==digest({k:v for k,v in p.items() if k!='hash'}),'candidate pool hash')
    require(p['Base_binding_hash']==file_hash(config.root/'BASE_BINDING.json'),'pool Base mismatch')
    return p


def canonical_comparison_mask(trajectories,valid,scores):
    mask=np.asarray(valid).copy()
    for row in range(len(mask)):
        seen={}
        for k in np.flatnonzero(mask[row]):
            key=np.ascontiguousarray(trajectories[row,k]).tobytes()
            if key in seen:
                require(abs(float(scores[row,k]-scores[row,seen[key]]))<=1e-8,'identical trajectory official labels changed')
                mask[row,k]=False
            else:seen[key]=int(k)
    return mask


def directory(config,kind,role):
    require(kind in ('deploy','rich') and role in ('scorer_fit','stage_val','selector_cal','dev_report'),'bank role/kind')
    require(kind!='rich' or role=='scorer_fit','rich supervision only permitted in D_S')
    return config.root/'banks'/kind/role


def save_bank(config,kind,role,scenes,rows,provenance):
    p=pool(config);folder=directory(config,kind,role);count=max(len(row) for row in rows);proto=protocol(config)
    horizon=read_json(config.root/'OFFLINE_BASE_CONTRACT.json')['trajectory']['horizon']
    trajectories=np.zeros((len(scenes),count,horizon,3),np.float32)
    scores=np.zeros((len(scenes),count),np.float32);valid=np.zeros_like(scores,dtype=bool)
    components=np.full((*scores.shape,len(COMPONENTS)),np.nan,np.float32);cv=np.zeros_like(components,dtype=bool)
    refs=[]
    for n,(scene,row) in enumerate(zip(scenes,rows)):
        references=[]
        for k,(candidate,label) in enumerate(row):
            require(candidate.scene_id==scene.scene_id and label.candidate_key==candidate.key,'scene/candidate score join')
            require(file_hash(candidate.trajectory_ref)==candidate.trajectory_hash,'trajectory checksum')
            require(label.score_valid and label.metric_context_hash==scene.metric_context_hash and label.score_protocol_hash==proto['protocol_hash'],'official label incomplete or stale')
            with np.load(candidate.trajectory_ref) as f:trajectory=f['trajectory'].copy()
            require(np.isfinite(trajectory).all(),'invalid physical candidate is not zero score')
            trajectories[n,k]=trajectory;scores[n,k]=label.total_score_01;valid[n,k]=True
            for j,c in enumerate(COMPONENTS):
                cv[n,k,j]=label.component_valid_masks[c]
                components[n,k,j]=label.named_components[c] if cv[n,k,j] else np.nan
            references.append({'candidate':asdict(candidate),'label':asdict(label)})
        refs.append(references)
    comparison_valid=canonical_comparison_mask(trajectories,valid,scores)
    data=io.BytesIO();np.savez(data,trajectories=trajectories,scores=scores,valid=valid,components=components,component_valid=cv,comparison_valid=comparison_valid)
    atomic_bytes(folder/'arrays.npz',data.getvalue(),immutable=True)
    metadata={'schema':'variable_K_official_research_bank_v1','kind':kind,'role':role,'scene_ids':[s.scene_id for s in scenes],
        'source_logs':[s.source_log_id for s in scenes],'pool_hash':p['hash'],'Base_binding_hash':file_hash(config.root/'BASE_BINDING.json'),
        'score_protocol_hash':proto['protocol_hash'],'array_hash':file_hash(folder/'arrays.npz'),
        'records':refs,'provenance':provenance,'components':list(COMPONENTS),'score_scale':'zero_one',
        'requested_scenes':len(scenes),'fully_scored_scenes':len(scenes),'invalid_prediction_count':0,'unresolved_scoring_failures':0}
    metadata['hash']=digest(metadata);atomic_json(folder/'BANK.json',metadata,immutable=True)
    from .objectives import pair_population
    pair,weights=pair_population(scores,comparison_valid)
    gaps=np.array([abs(scores[x,i]-scores[x,j]) for x,i,j in pair])
    stats={'scenes':len(scenes),'logs':len(set(metadata['source_logs'])),'candidates':int(valid.sum()),'K_max':count,
        'valid_candidates_distribution':dict(Counter(map(int,valid.sum(1)))),'non_tie_pairs':len(pair),'comparison_weight_sum':float(weights.sum()),
        'comparison_effective_pair_count':float(weights.sum()**2/(weights**2).sum()) if len(weights) else 0.,
        'base_argmax_wins':int((np.where(valid,scores,-np.inf).argmax(1)==0).sum()),'all_low_0_8':int(((scores<.8)|~valid).all(1).sum()),
        'gap_quantiles':np.quantile(gaps,[0,.25,.5,.75,.95,1]).tolist() if len(gaps) else [],
        'gap_gt_0_1':int((gaps>.1).sum()),'gap_gt_0_5':int((gaps>.5).sum()),
        'duplicate_trajectory_count':int((valid&~comparison_valid).sum()),
        'comparison_signal':'PRESENT' if weights.sum()>0 else 'SCORER_COMPARISON_SIGNAL_ABSENT','bank_hash':metadata['hash']}
    atomic_json(folder/'STATISTICS.json',stats,immutable=True)
    return stats


def load_bank(config,kind,role):
    folder=directory(config,kind,role);meta=read_json(folder/'BANK.json')
    require(meta['hash']==digest({k:v for k,v in meta.items() if k!='hash'}),'bank metadata corrupted')
    require(meta['pool_hash']==pool(config)['hash'] and meta['Base_binding_hash']==file_hash(config.root/'BASE_BINDING.json'),'bank/pool/Base stale')
    require(file_hash(folder/'arrays.npz')==meta['array_hash'],'bank arrays corrupt')
    require(meta['fully_scored_scenes']==meta['requested_scenes'] and meta['unresolved_scoring_failures']==0,'bank incomplete')
    with np.load(folder/'arrays.npz') as f:data={k:f[k].copy() for k in f.files}
    return meta,data


def base_records(config,scenes):
    cache=ReusedContext(config,capacity=0);binding=read_json(config.root/'BASE_BINDING.json');oldroot=Path(config.reuse_root)
    # Reuse actual scored trajectory identities, never aggregate means or another Base.
    labels=read_json(oldroot/'BASE_FIT_VAL_SCORES.json');candidates=[];scores=[];missing=[];proto=protocol(config)
    for scene in scenes:
        path=oldroot/'candidates/expert_0'/binding['checkpoint_hash']/(scene.scene_id+'.json')
        if path.exists() and scene.scene_id in labels:
            c=strict_record(CandidateRecord,read_json(path));s=strict_record(ScoreRecord,labels[scene.scene_id])
            require(c.expert_checkpoint_hash==binding['checkpoint_hash'] and c.feature_contract_hash==binding['contract_hash'],'Base candidate checkpoint/contract mismatch')
            require(file_hash(c.trajectory_ref)==c.trajectory_hash and s.candidate_key==c.key and s.metric_context_hash==scene.metric_context_hash and s.score_valid,'Base candidate/score mismatch')
            require(s.score_protocol_hash==proto['protocol_hash'],'Base official protocol changed')
            candidates.append(c);scores.append(s)
        else:
            c=save_candidate(config,scene,'expert_0',binding['checkpoint_hash'],cache.read(scene)['base_physical'].numpy())
            missing.append((len(candidates),scene,c));candidates.append(c);scores.append(None)
    if missing:
        results=score_candidates(config,[x[1] for x in missing],[x[2] for x in missing],'study_Base')
        for (i,_,_),r in zip(missing,results):scores[i]=r
    return list(zip(candidates,scores))


def build_role(config,role,device):
    folder=directory(config,'deploy',role)
    if (folder/'BANK.json').exists():load_bank(config,'deploy',role);return read_json(folder/'STATISTICS.json')
    scenes=scenes_by_role(config)[role];p=pool(config);columns=[base_records(config,scenes)]
    for item in p['experts']:
        model=load_endpoint(config,item,device)
        candidates,labels=candidate_labels(config,model,scenes,item['expert_id'],device)
        columns.append(list(zip(candidates,labels)));del model
    rows=[list(x) for x in zip(*columns)]
    return save_bank(config,'deploy',role,scenes,rows,{'sources':p['expert_ids'],'GT_deployment_candidate':False,'intermediate_endpoints_in_deploy':False})


def build_deploy(config,device='cuda:0'):
    if (config.root/'runtime/BANK_ROUTING.json').exists():
        from .bank_parallel import build_parallel
        return build_parallel(config)
    return {role:build_role(config,role,device) for role in ('scorer_fit','stage_val','selector_cal','dev_report')}


CONSTRUCTION_LIMITS = {'max_speed_mps': 70., 'max_abs_acceleration_mps2': 10.,
                       'max_lateral_acceleration_mps2': 8., 'max_yaw_rate_rad_s': 1.2,
                       'max_jerk_mps3': 30.}


def duplicates_original_gt(candidate, original, tolerance=1e-5):
    """Do not smuggle unchanged GT into B_rich under an optimized-source name."""
    a,b=np.asarray(candidate),np.asarray(original)
    require(a.shape==b.shape and a.shape[-1]==3,'GT comparison trajectory contract')
    yaw=np.arctan2(np.sin(a[...,2]-b[...,2]),np.cos(a[...,2]-b[...,2]))
    return bool(np.abs(a[...,:2]-b[...,:2]).max()<=tolerance and np.abs(yaw).max()<=tolerance)


def constructed_geometry(physical, dt, current_speed):
    """Checks construction plausibility, never removes safety-negative labels.

    Current speed is the real observation's ego_dynamic_state norm in m/s, not
    the normalized four-channel planning ego feature. The initial pose is the
    current rear axle at (0,0,0). Collision/road checks belong to official labels.
    """
    p = np.asarray(physical, dtype=np.float64)
    coarse = geometry(p, dt, horizon=len(p))
    if not coarse['valid'] or not np.isfinite(current_speed) or not 0 <= current_speed < 70:
        return {'valid': False, 'reason': 'geometry_or_current_speed'}
    velocity = np.diff(np.concatenate((np.zeros((1, 2)), p[:, :2])), axis=0)/dt
    speed = np.linalg.norm(velocity, axis=1)
    acceleration = np.diff(np.r_[current_speed, speed])/dt
    # First future point represents an interval-average speed at dt/2.
    acceleration[0] *= 2
    yaw_rate = np.diff(np.unwrap(np.r_[0., p[:, 2]]))/dt
    lateral = speed*np.abs(yaw_rate)
    jerk = np.diff(acceleration)/dt
    maxima = {'max_speed_mps': float(speed.max()), 'max_abs_acceleration_mps2': float(np.abs(acceleration).max()),
              'max_lateral_acceleration_mps2': float(lateral.max()), 'max_yaw_rate_rad_s': float(np.abs(yaw_rate).max()),
              'max_jerk_mps3': float(np.abs(jerk).max()) if len(jerk) else 0.}
    failed = [key for key, value in maxima.items() if value > CONSTRUCTION_LIMITS[key]]
    return {'valid': not failed, 'reasons': failed, **maxima}


def trajectory_variants(physical,dt,current_speed=None,diagnostics=None):
    """Small deterministic braking/speed and lateral-curvature perturbations.

    They are interventions on an own predicted trajectory, not fabricated observed
    future. Every variant receives a fresh official outcome label. Beyond-horizon
    extrapolation is forbidden; time warps stay within the original time support.
    """
    require(current_speed is not None,'current physical speed is required for constructed candidates')
    p=np.asarray(physical,dtype=np.float64);n=len(p);times=np.arange(n+1)*dt
    xy=np.concatenate((np.zeros((1,2)),p[:,:2]),0);result=[]
    def append(name,value):
        checks=constructed_geometry(value,dt,current_speed)
        if diagnostics is not None:diagnostics.append({'source':name,**checks})
        if checks['valid']:result.append((name,value))
    for scale in (.85,.95):
        query=times[1:]*scale
        altered=np.column_stack([np.interp(query,times,xy[:,j]) for j in range(2)])
        delta=np.diff(np.concatenate((np.zeros((1,2)),altered),0),axis=0)
        yaw=np.arctan2(delta[:,1],delta[:,0]);speed=np.linalg.norm(delta,axis=1)/dt
        yaw=np.where(speed<.05,p[:,2],yaw)
        value=np.column_stack((altered,yaw)).astype(np.float32)
        append('time_warp_'+str(scale),value)
    for amplitude in (-.2,.2):
        fraction=np.arange(1,n+1)/n;normal=np.column_stack((-np.sin(p[:,2]),np.cos(p[:,2])))
        altered=p[:,:2]+(amplitude*np.sin(np.pi*fraction)**2)[:,None]*normal
        delta=np.diff(np.concatenate((np.zeros((1,2)),altered),0),axis=0)
        yaw=np.arctan2(delta[:,1],delta[:,0]);value=np.column_stack((altered,yaw)).astype(np.float32)
        append('lateral_bend_'+str(amplitude),value)
    return result


@torch.no_grad()
def unscored_rich_column(config,module,scenes,expert_id,device):
    """Keep the existing 32-scene generation order; screen duplicates before labels."""
    module.eval();identity=module_hash(module);cache=ReusedContext(config);candidates=[]
    from .rich_precompute import column, batch
    precomputed=column(config,expert_id,identity,scenes)
    for start in range(0,len(scenes),32):
        sub=scenes[start:start+32]
        if precomputed is not None:
            p,item=precomputed
            rows,_=batch(config,p,item,sub,start//32,module,device,cache)
            candidates+=rows
            continue
        physical=module(cache.batch(sub,device)).physical.cpu().numpy()
        with ThreadPoolExecutor(max_workers=4) as workers:
            candidates+=list(workers.map(lambda pair:save_candidate(config,pair[0],expert_id,identity,pair[1]),zip(sub,physical)))
    return candidates


def append_unique_rich_column(config,scenes,candidates,rows,hashes,sources,source,job_name):
    require(len(scenes)==len(candidates)==len(rows)==len(hashes),'rich column scene join')
    pending=[];counts=Counter()
    for n,(scene,candidate) in enumerate(zip(scenes,candidates)):
        require(candidate.scene_id==scene.scene_id,'rich candidate reordered across scenes')
        if candidate.trajectory_hash in hashes[n]:
            counts['duplicate_official_requests_avoided']+=1
        elif len(rows[n])>=config.max_rich_candidates:
            counts['over_capacity_official_requests_avoided']+=1
        else:
            pending.append((n,scene,candidate))
    if pending:
        labels=score_candidates(config,[s for _,s,_ in pending],[c for _,_,c in pending],job_name)
        require(len(labels)==len(pending),'rich official label coverage')
        for (n,_,candidate),label in zip(pending,labels):
            rows[n].append((candidate,label));hashes[n].add(candidate.trajectory_hash);sources[source]+=1
    counts['unique_official_requests']+=len(pending)
    return counts


def deduplicate_rich_rows(rows):
    unique=[];removed=0
    for row in rows:
        seen={};retained=[]
        for candidate,label in row:
            previous=seen.get(candidate.trajectory_hash)
            if previous is None:
                seen[candidate.trajectory_hash]=label;retained.append((candidate,label))
            else:
                require(abs(previous.total_score_01-label.total_score_01)<=1e-8
                    and previous.named_components==label.named_components
                    and previous.component_valid_masks==label.component_valid_masks,
                    'duplicate physical trajectory has inconsistent official labels')
                removed+=1
        unique.append(retained)
    return unique,removed


def build_rich(config,device='cuda:0'):
    folder=directory(config,'rich','scorer_fit')
    if (folder/'BANK.json').exists():load_bank(config,'rich','scorer_fit');return read_json(folder/'STATISTICS.json')
    meta,data=load_bank(config,'deploy','scorer_fit');p=pool(config);scenes=scenes_by_role(config)['scorer_fit']
    require(meta['scene_ids']==[s.scene_id for s in scenes],'D_S bank ordering changed')
    rows=[[(strict_record(CandidateRecord,x['candidate']),strict_record(ScoreRecord,x['label'])) for x in row] for row in meta['records']]
    hashes=[{c.trajectory_hash for c,_ in row} for row in rows];sources=Counter();label_requests=Counter()
    for item in p['experts']:
        for step in (1000,3000):
            early=endpoint(config,item['cell'],step,'raw');model=load_endpoint(config,early,device)
            candidates=unscored_rich_column(config,model,scenes,item['cell']+f'_raw_{step}',device)
            label_requests.update(append_unique_rich_column(config,scenes,candidates,rows,hashes,sources,
                'own_intermediate',item['cell']+f'_raw_{step}_unique_'+early['model_hash'][:12]));del model
    import pandas as pd
    from .target_rules import verify_target_assets
    verify_target_assets(config)
    info=pd.read_parquet(Path(config.optimized_root)/'optimized_labels.parquet').set_index('token')
    with np.load(Path(config.optimized_root)/'optimized_trajectories.npz') as f:optimized={str(i):t for i,t in zip(f['tokens'],f['trajectories'])}
    asset=file_hash(Path(config.optimized_root)/'optimized_trajectories.npz');dt=read_json(config.root/'OFFLINE_BASE_CONTRACT.json')['trajectory']['dt']
    rules={'schema':'current_state_bounded_variants_v1','time_warps':[.85,.95],'lateral_bend_meters':[-.2,.2],
           'limits':CONSTRUCTION_LIMITS,'current_speed_source':'observation.ego_speed from norm(frame.ego_dynamic_state[:2]), m/s',
           't0':'current rear axle (0,0,0)','future_input':False,'official_safety_negatives_retained':True,
           'no_beyond_horizon_extrapolation':True,'source_scope':'constructed variants only; no filtering of real deployment negatives',
           'original_GT_not_added_as_optimized':True,'GT_duplicate_xy_and_periodic_yaw_tolerance':1e-5}
    atomic_json(config.root/'RICH_CONSTRUCTION_RULES_BEFORE_LABELS.json',rules,immutable=True)
    requests=[];construction_counts=Counter();target_cache=ReusedContext(config,capacity=1)
    for n,s in enumerate(scenes):
        if len(rows[n])>=config.max_rich_candidates:continue
        proposal=[]
        if s.scene_id in info.index and s.scene_id in optimized and admissible_optimized_source(str(info.loc[s.scene_id,'source'])):
            t=optimized[s.scene_id]
            if geometry(t,dt,horizon=t.shape[0])['valid']:
                if duplicates_original_gt(t,target_cache.read(s)['target_physical'].numpy()):
                    construction_counts['optimized_equals_original_GT_excluded']+=1
                else:proposal.append(('offline_optimized',t))
        observation=read_json(s.observation_ref)
        require(observation['token']==s.scene_id and np.isfinite(observation['ego_speed']),'invalid current state for variant constructor')
        require(digest({'record':observation,'images':{path:file_hash(path) for path in observation['image_paths']}})==s.observation_hash,
                'current observation changed before rich construction')
        checks=[]
        proposal+=trajectory_variants(data['trajectories'][n,0],dt,float(observation['ego_speed']),checks)
        for check in checks:
            construction_counts['attempted']+=1
            construction_counts['accepted' if check['valid'] else 'rejected_geometry']+=1
            for reason in check.get('reasons',[]):construction_counts[reason]+=1
        planned=0
        for source,t in proposal:
            if len(rows[n])+planned>=config.max_rich_candidates:break
            c=save_candidate(config,s,'rich_'+source,digest([asset,source]),t)
            if c.trajectory_hash not in hashes[n]:requests.append((n,s,c,source));hashes[n].add(c.trajectory_hash);planned+=1
    if requests:
        labels=score_candidates(config,[x[1] for x in requests],[x[2] for x in requests],'study_rich_variants')
        for (n,s,c,source),r in zip(requests,labels):rows[n].append((c,r));sources[source]+=1
    label_requests['unique_official_requests']+=len(requests)
    # Deployment keeps explicit expert columns, even for exact duplicate outputs.
    # Rich pointwise supervision retains each physical action only once. Keep
    # those original columns in the capacity accounting until here so avoiding
    # duplicates cannot silently expand the original twelve-label budget.
    require(sum(len(row) for row in rows)<=len(scenes)*config.max_rich_candidates,'rich unique label budget exceeded')
    rows,removed=deduplicate_rich_rows(rows)
    return save_bank(config,'rich','scorer_fit',scenes,rows,{'additional_sources':dict(sources),'deployment_bank_hash':meta['hash'],
            'label_budget_max':len(scenes)*12,'external_teacher_weights_loaded':False,'GT_added':False,
            'new_label_requests':dict(label_requests),'duplicates_removed_before_official_scoring':True,
            'exact_deployment_duplicates_removed_from_rich':removed,'all_unique_deployment_trajectories_retained':True,
            'construction_geometry':dict(construction_counts),'construction_rules_hash':file_hash(config.root/'RICH_CONSTRUCTION_RULES_BEFORE_LABELS.json')})
