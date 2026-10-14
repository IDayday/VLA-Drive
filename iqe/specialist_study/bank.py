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
    return {role:build_role(config,role,device) for role in ('scorer_fit','stage_val','selector_cal','dev_report')}


def trajectory_variants(physical,dt):
    """Small deterministic braking/speed and lateral-curvature perturbations.

    They are interventions on an own predicted trajectory, not fabricated observed
    future. Every variant receives a fresh official outcome label. Beyond-horizon
    extrapolation is forbidden; time warps stay within the original time support.
    """
    p=np.asarray(physical,dtype=np.float64);n=len(p);times=np.arange(n+1)*dt
    xy=np.concatenate((np.zeros((1,2)),p[:,:2]),0);result=[]
    for scale in (.85,.95):
        query=times[1:]*scale
        altered=np.column_stack([np.interp(query,times,xy[:,j]) for j in range(2)])
        delta=np.diff(np.concatenate((np.zeros((1,2)),altered),0),axis=0)
        yaw=np.arctan2(delta[:,1],delta[:,0]);speed=np.linalg.norm(delta,axis=1)/dt
        yaw=np.where(speed<.05,p[:,2],yaw)
        value=np.column_stack((altered,yaw)).astype(np.float32)
        if geometry(value,dt,horizon=n)['valid']:result.append(('time_warp_'+str(scale),value))
    for amplitude in (-.2,.2):
        fraction=np.arange(1,n+1)/n;normal=np.column_stack((-np.sin(p[:,2]),np.cos(p[:,2])))
        altered=p[:,:2]+(amplitude*np.sin(np.pi*fraction)**2)[:,None]*normal
        delta=np.diff(np.concatenate((np.zeros((1,2)),altered),0),axis=0)
        yaw=np.arctan2(delta[:,1],delta[:,0]);value=np.column_stack((altered,yaw)).astype(np.float32)
        if geometry(value,dt,horizon=n)['valid']:result.append(('lateral_bend_'+str(amplitude),value))
    return result


def build_rich(config,device='cuda:0'):
    folder=directory(config,'rich','scorer_fit')
    if (folder/'BANK.json').exists():load_bank(config,'rich','scorer_fit');return read_json(folder/'STATISTICS.json')
    meta,data=load_bank(config,'deploy','scorer_fit');p=pool(config);scenes=scenes_by_role(config)['scorer_fit']
    require(meta['scene_ids']==[s.scene_id for s in scenes],'D_S bank ordering changed')
    rows=[[(strict_record(CandidateRecord,x['candidate']),strict_record(ScoreRecord,x['label'])) for x in row] for row in meta['records']]
    hashes=[{c.trajectory_hash for c,_ in row} for row in rows];sources=Counter()
    def append_column(candidates,labels,source):
        for n,(c,r) in enumerate(zip(candidates,labels)):
            if len(rows[n])<config.max_rich_candidates and c.trajectory_hash not in hashes[n]:
                rows[n].append((c,r));hashes[n].add(c.trajectory_hash);sources[source]+=1
    for item in p['experts']:
        for step in (1000,3000):
            early=endpoint(config,item['cell'],step,'raw');model=load_endpoint(config,early,device)
            candidates,labels=candidate_labels(config,model,scenes,item['cell']+f'_raw_{step}',device)
            append_column(candidates,labels,'own_intermediate');del model
    import pandas as pd
    info=pd.read_parquet(Path(config.optimized_root)/'optimized_labels.parquet').set_index('token')
    with np.load(Path(config.optimized_root)/'optimized_trajectories.npz') as f:optimized={str(i):t for i,t in zip(f['tokens'],f['trajectories'])}
    asset=file_hash(Path(config.optimized_root)/'optimized_trajectories.npz');dt=read_json(config.root/'OFFLINE_BASE_CONTRACT.json')['trajectory']['dt']
    requests=[]
    for n,s in enumerate(scenes):
        if len(rows[n])>=config.max_rich_candidates:continue
        proposal=[]
        if s.scene_id in info.index and s.scene_id in optimized and admissible_optimized_source(str(info.loc[s.scene_id,'source'])):
            t=optimized[s.scene_id]
            if geometry(t,dt,horizon=t.shape[0])['valid']:proposal.append(('offline_optimized',t))
        proposal+=trajectory_variants(data['trajectories'][n,0],dt)
        planned=0
        for source,t in proposal:
            if len(rows[n])+planned>=config.max_rich_candidates:break
            c=save_candidate(config,s,'rich_'+source,digest([asset,source]),t)
            if c.trajectory_hash not in hashes[n]:requests.append((n,s,c,source));hashes[n].add(c.trajectory_hash);planned+=1
    if requests:
        labels=score_candidates(config,[x[1] for x in requests],[x[2] for x in requests],'study_rich_variants')
        for (n,s,c,source),r in zip(requests,labels):rows[n].append((c,r));sources[source]+=1
    return save_bank(config,'rich','scorer_fit',scenes,rows,{'additional_sources':dict(sources),'deployment_bank_hash':meta['hash'],
            'label_budget_max':len(scenes)*12,'external_teacher_weights_loaded':False,'GT_added':False})
