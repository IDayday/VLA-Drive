"""Fixed-query absolute/difference errors and scene-dependence controls."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import atomic_json,check_host,digest,read,source_sha
from tools.decision_local.train_small_model import build_model
from starVLA.dataloader.decision_local import FrozenConsequenceData,inputs_queries_targets
from starVLA.model.modules.decision_local.consequence_contract import predicted_relations,CONTINUOUS,CHANNELS


LEGACY_SENSITIVITY = 'any_supported_sample_v1'
LOCAL_SENSITIVITY = 'local_known_sensitive_complete_insensitive_unknown_v2'


def sensitivity_population(rows):
    """Count pairs and independent scenes/logs separately, never as replicas."""
    groups=('near_sensitive','near_insensitive','far_sensitive','far_insensitive',
            'near_unknown','far_unknown','ambiguous_spatial')
    result={}
    for group in groups:
        key=group+'_pairs'
        if not rows or key not in rows[0]:continue
        supported=[r for r in rows if r[key]>0]
        result[group]={'pairs':int(sum(r[key] for r in rows)),
            'scenes':len({r['token'] for r in supported}),
            'logs':len({r['log'] for r in supported})}
    return result


def scene_metrics(pred,truth,valid,poses,*,local_evidence=False):
    """All masks loss/evaluation-side; no score-dependent population filtering."""
    B,K,T,C=pred.shape
    def mean(x,m):
        return torch.where(m,x,torch.zeros_like(x)).flatten(1).sum(1)/m.flatten(1).sum(1).clamp_min(1)
    out={}
    for c,name in enumerate(CHANNELS):
        m=valid[...,c]
        error=torch.where(m,(pred[...,c]-truth[...,c]).abs(),torch.zeros_like(pred[...,c]))
        out[name+'_valid_points']=m.sum((1,2))
        out[name+'_abs_error_sum']=error.sum((1,2))
        out[name+'_coverage']=m.float().mean((1,2))
        out[name+'_MAE']=mean(error,m)
        for t in range(T):
            out[name+f'_t{t+1}_valid_points']=m[:,:,t].sum(1)
            out[name+f'_t{t+1}_abs_error_sum']=error[:,:,t].sum(1)
        if c in CONTINUOUS:
            pair=m[:,1:]&m[:,:1]
            pd=pred[:,1:,:,c]-pred[:,:1,:,c];td=truth[:,1:,:,c]-truth[:,:1,:,c]
            certain=pair&(td.abs()>=.5)
            out[name+'_pair_MAE']=mean((pd-td).abs(),pair)
            out[name+'_pair_abs_error_sum']=torch.where(pair,(pd-td).abs(),torch.zeros_like(pd)).sum((1,2))
            out[name+'_pair_valid_times']=pair.sum((1,2))
            # Distance quadrants mix lateral variants and arrival-time warps;
            # report perturbation type separately rather than calling that
            # contrast a pure causal effect of distance from q0.
            for group,sl in [('lateral',slice(0,4)),('arrival_time',slice(4,6))]:
                pm=pair[:,sl]
                out[name+'_'+group+'_pair_abs_error_sum']=torch.where(pm,(pd[:,sl]-td[:,sl]).abs(),torch.zeros_like(pd[:,sl])).sum((1,2))
                out[name+'_'+group+'_pair_valid_times']=pm.sum((1,2))
            out[name+'_direction_accuracy']=mean(((pd*td)>0).float(),certain)
            out[name+'_direction_correct_pairs']=(((pd*td)>0)&certain).sum((1,2))
            out[name+'_direction_valid_pairs']=certain.sum((1,2))
        else:
            p=pred[...,c]>=.5;y=truth[...,c]>.5
            for key,mask in [('TP',p&y),('FN',~p&y),('FP',p&~y),('TN',~p&~y)]:out[name+'_'+key]=(mask&m).sum((1,2))
            # A discrete transition is evaluated as two labels, never regressed
            # as a continuous physical distance in the difference objective.
            pair=m[:,1:]&m[:,:1];changed=y[:,1:]!=y[:,:1]
            predicted_change=p[:,1:]!=p[:,:1]
            out[name+'_change_valid']=pair.sum((1,2))
            out[name+'_change_correct']=((predicted_change==changed)&pair).sum((1,2))
            out[name+'_change_FN']=(changed&~predicted_change&pair).sum((1,2))
            full=m.all(-1);has_true=y.any(-1);has_pred=p.any(-1)
            gt_first=y.to(torch.int64).argmax(-1);pred_first=p.to(torch.int64).argmax(-1)
            both=full&has_true&has_pred
            out[name+'_first_conflict_full_sequences']=full.sum(1)
            out[name+'_first_conflict_both_present']=both.sum(1)
            out[name+'_first_conflict_absolute_time_error_sum_s']=torch.where(both,(gt_first-pred_first).abs()*.5,0.).sum(1)
            out[name+'_full_sequence_event_FN']=(full&has_true&~has_pred).sum(1)
            out[name+'_full_sequence_event_FP']=(full&~has_true&has_pred).sum(1)
    # Physical pre-registration: distinguish one-to-two grid-cell local changes
    # from wider temporal reparameterizations. Tiny spatial changes are ambiguous.
    d=(poses[:,1:,:,:2]-poses[:,:1,:,:2]).norm(dim=-1).amax(-1)
    sensitive=torch.zeros_like(d,dtype=torch.bool); supported=torch.zeros_like(d,dtype=torch.bool)
    complete=torch.ones_like(d,dtype=torch.bool)
    for c in (0,3):
        a,b=truth[:,:1,:,c],truth[:,1:,:,c];pair=valid[:,:1,:,c]&valid[:,1:,:,c]
        if c==0:cross=((a>=.25)&(b<=-.25))|((b>=.25)&(a<=-.25))
        else:cross=((a>=1.)&(b<=0.))|((b>=1.)&(a<=0.))
        sensitive|=(cross&pair).any(-1);supported|=pair.any(-1)
        complete&=pair.all(-1)
    a,b=truth[:,:1,:,4]>.5,truth[:,1:,:,4]>.5;pair=valid[:,:1,:,4]&valid[:,1:,:,4]
    sensitive|=((a!=b)&pair).any(-1);supported|=pair.any(-1)
    complete&=pair.all(-1)
    nonambiguous=d>=.5-1e-6
    for near in (True,False):
        if local_evidence:
            # A single known crossing proves sensitivity even when later facts
            # are unknown. Absence of a crossing is only conclusive when both
            # trajectories have all sampled facts for EVERY defining criterion.
            # This is not dense 4-second repair qualification or a safety claim.
            categories={'sensitive':sensitive,'insensitive':~sensitive&complete,
                        'unknown':~sensitive&~complete}
        else:
            # Historical reports retain their original grouping for replay.
            categories={'sensitive':sensitive&supported,'insensitive':~sensitive&supported}
        for category,mask in categories.items():
            selected=((d<=1.+1e-6)==near)&mask&nonambiguous
            name=('near' if near else 'far')+'_'+category
            out[name+'_pairs']=selected.sum(1)
            for c in CONTINUOUS:
                m=(valid[:,1:,:,c]&valid[:,:1,:,c])&selected[...,None]
                error=((pred[:,1:,:,c]-pred[:,:1,:,c])-(truth[:,1:,:,c]-truth[:,:1,:,c])).abs()
                out[name+'_'+CHANNELS[c]+'_pair_abs_error_sum']=torch.where(m,error,torch.zeros_like(error)).sum((1,2))
                out[name+'_'+CHANNELS[c]+'_valid_times']=m.sum((1,2))
    out['ambiguous_spatial_pairs']=(~nonambiguous).sum(1)
    return {k:v.detach().cpu().numpy() for k,v in out.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('features','labels','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--batch',type=int,default=32);p.add_argument('--controls',action='store_true')
    p.add_argument('--distribution',choices=('NATURAL','RECOVERY'),default='NATURAL')
    p.add_argument('--mean-features',type=Path);p.add_argument('--max-seconds',type=float,required=True)
    p.add_argument('--generalization-config',type=Path,
        help='Explicit preregistered query-only amplitude check; never changes trained weights')
    a=p.parse_args();resources=check_host();started=time.monotonic()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    s=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    natural=s['identity'].get('protocol')=='natural_local_v1'
    repair=s['identity'].get('protocol') in ('minimal_repair_v1','natural_local_v1')
    if repair:
        from starVLA.dataloader.recovery_plan_dataset import RecoveryPlanData,repair_inputs_targets
        from starVLA.model.modules.decision_local.minimal_repair_contract import execution_view
        from tools.decision_local.minimal_repair_training import label_semantics
        if natural:
            from starVLA.dataloader.natural_local_dataset import NaturalLocalData
            data=NaturalLocalData(a.features,a.labels)
        else:data=RecoveryPlanData(a.features,a.labels)
        if s['identity']['repair_label_semantics']!=label_semantics(data.label_identity):raise ValueError('Predictor and labels have different consequence definitions')
    else:
        if a.distribution!='NATURAL':raise ValueError('Recovery inputs require the versioned plan contract')
        data=FrozenConsequenceData(a.features,a.labels)
    if not s['identity']['kind'].startswith('P_'):raise ValueError('Predictor checkpoint required')
    if s['identity']['anchor_identity']!=data.feature_identity['anchor_identity']:
        raise ValueError('Evaluation must retain the trained foundation')
    trained_query=s['identity']['query_config'];evaluated_query=data.label_identity['query_config']
    if a.generalization_config:
        declared=read(a.generalization_config)
        if declared!=evaluated_query or set(declared)!=set(trained_query):
            raise ValueError('Generalization queries must match their explicit fixed registration')
        changed={key for key in declared if declared[key]!=trained_query[key]}
        if not changed or changed-{'lateral_m','time_shift_s'}:
            raise ValueError('Only registered lateral/time amplitudes may change in this diagnostic')
    elif trained_query!=evaluated_query:
        raise ValueError('Evaluation must retain the trained query contract')
    model=build_model(s['identity']['kind'],s['identity']['seed'],s['identity']['dim']).cuda().eval();model.load_state_dict(s['model'],strict=True)
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'metrics.jsonl').exists():raise FileExistsError('Immutable evaluation output')
    mean=None
    if a.controls:
        if a.mean_features is None:raise ValueError('Scene controls require TRAIN-only mean cache')
        mean=torch.load(a.mean_features,weights_only=True,map_location='cuda')
        if mean['anchor_identity']!=data.feature_identity['anchor_identity'] or mean['split']!='train':raise ValueError('Mean source must be shared train anchor')
    rng=torch.Generator().manual_seed(812);order=torch.randperm(len(data.index),generator=rng).tolist()
    partner={order[i]:order[(i+len(order)//2)%len(order)] for i in range(len(order))}
    all_metrics={};predictions=[];scene_count=0;population_rows=[]
    sensitivity_protocol=LOCAL_SENSITIVITY if natural else LEGACY_SENSITIVITY
    with torch.no_grad(),(a.output/'metrics.jsonl').open('w') as stream:
        for start in range(0,len(data.index),a.batch):
            if time.monotonic()-started>=a.max_seconds:raise TimeoutError('Finite evaluation budget reached; partial result not complete')
            ids=list(range(start,min(len(data.index),start+a.batch)))
            kwargs={'distribution':a.distribution} if repair else {}
            batch=data.batch(ids,'cuda',evaluation=True,**kwargs)
            if repair:
                current,plan,queries,targets=repair_inputs_targets(batch);inputs=execution_view(current,plan)
                target_relations,target_valid=targets.relations,targets.relation_validity
            else:
                inputs,queries,targets=inputs_queries_targets(batch);target_relations,target_valid=targets.relations,targets.valid
            variants={'normal':inputs}
            if a.controls:
                shuffled=data.batch([partner[i] for i in ids],'cuda',evaluation=True,**kwargs)
                variants['permuted_W_H_A']=replace(inputs,W=shuffled['W'],H_A=shuffled['H_A'])
                variants['training_mean_W_H_A']=replace(inputs,W=mean['W'][None].expand(len(ids),-1,-1),H_A=mean['H_A'][None].expand(len(ids),-1,-1))
            for condition,current in variants.items():
                pred=predicted_relations(model(current,queries));metrics=scene_metrics(pred,target_relations,target_valid,queries.poses,local_evidence=natural)
                if condition=='normal':predictions.append(pred.cpu().numpy())
                for k,v in metrics.items():all_metrics.setdefault(condition,{}).setdefault(k,[]).append(v)
                for j,i in enumerate(ids):
                    row={**data.index[i],'condition':condition,'sensitivity_protocol':sensitivity_protocol,
                         **{k:float(v[j]) for k,v in metrics.items()}}
                    stream.write(json.dumps(row)+'\n')
                    if condition=='normal':
                        population_rows.append({k:v for k,v in row.items()
                            if k in ('token','log') or k.endswith('_pairs')})
            scene_count+=len(ids)
    np.savez_compressed(a.output/'predictions.npz',relations=np.concatenate(predictions),tokens=np.array([r['token'] for r in data.index]))
    result={}
    for condition,values in all_metrics.items():
        result[condition]={k:float(np.concatenate(v).mean()) for k,v in values.items()}
        for name in CHANNELS:
            counts=np.concatenate(values[name+'_valid_points'])
            errors=np.concatenate(values[name+'_abs_error_sum'])
            valid_scene=counts>0
            result[condition][name+'_supported_scene_MAE']=float((errors[valid_scene]/counts[valid_scene]).mean()) if valid_scene.any() else None
            result[condition][name+'_pooled_MAE']=float(errors.sum()/counts.sum()) if counts.sum() else None
            result[condition][name+'_supported_scenes']=int(valid_scene.sum())
            if name+'_pair_valid_times' in values:
                den=np.concatenate(values[name+'_pair_valid_times']).sum()
                result[condition][name+'_pooled_pair_MAE']=float(np.concatenate(values[name+'_pair_abs_error_sum']).sum()/den) if den else None
                den=np.concatenate(values[name+'_direction_valid_pairs']).sum()
                result[condition][name+'_pooled_direction_accuracy']=float(np.concatenate(values[name+'_direction_correct_pairs']).sum()/den) if den else None
        for c in ('current_overlap','future_overlap'):
            counts={suffix:float(np.concatenate(values[c+'_'+suffix]).sum()) for suffix in ('TP','FN','FP','TN')}
            result[condition][c+'_counts']=counts
            result[condition][c+'_precision']=counts['TP']/max(1,counts['TP']+counts['FP'])
            result[condition][c+'_recall']=counts['TP']/max(1,counts['TP']+counts['FN'])
    atomic_json(a.output/'COMPLETE.json',{'scenes':scene_count,'results':result,'resource_check':resources,'GPU_hours':(time.monotonic()-started)/3600.,
        'checkpoint_sha256':digest(a.checkpoint),'features':data.feature_identity['identity'],'labels':data.label_identity['identity'],
        'source_sha':source_sha(ROOT),'precision':'FP32 head; TF32off','query_protocol':'actual plan seed42, seven fixed complete trajectories' if repair else 'q0 seed42, seven fixed complete trajectories',
        'distribution':a.distribution,'repair_protocol':repair,
        'local_evidence_protocol':natural,
        'sensitivity_protocol':sensitivity_protocol,
        'sensitivity_population':sensitivity_population(population_rows),
        'sensitivity_scope':'A known road/clearance/overlap crossing suffices for sensitive. Insensitive requires all defining sampled channel/time pairs supported; otherwise unknown. Not full-trajectory certification.' if natural else 'Historical any-supported grouping retained unchanged',
        'distance_interpretation':'conditional annotated-object distance, not proven free space' if natural else 'registered historical label contract',
        'generalization_only':bool(a.generalization_config),'trained_query_config':trained_query,
        'evaluated_query_config':evaluated_query,
        'generalization_config_sha256':digest(a.generalization_config) if a.generalization_config else None,
        'thresholds':{'spatial_ambiguous_below_m':.5,'near_max_displacement_m':1.,'road_sign_ambiguity_m':.25,'clearance_band_m':[0.,1.],'overlap_score':.5},
        'scene_mean_rule':'all scenes retained, zero numerator for unsupported scene; coverage and valid counts reported separately'})


if __name__=='__main__':main()
