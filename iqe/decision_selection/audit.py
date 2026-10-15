"""Existing official banks/predictions only; post-hoc diagnostics are not routing rules."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import csv
import io
import numpy as np
import torch
from iqe.contracts import require
from iqe.io import read_json, atomic_json, atomic_bytes, file_hash, digest
from iqe.specialist_study.bank import load_bank
from iqe.specialist_study.metrics import select, report_metrics
from iqe.scoring.reference import COMPONENTS
from .binding import verify_binding, r00_prediction_source
from .data import canonical_bank, ROLES, read_split


def csv_bytes(rows):
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(rows[0]))
    writer.writeheader();writer.writerows(rows);return buffer.getvalue().encode()


def read_reference(config,role,meta):
    source=r00_prediction_source(config,role)
    with np.load(source['path'],allow_pickle=False) as f:values=f['values'].copy()
    values=values[:,meta['canonical_order']]
    require(values.shape==(len(meta['scene_ids']),len(meta['expert_ids'])) and np.isfinite(values).all(),'R00 aligned prediction shape')
    binding=verify_binding(config)
    require(source==binding['R00'][role],'reference changed after binding')
    return values


def _stats(config,meta,bank,values,indices=None):
    ids=meta['expert_ids']
    if indices is None:indices=list(range(len(ids)))
    require(indices[0]==0,'canonical Base required before metric function')
    r=bank['scores'][:,indices];c=np.where(bank['component_valid'][:,indices],bank['components'][:,indices],np.nan)
    p=values[:,indices];subset_ids=[ids[i] for i in indices]
    choice=select(p,np.isfinite(bank['trajectories'][:,indices]).all((-1,-2)),expert_ids=subset_ids)
    result=report_metrics(r,c,choice,meta['source_logs'],seed=config.bootstrap_seed,repetitions=config.bootstrap_repetitions)
    return result,np.asarray(indices)[choice]


def run(config):
    bound=verify_binding(config);split=read_split(config);root=config.root/'existing_audit'
    rule=dict(schema='existing_error_descriptive_v1',binding=bound['hash'],split_hash=split['hash'],
              speed_bins_mps=[0.,2.,5.,10.,20.,100.],gap_bins=[0.,.01,.1,.5,1.000001],
              ADE_bins_m=[0.,.1,.5,1.,2.,5.,1000.],high_threshold=.8,zero_epsilon=1e-8,
              no_new_training_scoring_or_model_inference=True,subpool_status='POSTHOC_SUBPOOL_DIAGNOSTIC',
              new_model_selection_uses_this_audit=False)
    atomic_json(root/'RULES.json',rule,immutable=True)
    done=root/'COMPLETE.json'
    if done.exists():
        r=read_json(done);require(r['rules_hash']==digest(rule),'audit identity changed')
        for path,h in r['outputs'].items():require(file_hash(path)==h,'audit output changed')
        return dict(status='COMPLETE',reused=True,report=str(root/'ERROR_DECOMPOSITION.md'))
    source=config.source();pilot=read_json(source.root/'PILOT_SELECTION.json')['specialty']
    specialty={'expert_0':'Base'}
    for row in bound['experts']:
        specialty[row['expert_id']]=row['cell'].removeprefix('EXT_') if row['cell'].startswith('EXT_') else pilot
    allstats={};switches=[];subsets=[];cohorts=[];outputs=[];fit_errors=[]
    for role in ROLES:
        meta,bank,scenes=canonical_bank(config,role);values=read_reference(config,role,meta)
        stats,chosen=_stats(config,meta,bank,values);allstats[role]=stats
        r=bank['scores'].astype(float);c=bank['components'];row=np.arange(len(r));base=r[:,0];selected=r[row,chosen]
        delta=selected-base;regret=r.max(1)-selected;high_zero=(base>=.8)&(selected<=1e-8)
        scene_rows=[]
        for n,s in enumerate(scenes):
            scene_rows.append(dict(scene_id=s.scene_id,source_log=s.source_log_id,
                  base=float(base[n]),candidate_scores=r[n].tolist(),R00_predictions=values[n].tolist(),
                  selected_id=meta['expert_ids'][chosen[n]],best_ids=[meta['expert_ids'][j] for j in np.flatnonzero(r[n]==r[n].max())],
                  selected=float(selected[n]),regret=float(regret[n]),positive_gain=max(float(delta[n]),0.),
                  negative_cost=max(float(-delta[n]),0.),high_to_zero=bool(high_zero[n]),
                  new_safety_failures=[name for name in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound')
                      if c[n,0,COMPONENTS.index(name)]>=1-1e-8 and c[n,chosen[n],COMPONENTS.index(name)]<1-1e-8]))
        path=root/(role+'_per_scene.jsonl')
        import json
        atomic_bytes(path,(''.join(json.dumps(x,allow_nan=False)+'\n' for x in scene_rows)).encode(),immutable=True);outputs.append(path)
        for j,eid in enumerate(meta['expert_ids']):
            m=chosen==j;cost=np.maximum(-delta,0)*m
            switches.append(dict(role=role,expert_id=eid,specialty=specialty[eid],scene_denominator=len(r),selected_count=int(m.sum()),
                  positive_gain_sum=float((np.maximum(delta,0)*m).sum()),negative_cost_sum=float(cost.sum()),
                  positive_mean_over_population=float((np.maximum(delta,0)*m).mean()),negative_mean_over_population=float(cost.mean()),
                  mean_selected_regret=float(regret[m].mean()) if m.any() else None,
                  p95_selected_regret=float(np.quantile(regret[m],.95)) if m.any() else None,
                  high_to_zero=int((high_zero&m).sum()),regret_above_0_1=int(((regret>.1)&m).sum()),
                  regret_above_0_5=int(((regret>.5)&m).sum()),selected_logs=len({scenes[i].source_log_id for i in np.flatnonzero(m)})))
        # All Base-containing subsets; explicit post-hoc designation, never main-pool replacement.
        for bit in range(8):
            ix=[0]+[j+1 for j in range(3) if bit&(1<<j)]
            sub,_=_stats(config,meta,bank,values,ix)
            subsets.append(dict(role=role,status='POSTHOC_SUBPOOL_DIAGNOSTIC',ids='+'.join(meta['expert_ids'][j] for j in ix),
                 scenes=len(r),logs=stats['logs'],base=sub['base_01'],oracle=sub['oracle_01'],selected=sub['selected_01'],
                 oracle_gain=sub['oracle_gain_01'],selected_gain=sub['selected_gain_01'],
                 Oracle_loss_vs_K4=stats['oracle_01']-sub['oracle_01'],high_to_zero=sub['high_to_zero'],selected_zero=sub['selected_zero'],
                 new_NC=sub['safety']['no_at_fault_collisions']['new_failures'],new_DAC=sub['safety']['drivable_area_compliance']['new_failures'],
                 positive_gain=sub['positive_switch_gain_01'],negative_cost=sub['negative_switch_cost_01'],
                 gain_CI_low=sub['gain_ci']['lower_95'],gain_CI_high=sub['gain_ci']['upper_95']))
        def current(s):
            o=read_json(s.observation_ref)
            require(o['token']==s.scene_id and o['log']==s.source_log_id,'current metadata join')
            return o.get('ego_speed'),o.get('navigation'),file_hash(s.observation_ref)
        with ThreadPoolExecutor(max_workers=4) as workers:obs=list(workers.map(current,scenes))
        speed=np.asarray([x[0] if x[0] is not None else np.nan for x in obs]);nav=np.asarray([str(x[1]) for x in obs])
        t=bank['trajectories'];xy=t[row,chosen,:,:2]-t[:,0,:,:2]
        ade=np.linalg.norm(xy,axis=-1).mean(-1);lat=np.abs(xy[...,1]).mean(-1);lon=np.abs(xy[...,0]).mean(-1)
        endpoint=np.linalg.norm(xy[:,-1],axis=-1)
        dt=bound['trajectory']['dt'];chosen_speed=np.linalg.norm(np.diff(t[row,chosen,:,:2],axis=1),axis=-1).mean(-1)/dt
        base_speed=np.linalg.norm(np.diff(t[:,0,:,:2],axis=1),axis=-1).mean(-1)/dt
        strata={'Base_low':base<.8,'Base_high':base>=.8}
        for name,values_,bins in [('ego_speed',speed,rule['speed_bins_mps']),('chosen_Base_ADE',ade,rule['ADE_bins_m']),
                                  ('chosen_Base_lateral',lat,rule['ADE_bins_m']),('chosen_Base_longitudinal',lon,rule['ADE_bins_m']),
                                  ('chosen_Base_endpoint',endpoint,rule['ADE_bins_m']),('absolute_quality_gap',np.abs(delta),rule['gap_bins']),
                                  ('chosen_Base_speed_difference',np.abs(chosen_speed-base_speed),rule['speed_bins_mps'])]:
            for lo,hi in zip(bins[:-1],bins[1:]):strata[f'{name}:[{lo},{hi})']=(values_>=lo)&(values_<hi)
        for n in sorted(set(nav)):strata['current_navigation:'+n]=nav==n
        for log in sorted(set(meta['source_logs'])):strata['source_log:'+log]=np.asarray(meta['source_logs'])==log
        for name in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound'):
            j=COMPONENTS.index(name);strata['selected_failure:'+name]=c[row,chosen,j]<1-1e-8
        if role=='scorer_fit':
            inner_fit=set(split['fit']['scene_ids'])
            strata['internal_fit_R00_seen']=np.asarray([s.scene_id in inner_fit for s in scenes])
            strata['internal_val_R00_seen']=~strata['internal_fit_R00_seen']
            fit_errors=[(n,int(chosen[n])) for n in range(len(r)) if regret[n]>.5]
        for label,m in strata.items():
            if not m.any():continue
            cohorts.append(dict(role=role,stratum=label,scenes=int(m.sum()),logs=len(set(np.asarray(meta['source_logs'])[m])),
                                selected_gain=float(delta[m].mean()),regret=float(regret[m].mean()),
                                positive_gain=float(np.maximum(delta[m],0).mean()),negative_cost=float(np.maximum(-delta[m],0).mean()),
                                high_to_zero=int(high_zero[m].sum())))
        path=root/(role+'_observation_metadata_hashes.json');atomic_json(path,{s.scene_id:o[2] for s,o in zip(scenes,obs)},immutable=True);outputs.append(path)
    # Existing rich-bank coverage of the same in-training large-cost failures.
    rich_meta,rich=load_bank(source,'rich','scorer_fit');dm,db,ds=canonical_bank(config,'scorer_fit')
    require(rich_meta['scene_ids']==dm['scene_ids'],'rich/deploy scene alignment')
    coverage=[]
    for epsilon in (.1,.25,.5,1.):
        hits=[];count=0
        for n,k in fit_errors:
            distance=np.linalg.norm(rich['trajectories'][n,:,:,:2]-db['trajectories'][n,k,:,:2],axis=-1).mean(-1)
            mask=rich['valid'][n] & (distance<=epsilon) & (rich['scores'][n]<=.2)
            count+=int(mask.sum())
            if mask.any():hits.append(n)
        coverage.append(dict(ADE_m=epsilon,severe_fit_scene_denominator=len(fit_errors),near_low_quality_candidates=count,
                              supported_scenes=len(hits),supported_logs=len({ds[i].source_log_id for i in hits}),
                              includes_exact_deployment_candidates=True))
    for name,rows in [('PER_EXPERT_SWITCH_COST.csv',switches),('SUBPOOL_DIAGNOSTICS.csv',subsets),('ERROR_COHORTS.csv',cohorts)]:
        path=root/name;atomic_bytes(path,csv_bytes(rows),immutable=True);outputs.append(path)
    path=root/'RICH_COVERAGE.json';atomic_json(path,dict(bank_hash=rich_meta['hash'],rows=coverage,
             caveat='Includes B_deploy exact matches by construction; presence is not evidence that the model learned the boundary. Cross-log transfer is unproven.'),immutable=True);outputs.append(path)
    path=root/'SUMMARY.json';atomic_json(path,allstats,immutable=True);outputs.append(path)
    lines=['# Existing fixed-pool selection audit','',
        'All scores below are percent PDMS. No new inference or simulator calls. Subpools are post-hoc descriptions, not selected systems.',
        'R00 has seen all D_S, including the new inner validation logs; its inner results are in-sample background only.','',
        '| Role | Scenes/logs | Base | Oracle | R00 selected | Gain | High→0 | New NC/DAC/TTC |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for role,s in allstats.items():
        safety='/'.join(str(s['safety'][x]['new_failures']) for x in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound'))
        lines.append(f"| {role} | {s['scenes']}/{s['logs']} | {100*s['base_01']:.4f} | {100*s['oracle_01']:.4f} | {100*s['selected_01']:.4f} | {100*s['selected_gain_01']:+.4f} | {s['high_to_zero']} | {safety} |")
    lines+=['','Per-expert signed cost, tail regret and source-log contributions are in the accompanying CSVs. Counts are not substituted for loss magnitude.',
            'The new fit/val split and strata were fixed before new model results. Current metadata (speed/navigation) is read-only diagnostic information, not a learned routing rule.',
            'Rich coverage uses actual existing trajectory distances and true low labels; no new candidate construction or hidden target filtering.']
    path=root/'ERROR_DECOMPOSITION.md';atomic_bytes(path,('\n'.join(lines)+'\n').encode(),immutable=True);outputs.append(path)
    path=root/'TRAIN_HELDOUT_ERROR_GAP.md'
    atomic_bytes(path,('R00 training and held-out outcomes use the same fixed four generators. See SUMMARY.json for exact common denominators, signed switch benefit/cost and 2000-log bootstrap intervals.\n'
        'D_router_val is new-model-held-out only; R00 used the entire parent D_S. Do not use its internal score as an unseen-validation control. The original dev has been repeatedly observed and remains exploratory.\n').encode(),immutable=True);outputs.append(path)
    atomic_json(done,dict(status='COMPLETE',rules_hash=digest(rule),outputs={str(p):file_hash(p) for p in outputs},
                         model_inference_calls=0,official_scoring_calls=0),immutable=True)
    return dict(status='COMPLETE',report=str(root/'ERROR_DECOMPOSITION.md'),new_model_calls=0,new_scoring_calls=0)
