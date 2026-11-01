"""CPU-only full NAVTEST comparisons and finite review owner.

Never changes a training stream, budget, checkpoint, official summary or score.
"""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from starVLA.rl.qwen_world_safety_exploration.contracts import ROOT, fingerprint, file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json
from tools.qwen_world_research.evaluate import load_evaluation, read_sealed, physical_log
from tools.qwen_world_postcot.summarize import load as load_postcot, paired_statistics
from .notify import refresh

OLD=Path('/mnt/project/qwen-world-postcot-20261025')
BASELINES={
 'IL70':OLD/'IL_late_checkpoints_NAVTEST_20261029T0310Z/WORLD_IL_070000/evaluation',
 'SafetyEP10k':OLD/'rl_optimization_20261030/evaluations/PPO_SAFETY_EP_u010000_NAVTEST',
 'SafetyEP1k':OLD/'rl_optimization_20261030/evaluations/PPO_SAFETY_EP_u001000_NAVTEST',
}
BINS=('zero','low','middle','high')


def bin_index(x):
    return 0 if x<=1e-8 else 1 if x<.5 else 2 if x<.9 else 3


def cohort_transitions(rows,base):
    """Fixed baseline cohorts; rows never silently filtered by outcome."""
    if set(rows)!=set(base) or not rows:raise ValueError('Exact paired population required')
    table=np.zeros((4,4),dtype=np.int64);cohorts={}
    for token,row in rows.items():
        a=float(base[token]['feedback']['metrics']['score']);b=float(row['feedback']['metrics']['score'])
        table[bin_index(a),bin_index(b)]+=1
    for k,name in enumerate(BINS):
        keys=[t for t in rows if bin_index(float(base[t]['feedback']['metrics']['score']))==k]
        delta=np.asarray([rows[t]['feedback']['metrics']['score']-base[t]['feedback']['metrics']['score'] for t in keys],dtype=np.float64)
        cohorts[name]=dict(count=len(keys),mean_delta_PDMS=None if not keys else 100*float(delta.mean()),
            global_contribution_PDMS=100*float(delta.sum())/len(rows),
            to_zero=int(table[k,0]),below_high=int(table[k,:3].sum()))
    ds=np.asarray([rows[t]['feedback']['metrics']['score']-base[t]['feedback']['metrics']['score'] for t in rows],dtype=np.float64)
    return dict(bins=['score<=1e-8','0<score<.5','.5<=score<.9','score>=.9'],
        row_is_fixed_baseline=True,transition_counts=table.tolist(),fixed_cohorts=cohorts,
        G_plus_PDMS=100*float(np.maximum(ds,0).sum())/len(rows),G_minus_PDMS=100*float(np.minimum(ds,0).sum())/len(rows),
        losses_are_signed=True,full_population=len(rows))


def normalized_inference(value):
    canonical=dict(sampler='ODE',steps=10,shift=5.0,seed=2101,scene_seed_namespace='native_qwen_sourceflow_ODE',candidates=1)
    if value=='single_candidate_cleanx_K10_shift5' or value==canonical:return canonical
    raise ValueError('Unknown deployment sampler representation')


def compatible(a,b):
    if normalized_inference(a['inference'])!=normalized_inference(b['inference']):raise ValueError('Different paired inference')
    for k in ('cohort','token_sha256','data_sha256','seed','numerics','qualification'):
        if a.get(k)!=b.get(k):raise ValueError('Different paired protocol: '+k)
    if a['protocol']['sha256']!=b['protocol']['sha256']:raise ValueError('Different canonical scorer')
    if a.get('parent_model_sha256') and b.get('parent_model_sha256') and a['parent_model_sha256']!=b['parent_model_sha256']:
        raise ValueError('Different IL family')


def load_full(path):
    path=Path(path);v=read_sealed(path/'identity.json')
    ident,rows=(load_evaluation(path) if v['schema']=='world_rl_research_evaluation_v1' else load_postcot(path))
    if ident['cohort']!='NAVTEST' or len(rows)!=12146 or ident.get('qualification') is not False:
        raise ValueError('Only complete formal full NAVTEST')
    return ident,rows


def paired(rows,reference):
    result=paired_statistics(rows,reference)
    result['PDMS_delta']=100*result['mean_delta'];result['PDMS_CI95']=[100*x for x in result['CI95']]
    result['fixed_0p9_cohorts']=cohort_transitions(rows,reference)
    result['legacy_high_0p8_caveat']='Original paired high_to_zero uses .8; fixed_0p9_cohorts explicitly uses .9.'
    return result


class Review:
    def __init__(self):
        self.out=ROOT/'analysis/PILOT_COMPARISONS';self.out.mkdir(parents=True,exist_ok=True)
        self.lock=(self.out/'owner.lock').open('a');fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.cache={}
        atomic_json(self.out/'OWNER.json',dict(pid=os.getpid(),start_ticks=int(Path('/proc/self/stat').read_text().split()[21]),
            source=__file__,source_sha256=file_sha(__file__),no_GPU=True,started_unix=time.time()))

    def get(self,path):
        path=str(path)
        if path not in self.cache:self.cache[path]=load_full(path)
        return self.cache[path]

    def compare(self,target,base,*,label):
        i,rs=self.get(target);j,bs=self.get(base);compatible(i,j)
        return dict(label=label,baseline=str(base),baseline_identity_sha256=j['sha256'],
            result=paired(rs,bs))

    def tick(self):
        registry=json.loads((ROOT/'RUNS.json').read_text());ready={};pending=[];published=[]
        for run in registry['runs']:
            for step in run['gates']:
                label=f"{run['id']}_u{step:06}"
                ev=ROOT/'evaluations'/label/'NAVTEST';summary=ev/'SUMMARY.json'
                if not summary.exists():pending.append(label);continue
                s=read_sealed(summary)
                if s.get('status')!='COMPLETE' or s.get('scenes')!=12146 or s.get('invalid_scenes')!=0:
                    raise ValueError('Incomplete purported full summary: '+label)
                ready[(run['id'],step)]=(ev,s)
        for (arm,step),(ev,s) in ready.items():
            p=self.out/(arm+f'_u{step:06}.json')
            if p.exists():continue
            comparisons={name:self.compare(ev,base,label=name)for name,base in BASELINES.items()}
            payload=dict(schema='world_Safety_exploration_full_population_review_v1',status='COMPLETE',arm=arm,updates=step,
                evaluation=str(ev),summary_sha256=s['sha256'],PDMS=s['PDMS'],comparisons=comparisons,
                official_EP_is_safety_gated=True,EP_not_physical_progress=True,
                NAVTEST_never_enters_training_selection=True)
            payload['sha256']=fingerprint(payload);atomic_json(p,payload);published.append(str(p))
        contrasts=[('SAFETY_G16',500,'SAFETY_G8_CONTROL',1000,'same_16000_candidates_different_scenes_updates'),
            ('SAFETY_G16',1000,'SAFETY_G8_CONTROL',1000,'same_scenes_updates_twice_candidates')]
        for arm in ('SAFETY_DAC_DENSE_G8','SAFETY_ACTIVE_DATA_G8','SAFETY_SATURATED_G8','SAFETY_PROTECTED_HIGH_G8','SAFETY_INVERSION_CANCEL_G8'):
            contrasts.extend((arm,u,'SAFETY_G8_CONTROL',u,'same_scene_exposures_updates_candidates_actual_token_coverage_may_differ')for u in (500,1000))
        for a,u,b,v,budget in contrasts:
            if (a,u)not in ready or (b,v)not in ready:continue
            p=self.out/(f'{a}_{u}_vs_{b}_{v}.json')
            if p.exists():continue
            payload=dict(schema='world_Safety_exploration_between_arms_v1',status='COMPLETE',arm=a,updates=u,
                reference_arm=b,reference_updates=v,budget_contract=budget,
                comparison=self.compare(ready[(a,u)][0],ready[(b,v)][0],label=b))
            payload['sha256']=fingerprint(payload);atomic_json(p,payload);published.append(str(p))
        status=dict(status='WAITING_COMPLETE_NAVTEST' if pending else 'COMPLETE_AT_REGISTERED_GATES',
            ready=len(ready),pending=pending,published=published,heartbeat_unix=time.time(),no_training_mutations=True)
        atomic_json(self.out/'STATE.json',status)
        if published:refresh(self.out/'STATE.json','Full paired pilot analysis completed')
        return status


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=float,default=60)
    a=p.parse_args()
    if a.interval<10:p.error('interval>=10')
    review=Review()
    while True:
        try:state=review.tick()
        except Exception:
            atomic_json(review.out/'ERROR.json',dict(status='REVIEW_REQUIRED',error=traceback.format_exc(),at_unix=time.time()));raise
        if not a.watch or not state['pending']:break
        if (review.out/'STOP_OWNER').exists():
            atomic_json(review.out/'OWNER_STOPPED.json',dict(pid=os.getpid(),at_unix=time.time(),existing_scores_untouched=True));break
        time.sleep(a.interval)

if __name__=='__main__':main()
