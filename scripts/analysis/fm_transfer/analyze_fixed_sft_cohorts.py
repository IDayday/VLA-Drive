"""Describe RL effects in SFT-defined cohorts using the registered 0.9 threshold.

Cohorts never depend on the RL outcome. No fitting, curriculum or new rollout.
TRAIN diagnostic, DEV and Navtest are reported independently.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from scripts.analysis.simwam_recipe.analyze_evaluation import read,METRICS
from scripts.analysis.fm_transfer.analyze_evaluation import match_transfer
from scripts.analysis.fm_transfer.analyze_safety_transitions import source_reward
from starVLA.rl.fm_transfer.contracts import require_dispatch
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference
from starVLA.rl.flow_grpo.transactions import atomic_json


def cohort_statistics(base,current,logs):
    safe=(base[:,1:4]==1).all(1)
    masks=dict(SFT_safe_score_ge09=safe&(base[:,0]>=.9),
        SFT_safe_score_lt09=safe&(base[:,0]<.9),SFT_not_fully_safe=~safe)
    if not np.all(np.stack(list(masks.values())).sum(0)==1):raise ValueError('Cohorts must partition scenes')
    new_safe=(current[:,1:4]==1).all(1);delta=current[:,0]-base[:,0]
    rdelta=source_reward(current)-source_reward(base)
    out={}
    for name,mask in masks.items():
        indices=np.flatnonzero(mask);n=len(indices)
        out[name]=dict(scenes=n,logs=len({logs[i] for i in indices}),
            reference_mean={k:float(base[mask,j].mean()) if n else None for j,k in enumerate(METRICS)},
            current_mean={k:float(current[mask,j].mean()) if n else None for j,k in enumerate(METRICS)},
            PDMS_delta=log_bootstrap_difference(delta[mask],[logs[i] for i in indices]) if n else None,
            source_reward_mean_delta=float(rdelta[mask].mean()) if n else None,
            positive_gain_sum=float(delta[mask&(delta>0)].sum()),negative_degradation_sum=float(-delta[mask&(delta<0)].sum()),
            new_joint_violations=int((mask&safe&~new_safe).sum()),
            joint_recoveries=int((mask&~safe&new_safe).sum()),
            nonzero_to_zero=int((mask&(base[:,0]>0)&(current[:,0]==0)).sum()),
            zero_to_nonzero=int((mask&(base[:,0]==0)&(current[:,0]>0)).sum()),
            safety={k:dict(new=int((mask&(base[:,j]==1)&(current[:,j]<1)).sum()),
                recovered=int((mask&(base[:,j]<1)&(current[:,j]==1)).sum())) for j,k in enumerate(METRICS[1:4],1)})
    return out


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--variant',required=True,choices=['A_DDP_TIME','B_SOURCE_ACTION_DIT'])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();require_dispatch();start=time.monotonic()
    if a.output.exists():raise FileExistsError(a.output)
    root=Path('/mnt/project/fm-transfer-20261016');sft_step=20000 if a.variant=='A_DDP_TIME' else 100000
    result=dict(status='COMPLETE_FIXED_SFT_COHORT_ANALYSIS',variant=a.variant,source_sha256=file_sha(__file__),
        threshold=.9,threshold_source='unchanged registered SFT hard-selection cutoff',populations={},new_inference=0,new_scoring=0)
    for pop in ('TRAIN_DIAGNOSTIC','DEV','NAVTEST'):
        if pop=='NAVTEST':
            cur=root/'evaluations_scaleout'/f'{a.variant}_RL_final_NAVTEST';ref=root/'evaluations_scaleout'/f'{a.variant}_SFT_final_NAVTEST'
        else:
            cur=root/'evaluations'/f'{a.variant}_RL_update020000_{pop}';ref=root/'evaluations'/f'{a.variant}_SFT_update{sft_step:06}_{pop}'
        ci,tokens,logs,current,_,_=read(cur);ri,rt,rl,base,_,_=read(ref)
        match_transfer(ci,ri)
        if tokens!=rt or logs!=rl:raise ValueError('Unpaired population')
        result['populations'][pop]=dict(scenes=len(tokens),logs=len(set(logs)),
            current_manifest_sha256=file_sha(cur/'manifest.json'),reference_manifest_sha256=file_sha(ref/'manifest.json'),
            modes={m:cohort_statistics(base[m],current[m],logs) for m in ci['modes']})
    result['wall_seconds']=time.monotonic()-start
    result['limits']=['SFT single-seed cohorts are not all-noise safety guarantees.',
        'Membership resembles hard-selection criteria but DEV/Navtest never enter updates.',
        'TRAIN diagnostic contains targeted cases; no unbiased full-TRAIN claim.',
        'Observed cohort differences do not isolate a causal data-curriculum intervention.']
    atomic_json(a.output,result);print(json.dumps(dict(status=result['status'],variant=a.variant,wall_seconds=result['wall_seconds'])))


if __name__=='__main__':main()
