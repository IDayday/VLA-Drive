"""Read-only complete B1 Navtest pairing against the fixed D44 IL parent."""
import argparse
from pathlib import Path
import numpy as np
from tools.qwen_world_compare.evaluate import load_evaluation,summarize
from tools.qwen_world_postcot.summarize import METRICS
from starVLA.rl.flow_grpo.transactions import atomic_json


def contributions(rows,base):
    if set(rows)!=set(base)or not rows:raise ValueError('Exactly paired population required')
    tokens=list(rows);old=np.array([base[t]['feedback']['metrics']['score']for t in tokens]);new=np.array([rows[t]['feedback']['metrics']['score']for t in tokens]);delta=new-old
    high=old>=.9;zero=old<=1e-8;low=old<.6
    safe=lambda source:np.array([all(source[t]['feedback']['metrics'][m]==1 for m in METRICS[1:4])for t in tokens])
    oldsafe,newsafe=safe(base),safe(rows)
    cohorts={}
    for name,mask in [('IL44_high_ge_0.9',high),('IL44_zero',zero),('IL44_other',~(high|zero)),('IL44_joint_safe',oldsafe)]:
        cohorts[name]=dict(scenes=int(mask.sum()),mean_delta=None if not mask.any()else float(delta[mask].mean()),
            population_contribution=float(delta[mask].sum()/len(tokens)),new_zero=int((mask&(old>1e-8)&(new<=1e-8)).sum()))
    return dict(G_plus=float(np.maximum(delta,0).mean()),G_minus=float(np.minimum(delta,0).mean()),net=float(delta.mean()),
        fixed_reference_cohorts=cohorts,high_to_low=int((high&(new<.6)).sum()),low_to_high=int((low&(new>=.9)).sum()),
        initial_high_retained_ge_0_9=int((high&(new>=.9)).sum()),initial_high_denominator=int(high.sum()),
        new_joint_unsafe=int((oldsafe&~newsafe).sum()),repaired_joint_unsafe=int((~oldsafe&newsafe).sum()),
        score_units='raw [0,1]; multiply by100 for PDMS points',
        limits='Reference cohorts fixed by IL44. Single training seed42/evaluation seed2101; no seed uncertainty estimated. Zero observed failures are not a zero-risk guarantee.')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--evaluation',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    value=summarize(a.evaluation,a.baseline);identity,rows=load_evaluation(a.evaluation);baseid,base=load_evaluation(a.baseline)
    if identity['qualification']or identity['cohort']!='NAVTEST'or len(rows)!=12146 or value['physical_logs']!=44:
        raise ValueError('Research performance analysis requires complete12146/44 full Navtest; qualification excluded')
    value['gain_loss']=contributions(rows,base)
    output=Path(a.output)
    if output.exists():raise FileExistsError('Preserve previous derived summary; select new analysis output')
    atomic_json(output,value)


if __name__=='__main__':main()
