"""Existing complete official scores: exact zero transfers and overlapping failures."""
import argparse,csv,json,math
from pathlib import Path
from collections import Counter
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import file_sha256
from tools.foresight.summarize_experiments import paired_difference,collapse_sampling_runs
from tools.local_interaction_mask_v2.compare_pdms import read

FACTORS={'NC':'no_at_fault_collisions','DAC':'drivable_area_compliance','TTC':'time_to_collision_within_bound',
         'EP':'ego_progress','Comfort':'comfort','DDC':'driving_direction_compliance'}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--results',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    # Fixed before reading subgroup outcomes; exploratory, not a confirmation set.
    atomic_json(out/'RULES.json',{'high_score_threshold':.95,'low_score_threshold':.5,
        'zero':'official score exactly zero','pairs':['S4-S3','S0-C1','S1-S0','S3-S2'],
        'failures':'overlapping factor<1 labels, no exclusive causal assignment','case_quantiles':[.05,.25,.5,.75,.95]})
    sources={}
    for r in csv.DictReader(Path(a.results).open()):
        if r['updates']=='100000':sources[(r['split'],r['arm'])]=Path(r['score_summary']).parent/'scenes.csv'
    reports={}
    for split in ('dev','navtest'):
      for first,base in [('S4','S3'),('S0','C1'),('S1','S0'),('S3','S2')]:
        pa,pb=sources[(split,first)],sources[(split,base)];aa,bb=read(pa),read(pb)
        if set(aa)!=set(bb):raise ValueError('Populations differ')
        counts=Counter();rows=[]
        for token in sorted(aa):
            x,y=aa[token],bb[token]
            if x['log']!=y['log'] or x['status']!='ok' or y['status']!='ok':raise ValueError('Incomplete official pairing')
            sx,sy=float(x['score']),float(y['score']);transition=('both_zero' if sx==sy==0 else 'old_zero_new_nonzero' if sy==0 else 'old_nonzero_new_zero' if sx==0 else 'both_nonzero')
            counts[transition]+=1
            row={'token':token,'log':x['log'],'first':sx,'base':sy,'delta':sx-sy,'zero_transition':transition}
            for short,key in FACTORS.items():
                if key in x and key in y:
                    row[short+'_first']=float(x[key]);row[short+'_base']=float(y[key]);row[short+'_delta']=float(x[key])-float(y[key])
                    if float(x[key])<1:counts['new_factor_'+short+'_below1']+=1
                    if float(y[key])<1:counts['old_factor_'+short+'_below1']+=1
            rows.append(row)
        name=split+'_'+first+'-'+base
        collapsed_a,_=collapse_sampling_runs({42:aa},[42]);collapsed_b,_=collapse_sampling_runs({42:bb},[42])
        paired,_,_=paired_difference(collapsed_a,collapsed_b)
        ordered=sorted(rows,key=lambda r:(r['delta'],r['token']))
        cases=[ordered[round(q*(len(ordered)-1))] for q in (.05,.25,.5,.75,.95)]
        reports[name]={'scenes':len(rows),'counts':dict(counts),'paired':paired,'source_hashes':[file_sha256(pa),file_sha256(pb)],'cases_by_fixed_delta_quantile':cases,
            'event_times':'NOT_AVAILABLE_IN_SCORE_CSV; factor statistics do not establish event timing or causal failure type'}
        with (out/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    atomic_json(out/'SUMMARY.json',reports)

if __name__=='__main__':main()
