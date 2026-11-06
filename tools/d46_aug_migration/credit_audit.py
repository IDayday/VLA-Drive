"""TRAIN-only counterfactual credit audit over complete native collections.

No scores are regenerated. Advantage mass is not a parameter-gradient norm;
alternative weights here are diagnostics, not a new training result.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from starVLA.rl.d46_augustus.split_credit import action_reward,normalize_centered,reverse_travel
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec

SAFETY=('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound')


def temporal_cost(records):
    result=[]
    for r in records:
        m=r['metrics'];cost=sum(w*(1-m[k])for w,k in zip((.4,.35,.25),SAFETY))
        for key,weight in [('official_first_at_fault_collision_seconds',.15),('official_first_ttc_infraction_seconds',.10)]:
            value=r.get(key)
            if value is not None:
                if not np.isfinite(value)or not 0<=value<=4:raise ValueError('Invalid real event time')
                cost+=weight*(1-value/4)
        flags=r['non_drivable_body_by_time']
        if len(flags)<41:raise ValueError('Missing full real safety horizon')
        cost+=.10*np.mean(flags[1:41]);result.append(cost)
    return np.array(result)


def group_standardize(raw):
    return raw/(np.std(raw,ddof=1)+1e-6)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();run=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    identity=json.loads((run/'identity.json').read_text());assert identity['recipe']['training_uses_navtest'] is False
    allowed=set(json.loads(Path(identity['recipe']['population_manifest']).read_text())['tokens'])
    counts=Counter();mass=defaultdict(Counter);square=defaultdict(Counter);seen=set();selection=defaultdict(list);start=time.monotonic()
    for directory in sorted((run/'behavior').glob('*/collection*')):
        paths=[directory/f'rank{r}.pt'for r in range(identity['world'])]
        if not all(p.exists()for p in paths):continue
        scenes=[]
        for path in paths:
            b=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            scenes += [s for s in b['scenes']if not s['padding']]
        pdms=[np.array([r['metrics']['score']for r in s['records']],dtype=np.float64)for s in scenes]
        raw=[x-x.mean()for x in pdms];std=np.std(np.concatenate(raw))
        for s,scores,canonical_raw in zip(scenes,pdms,raw):
            if s['token']not in allowed:raise ValueError('Outside full legal NAVTRAIN')
            records=s['records'];safe=np.array([all(r['metrics'][k]==1 for k in SAFETY)for r in records]);category='all_safe'if safe.all()else'all_unsafe'if not safe.any()else'mixed'
            current=s['advantages'].double().numpy()
            released=action_reward(s);released=normalize_centered(released-released.mean()).numpy()
            values=dict(SafetyEP_global=current,PDMS_group_sample_std_proxy=group_standardize(5*canonical_raw),
                PDMS_global=np.clip(canonical_raw/(std+1e-8),-5,5),released_split_action=released)
            backward=reverse_travel(ActionCodec.denorm_odo(s['chain'][:,-1]))>0
            counts['reverse_gated_candidates']+=int(backward.sum())
            counts['reverse_gated_groups']+=int(backward.any())
            counts['released_signal_but_no_SafetyEP']+=int(np.max(np.abs(released))>1e-8 and np.max(np.abs(current))<1e-8)
            counts['scenes']+=1;counts[category]+=1;counts['candidates']+=len(scores);seen.add(s['token'])
            for name,adv in values.items():
                mass[name][category]+=float(np.abs(adv).sum());square[name][category]+=float(np.square(adv).sum())
                counts[name+'_active_groups']+=int(np.max(np.abs(adv))>1e-8)
            counts['opposite_candidate_sign']+=int(((current*values['released_split_action'])<-1e-8).sum())
            counts['PDMS_signal_but_no_SafetyEP']+=int(np.ptp(scores)>1e-8 and np.max(np.abs(current))<1e-8)
            if np.max(scores)<1e-8:
                counts['all_zero']+=1
                if np.max(np.abs(current))<1e-8:
                    counts['all_zero_no_current_signal']+=1
                    counts['all_zero_temporal_has_signal']+=int(np.ptp(temporal_cost(records))>1e-8)
                if len(selection['all_zero'])<16:selection['all_zero'].append(s['token'])
            if len(selection[category])<16:selection[category].append(s['token'])
        counts['collections']+=1
        if counts['collections']%100==0:
            print(json.dumps(dict(collections=counts['collections'],scenes=counts['scenes'],seconds=time.monotonic()-start)),flush=True)
    result=dict(status='COMPLETE_TRAIN_COUNTERFACTUAL_CREDIT',run=str(run),counts=counts,unique_scenes=len(seen),
                absolute_advantage_mass=mass,squared_advantage_mass=square,NAVTEST_used=False,
                seconds=time.monotonic()-start,normalization_revision='released_population_std_no_epsilon_reverse_gate',
                limits='Different weights on identical collected trajectories; not a training result, not gradient mass. Scene counts are valid exposures; unique scenes separately counted. Temporal variation may provide guidance but is not proof of recoverability.')
    (out/'SUMMARY.json').write_text(json.dumps(result,indent=2))
    (out/'TRAIN_DIAGNOSTIC_SELECTION.json').write_text(json.dumps(dict(source_run=str(run),TRAIN_only=True,strata=selection),indent=2))
    print(json.dumps(result))


if __name__=='__main__':main()
