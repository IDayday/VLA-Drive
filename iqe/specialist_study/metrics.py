import numpy as np
from iqe.contracts import require
from iqe.scoring.reference import COMPONENTS
from iqe.internal_baseline.evaluation import paired_bootstrap


def stable_order(count,expert_ids=None):
    ids=list(expert_ids) if expert_ids is not None else [f'expert_{i}' for i in range(count)]
    require(len(ids)==count and len(set(ids))==count and ids.count('expert_0')==1,'stable candidate IDs/Base required')
    return ids.index('expert_0'),sorted(range(count),key=lambda i:(ids[i]!='expert_0',ids[i]))


def select(predicted,valid=None,margin=0.,*,expert_ids=None):
    p=np.asarray(predicted);v=np.ones_like(p,dtype=bool) if valid is None else np.asarray(valid,dtype=bool)
    require(p.ndim==2 and v.shape==p.shape and np.isfinite(margin) and margin>=0,'candidate selection shape/margin')
    base,order=stable_order(p.shape[1],expert_ids)
    require(v[:,base].all(),'invalid Base cannot be silently replaced')
    require(np.isfinite(p[v]).all(),'nonfinite Scorer output on valid candidate')
    adjusted=np.where(v,p,-np.inf);winner=np.asarray(order)[adjusted[:,order].argmax(1)]
    chosen=np.where(adjusted[np.arange(len(p)),winner]-adjusted[:,base]>margin,winner,base)
    return chosen


def metrics(scores,components,selected,logs,*,seed=11235,repetitions=2000):
    r=np.asarray(scores,dtype=float);c=np.asarray(components,dtype=float);ix=np.arange(len(r));s=np.asarray(selected,dtype=int)
    require(r.ndim==2 and np.isfinite(r).all() and len(r)>0,'complete matched candidate scores required')
    b=r[:,0];o=r.max(1);v=r[ix,s];delta=v-b;g=float((o-b).mean());regret=float((o-v).mean())
    require(abs(delta.mean()-(g-regret))<1e-12,'Oracle-regret identity failed')
    switch=s!=0;gain=(delta>1e-8)&switch;loss=(delta< -1e-8)&switch
    result={'scenes':len(r),'logs':len(set(logs)),'base_01':float(b.mean()),'fixed_policy_01':r.mean(0).tolist(),
         'oracle_01':float(o.mean()),'selected_01':float(v.mean()),'oracle_gain_01':g,'regret_01':regret,
         'selected_gain_01':float(delta.mean()),'selected_over_best_fixed_01':float(v.mean()-r.mean(0).max()),
         'oracle_capture':float(delta.mean()/g) if abs(g)>1e-8 else None,'common_zero':int((r<=1e-8).all(1).sum()),
         'selected_zero':int((v<=1e-8).sum()),'switch_count':int(switch.sum()),'switch_rate':float(switch.mean()),
         'switch_precision_excluding_ties':float(gain.sum()/(gain.sum()+loss.sum())) if (gain|loss).any() else None,
         'base_fallback_rate':float((~switch).mean()),'positive_switch_gain_01':float(np.maximum(delta,0).mean()),
         'negative_switch_cost_01':float(np.maximum(-delta,0).mean()),'high_to_zero':int(((b>=.8)&(v<=1e-8)).sum()),
         'epsilon_optimal_rate':float((o-v<=.01).mean()),'large_gap_regret_0_1':int((o-v>.1).sum()),
         'large_gap_regret_0_5':int((o-v>.5).sum()),'gain_ci':paired_bootstrap(delta,logs,repetitions,seed),
         'oracle_gain_ci':paired_bootstrap(o-b,logs,repetitions,seed),'safety':{}}
    for name in ('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound'):
        j=COMPONENTS.index(name);old=c[:,0,j];now=c[ix,s,j];valid=np.isfinite(old)&np.isfinite(now)
        result['safety'][name]={'valid_denominator':int(valid.sum()),'base_failures':int((valid&(old<1-1e-8)).sum()),
             'selected_failures':int((valid&(now<1-1e-8)).sum()),'new_failures':int((valid&(old>=1-1e-8)&(now<1-1e-8)).sum())}
    return result
