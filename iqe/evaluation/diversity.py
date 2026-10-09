"""Trajectory geometry is descriptive; useful coverage requires real quality labels."""
import numpy as np


def candidate_diagnostics(trajectories,scores,valid,components,component_valid,expert_ids,groups,high=.8):
    tau,scores,valid=np.asarray(trajectories),np.asarray(scores),np.asarray(valid,bool)
    quality=valid & (scores>=high)
    for c in ('no_at_fault_collisions','drivable_area_compliance'):
        quality &= np.asarray(component_valid[c],bool) & (np.asarray(components[c])>=1-1e-8)
    pairs={}
    for i,eid in enumerate(expert_ids):
        for j,other in enumerate(expert_ids[:i]):
            mask=valid[:,i]&valid[:,j]
            diff=np.linalg.norm(tau[mask,i,:,:2]-tau[mask,j,:,:2],axis=-1)
            pairs[eid+'__'+other]={'valid_scenes':int(mask.sum()),'ADE_m':float(diff.mean()) if mask.any() else None,
                'FDE_m':float(diff[:,-1].mean()) if mask.any() else None,
                'safe_high_quality_overlap':int((quality[:,i]&quality[:,j]).sum())}
    return {'pairwise':pairs,'geometry_is_not_quality':True,
        'distribution_centers_xy':{eid:tau[valid[:,i],i,:,:2].mean(axis=(0,1)).tolist() if valid[:,i].any() else None for i,eid in enumerate(expert_ids)},
        'per_group_scores':{g:{eid:float(scores[(np.array(groups)==g)&valid[:,i],i].mean()) if ((np.array(groups)==g)&valid[:,i]).any() else None for i,eid in enumerate(expert_ids)} for g in sorted(set(groups))},
        'exclusive_safe_high_quality':{eid:int((quality[:,i] & (quality.sum(1)==1)).sum()) for i,eid in enumerate(expert_ids)}}
