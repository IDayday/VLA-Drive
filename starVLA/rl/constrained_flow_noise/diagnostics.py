"""Read-only exploration summaries; finite-group opportunity is not deployment."""
import math
import numpy as np
import torch

OFFICIAL_SAFETY=('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound')


def summarize_rows(rows, spec, accepted=None):
    if not rows:raise ValueError('No scenes in exploration denominator')
    rewards=torch.stack([r['rewards'].double() for r in rows]).numpy()
    costs=torch.stack([r['costs'].double() for r in rows]).numpy()
    sigma=torch.cat([r['chain']['old_std'].double() for r in rows]).numpy()
    safe=(costs[...,:3]==0).all(-1)
    official={key:np.array([[r['metrics'][key] for r in scene['records']] for scene in rows])
              for key in rows[0]['records'][0]['metrics']}
    same=lambda a:(a==a[:,0:1]).all(axis=1)
    zero_reward=same(rewards)
    # Exact official-component equality: no inference that equal rewards mean all fail.
    non_progress=[key for key in official if key not in ('score','ego_progress')]
    progress_only=(~zero_reward)&np.logical_and.reduce([same(official[k]) for k in non_progress])
    margins_vary=(costs[...,3:].ptp(axis=1)>0).any(axis=1)
    zero_final=np.array([bool((r['advantage']==0).all()) for r in rows])
    safe_better=(safe & (rewards>rewards.mean(1,keepdims=True)+.001)).any(1)
    ade=[];jitter=[];speed_std=[];forward_std=[];lateral_std=[];curvature_std=[]
    for row in rows:
        request=np.asarray(row['requests']);xy=request[...,:2]
        pair=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1).mean(-1)
        ade.append(float(pair[np.triu_indices(16,1)].mean()))
        jitter.append(float(np.linalg.norm(np.diff(xy,n=2,axis=1),axis=-1).mean()))
        simulated=np.array([r['simulated_states'] for r in row['records']])
        start=simulated[:,0];dx=simulated[:,-1,0]-start[:,0];dy=simulated[:,-1,1]-start[:,1]
        forward=np.cos(start[:,2])*dx+np.sin(start[:,2])*dy
        lateral=-np.sin(start[:,2])*dx+np.cos(start[:,2])*dy
        speed=np.linalg.norm(simulated[:,1:,3:5],axis=-1)
        speed_std.append(float(speed.mean(1).std()));forward_std.append(float(forward.std()));lateral_std.append(float(lateral.std()))
        # Wrapped heading change per travelled metre: a descriptive turn measure.
        heading=np.unwrap(simulated[:,:,2],axis=1)
        distance=np.linalg.norm(np.diff(simulated[:,:,:2],axis=1),axis=-1).sum(1)
        curvature_std.append(float(((heading[:,-1]-heading[:,0])/np.maximum(distance,.1)).std()))
    entropy=np.log(sigma)+.5*math.log(2*math.pi*math.e)
    if spec.variant=='C_SDE':entropy+=.5*7/8*math.log(1-spec.correlation**2)
    result=dict(scenes=len(rows),candidates=int(rewards.size),PDMS_mean=float(rewards.mean()),
        official_means={k:float(v.mean()) for k,v in official.items()},cost_means=costs.mean((0,1)).tolist(),
        zero_PDMS_advantage_groups=int(zero_reward.sum()),all_zero_PDMS_groups=int((rewards==0).all(1).sum()),
        zero_final_advantage_groups=int(zero_final.sum()),progress_only_official_reward_groups=int(progress_only.sum()),
        official_safety_cost_varies_groups=int((costs[...,:3].ptp(1)>0).any(1).sum()),
        geometric_cost_varies_groups=int(margins_vary.sum()),joint_safe_candidate_rate=float(safe.mean()),
        groups_with_safe_candidate=int(safe.any(1).sum()),
        groups_with_safe_candidate_better_than_group_mean_by_0_001=int(safe_better.sum()),
        opportunity_comparator='within-group stochastic mean; never an ODE gain or an oracle policy',
        safe_candidate_mean_PDMS=float(rewards[safe].mean()) if safe.any() else None,
        mean_finite_safe_max_PDMS=float(np.where(safe,rewards,-np.inf).max(1)[safe.any(1)].mean()) if safe.any() else None,
        pairwise_ADE_mean_m=float(np.mean(ade)),request_second_difference_mean_m=float(np.mean(jitter)),
        simulated_scene_mean_speed_std_mean_mps=float(np.mean(speed_std)),simulated_forward_end_std_mean_m=float(np.mean(forward_std)),
        simulated_lateral_end_std_mean_m=float(np.mean(lateral_std)),simulated_turn_per_metre_std_mean=float(np.mean(curvature_std)),
        sigma_mean_by_generation_step_and_channel=sigma.mean((0,1,3)).tolist(),
        sigma_quantiles_0_01_50_99_100_by_generation_step_and_channel=np.quantile(sigma,[0,.01,.5,.99,1],axis=(0,1,3)).tolist(),
        conditional_entropy_per_dimension=float(entropy.mean()),
        seconds={k:sum(float(r[k]) for r in rows) for k in ('rollout_seconds','scoring_seconds','reference_seconds','scoring_cpu_seconds')})
    if spec.variant=='N_FLOW_NOISE':
        fraction=(np.log(sigma)-math.log(spec.min_std))/math.log(spec.max_std/spec.min_std)
        result['sigma_lower_one_percent_logrange_fraction']=float((fraction<=.01).mean())
        result['sigma_upper_one_percent_logrange_fraction']=float((fraction>=.99).mean())
    if accepted is not None:
        result['accepted_groups']=len(accepted)
        result['accepted_cost_means']=costs[accepted].mean((0,1)).tolist() if accepted else None
        result['accepted_PDMS_mean']=float(rewards[accepted].mean()) if accepted else None
        result['discarded_zero_advantage_groups']=int(zero_final.sum())
        result['unused_effective_surplus_groups']=int((~zero_final).sum())-len(accepted)
        # Actual padded graphs are counted, even when their loss factor is zero.
        first=min(16,len(rows));distinct=0
        for slot in range(16):
            pg=accepted[slot] if slot<len(accepted) else (accepted[0] if accepted else 0)
            retention=slot if slot<first else 0
            distinct+=int(pg!=retention)
        result['velocity_evaluations']=dict(rollout_calls=160*len(rows),rollout_examples=160*len(rows),
            frozen_reference_calls=first,frozen_reference_examples=160*first,
            per_inner_PG_reference_forward_calls=16+distinct,per_inner_PG_reference_backward_calls=16+distinct,
            per_inner_PG_reference_examples=160*(16+distinct),
            per_inner_GT_replay_forward_calls=16,per_inner_GT_replay_backward_calls=16,
            per_inner_GT_replay_examples=16*8,GT_FM_repeats=8,inner_epochs=2,
            semantics='DiT velocity evaluations; one pass also produces noise hidden; padding work counted')
    return result


def saturation_streak(previous, summary):
    # Predeclared numerical/exploration pause, not a reward-based early stop.
    saturated=max(summary.get('sigma_lower_one_percent_logrange_fraction',0.),
                  summary.get('sigma_upper_one_percent_logrange_fraction',0.))>=.95
    return previous+1 if saturated else 0
