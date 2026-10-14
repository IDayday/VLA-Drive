from types import SimpleNamespace
import numpy as np
import torch
from starVLA.rl.constrained_flow_noise.diagnostics import summarize_rows,saturation_streak


def test_equal_success_is_not_all_failure_and_covariance_entropy_is_accounted():
    states=np.zeros((41,11));states[:,0]=np.arange(41)*.1;states[:,3]=1
    metrics=dict(score=1.,ego_progress=1.,no_at_fault_collisions=1.,drivable_area_compliance=1.,time_to_collision_within_bound=1.,comfort=1.)
    row=dict(rewards=torch.ones(16),costs=torch.zeros(16,5),advantage=torch.zeros(16),
             chain={'old_std':torch.full((1,16,10,8,4),.02)},requests=np.zeros((16,8,3)),
             records=[dict(metrics=metrics,simulated_states=states)]*16,
             rollout_seconds=1.,scoring_seconds=1.,reference_seconds=1.,scoring_cpu_seconds=1.)
    spec=SimpleNamespace(variant='N_FLOW_NOISE',min_std=.01,max_std=.04,correlation=.8)
    n=summarize_rows([row],spec,[])
    assert n['zero_PDMS_advantage_groups']==1 and n['all_zero_PDMS_groups']==0
    assert n['zero_final_advantage_groups']==1 and n['joint_safe_candidate_rate']==1
    assert n['groups_with_safe_candidate_better_than_group_mean_by_0_001']==0
    assert n['accepted_cost_means'] is None
    assert np.asarray(n['sigma_mean_by_generation_step_and_channel']).shape==(10,4)
    spec.variant='C_SDE';c=summarize_rows([row],spec)
    assert c['conditional_entropy_per_dimension']<n['conditional_entropy_per_dimension']


def test_sigma_pause_requires_consecutive_near_boundary_observations():
    near={'sigma_upper_one_percent_logrange_fraction':.95}
    counter=0
    for _ in range(19):counter=saturation_streak(counter,near)
    assert counter==19
    assert saturation_streak(counter,{'sigma_upper_one_percent_logrange_fraction':.94})==0
    assert saturation_streak(counter,near)==20


def test_arrival_uses_physical_grid_and_separates_horizon_censoring():
    from scripts.analysis.constrained_flow_noise.analyze_ode import arrival_at_distance
    t=np.arange(41)*.1
    assert arrival_at_distance(2*t,3.)==(1.5,False)
    assert arrival_at_distance(t,5.)==(4.,True)
    assert arrival_at_distance(t,4.)==(4.,False)
