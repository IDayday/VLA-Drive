"""Pure diagnostic arithmetic tests; fixtures are not physical repair evidence."""
from copy import deepcopy

import pytest

from tools.action_effect.analyze_physical_interventions import summarize


def fixture_population():
    names=['score','no_at_fault_collisions','drivable_area_compliance','ego_progress',
        'time_to_collision_within_bound','comfort','driving_direction_compliance',
        'progress_raw_meters','first_at_fault_time_seconds','first_ttc_state_time_seconds','first_road_time_seconds']
    def row(cid,kind,score,nc=1.,distance=.1,coeff=None,eligible=False,onset=0.):
        return {'candidate_id':cid,'kind':kind,'status':'VALID_SCORED','distance_m':distance,
            'metrics':dict(zip(names[:7],[score,nc,1.,1.,1.,1.,1.])),
            'progress_raw_meters':10.,'executable':True,'eligible_repair':eligible,
            'coefficients':coeff,'trajectory_sha256':cid,'ramp_intervals_seconds':[[onset,2. if onset==0 else 4.]],
            'active_intervals_seconds':[[onset,4.]]}
    along=[-.5,0,0,0,0,0,0,0];lateral=[0,0,0,0,.15,0,0,0]
    joint=[-.5,0,0,0,.15,0,0,0];late=[0,0,0,0,0,0,.15,0]
    first=[row('b','identity',.2,nc=0.,coeff=[0.]*8),
        row('long','longitudinal',.4,coeff=along),row('lat','lateral',.5,nc=0.,coeff=lateral),
        row('joint','joint',.9,distance=.4,coeff=joint,eligible=True),
        row('late','lateral',.5,nc=0.,coeff=late,onset=2.),
        row('large','larger_support',.99,distance=1.,coeff=[0,0,0,-2.,0,0,0,0],eligible=True),
        row('sample','policy_resample',1.,distance=2.,eligible=True)]
    second=[row('b','identity',.9,coeff=[0.]*8),row('bad','lateral',0.,nc=0.,coeff=lateral)]
    scenes=[{'scene_id':'fit','source_log':'log-fit','role':'fit','candidates':first,
             'repair':{'status':'VERIFIED_LOCAL_REPAIR','selected':'joint'}},
            {'scene_id':'validation','source_log':'log-validation','role':'validation','candidates':second,
             'repair':{'status':'NO_CHANGE_REQUIRED','selected':'b'}}]
    repairs=[{'scene_id':s['scene_id'],'source_log':s['source_log'],'role':s['role']} for s in scenes]
    pairs=[]
    def labels(r):
        values=[r['metrics'][n] for n in names[:7]]+[r['progress_raw_meters'],3. if r['metrics']['no_at_fault_collisions']==0 else 0.,0.,0.]
        return {'names':names,'values':values,'valid':[True]*8+[r['metrics']['no_at_fault_collisions']==0,False,False]}
    for s in scenes:
        base=s['candidates'][0]
        for r in s['candidates'][1:]:
            if r['kind']=='policy_resample':continue
            pairs.append({'scene_id':s['scene_id'],'candidate_id':r['candidate_id'],'source_log':s['source_log'],
                'role':s['role'],'base_labels':labels(base),'modified_labels':labels(r),
                'event_transition':{name:{'repaired':name=='collision' and base['metrics']['no_at_fault_collisions']==0 and r['metrics']['no_at_fault_collisions']==1,
                    'introduced':name=='collision' and base['metrics']['no_at_fault_collisions']==1 and r['metrics']['no_at_fault_collisions']==0}
                    for name in ('collision','road','ttc')}})
    return scenes,pairs,repairs


def test_fixed_failure_denominators_hold_tail_and_nullable_event_times():
    result=summarize(*fixture_population())
    assert result['fixed_failure_scenes']==1 and result['roles']['validation']['fixed_failures']==0
    late=next(r for r in result['temporal_intervention_event_observations'] if r['candidate_id']=='late')
    assert late['actual_active_intervals_seconds']==[[2.,4.]]
    assert late['events']['collision']['base_event_minus_onset_seconds']==1.
    assert late['events']['collision']['base_event_minus_ramp_end_seconds']==-1.
    assert late['events']['road']['base_first_seconds'] is None
    assert late['events']['road']['modified_minus_base_first_seconds'] is None
    assert late['events']['road']['time_delta_valid'] is False
    assert result['continuous_road_margin']['status']=='UNAVAILABLE'


def test_joint_factorial_interaction_and_minimal_repair_are_distinct_from_support():
    result=summarize(*fixture_population())
    assert result['joint_score_interaction']['n']==1
    assert result['joint_score_interaction']['mean']==pytest.approx(.2)
    selected=result['selected_minimal_repairs'][0]
    assert selected['selected_candidate']=='joint' and selected['maximum_score_eligible_candidate']=='joint'
    assert result['proposal_budget_curve'][-1]['local_repair_scenes']==1
    assert result['proposal_budget_curve'][-1]['support_success_scenes']==1
    assert result['minimal_repair_distance_m']['mean']==.4


def test_high_score_light_damage_and_safety_progress_conflicts_not_hidden():
    scenes,pairs,repairs=fixture_population()
    scenes[1]['candidates'][1]['progress_raw_meters']=12.
    result=summarize(scenes,pairs,repairs)
    lateral=result['by_kind']['lateral']
    assert lateral['high_light_pairs']==1 and lateral['high_light_to_zero']==1
    assert lateral['new_NC']==1 and lateral['progress_increase_with_safety_loss']==1
    assert lateral['original_high_pairs']==1


def test_wrong_scene_join_and_missing_valid_pair_rejected():
    scenes,pairs,repairs=fixture_population()
    wrong=deepcopy(pairs);wrong[0]['source_log']='other-log'
    with pytest.raises(ValueError,match='Pair/source join'):summarize(scenes,wrong,repairs)
    with pytest.raises(ValueError,match='lacks pinned effect pair'):summarize(scenes,pairs[1:],repairs)


def test_nonminimal_selected_repair_is_rejected():
    scenes,pairs,repairs=fixture_population()
    scenes[0]['candidates'][1]['eligible_repair']=True
    with pytest.raises(ValueError,match='not distance minimal'):summarize(scenes,pairs,repairs)
