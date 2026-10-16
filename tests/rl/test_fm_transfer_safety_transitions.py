import numpy as np
import pytest
from scripts.analysis.fm_transfer.analyze_safety_transitions import (
    physical_event_pair, source_reward, subsets, summarize_subset, distribution)


def row(ep, ttc, nc=1., dac=1., comfort=1.):
    return [nc*dac*(5*ep+5*ttc+2*comfort)/12,nc,dac,ttc,ep,comfort,1.]


def test_source_improvement_can_disagree_with_canonical_but_is_not_assumed():
    old=np.array([row(.4,1),row(.8,1)])
    new=np.array([row(1,0),row(.9,0)])
    assert source_reward(new)[0]>source_reward(old)[0] and new[0,0]<old[0,0]
    motion=[dict(speed=1.,time90=None)]*2
    result=summarize_subset(np.array([True,True]),old,new,motion,motion,['one','one'])
    assert result['source_improved_metric_degraded']==1
    assert result['source_improved_metric_degraded_both_ge005']==1
    assert result['scenes']==2 and result['logs']==1
    assert result['motion']['time90']['delta']['defined']==0


def test_subsets_retain_fractional_nc_and_separate_new_recovered_retained():
    old=np.array([row(.5,1),row(.5,0),row(.5,0),row(.5,1,nc=.5)])
    new=np.array([row(.7,0),row(.5,1),row(.7,0),row(.7,1,nc=.5)])
    selected=subsets(old,new)
    assert selected['new_TTC'].tolist()==[True,False,False,False]
    assert selected['recovered_TTC'].tolist()==[False,True,False,False]
    assert selected['retained_TTC_failure'].tolist()==[False,False,True,False]
    assert selected['SFT_safe_new_joint_violation'].tolist()==[True,False,False,False]
    assert not selected['both_safe_practical_improvement'].any()


def records():
    old=np.zeros((41,11));new=old.copy()
    old[:,0]=new[:,0]=100.;old[:,1]=new[:,1]=200.
    old[:,2]=new[:,2]=np.pi/2
    old[:,1]+=np.arange(41)*.1;new[:,1]+=np.arange(41)*.2
    return dict(simulated_states=old),dict(simulated_states=new)


def test_physical_event_pair_uses_same_time_and_initial_vehicle_frame():
    old,new=records();x=physical_event_pair(old,new,1.2)
    assert x['initial_states_exactly_equal']
    assert x['forward_difference_m']==pytest.approx(1.2)
    assert x['preceding_segment_speed_difference_mps']==pytest.approx(1.)
    assert x['reference_abs_lateral_m']==pytest.approx(0,abs=1e-10)
    assert physical_event_pair(old,new,0.)['preceding_segment_speed_difference_mps'] is None


@pytest.mark.parametrize('timestamp',[-.1,1.25,4.1])
def test_no_flow_step_or_off_grid_event_interpretation(timestamp):
    with pytest.raises(ValueError): physical_event_pair(*records(),timestamp)


def test_changed_initial_state_and_invalid_score_cannot_be_ignored():
    old,new=records();new['simulated_states'][0,0]+=1
    with pytest.raises(ValueError): physical_event_pair(old,new,1.)
    values=np.array([row(.5,1)]);values[0,0]=0
    with pytest.raises(ValueError): source_reward(values)
    with pytest.raises(ValueError): distribution([1.,np.nan])


def test_event_summary_serializes_numpy_scorer_scalars(monkeypatch):
    import json
    from scripts.analysis.fm_transfer import analyze_safety_transitions as module
    old,new=records()
    for item in (old,new):
        item.update(cache={'sha256':'same'},protocol_sha256='same',
            metrics={'no_at_fault_collisions':np.float64(1.)},
            official_first_ttc_infraction_seconds=np.float64(1.2))
    monkeypatch.setattr(module,'stored_record',lambda root,*args: old if root=='reference' else new)
    masks={'new_TTC':np.array([True]),'recovered_TTC':np.array([False]),'retained_TTC_failure':np.array([False])}
    result=module.events('current','reference','E_NATIVE',['token'],masks)
    assert json.loads(json.dumps(result,allow_nan=False))['new_TTC']['current_NC_not_fully_pass']==0
