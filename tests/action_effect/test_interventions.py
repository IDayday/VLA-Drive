import numpy as np
import pytest
from starVLA.model.modules.action_effect.intervention_basis import (
    BasisSpec, basis_matrix, reconstruct, intervention_grid, smoothstep)
from starVLA.model.modules.action_effect.trajectory_constraints import (
    ConstraintSpec, geometric_screen, tracking_screen)


def straight():
    return np.column_stack((np.arange(1,9)*2.5,np.zeros(8),np.zeros(8)))


def test_zero_is_bit_exact_even_heading_wrap():
    a=straight();a[:,2]=np.linspace(3.12,3.15,8)
    np.testing.assert_array_equal(reconstruct(a,np.zeros(8),BasisSpec()),a)


def test_grid_is_fixed_and_has_controls_and_larger_support():
    a,b=intervention_grid(BasisSpec()),intervention_grid(BasisSpec())
    assert a==b and len(a)==24
    assert {x['kind'] for x in a}=={'identity','longitudinal','lateral','joint','larger_support'}


def test_late_intervention_has_no_earlier_change():
    a=straight();c=np.zeros(8);c[6]=.3
    b=reconstruct(a,c,BasisSpec())
    np.testing.assert_array_equal(a[:4],b[:4]);assert b[-1,1]==pytest.approx(.3)


def test_hold_tail_is_in_actual_intervention_support():
    rows=intervention_grid(BasisSpec())
    assert rows[1]['ramp_intervals_seconds']==[[0.,2.]]
    assert rows[1]['active_intervals_seconds']==[[0.,4.]]


def test_low_speed_turn_is_not_hidden_by_curvature_mask():
    a=straight();a[:,:2]*=.01;a[:,2]=np.arange(1,9)*.2
    r=geometric_screen(a,BasisSpec(),ConstraintSpec())
    assert not r['geometrically_admissible'] and 'curvature_inv_m' in r['failures']
    a[:,:2]=0
    r=geometric_screen(a,BasisSpec(),ConstraintSpec())
    assert 'yaw_change_without_translation' in r['failures']


def test_smoothstep_support_derivatives():
    h=1e-4
    assert smoothstep(-1)==0 and smoothstep(2)==1
    assert abs(float(smoothstep(h))/h)<1e-6
    assert abs(float(1-smoothstep(1-h))/h)<1e-6


def test_coefficients_have_metres_not_normalized_units():
    c=np.zeros(8);c[3]=1
    b=reconstruct(straight(),c,BasisSpec())
    assert b[-1,0]-straight()[-1,0]==pytest.approx(1)
    assert basis_matrix(straight(),BasisSpec()).shape==(8,2,8)


def test_dynamic_rejection_is_not_zero_reward():
    a=straight();a[3,0]+=15
    j=geometric_screen(a,BasisSpec(),ConstraintSpec())
    assert not j['geometrically_admissible'] and not j['is_safety_label']
    assert 'score' not in j


def test_true_native_global_tracking_transform():
    a=straight();t=np.arange(41)*.1;yaw=.7;c,s=np.cos(yaw),np.sin(yaw)
    states=np.zeros((41,11));states[:,0]=100+5*t*c;states[:,1]=200+5*t*s;states[:,2]=yaw
    r=tracking_screen(a,states,BasisSpec(),ConstraintSpec())
    assert r['controller_tracking_pass'] and r['max_xy_error_m']<1e-12


def test_tracking_mismatch_and_time_contract_rejected():
    states=np.zeros((41,11));r=tracking_screen(straight(),states,BasisSpec(),ConstraintSpec())
    assert not r['controller_tracking_pass']
    with pytest.raises(ValueError):tracking_screen(straight(),states[:20],BasisSpec(),ConstraintSpec())


def test_nonfinite_and_foreign_dimensions_fail():
    a=straight();a[1,0]=np.nan
    with pytest.raises(ValueError):reconstruct(a,np.zeros(8),BasisSpec())
    with pytest.raises(ValueError):reconstruct(straight(),np.zeros(4),BasisSpec())
