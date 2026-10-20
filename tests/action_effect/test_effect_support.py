"""Analytic G0-family numeric gate tests, not learned-teacher qualification."""
import numpy as np
import pytest
import torch
from starVLA.dataloader.foresight_dataset import encode_ego
from starVLA.model.modules.action_effect.intervention_basis import BasisSpec,reconstruct
from starVLA.model.modules.action_effect.effect_support import endpoint_support,EndpointSupportSpec


def trajectory(coeff=None):
    base=np.column_stack((np.arange(1,9)*5.,np.zeros(8),np.zeros(8)))
    modified=base if coeff is None else reconstruct(base,coeff,BasisSpec())
    return torch.from_numpy(encode_ego(base))[None],torch.from_numpy(encode_ego(modified))[None]


def test_identity_and_exact_exposed_G0_intervention_are_inside_numeric_envelope():
    base,same=trajectory();assert endpoint_support(base,same)['mask'].tolist()==[True]
    coeff=np.zeros(8);coeff[0]=-.5;coeff[4]=.25
    base,changed=trajectory(coeff);changed.requires_grad_()
    result=endpoint_support(base,changed)
    assert result['mask'].tolist()==[True] and not result['mask'].requires_grad
    assert not result['interpolation_certified'] and not result['is_safety_label']
    assert result['real_executability'].startswith('UNTESTED')


@pytest.mark.parametrize('index,amount',[(0,3.),(4,1.),(1,.3),(5,.3)])
def test_large_radius_and_unexposed_windows_abstain(index,amount):
    coeff=np.zeros(8);coeff[index]=amount
    base,changed=trajectory(coeff);result=endpoint_support(base,changed)
    assert not result['mask'].item() and result['statistics']['status']=='NOT_IDENTIFIABLE'
    assert result['rows'][0]['reasons']


def test_arbitrary_noise_proxy_or_yaw_change_is_not_effect_teacher_support():
    base,changed=trajectory();changed[:,3,0]+=.05
    result=endpoint_support(base,changed)
    assert not result['mask'].item() and 'outside_exposed_smooth_basis' in result['rows'][0]['reasons']
    base,changed=trajectory();changed[...,2]=1.;changed[...,3]=0.
    assert not endpoint_support(base,changed)['mask'].item()
    base,changed=trajectory();changed[...,2:]=0
    assert endpoint_support(base,changed)['rows'][0]['reasons']==['undefined_endpoint_heading']


def test_combining_legal_coefficients_cannot_escape_physical_displacement_radius():
    coeff=np.zeros(8);coeff[[0,2,3]]=.8
    base,changed=trajectory(coeff);result=endpoint_support(base,changed)
    assert not result['mask'].item() and 'longitudinal_displacement_radius' in result['rows'][0]['reasons']


def test_nonfinite_is_model_failure_not_silent_abstention_and_spec_is_fixed():
    base,changed=trajectory();changed[0,0,0]=float('nan')
    with pytest.raises(FloatingPointError):endpoint_support(base,changed)
    with pytest.raises(ValueError,match='untested intervention windows'):
        EndpointSupportSpec(active_coefficients=(True,)*8).validate()
