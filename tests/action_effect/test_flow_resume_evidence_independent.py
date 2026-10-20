"""Independent equality primitive negatives; no real trainer or GPU calls."""
import copy
import numpy as np
import pytest
import torch
from tools.action_effect.audit_flow_resume_evidence import compare_values,exact_fingerprint


def test_bitwise_tensor_signed_zero_is_not_collapsed():
    assert torch.equal(torch.tensor([0.]),torch.tensor([-0.]))
    with pytest.raises(ValueError):compare_values(torch.tensor([0.]),torch.tensor([-0.]))


def test_scalar_moment_and_all_rng_types_roundtrip():
    state={'step':torch.tensor(9.),'rng':np.random.RandomState(42).get_state(),
           'cuda':[torch.tensor([1,2,3],dtype=torch.uint8)],'python':(3,(1,2),None)}
    assert compare_values(state,copy.deepcopy(state))['exact']


def test_dtype_shape_and_container_type_are_part_of_state():
    assert exact_fingerprint(torch.tensor([1.]))!=exact_fingerprint(torch.tensor([1.],dtype=torch.float64))
    assert exact_fingerprint(torch.ones(1,2))!=exact_fingerprint(torch.ones(2,1))
    assert exact_fingerprint([1,2])!=exact_fingerprint((1,2))


def test_optimizer_id_moment_swap_refused_even_if_values_same_multiset():
    state={'param_groups':[{'params':[0,1]}],'state':{0:torch.ones(2),1:torch.zeros(2)}}
    wrong=copy.deepcopy(state);wrong['state'][0],wrong['state'][1]=wrong['state'][1],wrong['state'][0]
    with pytest.raises(ValueError):compare_values(state,wrong)


def test_one_sampler_byte_and_one_history_scalar_difference_refused():
    state={'sampler':torch.tensor([1,2,3],dtype=torch.uint8),'history':[{'lr':1e-5}]}
    for field in ('sampler','history'):
        wrong=copy.deepcopy(state)
        if field=='sampler':wrong['sampler'][0]=0
        else:wrong['history'][0]['lr']=2e-5
        with pytest.raises(ValueError):compare_values(state,wrong)


def test_nonfinite_state_is_never_an_exact_pass():
    with pytest.raises(ValueError):compare_values(torch.tensor(float('nan')),torch.tensor(float('nan')))
