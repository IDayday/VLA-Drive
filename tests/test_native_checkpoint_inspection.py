"""Check exact rigid ZeRO metadata semantics, not a relaxed tensor tolerance."""
import pytest
import torch

from tools.decision_local.inspect_native_training_checkpoint import inspect_rigid_partition


def partition(padding=0):
    return {
        'ds_version':'0.16.9','overflow':False,'partition_count':[8],
        'group_paddings':[padding],
        'single_partition_of_fp32_groups':[torch.empty(5)],
        'base_optimizer_state':{
            'param_groups':[{'params':[0],'lr':1e-5}],
            'state':{0:{'step':2000,'exp_avg':torch.empty(5+padding),
                        'exp_avg_sq':torch.empty(5+padding)}}}}


@pytest.mark.parametrize('padding',[0,12])
def test_native_rigid_padding_is_exactly_accounted_for(padding):
    result=inspect_rigid_partition(partition(padding),2000,8)
    assert result['master_elements']==[5]
    assert result['moment_elements']==[5+padding]
    assert result['group_padding']==[padding]


@pytest.mark.parametrize('mutation',[
    'undeclared_padding','wrong_dtype','wrong_step','wrong_world','duplicate_group'])
def test_incomplete_or_incompatible_state_is_still_rejected(mutation):
    state=partition(12)
    if mutation=='undeclared_padding':state['group_paddings']=[0]
    if mutation=='wrong_dtype':state['base_optimizer_state']['state'][0]['exp_avg_sq']=torch.empty(17,dtype=torch.bfloat16)
    if mutation=='wrong_step':state['base_optimizer_state']['state'][0]['step']=1999
    if mutation=='wrong_world':state['partition_count']=[4]
    if mutation=='duplicate_group':state['base_optimizer_state']['param_groups']*=2
    with pytest.raises(ValueError):inspect_rigid_partition(state,2000,8)
