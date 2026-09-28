import pytest
import torch
from tools.ddpolicy_vehicle.optimizer_safety import bounded_parameter_groups, capture_master_samples, master_update_evidence, sample_indices


def test_large_parameter_grouping_preserves_every_parameter_once():
    parameters=[(str(i),torch.nn.Parameter(torch.empty(300_000_000,device='meta'))) for i in range(16)]
    groups,records=bounded_parameter_groups(parameters)
    assert sum(r['elements'] for r in records)==4_800_000_000
    assert max(r['elements'] for r in records)<2**31
    assert [id(p) for g in groups for p in g['params']]==[id(p) for _,p in parameters]
    with pytest.raises(ValueError,match='Individual parameter'):
        bounded_parameter_groups([('bad',torch.nn.Parameter(torch.empty(2**31,device='meta')))])


def test_optimizer_noop_rejected_and_real_master_change_recorded():
    from types import SimpleNamespace
    optimizer=SimpleNamespace(single_partition_of_fp32_groups=[torch.ones(1024)])
    before=capture_master_samples(optimizer)
    with pytest.raises(RuntimeError,match='changed no'):master_update_evidence(optimizer,before,'cpu')
    optimizer.single_partition_of_fp32_groups[0][0]+=.001
    evidence=master_update_evidence(optimizer,before,'cpu')
    assert evidence['fp32_master_update_verified'] and evidence['sampled_changed_elements']==1


def test_large_master_sample_indices_remain_exact_and_in_bounds():
    for size in (1,155582464,500000000,2**31-1):
        indices=sample_indices(size,512,'cpu')
        assert indices[0]==0 and indices[-1]==size-1
        assert (indices>=0).all() and (indices<size).all()
