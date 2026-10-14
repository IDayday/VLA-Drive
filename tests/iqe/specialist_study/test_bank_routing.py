from copy import deepcopy
from types import SimpleNamespace

import pytest

from iqe.io import atomic_json, file_hash, ContractError
from iqe.specialist_study.bank_parallel import ROLES, validate_routes


def fixture(tmp_path):
    atomic_json(tmp_path/'BASE_BINDING.json', {'checkpoint': 'fixed'})
    config = SimpleNamespace(root=tmp_path, identity='same-scientific-config', scoring_workers=8)
    hosts = ['training-vlawm-tsinghua-zt5', 'training-vlawm-tsinghua-zt6']
    plan = {'schema': 'whole_role_bank_routing_v1', 'config_hash': config.identity,
            'Base_binding_hash': file_hash(tmp_path/'BASE_BINDING.json'), 'deadline_hours': 12,
            'roles': {role: {'host': hosts[i//2], 'gpu': i % 2, 'scoring_workers': 4}
                      for i, role in enumerate(ROLES)}}
    return config, plan, {'allowed_hosts': {h: {'aliases': [h+'-worker-0']} for h in hosts}}


def test_runtime_route_does_not_change_scientific_identity(tmp_path):
    config, plan, policy = fixture(tmp_path)
    assert validate_routes(plan, config, policy) == plan
    assert config.identity == 'same-scientific-config'
    assert config.scoring_workers == 8


@pytest.mark.parametrize('change', ['forbidden_host', 'same_gpu', 'missing_role', 'Base', 'config', 'unbounded_CPU'])
def test_route_rejects_unsafe_or_stale_input(tmp_path, change):
    config, plan, policy = fixture(tmp_path)
    if change == 'forbidden_host': plan['roles']['scorer_fit']['host'] = 'withdrawn-host'
    elif change == 'same_gpu': plan['roles']['stage_val'] = deepcopy(plan['roles']['scorer_fit'])
    elif change == 'missing_role': plan['roles'].pop('dev_report')
    elif change == 'Base': plan['Base_binding_hash'] = 'different-Base'
    elif change == 'config': plan['config_hash'] = 'different-science'
    elif change == 'unbounded_CPU': plan['roles']['scorer_fit']['scoring_workers'] = 32
    with pytest.raises(ContractError): validate_routes(plan, config, policy)
