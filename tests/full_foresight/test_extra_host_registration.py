import copy
import pytest
from tools.full_foresight.register_priority import checked_extra_host


def inventory():
    return {'host_alias': 'training-rl-zt2', 'hostname': 'training-rl-zt2-worker-0',
            'authorization': 'User requests C4 on rl-zt2', 'gpu_count': 8, 'all_gpus_free': True,
            'gpus': [{'index': i, 'uuid': str(i), 'memory_total_mib': 81920, 'memory_used_mib': 0}
                     for i in range(8)]}


def test_current_registrations_unchanged_and_explicit_extra_host_required():
    assert checked_extra_host('C0', 8, None, None) is None
    assert checked_extra_host('C1', 8, None, None) is None
    with pytest.raises(ValueError):
        checked_extra_host('C4', 8, None, None)
    assert checked_extra_host('C4', 8, 'training-rl-zt2', inventory())['host_alias'] == 'training-rl-zt2'


@pytest.mark.parametrize('change', ['occupied', '40GB', 'missing', 'duplicate', 'wrong_host'])
def test_extra_host_rejects_incompatible_or_unavailable_devices(change):
    data = copy.deepcopy(inventory())
    if change == 'occupied': data['gpus'][0]['memory_used_mib'] = 1000
    if change == '40GB': data['gpus'][0]['memory_total_mib'] = 40960
    if change == 'missing': data['gpus'].pop()
    if change == 'duplicate': data['gpus'][1]['uuid'] = '0'
    if change == 'wrong_host': data['host_alias'] = 'another-host'
    with pytest.raises(ValueError):
        checked_extra_host('C4', 8, 'training-rl-zt2', data)
