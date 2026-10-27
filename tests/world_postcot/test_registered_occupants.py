import copy
import pytest
from starVLA.rl.qwen_world_postcot.contracts import TASK, fingerprint
from starVLA.rl.qwen_world_postcot.registered_occupants import validate_registered_occupant


def case():
    live = dict(pid=10, start_ticks=22, argv_sha256='cmd', cwd='/existing')
    row = dict(pid=10, uuid='gpu0', used_MiB=2048)
    receipt = dict(schema='world_registered_GPU_occupants_v1', task=TASK, host='fixed',
                   authorization_sha256='auth', valid_from=100, valid_until=200,
                   occupants=[dict(live, uuid='gpu0', maximum_used_MiB=8192)])
    receipt['sha256'] = fingerprint(receipt)
    return row, live, receipt


def test_exact_registered_small_job_can_be_shared():
    row, live, receipt = case()
    assert validate_registered_occupant(row, live, receipt, host='fixed', authorization_sha256='auth', now=150)


@pytest.mark.parametrize('change', ['pid', 'start_ticks', 'argv_sha256', 'cwd', 'memory', 'uuid', 'host', 'auth', 'expired'])
def test_changed_or_unregistered_occupant_is_rejected(change):
    row, live, receipt = case()
    args = dict(host='fixed', authorization_sha256='auth', now=150)
    if change in live: live[change] = 'changed'
    elif change == 'memory': row['used_MiB'] = 8193
    elif change == 'uuid': row['uuid'] = 'other'
    elif change == 'host': args['host'] = 'withdrawn'
    elif change == 'auth': args['authorization_sha256'] = 'withdrawn'
    else: args['now'] = 201
    with pytest.raises(ValueError): validate_registered_occupant(row, live, receipt, **args)
