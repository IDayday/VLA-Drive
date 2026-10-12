"""Check live task routing before allocating a producer GPU."""
import socket
from ..contracts import require
from ..io import read_json


def canonical(host):
    return host.removesuffix('-worker-0')


def producer_placement(policy_path='/mnt/project/server_dispatch_policy.json', *, host=None):
    policy = read_json(policy_path)
    host = canonical(host or socket.gethostname())
    task = policy['task_authorizations']['iqe_v1']
    placement = task['workload_placement']
    require(placement['forbid_cache_compute_on_base_host'] is True,
            'remote-only cache placement must be explicit')
    require(host != canonical(placement['base_training_host']),
            'BASE_HOST_RESERVED: visual producers must run on another server')
    require(host not in policy.get('global_dispatch_prohibitions', {}), 'withdrawn producer host')
    require(host in task['allowed_hosts'] and host in placement['cache_producer_hosts'],
            'cache producer outside current task placement')
    return {'host': host, 'base_host': placement['base_training_host'], 'policy': policy_path,
            'visual_compute_on_base': False}
