"""Whole-role bank execution routing; scientific order and identities are fixed."""
import argparse
import os
from pathlib import Path
import socket
import time

from iqe.contracts import require
from iqe.io import atomic_json, digest, file_hash, lock, read_json

ROLES = ('scorer_fit', 'stage_val', 'selector_cal', 'dev_report')


def validate_routes(plan, config, policy):
    require(plan['schema'] == 'whole_role_bank_routing_v1', 'bank routing schema')
    require(plan['config_hash'] == config.identity, 'bank routing config changed')
    require(plan['Base_binding_hash'] == file_hash(config.root/'BASE_BINDING.json'), 'bank routing Base changed')
    require(set(plan['roles']) == set(ROLES), 'all four immutable bank roles must have one route')
    require(0 < plan['deadline_hours'] <= 24, 'bounded bank routing deadline')
    allowed = set(policy['allowed_hosts'])
    for entry in policy['allowed_hosts'].values(): allowed.update(entry.get('aliases', []))
    assigned = set()
    for role in ROLES:
        route = plan['roles'][role]
        require(set(route) == {'host', 'gpu', 'scoring_workers'}, 'unknown bank route option')
        require(route['host'] in allowed and not route['host'].startswith('training-vla-zt2'), 'forbidden bank host')
        require(isinstance(route['gpu'], int) and 0 <= route['gpu'] < 8, 'explicit physical GPU required')
        require(1 <= route['scoring_workers'] <= config.scoring_workers, 'bounded scoring worker allocation')
        key = (route['host'], route['gpu'])
        require(key not in assigned, 'role workers need distinct registered GPU allocations')
        assigned.add(key)
    return plan


def worker(config_path, role, routing_hash, wait_for_dispatch=False):
    from .config import load_config
    from .resource_guard import check
    from .bank import build_role
    config = load_config(config_path)
    require(role in ROLES, 'deployment bank role required')
    route_file = config.root/'runtime/BANK_ROUTING.json'
    require(file_hash(route_file) == routing_hash, 'bank routing changed after dispatch')
    plan = validate_routes(read_json(route_file), config, read_json('/mnt/project/server_dispatch_policy.json'))
    route = plan['roles'][role]
    require(socket.gethostname() in (route['host'], route['host']+'-worker-0'), 'bank worker on wrong host')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(route['gpu']), 'bank worker GPU route mismatch')
    require(os.environ.get('IQE_OFFICIAL_SCORE_WORKERS') == str(route['scoring_workers']), 'scoring worker route mismatch')
    check(config, 'build-deployment-bank')
    folder = config.root/'runtime/bank_parallel'/role
    with lock(folder/'RUN.lock'):
        receipt = {'pid': os.getpid(), 'host': socket.gethostname(), 'role': role, 'route': route,
                   'routing_hash': routing_hash, 'started_unix': time.time(),
                   'original_scene_order_and_generation_batches': True}
        if wait_for_dispatch:
            deadline = time.monotonic()+plan['deadline_hours']*3600
            while not (folder/'START.json').exists():
                require(time.monotonic() < deadline, 'bounded bank dispatch wait expired')
                atomic_json(folder/'STATUS.json', {'status': 'READY_WAITING_DISPATCH', **receipt,
                                                  'heartbeat_unix': time.time()})
                time.sleep(20)
            start = read_json(folder/'START.json')
            require(start['routing_hash'] == routing_hash and start['config_hash'] == config.identity,
                    'bank dispatch identity mismatch')
            require(start['pool_hash'] == read_json(config.root/'SERVING_CANDIDATE_POOL.json')['hash'],
                    'bank dispatch pool mismatch')
        receipt['pool_hash'] = read_json(config.root/'SERVING_CANDIDATE_POOL.json')['hash']
        atomic_json(folder/'STATUS.json', {'status': 'RUNNING', **receipt})
        try:
            result = build_role(config, role, 'cuda:0')
        except Exception as error:
            atomic_json(folder/'STATUS.json', {'status': 'FAILED', **receipt, 'error': repr(error)})
            raise
        atomic_json(folder/'STATUS.json', {'status': 'COMPLETE', **receipt, 'result': result,
                                         'finished_unix': time.time()})


def build_parallel(config):
    from .bank import load_bank, directory
    from .resource_guard import check
    path = config.root/'runtime/BANK_ROUTING.json'
    plan = validate_routes(read_json(path), config, read_json('/mnt/project/server_dispatch_policy.json'))
    check(config, 'build-deployment-bank')
    routing_hash = file_hash(path)
    folder = config.root/'runtime/bank_parallel'
    deadline = time.monotonic() + 3600*plan['deadline_hours']
    with lock(folder/'COORDINATOR.lock'):
        roles = []
        for role in ROLES:
            if (directory(config, 'deploy', role)/'BANK.json').exists():
                load_bank(config, 'deploy', role)
                continue
            status_file = folder/role/'STATUS.json'
            require(status_file.exists(), 'role worker must be deployed explicitly before bank dispatch: '+role)
            status = read_json(status_file)
            require(status['routing_hash'] == routing_hash and status['status'] in ('READY_WAITING_DISPATCH', 'RUNNING', 'COMPLETE'),
                    'role worker failed/stale, inspect before retry: '+role)
            if status['status'] == 'READY_WAITING_DISPATCH':
                require(time.time()-status['heartbeat_unix'] < 120, 'role worker heartbeat stale: '+role)
            atomic_json(folder/role/'START.json', {'routing_hash': routing_hash, 'config_hash': config.identity,
                        'pool_hash': read_json(config.root/'SERVING_CANDIDATE_POOL.json')['hash'], 'role': role}, immutable=True)
            roles.append(role)
        while True:
            statuses = {role: read_json(folder/role/'STATUS.json') for role in roles}
            failures = {role: s for role, s in statuses.items() if s['status'] == 'FAILED'}
            require(not failures, 'bank worker failed; preserve artifacts and inspect runtime/bank_parallel: '+str(failures))
            if all(s['status'] == 'COMPLETE' for s in statuses.values()):
                break
            atomic_json(folder/'STATUS.json', {'status': 'RUNNING', 'routing_hash': routing_hash,
                        'workers': statuses,
                        'updated_unix': time.time()})
            require(time.monotonic() < deadline, 'bounded bank coordinator deadline; preserve worker evidence, do not restart blindly')
            time.sleep(10)
        results = {}
        for role in ROLES:
            load_bank(config, 'deploy', role)
            results[role] = read_json(directory(config, 'deploy', role)/'STATISTICS.json')
        atomic_json(folder/'STATUS.json', {'status': 'COMPLETE', 'routing_hash': routing_hash, 'result': results})
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--role', required=True, choices=ROLES)
    parser.add_argument('--routing-hash', required=True)
    parser.add_argument('--wait-for-dispatch', action='store_true')
    args = parser.parse_args()
    worker(args.config, args.role, args.routing_hash, args.wait_for_dispatch)


if __name__ == '__main__': main()
