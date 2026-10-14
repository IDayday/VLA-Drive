#!/usr/bin/env python3
"""Register a bounded slice of existing candidate/score work under an absolute lease."""
import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from iqe.contracts import require
from iqe.io import atomic_json, file_hash, lock, read_json
from iqe.specialist_study.capacity import probe
from iqe.specialist_study.temporary_lease import authorization, can_admit, process_environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lease', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpus', type=int, nargs='+', required=True)
    parser.add_argument('--rich-start', type=int, required=True)
    parser.add_argument('--rich-shards', type=int, required=True)
    parser.add_argument('--score-start', type=int, required=True)
    parser.add_argument('--score-count', type=int, required=True)
    parser.add_argument('--score-workers', type=int, required=True)
    parser.add_argument('--read-mib-per-second', type=float, default=32.)
    args = parser.parse_args()
    require(len(set(args.gpus)) == len(args.gpus) and all(0 <= g < 8 for g in args.gpus), 'explicit distinct GPUs')
    require(0 <= args.rich_start < args.rich_start+len(args.gpus) <= args.rich_shards <= 24, 'rich shard range')
    require(0 <= args.score_start < args.score_start+args.score_count <= args.score_workers <= 16, 'CPU worker range')
    require(0 < args.read_mib_per_second <= 64, 'bounded shared IO required')
    repo = Path(__file__).resolve().parents[2]
    lease_path = Path(args.lease).resolve()
    lease = read_json(lease_path)
    host = socket.gethostname()
    env = process_environment(lease, os.environ)
    env.update(PYTHONPATH=str(repo), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    policy = read_json('/mnt/project/server_dispatch_policy.json')
    authorization(policy, host, 'precompute-rich-candidates', env, require_guard=False)
    require(can_admit(lease), 'absolute admission deadline reached')
    output = lease_path.parent/'specialist_study'
    with lock(output/(host+'.LAUNCH.lock')):
        require(not (output/(host+'.LAUNCH.json')).exists(), 'host already dispatched; inspect original receipts')
        inventory = {str(g): probe(g) for g in args.gpus}
        require(all(value['ready'] for value in inventory.values()), 'physical GPU capacity/ownership unsafe')
        records = []
        jobs = [('rich-precompute', args.rich_start+i, args.rich_shards, g) for i,g in enumerate(args.gpus)]
        jobs += [('score-helper', args.score_start+i, args.score_workers, None) for i in range(args.score_count)]
        for kind, index, total, gpu in jobs:
            require(can_admit(lease), 'absolute admission closed during launch; retain already registered work')
            name = ('rich_' if kind == 'rich-precompute' else 'score_')+f'{index:02d}_of_{total:02d}'
            command = [sys.executable, '-u', '-m', 'iqe.specialist_study.temporary_worker',
                       '--lease', str(lease_path), '--config', str(Path(args.config).resolve()), '--name', name,
                       '--kind', kind, '--worker', str(index), '--workers', str(total)]
            if gpu is not None:
                command += ['--gpu', str(gpu), '--read-mib-per-second', str(args.read_mib_per_second)]
            logs = output/'wrapper_logs'/host
            logs.mkdir(parents=True, exist_ok=True)
            with open(logs/(name+'.log'), 'ab') as stream:
                child = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            records.append({'kind':kind, 'name':name, 'pid':child.pid, 'command':command, 'gpu':gpu})
            time.sleep(.3)
        atomic_json(output/(host+'.LAUNCH.json'), {'host':host, 'created_unix':time.time(),
                    'lease_hash':file_hash(lease_path), 'expires_unix':lease['expires_unix'],
                    'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                    'capacity_before':inventory, 'workers':records, 'markers_in_wrapper_initial_environment':True,
                    'new_models_trained':False, 'scientific_budget_changed':False}, immutable=True)


if __name__ == '__main__':
    main()
