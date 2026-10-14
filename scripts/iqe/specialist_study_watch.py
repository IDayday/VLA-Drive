#!/usr/bin/env python3
"""Adopt already-running fixed cells; never starts or stops a trainer."""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from iqe.contracts import require
from iqe.io import atomic_json, lock, read_json
from iqe.specialist_study.config import load_config
from iqe.specialist_study.context_parallel import verify


def alive(host, pid, marker):
    if host in (os.uname().nodename, 'local'):
        path = Path('/proc') / str(pid) / 'cmdline'
        return path.exists() and marker in path.read_bytes().replace(b'\0', b' ').decode()
    probe = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
                            'cat', f'/proc/{int(pid)}/cmdline'], capture_output=True, timeout=20)
    require(probe.returncode != 255, 'remote inspection unavailable; preserve running jobs')
    return probe.returncode == 0 and marker in probe.stdout.replace(b'\0', b' ').decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--topology', required=True)
    parser.add_argument('--hours', type=float, default=24.)
    args = parser.parse_args()
    require(0 < args.hours <= 36, 'bounded monitoring deadline')
    c = load_config(args.config)
    folder = c.root / 'runtime/expert_matrix_driver'
    topology_root = c.root / 'runtime/spatial_topologies' / args.topology
    plan = read_json(topology_root / 'PLAN.json')
    sources = {cell: folder / (cell+'_LAUNCH.json') for cell in ('E00', 'E01')}
    sources.update({cell: c.root/'runtime/rl_zt2_expansion'/(cell+'_LAUNCH.json') for cell in ('E10', 'E11')})
    launches = {cell: read_json(path) for cell, path in sources.items()}
    start = time.monotonic()
    with lock(c.root/'runtime/study_watch.lock'):
        while True:
            require(time.monotonic()-start < args.hours*3600, 'bounded watcher deadline reached')
            done, active, failed = {}, {}, {}
            for cell, receipt in launches.items():
                result = c.root/'experts'/cell/'result.json'
                if result.exists() and read_json(result)['status'] == 'COMPLETE':
                    done[cell] = {'status': 'COMPLETE'}
                    continue
                host = receipt.get('host', 'local').replace('-worker-0', '')
                if not alive(host, receipt['pid'], '--cell '+cell+' '):
                    # Close the result-read/process-exit race before failing.
                    if result.exists() and read_json(result)['status'] == 'COMPLETE':
                        done[cell] = {'status': 'COMPLETE'}
                        continue
                    failed[cell] = {'host': host, 'pid': receipt['pid']}
                else:
                    status = c.root/'experts'/cell/'status.json'
                    active[cell] = {'host': host, 'pid': receipt['pid'], 'training': read_json(status) if status.exists() else 'LOADING'}
            atomic_json(folder/'STATUS.json', {'status': 'FAILED' if failed else ('RUNNING' if active else 'FINISHED'),
                        'active': active, 'finished': done, 'failed': failed, 'queued': [], 'adopted': True,
                        'updated_unix': time.time()})
            require(not failed, 'expert process exited without complete result; no automatic restart')
            spatial_done = (c.root/'SPATIAL_CONTEXT_COMPLETE.json').exists()
            if not spatial_done and all((topology_root/f'worker_{i}/COMPLETE.json').exists() for i in range(plan['workers'])):
                verify(c, args.topology)
                spatial_done = True
            if not active and spatial_done:
                return
            time.sleep(30)


if __name__ == '__main__':
    main()
