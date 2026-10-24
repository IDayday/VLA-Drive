"""Operator-only admission extension of the existing durable process worker.

The extension is process-local: no source file or model function is patched.
The existing worker still owns launch, native child wait/reap, logs and result.
"""
import argparse
import fcntl
import json
from pathlib import Path
import socket
import sys


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--spec',required=True);args=parser.parse_args()
    spec=json.loads(Path(args.spec).read_text());source=Path(spec['source'])
    sys.path.insert(0,str(source))
    from scripts.analysis.qwen_sourceflow import job_worker
    task='qwen_flow_three_routes_20261024'
    def admitted():
        policy=json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
        host=socket.gethostname().removesuffix('-worker-0')
        if (host not in policy['allowed_hosts'] or host not in policy['task_authorizations'][task]['allowed_hosts'] or
            policy.get('global_dispatch_prohibitions',{}).get(host,{}).get('new_task_dispatch_allowed') is False):
            raise PermissionError('Withdrawn study worker: '+host)
        return host
    admitted()
    if not spec['module'].startswith('scripts.analysis.qwen_flow_exploration.'):
        raise ValueError('Only explicitly named study entries belong here')
    if spec['gpus']:
        if len(spec['gpus'])!=1:raise ValueError('One study process per GPU; no implicit cross-node training')
        directory=Path('/mnt/project/qwen-flow-three-routes-20261024/device_locks');directory.mkdir(exist_ok=True)
        lock=(directory/(spec['gpu_uuids'][0]+'.lock')).open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job_worker.MODULES=job_worker.MODULES|{spec['module']}
    job_worker.require_dispatch=admitted
    job_worker.main()


if __name__=='__main__':main()
