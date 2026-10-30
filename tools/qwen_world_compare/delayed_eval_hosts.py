"""Activate explicitly requested evaluation hosts after a not-before deadline.

This is a resource-only operator. Existing queue owners dispatch the work; it
never signals a process or creates a second evaluation/training owner.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import traceback

POLICY = Path('/mnt/project/server_dispatch_policy.json')
PYTHON = '/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
TASKS = ('qwen_sourceflow_native_20261018', 'qwen_world_postcot_20261025', 'qwen_world_compare_20261029')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def withdrawal_is_newer(policy, host, requested_at):
    ban = policy.get('global_dispatch_prohibitions', {}).get(host, {})
    if ban.get('new_task_dispatch_allowed') is not False:
        return False
    stamp = ban.get('utc')
    if not stamp:
        raise ValueError('Undated host prohibition requires review')
    return datetime.fromisoformat(stamp.replace('Z', '+00:00')).timestamp() > requested_at


def permit(policy, host, request, directory, now):
    if now < request['not_before_unix']:
        raise PermissionError('Ten-minute preparation window has not elapsed')
    if withdrawal_is_newer(policy, host, request['requested_at_unix']):
        raise PermissionError('A newer user withdrawal supersedes this request')
    result = deepcopy(policy)
    aliases = [host, host + '-worker-0', host.removeprefix('training-'), host.removeprefix('training-') + '-worker-0']
    result['allowed_hosts'][host] = dict(aliases=aliases, temporary=True, unbounded=True, record=str(directory/'REQUEST.json'))
    for alias in aliases:
        if alias not in result['allowed_aliases']:
            result['allowed_aliases'].append(alias)
    result.setdefault('global_dispatch_prohibitions', {})[host] = dict(
        host=host, aliases=aliases, new_task_dispatch_allowed=True,
        utc=datetime.fromtimestamp(now, timezone.utc).isoformat(), record=str(directory/'REQUEST.json'),
        scope='User-authorized registered checkpoint evaluation; preserve all unknown jobs and existing holds')
    grants = {}
    for task in TASKS:
        grant = deepcopy(result['task_authorizations'][task])
        previous = grant['record']
        field = 'allowed_hosts' if task == TASKS[0] else 'unbounded_temporary_hosts'
        grant.setdefault(field, [])
        if host not in grant[field]:
            grant[field].append(host)
        if task != TASKS[0]:
            # These evaluators intentionally use the measured sharing slot
            # even on an otherwise idle card. Temporary permission alone is
            # insufficient for the worker's separate sharing contract.
            grant.setdefault('shared_training_hosts', [])
            if host not in grant['shared_training_hosts']:
                grant['shared_training_hosts'].append(host)
        grant.update(previous_authorization_record=previous,
                     latest_evaluation_host_addition=str(directory/'REQUEST.json'),
                     old_training_and_run_retirement_unchanged=True)
        grant['record'] = str(directory/'grants'/f'{task}.{host}.json')
        result['task_authorizations'][task] = grant
        grants[task] = grant
    result.setdefault('idle_pressure_holds', {})[host] = dict(
        status='user_preparing_host_for_real_evaluation', resume_requires_explicit_instruction=True,
        record=str(directory/'REQUEST.json'), preserve_real_workloads=True)
    result['updated_utc'] = datetime.fromtimestamp(now, timezone.utc).isoformat()
    result['latest_world_resource_addition'] = dict(host=host, record=str(directory/'REQUEST.json'),
                                                    evaluation_only=True, activated_unix=now)
    return result, grants


def remote(host, argv, timeout=60):
    return subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', host,
                                    shlex.join(argv)], text=True, timeout=timeout)


def preflight(host, directory):
    # No CUDA allocations, process signals, environment mutation or pressure
    # preemption. Each real evaluator repeats admission under its physical lease.
    program = r'''import json,socket,subprocess,os
from pathlib import Path
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
gpus=[dict(index=int(i),uuid=u.strip(),total=int(t),free=int(f),util=int(v))for i,u,t,f,v in(line.split(',')for line in rows.splitlines())]
rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'],text=True)
owners=[]
for line in rows.splitlines():
 p,u,m=line.split(',');root=Path('/proc')/p.strip();o=dict(pid=int(p),uuid=u.strip(),used=int(m)if m.strip().isdigit()else None,known=False,pressure=False)
 try:
  args=(root/'cmdline').read_bytes().decode().split('\0');o['pressure']=any('gpu_stress' in a or 'gpu_pressure' in a for a in args)
  o['known']=any(a.startswith(('/mnt/project/qwen-','/mnt/project/VLA-Drive-qwen','/mnt/project/VLA-AD/experiments/recogdrive','/mnt/project/recogdrive'))for a in args)
 except OSError:pass
 owners.append(o)
print(json.dumps(dict(hostname=socket.gethostname(),gpus=gpus,owners=owners,shared_request_exists=Path(__REQUEST__).is_file(),python_exists=Path(__PYTHON__).is_file())))'''.replace('__REQUEST__', repr(str(directory/'REQUEST.json'))).replace('__PYTHON__', repr(PYTHON))
    samples = []
    for i in range(2):
        samples.append(json.loads(remote(host, ['/usr/bin/python3', '-c', program])))
        if i == 0:
            time.sleep(2)
    if any(s['hostname'].removesuffix('-worker-0') != host or not s['shared_request_exists'] for s in samples):
        raise ValueError('Unexpected host or nonshared filesystem')
    eligible = []
    for a, b in zip(samples[0]['gpus'], samples[1]['gpus']):
        if a['uuid'] != b['uuid']:
            raise ValueError('Physical GPU mapping changed')
        valid = True
        for gpu, sample in [(a, samples[0]), (b, samples[1])]:
            owners = [o for o in sample['owners'] if o['uuid'] == gpu['uuid']]
            if gpu['free'] < 32768 or any(not o['known'] or o['pressure'] or o['used'] is None for o in owners):
                valid = False
            if not owners and (gpu['util'] > 10 or gpu['total'] - gpu['free'] > 2048):
                valid = False
            if owners and all(o['used'] is not None for o in owners) and gpu['total']-gpu['free']-sum(o['used']for o in owners)>2048:
                valid = False
        if valid:
            eligible.append(b['index'])
    if not eligible:
        raise RuntimeError('No ready GPU: pressure/unknown occupancy or insufficient headroom; no processes stopped')
    if not samples[-1]['python_exists']:
        raise RuntimeError('Qualified model Python is absent; environment preparation required')
    check = "import torch,transformers,deepspeed;from transformers import Qwen3VLForConditionalGeneration;assert torch.__version__=='2.5.1+cu124';assert transformers.__version__=='4.57.0';assert deepspeed.__version__=='0.16.9';print('QUALIFIED_CPU_IMPORTS')"
    environment = remote(host, ['env', 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=1', PYTHON, '-c', check], timeout=180)
    return dict(samples=samples, eligible_gpu_indices=eligible, environment=environment, no_processes_signalled=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    directory = Path(args.request).resolve().parent
    request = read(args.request)
    lock = (directory/'operator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(directory/'OWNER.json', dict(pid=os.getpid(), start_ticks=int(Path('/proc/self/stat').read_text().split()[21])))
    while time.time() < request['not_before_unix']:
        if (directory/'CANCEL').exists():
            return
        time.sleep(min(30, request['not_before_unix']-time.time()))
    pending = set(request['hosts'])
    for attempt in range(1, 11):
        if not pending or (directory/'CANCEL').exists():
            break
        def inspect(host):
            try:
                if withdrawal_is_newer(read(POLICY), host, request['requested_at_unix']):
                    return host, None, 'NEWER_WITHDRAWAL'
                return host, preflight(host, directory), None
            except Exception:
                return host, None, traceback.format_exc()
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(inspect, sorted(pending)))
        for host, evidence, error in outcomes:
            write(directory/'preflight'/f'{host}.attempt{attempt:03d}.json', dict(at=time.time(), evidence=evidence, error=error))
            if error:
                if error == 'NEWER_WITHDRAWAL':
                    pending.remove(host)
                continue
            with ExitStack() as stack:
                for path in ['/mnt/project/server_dispatch_policy.lock', '/mnt/project/server_dispatch_policy.json.lock']:
                    handle = stack.enter_context(open(path, 'a'))
                    fcntl.flock(handle, fcntl.LOCK_EX)
                current = read(POLICY)
                amended, grants = permit(current, host, request, directory, time.time())
                write(directory/f'POLICY_BEFORE_{host}.json', current)
                for grant in grants.values():
                    write(grant['record'], grant)
                write(POLICY, amended)
                # Resource-only mirrors; actor/scientific identities do not
                # change. Both running owners read the live policy each route.
                for task, root in [(TASKS[1], '/mnt/project/qwen-world-postcot-20261025'),
                                   (TASKS[2], '/mnt/project/qwen-world-four-frameworks-20261029')]:
                    write(Path(root)/'RESOURCE_AUTHORIZATION.json', grants[task])
                write(directory/f'ACTIVATED_{host}.json', dict(at=time.time(), host=host, evidence=evidence,
                      policy_sha256=hashlib.sha256(POLICY.read_bytes()).hexdigest(), existing_queue_owners_preserved=True))
            pending.remove(host)
        if pending:
            time.sleep(30)
    write(directory/'COMPLETE.json', dict(status='ACTIVATED' if not pending else 'PARTIAL_OR_BLOCKED',
          blocked_hosts=sorted(pending), activated_hosts=[h for h in request['hosts']if(directory/f'ACTIVATED_{h}.json').exists()],
          at=time.time(), no_model_or_training_changes=True, no_processes_signalled=True))


if __name__ == '__main__':
    main()
