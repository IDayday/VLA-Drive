"""Bounded, cache-only dispatch after occupied GPUs become idle.

No process eviction, training start, phase repartition or automatic failed-worker
retry. The declared cohort reserves space for both hosts before either launches.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from iqe.contracts import require
from iqe.io import atomic_json, file_hash, read_json
from iqe.vision_cache.cli import hold_check
from iqe.vision_cache.plan import capacity_check, load_plan
from iqe.vision_cache.qualification import validate_cache_qualification
from build_phase import validate_partition


def idle(snapshot):
    return (len(snapshot['gpus']) == 8 and not snapshot['owners'] and
            all(g['memory_mib'] <= 1024 and g['utilization'] <= 5 for g in snapshot['gpus']))


def cohort_capacity(plan, certificate, reusable_bytes, output, jobs):
    retention = plan['checkpoint_retention_budget']
    protected = retention['remaining_count'] * retention['parent_bytes'] + (180 + 200) * 1024**3
    done = sum(p.stat().st_size for job in jobs for p in Path(job['root']).glob('shard_*.bin')
               if Path(str(p) + '.COMPLETE.json').exists())
    needed = max(0, math.ceil(certificate['representative_record_max_bytes'] * plan['requested_scenes'] * 1.05)
                 - reusable_bytes - done)
    value = capacity_check(output, needed, reserve_bytes=protected, temporary_bytes=len(jobs) * 8 * 2 * 1024**3)
    value.update(completed_new_bytes=done, reused_physical_bytes=reusable_bytes,
                 aggregate_requested_scenes=plan['requested_scenes'])
    return value


def remote(host, code, args):
    argv = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host,
            shlex.join([sys.executable, '-', *map(str, args)])]
    value = subprocess.run(argv, input=code, text=True, capture_output=True, timeout=50)
    require(value.returncode == 0, 'remote control failed: ' + host + ': ' + value.stderr[-2000:])
    return json.loads(value.stdout)


PROBE = '''import json, os, socket, subprocess, sys
from pathlib import Path
root = Path(sys.argv[1])
lines = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
gpus = [dict(index=int(i),memory_mib=int(m),utilization=int(u)) for i,m,u in (line.split(',') for line in lines)]
owners = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True).splitlines()
mount = json.loads(subprocess.check_output(['findmnt','-T',str(root.parent),'-J','-o','SOURCE,FSTYPE'],text=True))['filesystems'][0]
active = None
p = root/'REMOTE_CONTROLLER.json'
if p.exists():
    record = json.loads(p.read_text()); proc = Path('/proc')/str(record['pid'])
    try:
        stat = (proc/'stat').read_text().rsplit(')',1)[1].split()
        active = (stat[19] == record['start_ticks'] and 'build_phase.py' in (proc/'cmdline').read_text() and stat[0] != 'Z')
    except FileNotFoundError: active = False
print(json.dumps(dict(host=socket.gethostname(),gpus=gpus,owners=owners,mount=mount,controller_active=active)))
'''


LAUNCH = '''import json, os, socket, subprocess, sys, time
from pathlib import Path
repo, spec_path, job_index, admission = sys.argv[1:]
os.chdir(repo); sys.path.insert(0,repo)
from iqe.io import atomic_json, file_hash, lock, read_json
from iqe.resources import qualify
from iqe.contracts import require
spec = read_json(spec_path); job = spec['jobs'][int(job_index)]; root = Path(job['root'])
require(file_hash(Path(repo)/'scripts/iqe/fp32_cache/build_phase.py') == spec['builder_script_sha256'],'controller source changed')
with lock(root/'REMOTE_LAUNCH.lock'):
    require(not (root/'REMOTE_CONTROLLER.json').exists(),'already dispatched; inspect exact process before an explicit new attempt')
    qualify('cuda')
    owners = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    require(not owners, 'GPU owner appeared; preserving existing jobs')
    argv = [sys.executable,'-u','scripts/iqe/fp32_cache/build_phase.py','run',
        '--config',spec['config'],'--plan',job['plan'],'--root',str(root),
        '--qualification',spec['qualification'],'--hold-file',spec['hold_file'],
        '--cohort-admission',admission,'--workers','8','--rate-mib-s-per-worker',str(spec['rate_mib_s_per_worker']),
        '--max-seconds',str(spec['builder_max_seconds'])]
    if job.get('reuse_cache'): argv += ['--reuse-cache',job['reuse_cache']]
    atomic_json(root/'REMOTE_LAUNCH_INTENT.json',dict(argv=argv,created_unix=time.time(),spec_sha256=file_hash(spec_path)),immutable=True)
    with (root/'controller.log').open('ab') as log:
        child = subprocess.Popen(argv,cwd=repo,env=dict(os.environ,PYTHONPATH=repo,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    ticks = (Path('/proc')/str(child.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
    record = dict(pid=child.pid,start_ticks=ticks,host=socket.gethostname(),created_unix=time.time(),argv=argv,spec_sha256=file_hash(spec_path))
    atomic_json(root/'REMOTE_CONTROLLER.json',record,immutable=True)
    print(json.dumps(record))
'''


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--poll-seconds', type=int, default=60)
    parser.add_argument('--max-seconds', type=int, default=43200)
    args = parser.parse_args()
    require(30 <= args.poll_seconds <= 600 and 0 < args.max_seconds <= 86400, 'bounded dispatch/poll budget required')
    spec = read_json(args.manifest)
    require(spec['schema'] == 'iqe_fp32_phase_cohort_v1' and spec['no_training_or_migration'], 'cache-only cohort required')
    require(0 < spec['rate_mib_s_per_worker'] <= 64 and 0 < spec['builder_max_seconds'] <= 43200, 'bounded IO and time required')
    require([j['host'] for j in spec['jobs']] == ['training-vlawm-zt3', 'training-vlawm-zt4'], 'undeclared routing')
    root = Path(args.manifest).resolve().parent
    require(not (root/'SUPERSEDED_WITHOUT_GPU_WORK.json').exists(),
            'storage phase superseded before launch; use its declared successor, do not duplicate population writes')
    require(file_hash(Path(spec['repo'])/'scripts/iqe/fp32_cache/build_phase.py') == spec['builder_script_sha256'], 'builder source changed')
    aggregate = load_plan(spec['aggregate_plan'])
    require(aggregate['plan_hash'] == spec['aggregate_plan_hash'], 'aggregate plan changed')
    validate_partition(aggregate)
    parts = [load_plan(j['plan']) for j in spec['jobs']]
    for p, job in zip(parts, spec['jobs']):
        validate_partition(p)
        require(p['plan_hash'] == job['plan_hash'] and p['phase']['aggregate_plan_hash'] == aggregate['plan_hash'], 'foreign host partition')
        Path(job['root']).mkdir(parents=True, exist_ok=True)
    sets = [{r['scene_id'] for r in p['rows']} for p in parts]
    require(not sets[0] & sets[1] and sets[0] | sets[1] == {r['scene_id'] for r in aggregate['rows']}, 'cohort overlap or loss')
    cert = validate_cache_qualification(spec['qualification'], fit_hash=aggregate['fit_sha256'], for_build=True)
    from iqe.vision_cache.storage import CacheReader
    reuse = CacheReader(spec['jobs'][0]['reuse_cache'])
    require(reuse.complete['identity_hash'] == cert['visual_identity_hash'], 'foreign reuse precision')
    reused_bytes = reuse.complete['bytes']; reuse.close()
    began = time.monotonic()
    with (root/'DISPATCH.lock').open('ab') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(root/'DISPATCH_CONTROLLER.json', {'pid': os.getpid(), 'started_unix': time.time(),
                    'manifest_sha256': file_hash(args.manifest), 'args': vars(args), 'script_sha256': file_hash(__file__)})
        while True:
            hold_check(spec['hold_file'])
            # Re-read the shared policy before each prospective new dispatch.
            from iqe.resources import qualify
            qualify('cpu')
            policy = read_json('/mnt/project/server_dispatch_policy.json')
            require(all(j['host'] in policy['task_authorizations']['iqe_v1']['allowed_hosts'] for j in spec['jobs']), 'IQE host permission revoked')
            cap = cohort_capacity(aggregate, cert, reused_bytes, root, spec['jobs'])
            rows = []
            for i, job in enumerate(spec['jobs']):
                out = Path(job['root']); row = {'host': job['host'], 'root': str(out)}
                try:
                    snap = remote(job['host'], PROBE, [out]); row['snapshot'] = snap
                    require(snap['mount']['fstype'] == cap['mount']['fstype'] and snap['mount']['source'] == cap['mount']['source'], 'different shared project mount')
                    if (out/'PHASE_RESULT.json').exists():
                        result = read_json(out/'PHASE_RESULT.json'); row['status'] = result['status']
                    elif (out/'REMOTE_CONTROLLER.json').exists():
                        row['status'] = 'RUNNING' if snap['controller_active'] else 'STOPPED_INSPECT_PARTIAL'
                        if (out/'PHASE_STATUS.json').exists(): row['progress'] = read_json(out/'PHASE_STATUS.json')
                    elif (out/'REMOTE_LAUNCH_INTENT.json').exists():
                        row['status'] = 'AMBIGUOUS_LAUNCH_INSPECT_WITHOUT_RETRY'
                    elif not cap['sufficient']:
                        row['status'] = 'WAITING_STORAGE_RESERVE'
                    elif not idle(snap):
                        row['status'] = 'WAITING_IDLE_GPUS'
                    else:
                        admission = root/'admissions'/f'{time.time_ns()}.json'
                        atomic_json(admission, {'status': 'PASS', 'capacity': cap, 'created_unix': time.time(),
                                    'aggregate_plan_hash': aggregate['plan_hash'], 'partition_plan_hashes': [p['plan_hash'] for p in parts]}, immutable=True)
                        row['launch'] = remote(job['host'], LAUNCH, [spec['repo'], str(Path(args.manifest).resolve()), i, admission])
                        row['status'] = 'DISPATCHED'
                except Exception as error:
                    row.update(status='CONTROL_ERROR_RECHECK', error=str(error))
                rows.append(row)
            done = all(r['status'] == 'PHASE_COMPLETE_FULL_POPULATION_PENDING' for r in rows)
            status = {'status': 'PHASE_COMPLETE_FULL_POPULATION_PENDING' if done else 'ACTIVE',
                      'updated_unix': time.time(), 'elapsed_seconds': time.monotonic()-began,
                      'phase_scenes': aggregate['requested_scenes'], 'full_scenes': aggregate['fit_count'],
                      'capacity': cap, 'jobs': rows, 'no_training_or_migration': True}
            atomic_json(root/'COHORT_STATUS.json', status)
            if done: return 0
            if time.monotonic()-began >= args.max_seconds:
                atomic_json(root/'COHORT_STATUS.json', {**status, 'status': 'DISPATCH_TIME_BUDGET_ENDED_WORKERS_UNCHANGED'})
                return 3
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps({'status': 'BLOCKED', 'error': str(error)}), flush=True)
        sys.exit(2)
