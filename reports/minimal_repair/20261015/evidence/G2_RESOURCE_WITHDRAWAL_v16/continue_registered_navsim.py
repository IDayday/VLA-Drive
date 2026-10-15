"""Resume the registered native NAVSIM lineage; model/trainer/source stay frozen."""
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parent
ART = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008')
POLICY = Path('/mnt/project/server_dispatch_policy.json')
NAVTEST = Path('/mnt/project/VLA-Drive-structured-world-fgtr-perf-20261011/tools/structured_world/run_navsim_navtest.py')

def read(p):
    return json.loads(Path(p).read_text())

def write(p, value):
    p = Path(p)
    t = p.with_name(p.name + '.' + str(os.getpid()) + '.tmp')
    t.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(t, p)

def process(pid):
    p = Path('/proc') / str(pid)
    try:
        stat = (p/'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return {'pid': pid, 'start_ticks': int(stat[19]),
                'argv': [os.fsdecode(x) for x in (p/'cmdline').read_bytes().split(b'\0') if x]}
    except OSError:
        return None

def authorize(host):
    p = read(POLICY)
    assert host in p['allowed_hosts'] and host not in p.get('global_dispatch_prohibitions', {})
    assert host in p['task_authorizations']['structured_world_fgtr_round1']['allowed_hosts']
    reservation = p.get('host_reservations', {}).get(host, {})
    assert not reservation.get('exclude_new_other_workloads') or reservation.get('task') == 'structured_world_fgtr_round1'

def checkpoint(m):
    run = Path(m['run'])
    tag = (run/'checkpoints/latest').read_text().strip()
    assert Path(tag).name == tag
    p = run/'checkpoints'/tag
    d = read(p/'COMPLETE.json')
    assert d['identity'] == m['identity'] and d['ranks'] == 8
    assert len(d['file_sha256']) == 17 and all((p/f).is_file() for f in d['file_sha256'])
    return d

def enqueue(m, directory):
    bank = directory/'EVALUATION_REQUESTS.json'
    records = read(bank) if bank.exists() else {}
    run = Path(m['run'])
    for k in m['evaluation_observations']:
        if str(k) in records:
            continue
        candidates = list((run/'checkpoints').glob('*_'+str(k).zfill(7)))
        candidates = [p for p in candidates if not p.name.startswith('writing_') and (p/'COMPLETE.json').exists()]
        if not candidates:
            continue
        assert len(candidates) == 1
        receipt = read(candidates[0]/'COMPLETE.json')
        assert receipt['identity'] == m['identity'] and receipt['completed'] == k
        command = ['/usr/bin/python3', '-B', str(NAVTEST), '--enqueue', '--run', str(run),
                   '--tag', candidates[0].name, '--output', str(ART/'navsim_navtest_dispatch_20261011T0225')]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Evaluation preparation must not terminate the scientific training process.
        if result.returncode:
            write(directory/'EVALUATION_ENQUEUE_ERROR.json', {'update': k, 'returncode': result.returncode,
                  'stderr': result.stderr[-4000:], 'UTC': datetime.now(timezone.utc).isoformat()})
            continue
        records[str(k)] = {'command': command, 'receipt': result.stdout,
                           'UTC': datetime.now(timezone.utc).isoformat()}
        write(bank, records)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', required=True, choices=['G1_FULL_UNIFORM', 'G2_EVENT_UNIFORM'])
    args = parser.parse_args()
    directory = ROOT/args.group
    m = read(directory/'REGISTRATION.json')
    run = Path(m['run'])
    host = socket.gethostname().removesuffix('-worker-0')
    authorize(host)
    assert host in m['authorized_hosts']
    assert not (directory/'TRAINING_LAUNCH.json').exists(), 'Never duplicate a recorded training launch'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=m['cwd'],text=True).strip() == m['training_source_sha']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=m['cwd'],text=True).strip()
    assert read(run/'identity.json')['identity'] == m['identity']
    assert not (run/'STOP_AFTER_CHECKPOINT').exists()
    before = checkpoint(m)
    assert before['completed'] == m['parent_completed'] and before['tag'] == m['parent_tag']
    assert os.statvfs(run).f_bavail*os.statvfs(run).f_frsize > 200*(1<<30)
    for p in Path('/proc').glob('[0-9]*'):
        row = process(int(p.name))
        assert not (row and m['run'] in row['argv'] and any(a.endswith('/train_vla.py') for a in row['argv'])), 'Same run already active'
    raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    rows = [line.split(',') for line in raw.splitlines() if line.strip()]
    assert host == m['destination_host']
    assert len(rows) == 8 and all(int(x[2]) >= m['minimum_free_MiB_per_GPU'] for x in rows), 'Insufficient measured headroom; preserve other jobs'
    assert read(ROOT/'SOURCE_NATIVE_BOUNDARY_VERIFIED.json')['passed']
    assert read(ROOT/'ENVIRONMENT_AND_DATA_ACCEPTED.json')['passed']
    assert read(ROOT/'ROLLBACK_CHECKPOINT_PIN.json')['same_inode_all_files']
    uuids = sorted(x[1].strip() for x in rows)
    lock_root = ART/'physical_gpu_locks'
    paths = [run/'HOST_MIGRATION_RUNTIME.lock',
             lock_root/('queue_'+hashlib.sha256(str(run.resolve()).encode()).hexdigest()+'.lock'),
             lock_root/(hashlib.sha256(json.dumps(uuids).encode()).hexdigest()+'.lock')]
    with ExitStack() as stack:
        handles = [stack.enter_context(p.open('a')) for p in paths]
        for handle in handles:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        env = dict(os.environ, OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false',
                   PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
                   TORCH_EXTENSIONS_DIR=str(ROOT/'runtime_cache/torch_extensions'),
                   TRITON_CACHE_DIR=str(ROOT/'runtime_cache/triton'))
        with (directory/'training.log').open('ab') as log:
            child = subprocess.Popen(m['command'], cwd=m['cwd'], env=env, stdin=subprocess.DEVNULL,
                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                       pass_fds=tuple(f.fileno() for f in handles))
        write(directory/'TRAINING_LAUNCH.json', {'UTC':datetime.now(timezone.utc).isoformat(),
              'host':host, **process(child.pid), 'cwd':m['cwd'], 'parent_completed':before['completed'],
              'identity':m['identity'], 'training_source_sha':m['training_source_sha'],
              'execution_mode':m['execution_mode'], 'physical_GPU_UUIDs':uuids,
              'metrics_prefix_bytes':(run/'metrics.jsonl').stat().st_size, 'locks':list(map(str,paths))})
        write(directory/'STATE.json', {'status':'RESTORING_NATIVE_CHECKPOINT','parent_completed':before['completed'],
              'endpoint':100000,'pid':child.pid,'host':host})
        next_enqueue = time.monotonic() + 30
        while child.poll() is None:
            if time.monotonic() >= next_enqueue:
                try:
                    enqueue(m, directory)
                except Exception as error:
                    write(directory/'EVALUATION_ENQUEUE_ERROR.json', {'error':str(error)})
                next_enqueue = time.monotonic() + 30
            report = read(run/'RESUME_BOUNDARY_VERIFIED.json')
            if report.get('completed') == before['completed'] and report.get('passed'):
                write(directory/'RESUME_VERIFIED.json', report)
            time.sleep(5)
        code = child.wait()
        if code:
            write(directory/'STATE.json', {'status':'FAILED','returncode':code,'automatic_retry':False})
            raise RuntimeError('Training failed; inspect logs without automatic retry')
        final = checkpoint(m)
        if final['completed'] != 100000:
            write(directory/'STATE.json', {'status':'PAUSED','completed':final['completed'],'tag':final['tag']})
            return
        assert read(run/'TRAINING_COMPLETE.json')['updates'] == 100000
        enqueue(m,directory)
        write(directory/'STATE.json', {'status':'TRAINING_COMPLETE','completed':100000,'evaluation_complete':False})

if __name__ == '__main__':
    main()
