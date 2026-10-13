"""Resume the exact saved G1 state once, preserving existing real GPU work."""
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
import sys
import time

OUT = Path(__file__).resolve().parent
ART = OUT.parent
REPO = Path('/mnt/project/VLA-Drive-structured-world-fgtr-io-formal-frozen-20261009')
SHA = '0c54f1c49e7ed2a76b22826b654d6c72625f5b6d'
PYTHON = '/mnt/project/structured-world-fgtr-execution-assets/20261012/qwen_runtime_tsinghua_v1/bin/python'
RUN = ART / 'formal_navsim_zt2_G1_G0_v3_io/G1_FULL_UNIFORM'
TASK = 'structured_world_fgtr_round1'

def read(path):
    return json.loads(Path(path).read_text())

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(name, value):
    path = OUT / name
    tmp = path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(tmp, path)

def process(pid):
    p = Path('/proc') / str(pid)
    st = (p/'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=int(st[19]),
                argv=[os.fsdecode(x) for x in (p/'cmdline').read_bytes().split(b'\0') if x],
                cwd=os.readlink(p/'cwd'), uid=p.stat().st_uid)

def checks():
    host = socket.gethostname().removesuffix('-worker-0')
    assert host == 'training-vla-zt'
    policy = read('/mnt/project/server_dispatch_policy.json')
    assert host in policy['task_authorizations'][TASK]['allowed_hosts']
    assert host not in policy['global_dispatch_prohibitions']
    assert policy['host_reservations'][host]['task'] == TASK
    for banned in ('training-vlawm-zt','training-vlawm-zt2','training-vlawm-zt3','training-vlawm-zt5'):
        assert policy['global_dispatch_prohibitions'][banned]['new_task_dispatch_allowed'] is False
    assert read(OUT/'DESTINATION_AUTHORIZATION.json')['host'] == host
    for file in ('ENVIRONMENT_PARITY.json','ASSET_METADATA_PARITY.json','CPU_PREFETCH_EQUIVALENCE.json'):
        assert read(OUT/file)['passed']
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip() == SHA
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    parent = read(OUT/'PARENT_CHECKPOINT_VERIFIED.json')
    assert parent['status'] == 'PASS' and parent['completed'] == 80000
    assert (RUN/'checkpoints/latest').read_text().strip() == parent['tag']
    assert read(RUN/'identity.json')['identity'] == parent['identity']
    assert digest(Path(parent['folder'])/'COMPLETE.json') == parent['receipt_sha256']
    for file in parent['files']:
        stat = (Path(parent['folder'])/file['file']).stat()
        assert (stat.st_ino, stat.st_size, stat.st_mtime_ns) == (file['inode'],file['bytes'],file['mtime_ns'])
    assert not (RUN/'STOP_AFTER_CHECKPOINT').exists()
    assert not (OUT/'TARGET_LAUNCH.json').exists(), 'Launch exists: inspect; never duplicate'
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == os.getpid():
            continue
        try:
            argv = process(int(p.name))['argv']
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        assert not (str(RUN) in argv and any(x.endswith('/train_vla.py') for x in argv)), 'G1 already active'
    assert os.statvfs(RUN).f_bavail * os.statvfs(RUN).f_frsize > 200 * 1024**3
    expected = {r['pid']:r for r in read(OUT/'DESTINATION_OCCUPANTS.json')}
    gpu = []
    for _ in range(3):
        raw = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
        rows = [dict(zip(('index','uuid','total','used','util'),[x.strip() for x in row.split(',')])) for row in raw.splitlines()]
        assert len(rows) == 8 and len({r['uuid'] for r in rows}) == 8
        assert all(int(r['total'])-int(r['used']) >= 55*1024 for r in rows), rows
        gpu.append(rows)
        raw = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
        owners = {int(x.strip()) for x in raw.splitlines() if x.strip()}
        assert owners <= expected.keys(), ('Unexpected GPU owner',owners-expected.keys())
        for pid in owners:
            current = process(pid)
            assert all(current[k] == expected[pid][k] for k in ('pid','start_ticks','argv','cwd','uid'))
            assert 'iqe.rolling_cache.cli' in current['argv'] and 'produce' in current['argv']
        time.sleep(1)
    return parent, gpu, owners

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--execute',action='store_true')
    args = parser.parse_args()
    parent, gpu, owners = checks()
    old = read(OUT/'SOURCE_G1_MANIFEST.json')
    original = next(p['command'] for p in old['training_plan'] if p['target'] == 100000)
    argv = list(original)
    argv[0] = PYTHON
    for key,value in [('--cache',str(ART/'navsim_train_full_label_v5_centered')),
                      ('--dino-root',str(ART/'navsim_current_C1_only_v1'))]:
        argv[argv.index(key)+1] = value
    assert '--resume' in argv and '--stop-after' not in argv
    assert argv[argv.index('--output')+1] == str(RUN)
    assert argv[argv.index('--execution-mode')+1] == 'io_preserving_v1'
    allowed_indices = {0,original.index('--cache')+1,original.index('--dino-root')+1}
    assert len(argv)==len(original) and all(a==b or i in allowed_indices for i,(a,b) in enumerate(zip(argv,original)))
    uuids = sorted(x['uuid'] for x in gpu[-1])
    lockroot = ART/'physical_gpu_locks'
    locks = [RUN/'HOST_MIGRATION_RUNTIME.lock',
             lockroot/('queue_'+hashlib.sha256(str(RUN.resolve()).encode()).hexdigest()+'.lock'),
             lockroot/(hashlib.sha256(json.dumps(uuids).encode()).hexdigest()+'.lock')]
    record = dict(status='PREPARED',utc=datetime.now(timezone.utc).isoformat(),host='training-vla-zt',
                  argv=argv,original_argv=original,cwd=str(REPO),training_source_sha=SHA,
                  execution_mode='io_preserving_v1',run=str(RUN),identity=parent['identity'],
                  parent_completed=80000,parent_tag=parent['tag'],parent_receipt_sha256=parent['receipt_sha256'],
                  horizon=100000,global_batch=32,microbatch=4,FM_repeat=8,world_size=8,
                  GPU_samples=gpu,physical_GPU_UUIDs=uuids,preexisting_cache_pids_preserved=sorted(owners),
                  process_signals_sent=False,from_scratch=False,metrics_pre_resume_bytes=(RUN/'metrics.jsonl').stat().st_size,
                  unsaved_updates_requiring_replay=4873,original_metrics_preserved_without_truncation=True,
                  log=str(OUT/'G1_RESUMED_VLA_ZT.log'),locks=[str(p) for p in locks])
    write('TARGET_LAUNCH_DRY_RUN.json',record)
    if not args.execute:
        print(json.dumps(record),flush=True)
        return
    with ExitStack() as stack:
        handles = [stack.enter_context(p.open('ab')) for p in locks]
        for f in handles:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        checks()
        env = dict(os.environ,OMP_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',
                   CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
                   TORCH_EXTENSIONS_DIR=str(OUT/'runtime_cache/torch_extensions'),
                   TRITON_CACHE_DIR=str(OUT/'runtime_cache/triton'))
        with (OUT/'G1_RESUMED_VLA_ZT.log').open('ab') as log:
            child = subprocess.Popen(argv,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,
                                     start_new_session=True,pass_fds=tuple(f.fileno() for f in handles))
        record.update(status='LAUNCHED',**process(child.pid))
        write('TARGET_LAUNCH.json',record)
        write('STATE.json',dict(status='RUNNING',pid=child.pid,parent_completed=80000,horizon=100000))
        print(json.dumps(dict(status='LAUNCHED',pid=child.pid,host=record['host'],log=record['log'])),flush=True)
        result = child.wait()
        if result != 0:
            write('STATE.json',dict(status='FAILED',returncode=result,automatic_retry=False,process_signals_sent=False))
            raise RuntimeError('Native G1 continuation failed; inspect without blind retry')
        if not (RUN/'TRAINING_COMPLETE.json').exists():
            tag = (RUN/'checkpoints/latest').read_text().strip()
            complete = read(RUN/'checkpoints'/tag/'COMPLETE.json')
            assert complete['status'] == 'PAUSED' and complete['identity'] == parent['identity']
            write('STATE.json',dict(status='PAUSED',completed=complete['completed'],tag=tag,automatic_retry=False))
            return
        assert read(RUN/'TRAINING_COMPLETE.json')['updates'] == 100000
        assert read(RUN/'RESUME_BOUNDARY_VERIFIED.json')['completed'] == 80000
        routing = Path('/mnt/project/VLA-Drive-structured-world-fgtr-perf-20261011/tools/structured_world/run_navsim_navtest.py')
        command = ['/usr/bin/python3','-B',str(routing),'--enqueue','--run',str(RUN),'--tag','milestone_0100000',
                   '--output',str(ART/'navsim_navtest_dispatch_20261011T0225')]
        receipt = subprocess.check_output(command,cwd=REPO,text=True)
        write('COMPLETE.json',dict(status='TRAINING_COMPLETE',updates=100000,evaluation_enqueue=receipt,evaluation_command=command))
        write('STATE.json',dict(status='TRAINING_COMPLETE',updates=100000,evaluation_enqueued=True))

if __name__ == '__main__':
    main()
