"""Lead-only C0--C4 launch/resume from immutable, independently reviewed inputs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time

ROOT = Path('/mnt/project/st-world-policy-20261016')
REPO = Path('/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017')
sys.path.insert(0, str(REPO))
from tools.st_world_policy.check_formal_approval import check_formal_approval, verify_registration, file_hash
from tools.st_world_policy.flow_training_config import validate_config
from tools.st_world_policy.training_state import resolve_latest
from starVLA.dataloader.st_world_targets import atomic_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--registration', required=True)
p.add_argument('--approval', required=True)
p.add_argument('--resume', action='store_true')
p.add_argument('--acknowledge-stop', action='store_true')
a = p.parse_args()
if a.acknowledge_stop and not a.resume:
    p.error('--acknowledge-stop requires --resume')
regpath = Path(a.registration).resolve()
assert regpath.is_relative_to(ROOT/'formal/consequence/registrations')
reg = json.loads(regpath.read_text())
cfg = json.loads(Path(reg['artifacts']['config']['path']).read_text())
validate_config(cfg)
assert cfg['execution']['scope'] == 'FORMAL'
assert cfg['execution']['max_updates'] == cfg['execution']['stop_after'] == 4096
assert cfg['execution']['world_size'] == 4 and cfg['execution']['global_batch'] == 32
assert cfg['execution']['micro_batch'] == 4 and cfg['execution']['sample_limit'] == 0
assert cfg['runtime']['max_seconds'] == 28800 and cfg['runtime']['save_every'] == 512
assert cfg['runtime'].get('profile_mode', 'none') == 'none'
fingerprint = verify_registration(regpath, repo=REPO)
# Resume only. Preserve original reviews and RUN_APPROVAL.json; an independent
# policy refresh binds the same scientific fingerprint and immutable evidence.
assert a.resume, 'Policy-refresh launcher only resumes the existing lineage'
approval = Path(a.approval).resolve()
assert approval.is_relative_to(regpath.parent/'policy_refresh'), 'Use versioned local approval'
gate = check_formal_approval(regpath, approval, repo=REPO)
host = socket.gethostname()
policy_path = Path('/mnt/project/server_dispatch_policy.json')
policy = json.loads(policy_path.read_text())
auth_path = ROOT/'resources/AUTHORIZATION_20261016T203442.json'
auth = json.loads(auth_path.read_text())
assert host in policy['allowed_aliases'] and host.removesuffix('-worker-0') in auth['allowed_hosts']
lease = json.loads(Path(reg['artifacts']['resource_lease']['path']).read_text())
assert host in lease['hostnames'] and lease['valid_from_unix'] <= time.time()
seconds = cfg['runtime']['max_seconds']
# Every invocation including final-save/cleanup grace must fit the immutable lease.
assert time.time() + seconds + 780 < lease['valid_until_unix'], 'Insufficient lease for invocation and save grace'
visible = lease['cuda_visible_devices_by_host'][host]
ids = [int(x) for x in visible.split(',')]
assert len(ids) == 4 and len(set(ids)) == 4
out = Path(cfg['runtime']['output_root'])/reg['run_id']
assert out.is_relative_to(ROOT/'formal/consequence/runs')
launch_lock = (regpath.parent/'DISPATCH.lock').open('a+')
fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
resolution = None
saved_directories = []
if a.resume:
    assert out.is_dir() and (regpath.parent/'LAUNCH.json').is_file()
    assert len(list((regpath.parent/'invocations').glob('*/LAUNCH.json'))) < 6, 'Finite six-invocation budget exhausted'
    identity = json.loads((out/'identity.json').read_text())
    assert identity['registration_sha256'] == file_hash(regpath)
    assert (out/'STOP_REQUESTED').exists() == a.acknowledge_stop
else:
    assert not out.exists() and not (regpath.parent/'LAUNCH.json').exists(), 'Duplicate output or implicit resume refused'
locks = [launch_lock]
for i in ids:
    lock = open(f'/tmp/foresight_gpu_{i}.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    locks.append(lock)
rows = [x.split(',') for x in subprocess.check_output(['nvidia-smi','--id='+visible,
    '--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True).splitlines()]
assert [r[1].strip() for r in rows] == lease['gpu_uuids_by_host'][host]
assert all(int(r[2]) < 64 and int(r[3]) < 5 for r in rows), 'Real or unknown jobs are protected'
if a.resume:
    # Do not estimate future writes from an unverified higher-step receipt:
    # the trainer may otherwise fall back to an older complete checkpoint.
    resolution = resolve_latest(out/'checkpoints', identity=identity['identity'], world_size=4)
    completed = resolution['selected']['receipt']['state']['completed']
    saved_directories = [x.name for x in (out/'checkpoints').iterdir() if x.is_dir()]
    initial_writes = 0
else:
    completed = 0
    initial_writes = 1
remaining_periodic = [x for x in range(512,4097,512) if x > completed]
extra_pause_and_partial = 2 if completed < 4096 else 0
maximum_directories = len(saved_directories)+initial_writes+len(remaining_periodic)+extra_pause_and_partial
assert maximum_directories <= 15, 'Remaining periodic/pause/partial native writes exceed reviewed15-directory budget'
free = shutil.disk_usage(ROOT).free
# Cache cap is separate: preserve remaining cache claims, all four W checkpoint
# reservations and the conservative full five-arm C reserve. Never delete files.
assert free >= (1024 + 123 + 4*240 + 187 + 4*209)*2**30, 'Insufficient separate disk reserves'
invocation = regpath.parent/'invocations'/str(time.time_ns())
invocation.mkdir(parents=True, exist_ok=False)
atomic_json(invocation/'NATIVE_STORAGE_ADMISSION.json', {
    'existing_directories':saved_directories, 'verified_completed':completed,
    'initial_writes':initial_writes, 'remaining_periodic_steps':remaining_periodic,
    'extra_pause_and_partial':extra_pause_and_partial,
    'maximum_directories':maximum_directories, 'reviewed_limit':15}, immutable=True)
if resolution is not None:
    atomic_json(invocation/'RESUME_RESOLUTION.json', resolution, immutable=True)
cmd = ['timeout','--signal=TERM','--kill-after=180s',str(seconds+600)+'s',
    '/root/miniconda3/envs/ddp/bin/python','-m','torch.distributed.run',
    '--standalone','--nnodes=1','--nproc_per_node=4','-m','tools.st_world_policy.train_flow_policy',
    'train','--registration',str(regpath),'--approval',str(approval)]
if a.resume:
    cmd.append('--resume')
if a.acknowledge_stop:
    cmd.append('--acknowledge-stop')
env = {**os.environ,'CUDA_VISIBLE_DEVICES':visible,'CUBLAS_WORKSPACE_CONFIG':':4096:8',
    'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
    'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}
env.pop('FLASH_ATTENTION_DETERMINISTIC', None)
with (invocation/'TRAIN.log').open('xb') as stream:
    child = subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
        pass_fds=tuple(x.fileno() for x in locks))
record = {'schema':'st_reviewed_formal_flow_policy_refresh_launch_v1','pid':child.pid,'host':host,
    'started_unix':time.time(),'argv':cmd,'cwd':str(REPO),'scope':'FORMAL',
    'source_sha':reg['source_sha'],'registration':str(regpath),
    'registration_sha256':file_hash(regpath),'approval_sha256':file_hash(approval),
    'dispatch_policy_sha256':file_hash(policy_path),'authorization_sha256':file_hash(auth_path),
    'gate':gate,'fresh_gpu_inventory':rows,'free_bytes_at_launch':free,
    'output':str(out),'resume':a.resume,'diagnostic_weights_used':False,
    'automatic_followup':False,'invocation_seconds':seconds,'save_cleanup_grace_seconds':780}
atomic_json(invocation/'LAUNCH.json',record,immutable=True)
if not a.resume:
    atomic_json(regpath.parent/'LAUNCH.json',record,immutable=True)
observer = ROOT/'resources/observe_formal_flow_cpu_v1.py'
with (invocation/'OBSERVER.log').open('xb') as stream:
    obs = subprocess.Popen(['/root/miniconda3/envs/ddp/bin/python',str(observer),
        '--invocation',str(invocation)],cwd=REPO,
        env={**env,'CUDA_VISIBLE_DEVICES':''},stdin=subprocess.DEVNULL,
        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
atomic_json(invocation/'OBSERVER_LAUNCH.json',{'pid':obs.pid,'script':str(observer),
    'sha256':file_hash(observer),'interval_seconds':10,'max_bytes':256*2**20,
    'max_seconds':seconds+780,'deadline_mode':'cooperative_monotonic_between_proc_samples',
    'signals_sent':False},immutable=True)
print(json.dumps({'status':'FORMAL_RESUMED' if a.resume else 'FORMAL_LAUNCHED',
    'pid':child.pid,'run_id':reg['run_id'],'invocation':str(invocation),'output':str(out)}))
