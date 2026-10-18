"""Lead-only first native SP2/SP3 20k continuation; preserve original locks/gate."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path('/mnt/project/st-world-policy-20261016')
REPO = Path('/mnt/project/VLA-Drive-st-official-runtime-v2-20261016')
sys.path.insert(0, str(REPO))
from tools.st_world_policy.check_formal_approval import check_formal_approval, verify_registration, file_hash
from starVLA.dataloader.st_world_targets import atomic_json

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--registration', required=True)
a = p.parse_args()
regpath = Path(a.registration).resolve()
assert regpath.is_relative_to(ROOT / 'formal/spatial/registrations')
reg = json.loads(regpath.read_text())
cfg = json.loads(Path(reg['artifacts']['config']['path']).read_text())
assert reg['run_id'] in ('SP2_fulltrain_seed42_official_20k_v1','SP3_fulltrain_seed42_official_20k_v1')
assert cfg['execution']['scope'] == 'FORMAL' and cfg['execution']['stop_after'] == 20000
assert cfg.get('continuation'), 'Exact native parent binding required'
from tools.st_world_policy.budget_extension import _recipe
parent = json.loads(Path(cfg['continuation']['parent_registration']['path']).read_text())
assert file_hash(cfg['continuation']['parent_registration']['path']) == cfg['continuation']['parent_registration']['sha256']
assert _recipe(cfg) == _recipe(json.loads(Path(parent['artifacts']['config']['path']).read_text()))
assert cfg['execution']['max_updates'] == 20000 and cfg['execution']['world_size'] == 4
fingerprint = verify_registration(regpath, repo=REPO)
approval = regpath.parent / 'RUN_APPROVAL.json'
gate = check_formal_approval(regpath, approval, repo=REPO)
host = socket.gethostname()
policy_path = Path('/mnt/project/server_dispatch_policy.json')
policy = json.loads(policy_path.read_text())
auth_path = ROOT / 'resources/AUTHORIZATION_20261016T203442.json'
auth = json.loads(auth_path.read_text())
assert host in policy['allowed_aliases'] and host.removesuffix('-worker-0') in auth['allowed_hosts']
lease = json.loads(Path(reg['artifacts']['resource_lease']['path']).read_text())
assert lease['run_id'] == reg['run_id'] and lease['source_sha'] == reg['source_sha']
assert lease['first_continuation_only'] is True and lease['no_automatic_resume'] is True
assert host in lease['hostnames'] and lease['valid_from_unix'] <= time.time()
assert lease['valid_until_unix'] >= time.time() + cfg['runtime']['max_seconds'] + 780
assert lease['max_invocation_seconds'] == cfg['runtime']['max_seconds'] == 28800
assert lease['maximum_optimizer_updates_this_stage'] == 15000
visible = lease['cuda_visible_devices_by_host'][host]
ids = [int(x) for x in visible.split(',')]
assert len(ids) == 4
out = Path(cfg['runtime']['output_root']) / reg['run_id']
assert not out.exists(), 'Refuse duplicate or implicit resume'
assert not (regpath.parent / 'LAUNCH.json').exists(), 'This registration was already dispatched'
locks = []
for i in ids:
    lock = open(f'/tmp/foresight_gpu_{i}.lock', 'a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    locks.append(lock)
rows = [x.split(',') for x in subprocess.check_output(['nvidia-smi', '--id=' + visible,
    '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).splitlines()]
assert [r[1].strip() for r in rows] == lease['gpu_uuids_by_host'][host]
assert all(int(r[2]) < 64 and int(r[3]) < 5 for r in rows), 'Real/unknown GPU jobs are protected'
free = shutil.disk_usage(ROOT).free
# Separate current retained obligations, new bounded slots, unfilled cache and floor.
cpref = lease['checkpoint_commitments']
assert file_hash(cpref['path']) == cpref['sha256']
cp = json.loads(Path(cpref['path']).read_text())
assert cp['schema'] == 'st_checkpoint_storage_commitments_v1' and cp['status'] == 'CURRENT'
assert cp['checked_at_unix'] <= time.time() and cp['valid_until_unix'] >= time.time() + 29580
assert cp['remaining_bytes'] == 4143 * 2**30 and cp['free_floor_bytes'] == 1024 * 2**30
assert free >= (cp['remaining_bytes'] + cp['free_floor_bytes']
                + lease['remaining_cache_growth_bytes'] + lease['separate_conservative_metadata_buffer_bytes'])
seconds = cfg['runtime']['max_seconds']
cmd = ['timeout', '--signal=TERM', '--kill-after=180s', str(seconds + 600) + 's',
    '/root/miniconda3/envs/ddp/bin/python', '-m', 'torch.distributed.run',
    '--standalone', '--nnodes=1', '--nproc_per_node=4', '-m', 'tools.st_world_policy.train_world',
    '--registration', str(regpath), '--approval', str(approval)]
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': visible, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
    'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
env.pop('FLASH_ATTENTION_DETERMINISTIC', None)
with (regpath.parent / 'TRAIN.log').open('xb') as stream:
    child = subprocess.Popen(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
        pass_fds=tuple(x.fileno() for x in locks))
atomic_json(regpath.parent / 'LAUNCH.json', {'schema': 'st_reviewed_formal_launch_v1',
    'pid': child.pid, 'host': host, 'started_unix': time.time(), 'argv': cmd, 'cwd': str(REPO),
    'registration_sha256': file_hash(regpath), 'approval_sha256': file_hash(approval),
    'dispatch_policy_sha256': file_hash(policy_path), 'authorization_sha256': file_hash(auth_path),
    'gate': gate, 'fresh_gpu_inventory': rows, 'free_bytes_at_launch': free,
    'output': str(out), 'diagnostic_weights_used': False, 'automatic_followup': False}, immutable=True)
print(json.dumps({'status': 'FORMAL_LAUNCHED', 'pid': child.pid, 'run_id': reg['run_id'], 'output': str(out)}))
