"""Lead-only bounded scoring of one already registered complete dev export.

No waiting controller, automatic retry, GPU context, training or navtest.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path('/mnt/project/st-world-policy-20261016')
REPO = Path('/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017')
PREP = Path('/mnt/project/VLA-Drive-st-evaluation-agent-20261016/reports/st_world_policy/20261016/evaluation/full_dev_execution_v1')
SOURCE = '6714b1e9fb2e0c8059b90b8b4d48cd40796fe345'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0, str(REPO))
from tools.st_world_policy.score_world import read_complete_export
from starVLA.dataloader.st_world_targets import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import identity_hash
from tools.st_world_policy.training_state import TERMINATIONS, require_native_inventory


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_pin(pin):
    assert sha(pin['path']) == pin['sha256'], pin['path']
    return json.loads(Path(pin['path']).read_text())


# Fixed host routing for this campaign; no arbitrary host/config dispatcher.
C3_V3_PREP = PREP.parent / 'full_dev_C3_execution_v1'
C3_V3_PINS = {
    'POLICY.json': '70c172ffb691ba893e0c99ad1e9bcfc78dce338174b17539ec225d1fa759b39d',
    'INTENTS.json': '22be1d966aea1a2b076f1d29f7a124a3d25285abbe8591cbdb7cbefe533b6be4',
    'COMMANDS.json': 'c4f4c6498b29f42548b98a17c8d398e8cf2182f4ecb19c11197209774af12195',
    'TRAINING_BINDINGS.json': 'ae2cac8211bb3eb94c454129ed976f6fbe0fc6ee465925fe8751dbf30c299168',
    'EXECUTION_PLAN.json': 'df7bc0cbc30a2666913945ed6e5dd6188acab4e036442a99b1de787efb83c7ed',
}
ROUTE_HOSTS_V3 = ('training-rl-zt2', 'training-vla-zt', 'training-vlawm-tsinghua-zt')
ROUTE_SHARED_SOURCE_V3 = '100.97.192.3@tcp:100.97.192.4@tcp:/hbTcp001/78ebbad2386e922193d44ac249bbb12e_kl4gv5vivlcf06o2[/mnt/project]'
ROUTE_INFERENCE_V3 = {
    'world_size': 1, 'rank': 0, 'seed': 42, 'requested_scenes': 1696,
    'precision': 'FP32_master_eval_TF32_off', 'attention': 'sdpa',
    'solver': 'original_Euler_10', 'candidates_per_scene': 1,
    'score_max_calls': 1728, 'score_workers': 2,
}


def c3_pin_refs_v3():
    return {name: {'path': str(C3_V3_PREP / name), 'sha256': value}
            for name, value in C3_V3_PINS.items()}


def canonical_host_v3(host, policy, authorization):
    matches = [name for name in ROUTE_HOSTS_V3
               if name in policy['allowed_hosts']
               and host in policy['allowed_hosts'][name]['aliases']]
    assert len(matches) == 1 and host in policy['allowed_aliases'], 'Host outside the exact three-host route allowlist'
    name = matches[0]
    assert name in authorization['allowed_hosts'], 'Host lacks campaign authorization'
    bans = policy['global_dispatch_prohibitions']
    assert name not in bans and host not in bans, 'Host dispatch prohibited'
    for entry in bans.values():
        assert host not in entry.get('aliases', []) and name not in entry.get('aliases', []), 'Host alias dispatch prohibited'
    return name


def check_route_time_v3(route, now, remaining_seconds):
    import math
    start, end = route['valid_from_unix'], route['valid_until_unix']
    assert type(start) in (int, float) and type(end) in (int, float)
    assert math.isfinite(start) and math.isfinite(end)
    assert start <= now and end - now >= remaining_seconds, 'Route expired/not yet valid or insufficient bounded invocation time'


def validate_route_v3(route, job, host, policy, authorization, *, checkpoint_tag,
                      complete_receipt, receipt_identity, environment_reference, now,
                      remaining_seconds):
    fields = {'schema', 'job_id', 'host', 'hostname', 'gpu', 'gpu_uuid',
              'valid_from_unix', 'valid_until_unix', 'source_sha', 'training_registration',
              'runtime_identity', 'expected_completed', 'evaluation_policy',
              'checkpoint_tag', 'complete_receipt', 'receipt_identity', 'environment', 'inference'}
    assert set(route) == fields and route['schema'] == 'st_fixed_dev_route_v3'
    canonical = canonical_host_v3(host, policy, authorization)
    assert route['hostname'] == host and canonical_host_v3(route['host'], policy, authorization) == canonical, 'Fixed job route is on another host'
    assert route['job_id'] == job['job_id'] and route['source_sha'] == SOURCE
    assert route['training_registration'] == job['training_binding']['registration']
    assert route['runtime_identity'] == job['runtime_identity']
    assert type(route['expected_completed']) is int and route['expected_completed'] == job['expected_completed']
    assert route['evaluation_policy'] == job['evaluation_policy']
    assert route['checkpoint_tag'] == checkpoint_tag and route['complete_receipt'] == complete_receipt
    assert route['receipt_identity'] == receipt_identity
    assert route['environment'] == environment_reference
    assert identity_hash(route['inference']) == identity_hash(ROUTE_INFERENCE_V3), 'Fixed inference/score budget changed'
    assert type(route['gpu']) is int and 0 <= route['gpu'] < 8
    uuid = route['gpu_uuid']
    assert isinstance(uuid, str) and len(uuid) == 40 and uuid.startswith('GPU-')
    assert all(c in '0123456789abcdef-' for c in uuid[4:]), 'Malformed GPU UUID'
    check_route_time_v3(route, now, remaining_seconds)
    return canonical


def read_route_v3(regroot, expected_sha):
    assert len(expected_sha) == 64 and all(c in '0123456789abcdef' for c in expected_sha)
    pin = {'path': str(regroot / 'ROUTE.json'), 'sha256': expected_sha}
    assert sha(pin['path']) == expected_sha, 'Changed immutable route'
    return json.loads(Path(pin['path']).read_text()), pin


def fresh_environment_v3(reference):
    from tools.st_world_policy.training_config import environment_record
    assert sha(reference['path']) == reference['sha256'], 'Changed environment pin'
    expected = json.loads(Path(reference['path']).read_text())
    actual = environment_record()  # CPU-only package/CUDA-build metadata; no model/context.
    assert actual == expected, 'Actual host environment differs from fixed evaluation environment'
    return actual


def shared_lock_mount_v3():
    record = json.loads(subprocess.check_output(
        ['findmnt', '--json', '--target', str(ROOT), '--output', 'TARGET,FSTYPE,OPTIONS,SOURCE'], text=True))
    entries = record['filesystems']
    assert len(entries) == 1
    entry = entries[0]
    options = set(entry['options'].split(','))
    assert entry['target'] == '/mnt/project' and entry['source'] == ROUTE_SHARED_SOURCE_V3, 'Campaign slots must share the recorded filesystem'
    assert entry['fstype'] == 'lustre' and 'flock' in options and not options & {'localflock', 'noflock'}, 'Campaign slots require shared global flock semantics'
    return entry


def cpu_capacity_v3():
    affinity = len(os.sched_getaffinity(0))
    v2 = Path('/sys/fs/cgroup/cpu.max')
    if v2.is_file():
        quota, period = v2.read_text().split()
        quota = -1 if quota == 'max' else int(quota)
        period = int(period)
    else:
        quota = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
        period = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
    assert affinity >= 2 and (quota < 0 or quota / period >= 2), 'Insufficient CPU affinity/quota for fixed worker/thread budget'
    return {'affinity_count': affinity, 'quota_us': quota, 'period_us': period}


def acquire_campaign_slot_v3(kind):
    assert kind in ('export', 'score')
    count = 4 if kind == 'export' else 2
    for slot in range(count):
        f = (ROOT / f'resources/dev_{kind}_slot_{slot}.lock').open('a+')
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f, slot  # Caller passes this FD to the child for its full lifetime.
        except BlockingIOError:
            f.close()
    raise RuntimeError(f'All {count} campaign {kind} slots are occupied')


def fresh_gpu_v3(route):
    row = subprocess.check_output(['nvidia-smi', '--id='+str(route['gpu']),
        '--query-gpu=index,uuid,name,memory.used,utilization.gpu,memory.total',
        '--format=csv,noheader,nounits'], text=True).strip().split(',')
    assert len(row) == 6 and int(row[0]) == route['gpu']
    assert row[1].strip() == route['gpu_uuid'], 'GPU UUID differs from fixed route'
    assert row[2].strip() == 'NVIDIA A800-SXM4-80GB' and int(row[5]) >= 80000
    assert int(row[3]) < 64 and int(row[4]) < 5, 'Protect occupied GPU'
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader,nounits'], text=True)
    for line in processes.splitlines():
        if line.strip():
            fields = line.split(',')
            assert len(fields) == 2
            assert fields[0].strip() != route['gpu_uuid'], 'Protect existing GPU process, including unknown/foreign PID'
    return row


def check_export_budget_v3(regroot, out, resume):
    # Every durable invocation consumes one of two, including failed/unknown Popen.
    prior = [p for p in (regroot/'invocations').glob('*') if p.is_dir()]
    assert len(prior) < 2, 'Two export invocation admissions already consumed'
    assert not (out/'export/COMPLETE_rank0.json').exists(), 'Already complete; no duplicate inference'
    if resume:
        assert prior and (out/'export/IDENTITY.json').is_file(), 'Explicit resume needs an existing bound export'
    else:
        assert not prior and not (out/'export').exists(), 'Use explicit same-identity resume'
    return prior


# Exact W launch supplement; training remains fdf25f99, evaluation remains6714.
W_ARMS_V4 = ('SP0', 'SP1', 'SP2', 'SP3')
W_TRAINING_SOURCE_V4 = 'fdf25f99b702754787904215bf1c2a2b7515ea75'
W_TRAINING_RUNTIME_V4 = '/mnt/project/VLA-Drive-st-official-runtime-v2-20261016'
W_LAUNCH_SUPPLEMENT_V4 = {
    'path': str(PREP.parent / 'W_LAUNCH_HELPER_v4/W_LAUNCH_BINDINGS.json'),
    'sha256': 'e57d2219bc79b738d3305683181c49a12d865f3892acad35dafefbbf160bbd53',
}


def read_w_launch_control_v4(pin):
    # Only the small supplement/registration/config/approval/launch controls.
    # The original large W identity is retained by reference, never read here.
    with Path(pin['path']).open('rb') as handle:
        data = handle.read(1024 * 1024 + 1)
    assert len(data) <= 1024 * 1024, 'W launch control exceeds metadata bound'
    assert hashlib.sha256(data).hexdigest() == pin['sha256'], 'Changed W launch control pin'
    return json.loads(data)


def validate_w_launch_v4(job, binding, registration, config, approval, launch):
    arm = job['arm']
    assert arm in W_ARMS_V4 and binding['arm'] == arm
    assert job['job_id'] == binding['job_id'] == f'{arm}_step005000_dev1696_seed42_v1'
    assert job['stage'] == 'SCREENING' and job['expected_completed'] == 5000
    assert job['export_world_size'] == 1 and job['export_rank'] == 0
    assert job['training_world_size'] == 4
    assert job['runtime_identity'] == binding['runtime_identity']
    original = job['training_binding']
    assert binding['launch']['path'] == str(Path(binding['registration']['path']).parent / 'LAUNCH.json')
    for key in ('registration', 'approval', 'run_identity_file'):
        assert original[key] == binding[key], 'Changed original W binding: ' + key
    if 'launch' in original:
        assert original['launch'] == binding['launch'], 'Changed existing W launch pin'
    assert job['training_config'] == binding['config'] == registration['artifacts']['config']
    assert job['training_source_sha'] == original['training_source_sha'] == registration['source_sha'] == W_TRAINING_SOURCE_V4
    assert registration['source_fingerprint'] == binding['source_fingerprint']
    assert registration['run_id'] == binding['run_id'] == f'{arm}_fulltrain_seed42_official_5k_v1'
    assert config['space_time']['cell'] == arm.replace('SP', 'S-P')
    assert config['execution'] == registration['execution']
    fingerprint = {
        'registration_sha256': binding['registration']['sha256'],
        'run_id': registration['run_id'],
        'source_fingerprint': registration['source_fingerprint'],
        'execution': registration['execution'],
        'artifacts': {name: pin['sha256'] for name, pin in registration['artifacts'].items()},
    }
    assert identity_hash(fingerprint) == binding['registration_fingerprint_sha256']
    assert approval['formal_launch_allowed'] is True and approval['fingerprint'] == fingerprint
    assert launch['schema'] == 'st_reviewed_formal_launch_v1'
    assert launch['registration_sha256'] == binding['registration']['sha256']
    assert launch['approval_sha256'] == binding['approval']['sha256']
    assert launch['gate'] == {'formal_launch_allowed': True,
                              'identity': identity_hash(fingerprint), 'status': 'VERIFIED'}
    assert launch['cwd'] == W_TRAINING_RUNTIME_V4
    # Historical W receipts have no source_sha field; their pinned cwd and
    # approved registration carry fdf25f99, independently of evaluation6714.
    assert launch.get('source_sha', W_TRAINING_SOURCE_V4) == W_TRAINING_SOURCE_V4
    assert launch['output'] == str(Path(config['runtime']['output_root']) / registration['run_id'])
    assert launch['diagnostic_weights_used'] is False
    assert launch['argv'] == [
        'timeout', '--signal=TERM', '--kill-after=180s', '29400s',
        '/root/miniconda3/envs/ddp/bin/python', '-m', 'torch.distributed.run',
        '--standalone', '--nnodes=1', '--nproc_per_node=4', '-m',
        'tools.st_world_policy.train_world',
        '--registration', binding['registration']['path'],
        '--approval', binding['approval']['path'],
    ]


def supplement_w_launch_v4(job):
    if job['arm'] not in W_ARMS_V4:
        return job, None  # Every existing C path remains byte-for-byte unchanged.
    document = read_w_launch_control_v4(W_LAUNCH_SUPPLEMENT_V4)
    assert document['schema'] == 'st_fixed_W_launch_supplement_v4'
    assert document['parent_intents'] == {
        'path': str(PREP / 'INTENTS.json'),
        'sha256': 'e749a4969803c2a9f0a6eb67cbd19e7544de33013e946517fb67709a6a1df1b6',
    }
    assert document['training_source_sha'] == W_TRAINING_SOURCE_V4
    assert document['training_runtime'] == W_TRAINING_RUNTIME_V4
    assert document['evaluation_source_sha'] == SOURCE
    assert set(document['arms']) == set(W_ARMS_V4)
    binding = document['arms'][job['arm']]
    registration, config, approval, launch = (
        read_w_launch_control_v4(binding[name])
        for name in ('registration', 'config', 'approval', 'launch'))
    validate_w_launch_v4(job, binding, registration, config, approval, launch)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'],
        cwd=W_TRAINING_RUNTIME_V4, text=True).strip() == W_TRAINING_SOURCE_V4
    assert not subprocess.check_output(['git', 'status', '--porcelain'],
        cwd=W_TRAINING_RUNTIME_V4, text=True).strip()
    normalized = dict(job)
    normalized['training_binding'] = {**job['training_binding'], 'launch': binding['launch']}
    return normalized, dict(W_LAUNCH_SUPPLEMENT_V4)


# Retain original C2 pins and the original C3-null rejection path.
ADDITIVE_V2_PREP = PREP.parent / 'full_dev_C2_C3_execution_v1'
ADDITIVE_V2_PINS = {
    'POLICY.json': '7b97826d9774a6a84f5541d89d18e5c25c2a028670bab949c9ffab03f3b80303',
    'INTENTS.json': '16dced163d61d0c32ba6c97901e7f2d4e1b0b2d212f3fbe6a2069adab11c98ce',
    'COMMANDS.json': '640832d3714e8f889183950b50147bc1dcd1c2238cd637f019221a3c2cb10467',
    'TRAINING_BINDINGS.json': 'ed571df22ad2c82a1b099547fe98d4e163de260d9ce91d560949a5252c150324',
    'EXECUTION_PLAN.json': '6c413cf0d9f1b539243131138dfd7437735e6b42a5c870631947aa15a9b55e2f',
}


def additive_pin_refs_v2():
    return {name: {'path': str(ADDITIVE_V2_PREP / name), 'sha256': value}
            for name, value in ADDITIVE_V2_PINS.items()}


def normalize_fixed_additive_job_v3(raw, binding, registration, config, identity, arm, refs):
    """Only the two explicit source-pinned C2/C3 preparations are accepted."""
    assert arm in ('C2', 'C3') and raw['arm'] == arm
    assert refs == (additive_pin_refs_v2() if arm == 'C2' else c3_pin_refs_v3())
    step = raw['expected_completed']
    assert step in (512, 1024, 4096)
    stage = {512: 'DIAGNOSTIC_512', 1024: 'DIAGNOSTIC_1024', 4096: 'COMMON_ENDPOINT'}[step]
    assert raw['stage'] == stage and raw['job_id'] == f'{arm}_step{step:06d}_dev1696_seed42_v1'
    assert raw['comparison'] == f'{arm}_minus_C0' and raw['control_arm'] == 'C0'
    assert raw['control_job_id'] == f'C0_step{step:06d}_dev1696_seed42_v1'
    assert raw['training_source_sha'] == raw['evaluation_source_sha'] == binding['source_sha'] == registration['source_sha'] == SOURCE
    assert raw['runtime_identity'] and raw['runtime_identity'] == binding['runtime_identity'] == identity['identity']
    assert identity_hash({k: v for k, v in identity.items() if k != 'identity'}) == identity['identity']
    assert identity['registration_sha256'] == binding['registration']['sha256']
    assert identity['config'] == config and binding['config'] == registration['artifacts']['config']
    assert binding['registered_execution'] == registration['execution']
    assert binding['source_fingerprint'] == registration['source_fingerprint']
    assert config['cell'] == arm and registration['execution']['world_size'] == 4
    assert raw['export_world_size'] == 1 and raw['export_rank'] == 0
    for raw_key, bound_key in (('training_registration', 'registration'),
                               ('training_approval', 'approval'),
                               ('training_launch', 'launch'),
                               ('runtime_identity_file', 'run_identity_file')):
        assert raw[raw_key] == binding[bound_key] and raw[raw_key] is not None
    out = ROOT / 'formal/evaluation/jobs' / raw['job_id']
    assert raw['output_root'] == str(out)
    assert raw['export_root'] == str(out / 'export') and raw['score_root'] == str(out / 'official_score')
    assert raw['evaluation_registration_path'] == str(ROOT / 'formal/evaluation/registrations' / raw['job_id'] / 'REGISTRATION.json')
    assert raw['training_binding_ref'] == refs['TRAINING_BINDINGS.json']
    assert all(raw['checkpoint_binding'][key] is None
               for key in ('tag', 'complete_receipt', 'native_file_hashes', 'receipt_identity'))
    normalized = dict(raw)  # Preserve every original pair/source/static field.
    normalized['training_binding'] = {key: binding[key] for key in
                                      ('registration', 'approval', 'launch', 'run_identity_file')}
    normalized['training_binding'].update(status=binding['status'], training_source_sha=SOURCE)
    normalized['training_config'] = binding['config']
    normalized['training_world_size'] = registration['execution']['world_size']
    normalized['evaluation_policy'] = refs['POLICY.json']
    return normalized


def normalize_fixed_c2_job_v2(raw, binding, registration, config, identity):
    return normalize_fixed_additive_job_v3(raw, binding, registration, config, identity,
                                            'C2', additive_pin_refs_v2())


def load_fixed_additive_v2(job_id, base_documents):
    if not job_id.startswith(('C2_', 'C3_')):
        return None
    # The old C3-null plan remains rejected even when called directly.
    return load_pinned_additive_v3(job_id, base_documents, 'C2', additive_pin_refs_v2())


def load_fixed_additive_v3(job_id, base_documents):
    if job_id.startswith('C3_'):
        return load_pinned_additive_v3(job_id, base_documents, 'C3', c3_pin_refs_v3())
    return load_fixed_additive_v2(job_id, base_documents)


def load_pinned_additive_v3(job_id, base_documents, arm, refs):
    assert arm in ('C2', 'C3')
    assert refs == (additive_pin_refs_v2() if arm == 'C2' else c3_pin_refs_v3())

    def pinned(ref):
        assert sha(ref['path']) == ref['sha256'], 'Changed additive pin: ' + ref['path']
        return json.loads(Path(ref['path']).read_text())

    docs = {name: pinned(ref) for name, ref in refs.items()}
    policy = docs['POLICY.json']
    base_policy = base_documents['POLICY.json']
    assert policy['parent_policy'] == {'path': str(PREP / 'POLICY.json'),
        'sha256': '5373e3eb57dc4bbb0e042f4f08df4569bc0a5cbb5c5fedcbd45bfbb0a29ed17f'}
    assert pinned(policy['parent_policy']) == base_policy
    assert policy['scientific_v4'] == base_policy['scientific_plan']
    scientific = pinned(policy['scientific_v4'])
    for key in ('population_manifest', 'inference', 'score', 'dedup_resume', 'resource_default'):
        assert policy[key] == base_policy[key], 'Changed common evaluation contract: ' + key
    for key, name in (('commands', 'COMMANDS.json'), ('intents', 'INTENTS.json'),
                      ('training_bindings', 'TRAINING_BINDINGS.json'), ('execution_plan', 'EXECUTION_PLAN.json')):
        assert policy[key] == refs[name]
    plan = docs['EXECUTION_PLAN.json']
    assert plan['statistics'] == scientific['statistics'] and plan['endpoints'] == scientific['endpoints']
    matches = [j for j in docs['INTENTS.json']['jobs'] if j['job_id'] == job_id]
    commands = [j for j in docs['COMMANDS.json']['checkpoint_jobs'] if j['job_id'] == job_id]
    assert len(matches) == len(commands) == 1, 'Only the fixed additive endpoint IDs'
    raw, command = matches[0], commands[0]
    binding = docs['TRAINING_BINDINGS.json']['bindings'][raw['arm']]
    if raw['arm'] != arm or not binding['runtime_identity'] or not binding['run_identity_file'] or not binding['approval'] or not binding['launch']:
        raise ValueError('Unresolved or wrong arm/runtime in this frozen additive plan')
    registration = pinned(binding['registration'])
    config = pinned(binding['config'])
    identity = pinned(binding['run_identity_file'])
    approval, launch = pinned(binding['approval']), pinned(binding['launch'])
    assert approval['formal_launch_allowed'] and approval['fingerprint'] == binding['registration_fingerprint']
    assert launch['registration_sha256'] == binding['registration']['sha256']
    assert launch['approval_sha256'] == binding['approval']['sha256'] and launch['source_sha'] == SOURCE
    for key in ('registration', 'approval', 'run_identity_file'):
        assert plan['formal_runs'][arm][key] == binding[key]
    normalized = normalize_fixed_additive_job_v3(raw, binding, registration, config, identity, arm, refs)
    assert command['cwd'] == str(REPO) and command['execution_source_sha'] == SOURCE
    assert command['required_training_runtime_identity'] == normalized['runtime_identity']
    assert docs['COMMANDS.json']['only_runtime'] == str(REPO) and docs['COMMANDS.json']['only_source_sha'] == SOURCE
    return {'job': normalized, 'command': command, 'pins': refs}


def validate_static_job(registration, intent):
    """Bind the endpoint to the unchanged original ten-job preparation."""
    if intent['arm'] in ('C2', 'C3'):
        assert registration['job'] == intent, 'Changed fixed additive arm/step/control/source fields'
        assert registration.get('additive_preparation') == (additive_pin_refs_v2() if intent['arm'] == 'C2' else c3_pin_refs_v3())
    if intent['arm'] in W_ARMS_V4:
        assert registration['job'] == intent, 'Changed fixed W endpoint fields'
        assert registration.get('w_launch_supplement') == W_LAUNCH_SUPPLEMENT_V4
    for key in ('job_id', 'arm', 'stage', 'expected_completed', 'output_root',
                'export_root', 'score_root', 'evaluation_registration_path',
                'export_rank', 'export_world_size', 'runtime_identity',
                'training_binding', 'training_config', 'training_source_sha',
                'training_world_size', 'evaluation_policy'):
        assert registration['job'][key] == intent[key], 'Static endpoint changed: ' + key
    assert registration['schema'] == 'st_formal_endpoint_evaluation_binding_v1'
    assert registration['evaluation_source_sha'] == SOURCE
    assert registration['requested_scenes'] == 1696 and registration['seed'] == 42
    assert registration['export_world_size'] == 1 and registration['navtest'] is False
    assert registration['output_root'] == intent['output_root']


def validate_native_reference(registration, intent, checkpoint, native):
    """Read/compare the small COMPLETE receipt; do not rehash native payloads."""
    assert native['schema'] == 'st_native_checkpoint_v1'
    assert identity_hash({k: v for k, v in native.items() if k != 'receipt_identity'}) == native['receipt_identity']
    assert native['receipt_identity'] == registration['receipt_identity'] == checkpoint['receipt_identity']
    assert native['files'] == registration['native_files'], 'Native file map changed'
    world = intent['training_world_size']
    assert native['world_size'] == world
    require_native_inventory(native['files'], world)
    state = native['state']
    assert state['identity'] == intent['runtime_identity'] == checkpoint['run_identity']
    assert state['completed'] == intent['expected_completed'] == checkpoint['completed']
    assert state['tag'] == registration['checkpoint_tag'] == checkpoint['tag']
    assert state['termination_reason'] in TERMINATIONS
    assert checkpoint['native_receipt_sha256'] == registration['complete_receipt']['sha256']
    assert checkpoint['native_checkpoint_sha256'] == identity_hash({k: v['sha256'] for k, v in native['files'].items()})
    assert identity_hash({k: v for k, v in checkpoint.items() if k != 'sha256'}) == checkpoint['sha256']
    assert checkpoint['scope'] == 'FORMAL' and checkpoint['weights_usable_for_formal_init'] is True
    assert checkpoint['training_source_sha'] == intent['training_source_sha']


def validate_export_metadata(export, scenes, registration, intent, plan):
    validate_static_job(registration, intent)
    population = read_pin(plan['population_manifest'])
    assert population['schema'] == 'st_fixed_scene_list_v1' and population['split'] == 'dev'
    assert scenes == population['scenes'], 'Exact ordered dev scenes/input hashes changed'
    assert len(scenes) == 1696 and len({r['log'] for r in scenes}) == 16
    assert export['observations']['split'] == 'dev'
    assert export['observations']['manifest_sha256'] == plan['population_manifest']['sha256']
    assert export['source_sha'] == SOURCE
    for key in ('world_size', 'seed', 'solver', 'precision', 'candidates_per_scene'):
        assert export[key] == plan['inference'][key], 'Inference contract changed: ' + key
    assert export['requested_scenes'] == 1696
    checkpoint = export['checkpoint']
    tag = registration['checkpoint_tag']
    assert tag and Path(tag).name == tag and tag not in ('.', '..')
    training = read_pin(intent['training_binding']['registration'])
    config = read_pin(intent['training_config'])
    expected_run = Path(config['runtime']['output_root']) / training['run_id']
    assert checkpoint['run'] == str(expected_run)
    expected_receipt = expected_run / 'checkpoints' / tag / 'COMPLETE.json'
    assert registration['complete_receipt']['path'] == str(expected_receipt)
    native = read_pin(registration['complete_receipt'])
    validate_native_reference(registration, intent, checkpoint, native)


def check_score_budget(regroot, out, resume):
    # Count every durable directory, including an unknown/failed Popen attempt.
    prior = [p for p in (regroot / 'score_invocations').glob('*') if p.is_dir()]
    assert len(prior) < 2, 'Two scoring invocation admissions already consumed'
    assert not (out / 'official_score/COMPLETE.json').exists()
    if resume:
        assert prior and (out / 'official_score/IDENTITY.json').is_file()
    else:
        assert not prior and not (out / 'official_score').exists()
    return prior


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job-id', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--route-sha256', required=True, help='SHA256 of this job registration directory ROUTE.json')
    a = p.parse_args()
    host = socket.gethostname()
    policy_path = Path('/mnt/project/server_dispatch_policy.json')
    policy = json.loads(policy_path.read_text())
    auth_path = ROOT/'resources/AUTHORIZATION_20261016T203442.json'
    auth = json.loads(auth_path.read_text())
    canonical_host_v3(host, policy, auth)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip() == SOURCE
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip()
    commands = read_pin({'path':str(PREP/'COMMANDS.json'),'sha256':'b86ded00ad852034fedfdc4f21b44a2ef6480332d321f71f4bf0059c65e9f421'})
    plan = read_pin({'path':str(PREP/'POLICY.json'),'sha256':'5373e3eb57dc4bbb0e042f4f08df4569bc0a5cbb5c5fedcbd45bfbb0a29ed17f'})
    for pin in plan['execution_runtime']['source_files']:
        assert sha(pin['path']) == pin['sha256']
    read_pin(plan['score']['metric_index'])
    intents = read_pin({'path':str(PREP/'INTENTS.json'),'sha256':'e749a4969803c2a9f0a6eb67cbd19e7544de33013e946517fb67709a6a1df1b6'})
    additive = load_fixed_additive_v3(a.job_id, {'POLICY.json': plan, 'INTENTS.json': intents, 'COMMANDS.json': commands})
    matching = ([additive['command']] if additive else [j for j in commands['checkpoint_jobs'] if j['job_id'] == a.job_id])
    assert len(matching) == 1, 'Only predeclared dev endpoints'
    command = matching[0]['score']
    intent_matches = ([additive['job']] if additive else [j for j in intents['jobs'] if j['job_id'] == a.job_id])
    assert len(intent_matches) == 1
    intent, w_launch_supplement = supplement_w_launch_v4(intent_matches[0])
    regroot = ROOT/'formal/evaluation/registrations'/a.job_id
    registration = json.loads((regroot/'REGISTRATION.json').read_text())
    out = Path(registration['job']['output_root'])
    root = out/'export'
    validate_static_job(registration, intent)
    route, route_pin = read_route_v3(regroot, a.route_sha256)
    training_registration = read_pin(intent['training_binding']['registration'])
    canonical_host = validate_route_v3(route, intent, host, policy, auth,
        checkpoint_tag=registration['checkpoint_tag'], complete_receipt=registration['complete_receipt'],
        receipt_identity=registration['receipt_identity'],
        environment_reference=training_registration['artifacts']['environment'],
        now=time.time(), remaining_seconds=4260)
    assert registration.get('route') == route_pin, 'Export registration is not bound to this immutable route'
    actual_environment = fresh_environment_v3(route['environment'])
    shared_mount = shared_lock_mount_v3()
    cpu_capacity = cpu_capacity_v3()
    locks = []
    for lock_path in (regroot/'SCORE_DISPATCH.lock', root/'rank_0.lock'):
        f = lock_path.open('a+')
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(f)
    # Hold the export rank lock through scoring, so another export cannot enter
    # after the exit/completeness checks. This does not signal the exporter.
    export, scenes = read_complete_export(root)
    validate_export_metadata(export, scenes, registration, intent, plan)
    assert export['environment'] == actual_environment, 'Scoring host/export environment changed'
    results = sorted((root/'attempts').glob('*/RESULT.json'))
    assert results, 'No completed export attempt'
    result_path = results[-1]
    result = json.loads(result_path.read_text())
    assert result['status'] == 'COMPLETE' and result['completed'] == 1696 and result['failed_this_attempt'] == 0
    assert result['identity'] == export['identity']
    assert result['rank'] == 0 and result['requested'] == 1696
    export_invocation = json.loads((result_path.parent/'INVOCATION.json').read_text())
    assert export_invocation['host'] == host and export_invocation['identity'] == export['identity']
    assert export_invocation['arguments']['output'] == str(root)
    assert export_invocation['arguments']['checkpoint_tag'] == registration['checkpoint_tag']
    launches = sorted((regroot/'invocations').glob('*/LAUNCH.json'))
    assert launches, 'No actual registered export launch'
    export_launch_path = launches[-1]
    export_launch = json.loads(export_launch_path.read_text())
    assert export_launch['host'] == host and export_launch['source_sha'] == SOURCE
    assert export_launch.get('route') == route_pin, 'Export launch route changed'
    assert export_launch['gpu'] == route['gpu'] and export_launch['gpu_uuid'] == route['gpu_uuid']
    assert export_launch['registration_sha256'] == sha(regroot/'REGISTRATION.json')
    assert str(root) in export_launch['argv']
    # A completed receipt can precede final interpreter teardown. Refuse while any
    # live process still owns this exact export output; never signal it.
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            argv = (proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        assert not (b'tools.st_world_policy.export_world' in argv and str(root).encode() in argv), 'Exporter has not exited'
    assert shutil.disk_usage(ROOT).free >= 1024*2**30
    mem = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    assert int(mem['MemAvailable'].split()[0])*1024 >= 16*2**30
    # Same two shared score slots as v1/v2; never multiply by host count.
    slot_lock, slot = acquire_campaign_slot_v3('score')
    locks.append(slot_lock)
    prior = check_score_budget(regroot, out, a.resume)
    argv = command['argv']
    assert argv[argv.index('--predictions')+1] == str(root)
    assert argv[argv.index('--output')+1] == str(out/'official_score')
    assert argv[argv.index('--workers')+1] == '2' and argv[argv.index('--max-calls')+1] == '1728'
    assert argv[argv.index('--run-id')+1] == 'st-world-policy-20261016'
    assert argv[argv.index('--retry-attempts')+1] == '1' and argv[argv.index('--max-seconds')+1] == '3600'
    assert command['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert all(command['env'][name] == '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))
    check_route_time_v3(route, time.time(), 4260)
    assert sha(route_pin['path']) == route_pin['sha256'], 'Route changed during admission'
    invocation = regroot/'score_invocations'/str(time.time_ns()); invocation.mkdir(parents=True,exist_ok=False)
    cmd = ['timeout','--signal=TERM','--kill-after=60s','4200s',*argv]
    env = {**os.environ,**command['env'],'PYTHONUNBUFFERED':'1'}
    admission = {'schema':'st_dev_score_invocation_admission_v1',
        'state':'ADMITTED_BEFORE_POPEN_OUTCOME_UNKNOWN_UNTIL_LAUNCH',
        'admitted_unix':time.time(), 'host':host, 'job_id':a.job_id,
        'ordinal':len(prior)+1, 'maximum_invocations':2, 'resume':a.resume,
        'registration_sha256':sha(regroot/'REGISTRATION.json'),
        'export_identity':export['identity'], 'max_calls_total_across_resume':1728,
        'argv':cmd, 'source_sha':SOURCE, 'route':route_pin,
        'campaign_score_slot':slot,'maximum_campaign_score_slots':2}
    # A crash or failed Popen consumes this admission; never remove its directory.
    atomic_json(invocation/'ADMISSION.json', admission, immutable=True)
    with (invocation/'SCORE.log').open('xb') as log:
        child = subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
            pass_fds=tuple(f.fileno() for f in locks))
    record = {'schema':'st_complete_dev_score_launch_v1','pid':child.pid,'host':host,
        'argv':cmd,'cwd':str(REPO),'started_unix':time.time(),'source_sha':SOURCE,
        'export_identity':export['identity'],'registration_sha256':sha(regroot/'REGISTRATION.json'),
        'export_complete':{'path':str(root/'COMPLETE_rank0.json'),'sha256':sha(root/'COMPLETE_rank0.json')},
        'export_result':{'path':str(result_path),'sha256':sha(result_path)},
        'authorization_sha256':sha(auth_path),'live_policy_sha256':sha(policy_path),
        'launcher_sha256':sha(__file__),'score_workers':2,'gpu_count':0,'requested_scenes':1696,
        'max_calls_total_across_resume':1728,'resume':a.resume,'navtest':False,'automatic_followup':False,
        'admission':{'path':str(invocation/'ADMISSION.json'),'sha256':sha(invocation/'ADMISSION.json')},
        'native_complete':registration['complete_receipt'],
        'population_manifest':plan['population_manifest'],
        'export_launch':{'path':str(export_launch_path),'sha256':sha(export_launch_path)},
        'export_rank_lock_held_through_score':True,'route':route_pin,
        'canonical_host':canonical_host,'environment':actual_environment,
        'shared_lock_mount':shared_mount,'cpu_capacity':cpu_capacity,
        'campaign_score_slot':slot,'maximum_campaign_score_slots':2}

    if additive:
        record['additive_preparation'] = additive['pins']
    atomic_json(invocation/'LAUNCH.json',record,immutable=True)
    print(json.dumps(record))


if __name__ == "__main__":
    main()
