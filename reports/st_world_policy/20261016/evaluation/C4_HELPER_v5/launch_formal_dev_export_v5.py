"""Lead-only launch of a predeclared, exact complete full-dev endpoint.

No polling, latest fallback, extra training, scoring or navtest dispatch.
Existing frozen export code performs full native checksum/strict model loading.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

ROOT = Path('/mnt/project/st-world-policy-20261016')
PREP = Path('/mnt/project/VLA-Drive-st-evaluation-agent-20261016/reports/st_world_policy/20261016/evaluation/full_dev_execution_v1')
REPO = Path('/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017')
SOURCE = '6714b1e9fb2e0c8059b90b8b4d48cd40796fe345'
sys.path.insert(0, str(REPO))
from starVLA.dataloader.st_world_targets import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import identity_hash
from tools.st_world_policy.training_state import require_native_inventory


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(ref):
    p = Path(ref['path'])
    assert p.is_absolute() and sha(p) == ref['sha256'], str(p)
    return json.loads(p.read_text())


# Only C4 is added; all older preparation pins and routes remain fixed.
C4_V5_PREP = Path('/mnt/project/VLA-Drive-st-c4-dev-preparation-20261017/reports/st_world_policy/20261016/evaluation/full_dev_C4_execution_v1')
C4_V5_PINS = {'POLICY.json': 'ed465e3bedc8974c0bc43edd64ed73f55090010c314cccce8b9c9718c3e2e1b2', 'INTENTS.json': '9bfea3670b0f7ea40a57eb5902fd94ae404c2e78038ebe76246885f522f9f8ea', 'COMMANDS.json': '334e8735f45da2f2fbf70d3b27e4e2cdabfdfa070c4549443533a3d57d125879', 'TRAINING_BINDINGS.json': 'c68375034eaf616c4a797d1b8e1fae4f441ff3588933efef34e0e8884f18338e', 'EXECUTION_PLAN.json': 'dddf0077c272e1310b873321006d469fcc261f82620ef6804135067bc40fca6c'}


def c4_pin_refs_v5():
    return {name: {'path': str(C4_V5_PREP / name), 'sha256': value}
            for name, value in C4_V5_PINS.items()}


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
    """Only the explicit source-pinned C2/C3/C4 preparations are accepted."""
    assert arm in ('C2', 'C3', 'C4') and raw['arm'] == arm
    assert refs == (additive_pin_refs_v2() if arm == 'C2' else c3_pin_refs_v3() if arm == 'C3' else c4_pin_refs_v5())
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
    if job_id.startswith('C4_'):
        return load_pinned_additive_v3(job_id, base_documents, 'C4', c4_pin_refs_v5())
    if job_id.startswith('C3_'):
        return load_pinned_additive_v3(job_id, base_documents, 'C3', c3_pin_refs_v3())
    return load_fixed_additive_v2(job_id, base_documents)


def load_pinned_additive_v3(job_id, base_documents, arm, refs):
    assert arm in ('C2', 'C3', 'C4')
    assert refs == (additive_pin_refs_v2() if arm == 'C2' else c3_pin_refs_v3() if arm == 'C3' else c4_pin_refs_v5())

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


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--job-id', required=True)
p.add_argument('--checkpoint-tag', required=True)
p.add_argument('--gpu', required=True, type=int, choices=range(8))
p.add_argument('--route-sha256', required=True, help='SHA256 of this job registration directory ROUTE.json')
p.add_argument('--resume', action='store_true')
a = p.parse_args()
assert re.fullmatch(r'step_[0-9]{6}_[0-9]+', a.checkpoint_tag), 'Explicit native tag required'
host = socket.gethostname()
policy_path = Path('/mnt/project/server_dispatch_policy.json')
policy = json.loads(policy_path.read_text())
auth_path = ROOT/'resources/AUTHORIZATION_20261016T203442.json'
auth = json.loads(auth_path.read_text())
canonical_host_v3(host, policy, auth)
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip() == SOURCE
assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip()
pins = {'POLICY.json':'5373e3eb57dc4bbb0e042f4f08df4569bc0a5cbb5c5fedcbd45bfbb0a29ed17f',
        'COMMANDS.json':'b86ded00ad852034fedfdc4f21b44a2ef6480332d321f71f4bf0059c65e9f421',
        'INTENTS.json':'e749a4969803c2a9f0a6eb67cbd19e7544de33013e946517fb67709a6a1df1b6'}
docs = {k:read({'path':str(PREP/k),'sha256':v}) for k,v in pins.items()}
cfg = docs['POLICY.json']
for ref in cfg['execution_runtime']['source_files']:
    assert sha(ref['path']) == ref['sha256']
for name in ('population_manifest','scientific_plan','assets','budget'):
    read(cfg[name])
read(cfg['score']['metric_index'])
review_path = ROOT/'audits/correctness/FORMAL_EVALUATION_REFERENCE_BINDING_v1.json'
assert sha(review_path) == 'd304b2154a447d2d7c0ed3bfb23bb44b434b2b86fdedfcb2728ae6db27c3ea7e'
additive = load_fixed_additive_v3(a.job_id, docs)
jobs = ([additive['job']] if additive else [r for r in docs['INTENTS.json']['jobs'] if r['job_id'] == a.job_id])
commands = ([additive['command']] if additive else [r for r in docs['COMMANDS.json']['checkpoint_jobs'] if r['job_id'] == a.job_id])
assert len(jobs) == len(commands) == 1
job, command = jobs[0], commands[0]
job, w_launch_supplement = supplement_w_launch_v4(job)
train_reg = read(job['training_binding']['registration'])
train_cfg = read(job['training_config'])
for name in ('approval','launch','run_identity_file'):
    read(job['training_binding'][name])
run = Path(train_cfg['runtime']['output_root'])/train_reg['run_id']
folder = run/'checkpoints'/a.checkpoint_tag
receipt_path = folder/'COMPLETE.json'
receipt = json.loads(receipt_path.read_text())
assert receipt['schema'] == 'st_native_checkpoint_v1'
assert receipt['receipt_identity'] == identity_hash({k:v for k,v in receipt.items() if k != 'receipt_identity'})
assert receipt['state']['completed'] == job['expected_completed']
assert receipt['state']['tag'] == a.checkpoint_tag
assert receipt['state']['identity'] == job['runtime_identity']
assert receipt['world_size'] == job['training_world_size']
assert receipt['state']['termination_reason'] in ('RUNNING_CHECKPOINT','OPERATOR_PAUSED','BUDGET_PAUSED','MAX_STEPS')
require_native_inventory(receipt['files'], receipt['world_size'])
assert set(receipt['files']) == {x.name for x in folder.glob('*.pt')}
for name, info in receipt['files'].items():
    assert Path(name).name == name and (folder/name).stat().st_size == info['bytes']
# Payload checksums are revalidated by both frozen identity/model readers before
# use. This admission binds the exact canonical receipt; it does not replace them.
out = Path(job['output_root'])
regroot = Path(job['evaluation_registration_path']).parent
route, route_pin = read_route_v3(regroot, a.route_sha256)
canonical_host = validate_route_v3(route, job, host, policy, auth,
    checkpoint_tag=a.checkpoint_tag,
    complete_receipt={'path':str(receipt_path),'sha256':sha(receipt_path)},
    receipt_identity=receipt['receipt_identity'], environment_reference=train_reg['artifacts']['environment'],
    now=time.time(), remaining_seconds=7860)
assert a.gpu == route['gpu'], 'Requested GPU differs from fixed job route'
actual_environment = fresh_environment_v3(route['environment'])
shared_mount = shared_lock_mount_v3()
cpu_capacity = cpu_capacity_v3()
locks = []
for path in (regroot/'DISPATCH.lock', Path(f'/tmp/foresight_gpu_{a.gpu}.lock')):
    f = path.open('a+'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB); locks.append(f)
slot_lock, slot = acquire_campaign_slot_v3('export')
locks.append(slot_lock)
row = fresh_gpu_v3(route)
assert shutil.disk_usage(ROOT).free >= (1024+123+960+1023+10)*2**30, 'Protect independent cache/checkpoint/evaluation reserves'
mem = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
assert int(mem['MemAvailable'].split()[0])*1024 >= 64*2**30, 'Insufficient host memory for native reconstruction'
prior = check_export_budget_v3(regroot, out, a.resume)
binding = {'schema':'st_formal_endpoint_evaluation_binding_v1','job':job,
    'checkpoint_tag':a.checkpoint_tag,'complete_receipt':{'path':str(receipt_path),'sha256':sha(receipt_path)},
    'receipt_identity':receipt['receipt_identity'],'native_files':receipt['files'],
    'policy_files_sha256':pins,'evaluation_source_sha':SOURCE,
    'evaluation_correctness':{'path':str(review_path),'sha256':sha(review_path)},
    'export_world_size':1,'seed':42,'requested_scenes':1696,'output_root':str(out),
    'formal_evaluation_authorized_by':'Lead under fixed campaign dev schedule',
    'training_launch_authorized':False,'navtest':False,'ACTIVE_update':False,
    'route':route_pin}
if additive:
    binding['additive_preparation'] = additive['pins']
if w_launch_supplement:
    binding['w_launch_supplement'] = w_launch_supplement
atomic_json(regroot/'REGISTRATION.json',binding,immutable=True)
argv = [a.checkpoint_tag if x == '__EXACT_COMPLETE_NATIVE_TAG_REQUIRED__' else x for x in command['export']['argv_template']]
assert not any('__' in x for x in argv)
cmd = ['timeout','--signal=TERM','--kill-after=60s','7800s',*argv]
env = {**os.environ,**command['export']['env'],'CUDA_VISIBLE_DEVICES':str(a.gpu),'PYTHONUNBUFFERED':'1'}
# Recheck remaining route time immediately before charging a durable invocation.
check_route_time_v3(route, time.time(), 7860)
assert sha(route_pin['path']) == route_pin['sha256'], 'Route changed during admission'
invocation = regroot/'invocations'/str(time.time_ns()); invocation.mkdir(parents=True,exist_ok=False)
admission = {'schema':'st_dev_export_invocation_admission_v3',
    'state':'ADMITTED_BEFORE_POPEN_OUTCOME_UNKNOWN_UNTIL_LAUNCH',
    'admitted_unix':time.time(),'host':host,'job_id':a.job_id,'route':route_pin,
    'ordinal':len(prior)+1,'maximum_invocations':2,'resume':a.resume,
    'registration_sha256':sha(regroot/'REGISTRATION.json'),'argv':cmd,'source_sha':SOURCE,
    'campaign_export_slot':slot,'maximum_campaign_export_slots':4}
atomic_json(invocation/'ADMISSION.json',admission,immutable=True)
with (invocation/'EXPORT.log').open('xb') as log:
    child = subprocess.Popen(cmd,cwd=REPO,env=env,stdin=subprocess.DEVNULL,stdout=log,
        stderr=subprocess.STDOUT,start_new_session=True,pass_fds=tuple(f.fileno() for f in locks))
record = {'pid':child.pid,'host':host,'argv':cmd,'cwd':str(REPO),'started_unix':time.time(),
    'gpu':a.gpu,'gpu_uuid':row[1].strip(),'source_sha':SOURCE,'resume':a.resume,
    'registration_sha256':sha(regroot/'REGISTRATION.json'),'authorization_sha256':sha(auth_path),
    'live_dispatch_policy_sha256':sha(policy_path),'launcher_sha256':sha(__file__),
    'output_reservation_bytes':2**30,'maximum_calls_later_score':1728,'score_started':False,
    'checkpoint_payload_verification':'frozen exporter identity/model readers before use',
    'launch_scope':'fixed formal full-dev single-trajectory export only',
    'route':route_pin,'canonical_host':canonical_host,'environment':actual_environment,
    'campaign_export_slot':slot,'maximum_campaign_export_slots':4,
    'shared_lock_mount':shared_mount,'cpu_capacity':cpu_capacity,
    'admission':{'path':str(invocation/'ADMISSION.json'),'sha256':sha(invocation/'ADMISSION.json')},
    'host_specific_export_latency':'NOT_MEASURED_BY_THIS_HELPER'}
atomic_json(invocation/'LAUNCH.json',record,immutable=True)
print(json.dumps(record))
