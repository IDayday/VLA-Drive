"""Independent T0 metadata resource review. Never dispatches either original CLI."""
from pathlib import Path
import ast
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
OWNER = Path('/mnt/project/VLA-Drive-st-temporal-profile-prepare-20261018')
PACKAGE = OWNER / 'reports/st_world_policy/20261018/temporal_profile_actual_preparation_v1'
RUNTIME = Path('/mnt/project/VLA-Drive-st-temporal-profile-runtime-v1-20261018')
ROOT = Path('/mnt/project/st-world-policy-20261016/profile/temporal_global32_qualification_v1/T0_capacity4/metadata')
PLAN_SHA = '9217f72273de0eeb80c2d872e061f69944a404fe1ff066a8f96e62ee0d79fbcc'
ENTRY_SHA = '81b2caf5c70d21c3f10a6cfceb212a6766ba4c65d6a648a79711b110680e665f'
LAUNCH_SHA = 'e5b83e2400a247841b4d41668a61fc085f52704fe6845b8b20dd2a29e3a4e573'
COMMIT = '38f86a15a92171e583bb5019cce89a63fb7a1d43'
evidence = []


def raw(path, expected=None):
    path = Path(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as f:
        before = os.fstat(f.fileno())
        assert stat.S_ISREG(before.st_mode) and before.st_size < 4 << 20
        value = f.read((4 << 20) + 1)
        after = os.fstat(f.fileno())
    version = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    assert version(before) == version(after) == version(path.stat())
    digest = hashlib.sha256(value).hexdigest()
    assert expected is None or digest == expected, str(path)
    pin = {'path': str(path), 'sha256': digest}
    if pin not in evidence:
        evidence.append(pin)
    return value


def pin(path):
    return {'path': str(path), 'sha256': hashlib.sha256(raw(path)).hexdigest()}


def document(ref):
    return json.loads(raw(ref['path'], ref['sha256']))


def git(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, text=True, timeout=15).strip()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def cpu_stats():
    return {line.split()[0]: list(map(int, line.split()[1:9]))
            for line in Path('/proc/stat').read_text().splitlines()
            if line.split() and line.split()[0] in ('cpu12', 'cpu13')}


def main():
    assert not (HERE / 'ACTUAL_ADMISSION.json').exists(), 'Never overwrite an admission attempt'
    plan = document({'path': str(PACKAGE / 'CPU_PLAN.json'), 'sha256': PLAN_SHA})
    raw(PACKAGE / 'prepare_once.py', ENTRY_SHA)
    launcher = raw(PACKAGE / 'launch_once.sh', LAUNCH_SHA).decode()
    a = document(pin(PACKAGE / 'CPU_ADMISSION_TEMPLATE.json'))
    owner_commit = git(OWNER, 'rev-parse', 'HEAD')
    for filename in ('CPU_PLAN.json', 'prepare_once.py', 'launch_once.sh', 'CPU_ADMISSION_TEMPLATE.json'):
        relative = str((PACKAGE / filename).relative_to(OWNER))
        committed = subprocess.check_output(['git', 'show', owner_commit + ':' + relative], cwd=OWNER, timeout=15)
        assert committed == raw(PACKAGE / filename), 'Owner packet must be committed: ' + filename
    assert git(RUNTIME, 'rev-parse', 'HEAD') == COMMIT
    assert not git(RUNTIME, 'status', '--porcelain=v1')
    assert socket.gethostname() == 'training-vla-zt-worker-0'
    assert {12, 13}.issubset(os.sched_getaffinity(0))
    assert ROOT.parent.is_dir() and not ROOT.exists() and not ROOT.is_symlink()
    for parent in ROOT.parents:
        assert parent.is_dir() and not parent.is_symlink()
    assert "CUDA_VISIBLE_DEVICES=''" in launcher and 'taskset -c 12,13' in launcher
    assert 'timeout --signal=TERM --kill-after=40s 1500s' in launcher
    assert plan['limits'] == dict(wall_seconds=1500, cleanup_seconds=40, read_bytes=64 << 30, rss_bytes=4 << 30)
    assert plan['max_output_bytes'] == 8 << 20
    assert plan['cpu_affinity'] == [12, 13]
    assert plan['gpu'] is False and plan['training_started'] is False
    assert plan['checkpoint_reservation_created'] is False and plan['T4_automatic'] is False
    reserved = plan['control_reservation']
    prior = document(reserved['prior_evidence'])
    assert prior['conservative_metadata_MiB']['total'] == 10098
    assert reserved['conservative_prior_bytes'] == 10098 << 20
    assert reserved['total_bytes'] == reserved['control_reservation_bytes'] == 8 << 20
    assert reserved['proposed_total_reserved_bytes'] == 10106 << 20
    assert reserved['metadata_envelope_bytes'] == 10240 << 20
    assert reserved['remaining_bytes'] == 134 << 20
    assert reserved['typed_claims_changed'] is False and reserved['ledger_written'] is False
    assert reserved == a['control_reservation']
    policy_path = Path('/mnt/project/server_dispatch_policy.json')
    policy = document(pin(policy_path))
    assert 'training-vla-zt' in policy['allowed_hosts']
    assert pin(policy_path)['sha256'] == '29ce8b89aec0e20d0b429d6a026634ec59e6ddee5933106d2ce9e378fa322d55'
    auth = document(pin('/mnt/project/st-world-policy-20261016/resources/AUTHORIZATION_20261016T203442.json'))
    assert 'training-vla-zt' in auth['allowed_hosts'] and auth['formal_reviews_required']

    wrapper = module(PACKAGE / 'prepare_once.py', 'reviewed_T0_adapter')
    controller = wrapper.controller()
    ast.parse(wrapper.bound_source(wrapper.donor()))
    commands = wrapper.commands(plan)
    assert len(commands) == 2
    assert commands[0][0] == 'tools.st_world_policy.qualify_indexed_provider'
    assert commands[0][1][0] == 'prepare-profile-migration'
    assert commands[1][0] == 'tools.st_world_policy.temporal_formal'
    second = dict(zip(commands[1][1][::2], commands[1][1][1::2]))
    assert second['--cell'] == 'T0' and second['--scope'] == 'PROFILE'
    assert second['--world-size'] == '4' and second['--micro-batch'] == '1'
    assert second['--profile-mode'] == 'none' and '--history-complete' not in second
    assert second['--max-seconds'] == '3600'  # Future GPU profile ceiling, not this CPU monitor.
    assert 'train_world' not in repr(commands) and 'profile_pipeline' not in repr(commands)
    # Inherited original supervisor/bound reader are imported, never executed.
    common = module(controller.REVIEWED_HOME.parent / 'w_storage_bridge_v1/storage_bridge.py', 'reviewed_reads')
    reads = common.Reads()
    now = time.time()
    a.update(status='APPROVED_T0_GLOBAL32_METADATA_PREPARATION_ONLY', reviewer='/root/spatial20k_efficiency',
             hostname=socket.gethostname(), issued_at_unix=now, latest_start_unix=now + 900,
             valid_until_unix=now + 2700)
    controller.validate(a, plan, SimpleNamespace(), reads, now)
    rejected = []
    for field, bad in [('gpu', True), ('cpu_affinity', [10, 11]), ('max_output_bytes', 16 << 20),
                       ('automatic_retry', True), ('checkpoint_reservation_created', True),
                       ('reviewer', '/root/temporal_profile_prepare'), ('output_root', str(ROOT.parent))]:
        changed = copy.deepcopy(a); changed[field] = bad
        try:
            controller.validate(changed, plan, SimpleNamespace(), reads, now)
        except ValueError:
            rejected.append(field)
        else:
            raise AssertionError('Invalid admission accepted: ' + field)
    for key in ('read_bytes', 'rss_bytes', 'wall_seconds', 'cleanup_seconds'):
        changed = copy.deepcopy(a); changed['limits'][key] += 1
        try:
            controller.validate(changed, plan, SimpleNamespace(), reads, now)
        except ValueError:
            rejected.append('limits.' + key)
        else:
            raise AssertionError('Changed budget accepted: ' + key)

    # Read-only native storage formula with already admitted TRAIN typed claim.
    cp_path = Path('/mnt/project/VLA-Drive-st-flow-resume-efficiency-inheritance-20261018/reports/st_world_policy/20261018/checkpoint_remaining_reconciliation_v1/CHECKPOINT_COMMITMENTS_REMAINING_V2.json')
    cp = document({'path': str(cp_path), 'sha256': '7b88fcbf0a4b4505aa251fdbf2e146c6b94dd3fcb15cef8269cee4d1989fa60c'})
    assert cp['remaining_bytes'] == 3701166716312
    ledger_path = Path('/mnt/project/st-world-policy-20261016/resources/cache_storage_ledger_v1/CLAIMS.json')
    ledger = document(pin(ledger_path))
    sys.path.insert(0, str(RUNTIME))
    from tools.st_world_policy.cache_storage_budget import StorageBudget
    b = StorageBudget.__new__(StorageBudget); b.core = ledger['core']; b.state_dir = ledger_path.parent
    b._check_disk(ledger, ledger_path.parent, free_reserve_bytes=(1 << 40) + cp['remaining_bytes'])
    free = shutil.disk_usage(ledger_path.parent).free
    required = cp['remaining_bytes'] + (1024 + 768 + 10) * (1 << 30)
    assert free >= required
    helper = Path('/mnt/project/VLA-Drive-st-future-finalize-runtime-v1-20261017/reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1')
    sys.path.insert(0, str(helper))
    import supervise_once
    supervise_once.shared_mount()
    lock = Path(supervise_once.LOCK)
    with os.fdopen(os.open(lock, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), 'rb') as f:
        assert stat.S_ISREG(os.fstat(f.fileno()).st_mode)
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(f, fcntl.LOCK_UN)
    c0 = cpu_stats(); time.sleep(3); c1 = cpu_stats()
    cpu_load = {}
    for cpu in c0:
        deltas = [b-a for a,b in zip(c0[cpu],c1[cpu])]
        cpu_load[cpu] = 1 - (deltas[3] + deltas[4]) / max(sum(deltas), 1)
    old_path = Path('/mnt/project/st-world-policy-20261016/audits/full_population_provider_preparation_v1/controls/RESULT.json')
    old = document(pin(old_path))
    assert old['status'] == 'PREPARATION_COMPLETE' and old['resources']['cleanup']['live_owned_remaining'] == 0
    out = HERE / 'ACTUAL_ADMISSION.json'
    out.write_text(json.dumps(a, indent=2) + '\n')
    command = dict(schema='st_reviewed_single_command_v1', cwd=str(RUNTIME),
        argv=['bash', str(PACKAGE/'launch_once.sh'), '--admission', str(out), '--admission-sha256', pin(out)['sha256']],
        run_by_reviewer=False, automatic_retry=False, resource_scope='CPU_METADATA_ONLY')
    (HERE/'COMMAND.json').write_text(json.dumps(command, indent=2)+'\n')
    review = dict(schema='st_T0_metadata_independent_efficiency_review_v1', status='PASS_CPU_METADATA_ONLY',
        reviewer='/root/spatial20k_efficiency', issued_at_unix=now, owner_commit=owner_commit,
        runtime_commit=COMMIT, admission=pin(out), command=pin(HERE/'COMMAND.json'),
        pure_admission_negative_cases=rejected, original_cli_commands=commands,
        original_full_population_observation=dict(result=pin(old_path),
            logical_read_bytes=old['charged_rchar_bytes'], wall_seconds=old['resources']['invocation_seconds'],
            peak_aggregate_rss_bytes=old['resources']['peak_aggregate_rss_bytes']),
        resource_check=dict(hostname=socket.gethostname(), cpu_affinity=[12,13], shared_not_exclusive=True,
            cpu_three_second_busy_fraction=cpu_load, machine_load=os.getloadavg(),
            metadata_total_MiB=10106, metadata_limit_MiB=10240, metadata_remaining_MiB=134,
            disk_free_bytes=free, required_free_bytes=required, disk_headroom_bytes=free-required,
            native_disk_check='PASS_WITH_EXISTING_TYPED_CLAIMS', full_cache_768GiB_retained=True,
            future_checkpoint_remaining_bytes=cp['remaining_bytes'], shared_metadata_lock='AVAILABLE_AT_REVIEW'),
        actions=dict(dispatch=False, GPU_context=False, large_payload_read=False, source_mutation=False,
            ledger_mutation=False, checkpoint_allocation=False, model_tests_repeated=False),
        boundaries=['Only one CPU metadata preparation; no automatic retry or T4 dispatch.',
            'Full population source migration and T0 descriptor only; native observer enforces read/RSS/wall/8MiB limits.',
            'T0 output PROFILE4 configuration is not GPU capacity permission or formal-training approval.',
            'No history build dependency for T0; no teacher tensor/checkpoint generation in this invocation.',
            'Native full legacy-cache reserve remains conservative; occupied checkpoint credit is inherited from independently audited evidence.',
            'CPU 12/13 are shared, not exclusive; history 10/11 remains untouched.',
            'Future capacity/timing and T1/T3/T4 need their own actual resource and source approvals.'],
        evidence_artifacts=evidence)
    (HERE/'REVIEW.json').write_text(json.dumps(review,indent=2)+'\n')
    print(json.dumps({'admission':pin(out),'command':pin(HERE/'COMMAND.json'),'review':pin(HERE/'REVIEW.json')},indent=2))


if __name__ == '__main__':
    main()
