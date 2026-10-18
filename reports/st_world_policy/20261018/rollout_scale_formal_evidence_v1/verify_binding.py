"""Bounded evidence binding check; never manufactures approval or runs training."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import resource
import subprocess
import time

HERE = Path(__file__).resolve().parent
RUNTIME = Path('/mnt/project/VLA-Drive-st-rollout-scale-runtime-v2-20261018')
REG = Path('/mnt/project/st-world-policy-20261016/formal/consequence/registrations/C1_rollout_std_fulltrain_seed42_official_4096_v1')
OLD = REG.parent / 'C1_fulltrain_seed42_official_4096_v1'
SHA = '751fcfa2b41549ae16f7b3204aef8dd18b912016'


def diff(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        result = []
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                result.append({'path': path + '/' + key, 'old': a.get(key), 'new': b.get(key)})
            else:
                result.extend(diff(a[key], b[key], path + '/' + key))
        return result
    return [] if a == b else [{'path': path, 'old': a, 'new': b}]


def read(path):
    return json.loads(path.read_bytes())


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    # Process-local Git optimization only; source/status verification still executes.
    os.environ.update(GIT_CONFIG_COUNT='1', GIT_CONFIG_KEY_0='core.preloadIndex', GIT_CONFIG_VALUE_0='false')
    regbytes = (REG / 'REGISTRATION.json').read_bytes()
    assert hashlib.sha256(regbytes).hexdigest() == '928b9788404c4dcb9316b8457dc1a627010f892b85308f52aacf07b2cbb0783f'
    reg = json.loads(regbytes)
    source = RUNTIME / 'tools/st_world_policy/check_formal_approval.py'
    assert source.read_bytes() == subprocess.check_output(['git', '-C', str(RUNTIME), 'show', SHA + ':tools/st_world_policy/check_formal_approval.py'])
    spec = importlib.util.spec_from_file_location('fixed_approval_verifier', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    wall0, cpu0 = time.monotonic(), time.process_time()
    fingerprint = module.verify_registration(REG / 'REGISTRATION.json', repo=RUNTIME, effective_execution=reg['execution'])
    hash_bytes = sum(Path(value['path']).stat().st_size for value in reg['artifacts'].values())
    execution = {
        'function': 'tools.st_world_policy.check_formal_approval.verify_registration',
        'runtime_repo': str(RUNTIME),
        'runtime_head': subprocess.check_output(['git', '-C', str(RUNTIME), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'status': 'VERIFY_REGISTRATION_PASS',
        'successful_attempt_self_cpu_seconds': time.process_time() - cpu0,
        'successful_attempt_wall_seconds': time.monotonic() - wall0,
        'max_self_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'successful_attempt_artifact_stream_hash_bytes': hash_bytes,
        'prior_failed_attempts': 2,
        'prior_failed_attempt_artifact_stream_hash_bytes': 2 * hash_bytes,
        'cumulative_artifact_stream_hash_bytes': 3 * hash_bytes,
        'artifact_hash_scope': 'Two failed bounded-environment attempts plus one successful mature verifier call; TRAIN manifest streamed only for hashing, never JSON parsed. No model/image/cache/rollout payloads.',
        'CPU_limit_seconds_per_attempt': 30, 'wall_limit_seconds_per_attempt': 60,
        'address_space_limit_bytes': None,
        'memory_boundary': 'stdlib stream hash chunks8MiB; virtual-address cap removed because Git pack mmap exceeded256MiB. Actual self RSS recorded; no model loading.',
        'git_config_process_local': {'core.preloadIndex': 'false'},
        'GPU_used': False, 'training_dispatch': False,
        'formal_launch_approval_checked_or_issued': False,
    }
    oldreg = read(OLD / 'REGISTRATION.json')
    config_diff = diff(read(OLD / 'CONFIG.json'), read(REG / 'CONFIG.json'))
    target_diff = diff(read(OLD / 'TARGET_CONTRACT.json'), read(REG / 'TARGET_CONTRACT.json'))
    assert {row['path'] for row in config_diff} == {'/fpo/advantage', '/runtime/official_execution/adaptation_reason'}
    assert {row['path'] for row in target_diff} == {'/policy/fpo/advantage'}
    assert reg['execution'] == oldreg['execution']
    common = ['base_binding', 'data_manifest', 'environment', 'teacher_manifest']
    for key in common:
        assert reg['artifacts'][key]['sha256'] == oldreg['artifacts'][key]['sha256']
    assert '/root/consequence_next_hypothesis' not in reg['implementer_agents']
    lease = read(REG / 'RESOURCE_LEASE.json')
    assert lease['max_invocation_seconds'] == 28800 and 'at most six8h invocations' in lease['purpose']
    assert reg['execution']['max_updates'] == reg['execution']['stop_after'] == 4096
    assert fingerprint['registration_sha256'] == hashlib.sha256(regbytes).hexdigest()
    binding = {
        'schema': 'rollout_scale_formal_evidence_binding_diff_v1',
        'status': 'MATCHES_SEALED_SINGLE_CANDIDATE',
        'control_run': oldreg['run_id'], 'candidate_run': reg['run_id'],
        'config_diff': config_diff, 'target_diff': target_diff,
        'execution_exactly_equal': True,
        'common_artifact_hashes': {key: reg['artifacts'][key]['sha256'] for key in common},
        'normalization_source_commit': reg['source_sha'],
        'run_label_change': 'Draft C1_rollout_scale_seed42_official_4096_v1 -> actual C1_rollout_std_fulltrain_seed42_official_4096_v1; naming only, no new arm.',
        'resource_scope': '8h per invocation, at most6 invocations, one cumulative4096 endpoint with original scheduler/native continuation. Six-invocation bound is stated in purpose text; enforcement and current host/cost admission belong to independent efficiency/resource review.',
        'not_signed': ['correctness', 'efficiency', 'resource_lease_admission', 'RUN_APPROVAL', 'evaluation_launch'],
        'method_selection_from_profile_quality': False,
    }
    for name, value in [('FINGERPRINT.json', fingerprint), ('VERIFY_EXECUTION.json', execution), ('BINDING_DIFF.json', binding)]:
        (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(execution, ensure_ascii=False, indent=2))
    print(json.dumps(fingerprint, indent=2))


if __name__ == '__main__':
    main()
