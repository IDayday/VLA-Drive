"""Read only tiny immutable receipts; no torch, model, image, GPU or dispatch."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import time

ROOT = Path('/mnt/project/st-world-policy-20261016/audits/loss_scale_calibration_real4_v1')
CONTROL = Path('/mnt/project/st-world-policy-20261016/resources/loss_scale_real4_v1/CONTROL.json')
HERE = Path(__file__).resolve().parent
inputs = []
checks = []
started = time.monotonic()


def io():
    return {k: int(v) for k, v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}


before = io()


def read(path, decode=True):
    path = Path(path)
    st = path.lstat()
    assert stat.S_ISREG(st.st_mode) and st.st_size <= 1024 * 1024, str(path)
    value = path.read_bytes()
    assert len(value) == st.st_size
    digest = hashlib.sha256(value).hexdigest()
    inputs.append({'path': str(path), 'bytes': len(value), 'sha256': digest})
    return (json.loads(value) if decode else value.decode()), digest


def check(name, condition):
    assert condition, name
    checks.append(name)
    print('PASS', name, flush=True)


start, _ = read(ROOT / 'START.json')
result, _ = read(ROOT / 'RESULT.json')
failed, _ = read(ROOT / 'result/FAILED.json')
resolved, resolved_hash = read(ROOT / 'result/RESOLVED_CONFIG.json')
control, control_hash = read(CONTROL)
admission, admission_hash = read(control['admission']['path'])
review, review_hash = read(control['resource_review']['path'])
config, config_hash = read(admission['config']['path'])

check('actual control and admission checksums match recorded identities',
      start['control'] == str(CONTROL) and start['sha256'] == control_hash
      and admission_hash == control['admission']['sha256']
      and review_hash == control['resource_review']['sha256']
      and config_hash == admission['config']['sha256'])
check('resolved actual config equals admitted fixed four-scene config', resolved == config
      and len(resolved['batch']['scene_indices']) == len(resolved['batch']['scene_tokens']) == 4
      and resolved['batch']['micro_batch'] == 1)
check('failed attempt has no complete measurement artifact', result['status'] == 'FAILED_NO_AUTO_RETRY'
      and failed['status'] == 'FAILED' and not (ROOT / 'result/MEASUREMENT.json').exists())
check('failure records CuBLAS determinism requirement', 'CUBLAS_WORKSPACE_CONFIG' in failed['error']
      and 'Deterministic behavior was enabled' in failed['error'])
check('no automatic retry or scientific calibration selection', start['automatic_retry'] is False
      and control['limits']['max_invocations'] == 1 and result['scientific_calibration_selected'] is False)
check('zero optimizer updates and zero newly persisted tensor output are recorded',
      failed['optimizer_steps'] == result['optimizer_updates'] == result['new_tensor_bytes'] == 0)
check('failure does not claim state/RNG after-equality',
      not any(k in failed for k in ('model_state_before_after_equal', 'RNG_restored', 'gradient_groups', 'proposal')))

r = result['resources']
check('workload failure is separate from resource monitor success',
      r['returncode'] == 2 and r['failure'] is None and r['monitor_error'] is None
      and r['child_accounting']['cli_result']['exit_code'] == 2)
check('owned descendants recorded drained', r['cleanup']['live_owned_remaining'] == 0)
check('actual recorded time/RSS/read/output stay within admitted numeric caps',
      r['invocation_seconds'] <= control['limits']['wall_seconds'] + control['limits']['cleanup_seconds']
      and r['peak_aggregate_rss_bytes'] <= control['limits']['rss_bytes']
      and result['charged_parent_child_rchar_bytes'] <= control['limits']['read_bytes']
      and sum(x['bytes'] for x in inputs if x['path'].startswith(str(ROOT))) <= control['limits']['max_disk_bytes'])
check('read accounting has terminal child and parent reconciliation',
      r['read_accounting']['status'] == 'COMPLETE'
      and r['read_accounting']['child_terminal_rchar_bytes'] == r['charged_owned_child_rchar_bytes']
      and result['charged_parent_child_rchar_bytes'] >= r['read_accounting']['charged_parent_child_rchar_bytes'])

sources = []
for record in control['sources']:
    text, digest = read(record['path'], decode=False)
    check('actual pinned controller dependency ' + Path(record['path']).name, digest == record['sha256'])
    sources.append({'path': record['path'], 'sha256': digest})
    if Path(record['path']).name == 'run_once.py':
        controller_text = text
check('actual controller did not pin CuBLAS workspace before child launch',
      'CUBLAS_WORKSPACE_CONFIG' not in controller_text)

runtime = Path(control['runtime'])
cli, cli_hash = read(runtime / 'tools/st_world_policy/calibrate_temporal_scale.py', decode=False)
measurement, measurement_hash = read(runtime / 'tools/st_world_policy/calibration_measurement.py', decode=False)
check('actual CLI enables deterministic algorithms and lacks workspace preflight',
      'torch.use_deterministic_algorithms(training["runtime"]["deterministic"])' in cli
      and 'CUBLAS_WORKSPACE_CONFIG' not in cli)
check('source retains failure cleanup assertions but they are not receipt evidence',
      'with preserved_rng(generators):' in cli and 'finally:' in measurement
      and 'Calibration changed model parameters/buffers/modes' in measurement
      and 'Calibration populated model .grad' in measurement)

after = io()
output = {
    'schema': 'st_temporal_scale_real4_actual_failure_review_v1',
    'status': 'PASS_FAILURE_CLASSIFICATION_ONLY',
    'reviewer': '/root/st_calibration_independent_correctness_v2',
    'review_base_commit': '112ef4c40c0b0f2e82df367b2a4a64e9fdcf3d30',
    'actual_runtime_commit': control['source_sha'],
    'actual_root': str(ROOT),
    'actual_measurement_status': 'FAILED_NO_AUTO_RETRY',
    'actual_error': failed['error'],
    'post_failure_model_and_RNG_equality': 'NOT_RECORDED',
    'actual_optimizer_updates': 0,
    'actual_new_persistent_tensor_bytes': 0,
    'actual_loss_or_shared_gradient_result': 'ABSENT',
    'actual_cost': {
        'measurement_elapsed_seconds': failed['elapsed_seconds'],
        'invocation_seconds': r['invocation_seconds'],
        'supervised_child_seconds': r['supervised_child_seconds'],
        'peak_aggregate_rss_bytes': r['peak_aggregate_rss_bytes'],
        'charged_parent_child_rchar_bytes': result['charged_parent_child_rchar_bytes'],
        'cuda_peak': r['child_accounting']['cli_result']['cuda'],
        'live_owned_remaining': r['cleanup']['live_owned_remaining'],
    },
    'source_hashes': {'cli': cli_hash, 'measurement': measurement_hash},
    'checks': checks,
    'input_artifacts': inputs,
    'audit_subprocess_cost': {
        'elapsed_seconds': time.monotonic() - started,
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'rchar_delta': after['rchar'] - before['rchar'],
        'physical_read_delta': after['read_bytes'] - before['read_bytes'],
        'declared_input_bytes': sum(x['bytes'] for x in inputs),
        'cpu_affinity': sorted(os.sched_getaffinity(0)),
        'GPU_calls': 0,
        'tensor_payload_reads': 0,
    },
    'retry': {'automatic': False, 'authorized_by_this_review': False,
              'required': ['minimal fix independent source review', 'fresh independent resource admission',
                           'new immutable attempt/output identity', 'Lead explicit one-shot launch decision']},
    'historical_T1_resource_result': 'FAIL_LIVE_DESCENDANT_SCOPE_PRESERVED',
    'limitations': [
        'This verifies small recorded artifacts, not checkpoint or teacher tensors.',
        'No new CUDA execution or determinism equivalence has been tested.',
        'No complete loss/gradient measurement or coefficient proposal exists for this failed attempt.',
        'No post-failure model hash/RNG equality receipt was produced; source finally code is not measured PASS.',
        'The RuntimeError identifies CuBLAS deterministic workspace setup, not an exact failing operator.',
        'Resource figures are the controller recorded observations; sampled enforcement is not a kernel hard limit.',
        'Zero new persistent tensor bytes does not mean zero model/teacher input reads or zero CUDA work.',
        'A corrected run must remain a separate bounded attempt and retain this failed attempt and its costs.',
    ],
}
assert output['audit_subprocess_cost']['elapsed_seconds'] < 120
assert output['audit_subprocess_cost']['peak_rss_bytes'] < 1 << 30
assert output['audit_subprocess_cost']['rchar_delta'] < 1 << 30
assert output['audit_subprocess_cost']['cpu_affinity'] == [6, 7]
assert not (HERE / 'FAILURE_REVIEW.json').exists()
(HERE / 'FAILURE_REVIEW.json').write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps({'status': output['status'], 'checks': len(checks), 'cost': output['audit_subprocess_cost']}))
