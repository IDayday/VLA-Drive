"""Read-only arithmetic/identity audit of completed real4-v2 small receipts.

No torch imports, tensor payloads, image reads, process signals or reruns.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/project/st-world-policy-20261016/audits/loss_scale_calibration_real4_v2')
OLD = ROOT.with_name('loss_scale_calibration_real4_v1')
CONTROL = Path('/mnt/project/st-world-policy-20261016/resources/loss_scale_real4_v2/CONTROL.json')
RUNTIME = Path('/mnt/project/VLA-Drive-st-loss-scale-real4-v3-20261017')
EXPECTED_SOURCE = 'f212c96cbb74af9b3c59d91e80241675dc55fec0'
INPUTS, CHECKS = [], []
total_reads = 0
started = time.monotonic()


def io():
    return {k: int(v) for k, v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}


baseline = io()


def read(path, expected=None):
    global total_reads
    path = Path(path)
    assert path.suffix in ('.json', '.py')
    before = path.lstat()
    assert stat.S_ISREG(before.st_mode) and before.st_size < 4 << 20
    assert total_reads + before.st_size < 32 << 20
    raw = path.read_bytes()
    after = path.lstat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    digest = hashlib.sha256(raw).hexdigest()
    if expected:
        assert digest == expected, str(path)
    total_reads += len(raw)
    INPUTS.append({'path': str(path), 'bytes': len(raw), 'sha256': digest})
    return json.loads(raw) if path.suffix == '.json' else raw.decode()


def check(name, condition):
    assert condition, name
    CHECKS.append(name)
    print('PASS', name, flush=True)


def jhash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-14)


# Terminal artifacts are required; no polling and no creating a completed status early.
result = read(ROOT / 'RESULT.json')
measurement = read(ROOT / 'result/MEASUREMENT.json')
start = read(ROOT / 'START.json')
preflight = read(ROOT / 'result/RUNTIME_PREFLIGHT.json')
resolved = read(ROOT / 'result/RESOLVED_CONFIG.json')
control = read(CONTROL, '76e00b6fe4b5840333263554441b8ff9e108b99959fbd97ca8ac42bf95c4efb9')
admission = read(control['admission']['path'], '00fc941208da760770cdeed367e45378af15c79e4ed70fa43852f4378f0c53a0')
cfg = read(admission['config']['path'], admission['config']['sha256'])
training = read(cfg['training_config']['path'], cfg['training_config']['sha256'])
manifest = read(cfg['data_manifest']['path'], cfg['data_manifest']['sha256'])
teacher = read(cfg['teacher_manifest']['path'], cfg['teacher_manifest']['sha256'])
requests = read(cfg['teacher_requests']['path'], cfg['teacher_requests']['sha256'])
binding = read(cfg['source']['binding']['path'], cfg['source']['binding']['sha256'])
prior = read(OLD / 'RESULT.json')
prior_failure = read(OLD / 'result/FAILED.json')
for entry in control['sources']:
    read(entry['path'], entry['sha256'])
read(RUNTIME / 'tools/st_world_policy/calibrate_temporal_scale.py',
     'f2de950a36ac39f76b96e9fb22a5cc1c7fab8facda9d1e01d217dfa31fd6a1e4')
read(RUNTIME / 'tools/st_world_policy/calibration_measurement.py',
     '49d04f0dd2f00db1d08b16a9066f820699cadb8724bb8afd5b8d38009bf8fef0')

check('exact launch/config/source identity', start['sha256'] == INPUTS[5]['sha256']
      and start['control'] == str(CONTROL) and resolved == cfg
      and measurement['config_sha256'] == admission['config']['sha256']
      and measurement['source_sha'] == control['source_sha'] == admission['source_sha'] == EXPECTED_SOURCE
      and admission['config']['sha256'] == 'd5358c65585a4852bc959e5795a0f0ae647b9bdf6dd99c64220e361b38eed560')
check('exact pre-CUDA workspace receipt', preflight == measurement['runtime']['calibration_preflight']
      and preflight['runtime_environment'] == cfg['runtime_environment'] == control['runtime_environment'] == {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
      and preflight['cuda_initialized_at_check'] is False and preflight['deterministic_algorithms'] is True
      and training['runtime']['deterministic'] is True)
check('fixed real four scenes and all declared random seeds', measurement['batch'] == cfg['batch']
      and measurement['microbatches'] == 4
      and [manifest['scenes'][i]['token'] for i in cfg['batch']['scene_indices']] == cfg['batch']['scene_tokens'])
check('training data and existing teacher identity', manifest['split'] == requests['split'] == measurement['dataset']['split'] == 'train'
      and measurement['dataset']['data_manifest_sha256'] == cfg['data_manifest']['sha256']
      and measurement['dataset']['teacher_cache_identity'] == teacher['identity']
      and requests['identity'] == teacher['request_manifest_identity']
      and measurement['dataset']['need_future'] is True)
check('common S0 initialization binding is the admitted source', measurement['source']['kind'] == cfg['source']['kind'] == 'common_s0'
      and measurement['source']['binding'] == binding['binding_sha256']
      and measurement['source']['initialization_reference'] == cfg['source']['initialization_reference']
      and measurement['source']['initialization_seed'] == training['execution']['seed'])
check('diagnostic measurement only and zero optimizer/new teacher output', measurement['status'] == 'MEASURED'
      and measurement['output_purpose'] == cfg['purpose'] == 'DIAGNOSTIC'
      and measurement['optimizer_steps'] == result['optimizer_updates'] == 0
      and measurement['new_teacher_tensor_bytes'] == result['new_tensor_bytes'] == 0
      and measurement['teacher_instantiated'] is False and result['scientific_calibration_selected'] is False)

patch, pairs = measurement['patch_indices'], measurement['future_pair']
check('recorded common patch/time request hash', jhash({'patch': patch, 'pair': pairs}) == measurement['requests_sha256'])
check('valid fixed patch and adjacent future request layout',
      len(patch) == len(pairs) == 4
      and all(len(scene) == 3 and all(len(view) == len(set(view)) == 96 and all(type(i) is int and 0 <= i < 18 * 24 for i in view) for view in scene) for scene in patch)
      and all(len(pair) == 2 and 0 <= pair[0] <= 2 and pair[1] == pair[0] + 1 for pair in pairs))
pair_counts = {str(i): sum(pair[0] == i for pair in pairs) for i in range(3)}
check('actual future-pair coverage reported without inventing missing pair', pair_counts == measurement['pair_scene_counts'])
check('original full valid scalar-element denominators recorded', measurement['counts'] ==
      {'current': 4 * 3 * 96 * 384, 'future_a': 4 * 3 * 96 * 384,
       'future_b': 4 * 3 * 96 * 384, 'delta': 4 * 3 * 96 * 384, 'ego_scenes': 4})
losses, groups = measurement['losses'], measurement['gradient_groups']
tasks = {'ego_fm', 'current', 'future', 'delta'}
check('all measured losses finite and positive', set(losses) == tasks and all(math.isfinite(x) and x > 0 for x in losses.values()))
check('all declared gradient norms finite positive and no missing parameters',
      set(groups) == {'last_language_block', 'world_queries', 'declared_shared_union'}
      and all(set(group['norms']) == tasks and all(math.isfinite(n) and n > 0 for n in group['norms'].values())
              and all(not names for names in group['missing_gradient_parameters'].values())
              and all(math.isfinite(cos) and -1.000000000001 <= cos <= 1.000000000001 for cos in group['cosines'].values())
              for group in groups.values()))
check('union norm combines the two disjoint declared groups', all(close(groups['declared_shared_union']['norms'][task] ** 2,
      groups['last_language_block']['norms'][task] ** 2 + groups['world_queries']['norms'][task] ** 2) for task in tasks))
subset = measurement['parameter_subset']
params = [p for group in subset.values() for p in group]
check('actual shared subset names/shapes/dtypes are explicit',
      set(subset) == {'last_language_block', 'world_queries'} and len(subset['last_language_block']) == 11
      and len(subset['world_queries']) == 1 and subset['world_queries'][0]['name'] == 'base.foresight_queries'
      and all('language_model.layers.27.' in p['name'] for p in subset['last_language_block'])
      and len({p['name'] for p in params}) == len(params)
      and all(p['dtype'] == 'torch.bfloat16' and len(p['initial_sha256']) == 64 and all(type(n) is int and n > 0 for n in p['shape']) for p in params)
      and measurement['shared_subset_not_whole_LLM'] is True and measurement['full_shared_model_gradient_measured'] is False)

norms = groups['declared_shared_union']['norms']
ratios = {task: {'loss_ratio': losses['current'] / losses[task],
                 'gradient_ratio': norms['current'] / norms[task]} for task in ('future', 'delta')}
weights = {'current': 1., **{task: min(values.values()) for task, values in ratios.items()}}
proposal = measurement['proposal']
check('fixed proposal arithmetic independently recomputed', proposal['rule'] == 'current_anchored_min_loss_and_shared_gradient_ratio_v1'
      and proposal['status'] == 'PROPOSAL_ONLY'
      and all(close(proposal['ratios'][task][key], value) for task, item in ratios.items() for key, value in item.items())
      and all(close(proposal['weights'][task], value) for task, value in weights.items())
      and proposal['ego_fm_weight_unchanged'] == 1.)
check('proposal is not applied or claimed statistically optimal', proposal['applied_to_training'] is False
      and proposal['statistical_optimality_claim'] is False)
check('reviewed source completed its model/mode/grad assertions and RNG cleanup',
      measurement['model_state_before_after_equal'] is True and measurement['RNG_restored'] is True)

r = result['resources']
check('CLI measurement completed while wrapper lifecycle explicitly failed', r['returncode'] == 0
      and r['child_accounting']['cli_result']['exit_code'] == 0 and r['child_accounting']['error'] is None
      and result['status'] == 'FAILED_NO_AUTO_RETRY' and r['failure'] == 'OWNED_DESCENDANTS_AFTER_LEADER_EXIT')
check('cleanup complete is separate from lifecycle qualification', r['cleanup']['live_owned_remaining'] == 0
      and r['cleanup']['verified_processes'] == 8 and r['monitor_error'] is None)
check('second measured caps and parent child read accounting remain within admission',
      r['invocation_seconds'] <= control['limits']['wall_seconds'] + control['limits']['cleanup_seconds']
      and r['peak_aggregate_rss_bytes'] < control['limits']['rss_bytes']
      and result['charged_parent_child_rchar_bytes'] < control['limits']['read_bytes']
      and r['read_accounting']['status'] == 'COMPLETE'
      and r['read_accounting']['child_terminal_rchar_bytes'] == r['charged_owned_child_rchar_bytes'])
total_seconds = prior['resources']['invocation_seconds'] + r['invocation_seconds']
total_read_bytes = prior['charged_parent_child_rchar_bytes'] + result['charged_parent_child_rchar_bytes']
check('both actual attempts remain below original aggregate budget', total_seconds < 3640 and total_read_bytes < 256 << 30)
check('first failure not promoted to state or RNG pass', prior['status'] == 'FAILED_NO_AUTO_RETRY'
      and prior_failure['status'] == 'FAILED' and 'RNG_restored' not in prior_failure
      and 'model_state_before_after_equal' not in prior_failure and not (OLD / 'result/MEASUREMENT.json').exists())
check('no automatic retry allowed', start['automatic_retry'] is False and control['limits']['max_invocations'] == 1)

after = io()
audit_cost = {'elapsed_seconds': time.monotonic() - started,
    'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    'logical_read_delta': after['rchar'] - baseline['rchar'], 'explicit_read_bytes': total_reads,
    'cpu_affinity': sorted(os.sched_getaffinity(0)), 'GPU_calls': 0, 'tensor_payload_reads': 0}
assert audit_cost['elapsed_seconds'] < 60 and audit_cost['logical_read_delta'] < 32 << 20
assert audit_cost['cpu_affinity'] == [6, 7]
value = {'schema': 'st_real4_v2_independent_actual_review_v1', 'review_status': 'COMPLETE',
    'measurement_receipt_and_arithmetic': 'PASS_RECORDED_DIAGNOSTIC',
    'actual_attempt': result['status'], 'lifecycle_qualification': 'FAIL_OWNED_DESCENDANTS_AFTER_LEADER_EXIT',
    'reviewer': '/root/st_calibration_independent_correctness_v2', 'source_sha': EXPECTED_SOURCE,
    'root': str(ROOT), 'checks': CHECKS, 'input_artifacts': INPUTS,
    'fixed4_mathematically_calibratable_under_declared_rule': True,
    'losses': losses, 'gradient_groups': groups, 'proposal_weights': weights, 'ratios': ratios,
    'weighted_gradient_relative_to_current': {k: weights[k] * norms[k] / norms['current'] for k in weights},
    'future_pair_counts': pair_counts, 'unobserved_future_pair_indices': [i for i in range(3) if pair_counts[str(i)] == 0],
    'recorded_state_assertion': 'PARAMETER_BUFFER_MODE_BYTES_CHECK_COMPLETED_IN_REVIEWED_SOURCE',
    'recorded_RNG_status': 'RESTORATION_PATH_COMPLETED_NO_INDEPENDENT_SNAPSHOTS',
    'independent_parameter_or_RNG_tensor_comparison': False,
    'actual_cost': {'measurement_seconds': measurement['elapsed_seconds'],
        'invocation_seconds': r['invocation_seconds'], 'peak_RSS_bytes': r['peak_aggregate_rss_bytes'],
        'logical_read_bytes': result['charged_parent_child_rchar_bytes'],
        'cuda': r['child_accounting']['cli_result']['cuda'], 'cleanup': r['cleanup']},
    'combined_attempt_cost': {'invocation_seconds': total_seconds, 'logical_read_bytes': total_read_bytes},
    'audit_subprocess_cost': audit_cost,
    'proposal_applied': False, 'scientific_coefficient_selected': False, 'formal_permission': False,
    'retry_authorized': False, 'historical_T1_resource_FAIL_preserved': True,
    'limitations': ['Four fixed training scenes, one initialization, two adjacent future-pair types only; not population/horizon coverage',
        'Gradient subset is last language block and W queries, not all VLM parameters',
        'No optimizer step, ZeRO/DDP communication parity or resumed-training equivalence is tested',
        'Receipt values/source assertions verified; no independent tensor hash scan or GPU rerun',
        'State invariance is before/after this diagnostic model; not original S0-output/history-adapter equivalence',
        'RNG restored flag means reviewed restoration path completed; before/after RNG tensors are not in the artifact',
        'Lifecycle failure remains despite successful measurement and eventual cleanup; no third automatic attempt',
        'Single-sample-batch proposal is not an optimality, convergence, safety or formal-training approval']}
path = HERE / 'ACTUAL_REVIEW.json'
assert not path.exists()
path.write_text(json.dumps(value, indent=2) + '\n')
print(json.dumps({'checks': len(CHECKS), 'measurement': value['measurement_receipt_and_arithmetic'],
                  'lifecycle': value['lifecycle_qualification'], 'cost': audit_cost}))
