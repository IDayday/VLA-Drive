"""Twelve prospective full-gradient observations; CPU planning by default.

Reuses cold68 process/forward verification and a separately frozen746 producer.
No optimizer updates, retry, process termination or automatic acceptance.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
COLD = Path('/mnt/project/VLA-Drive-structured-world-fgtr-perf-validation-v2-frozen-20261011')
PRODUCER = Path('/mnt/project/VLA-Drive-structured-world-fgtr-perf-fullgrad-v1-frozen-20261011')
PRODUCER_GIT = '74651848e5d9a0f3522f7abdd66c9557b83d1b2b'
REL = 'tools/structured_world/validate_p0_fullgrad_numerics.py'
PRODUCER_SHA = '75f51d573110c775229ca9363daed4a5c7317ed8f24a9ccb3a60f743220b5f55'
BINDING_SHA = '8691e3390cce77fcd55a38c90336fff732180b4a9e967b1d09743b9f72af0598'
QUALIFICATION_SHA = '09a40febff8a3876040b38159038981e5ba0ab7fe62f5971eb537ecc79a0115e'
sys.path.insert(0, str(COLD))
from tools.structured_world import run_p0_forward_probes as forward
from tools.structured_world import run_native_execution_continuation as native
bench = forward.bench
require, load, ref, write = forward.require, forward.load, forward.ref, forward.private_json
spec = importlib.util.spec_from_file_location('_p0_fullgrad_producer', PRODUCER/REL)
full = importlib.util.module_from_spec(spec)
require(bench.digest(PRODUCER/REL) == PRODUCER_SHA, 'Pinned full-gradient producer changed')
spec.loader.exec_module(full)


def checked_binding(new_ref, original_ref):
    """Only the actual prospective869 registration is accepted."""
    path = Path(new_ref['path']).resolve()
    actual = ref(path)
    expected = new_ref.get('sha256')
    require(expected == actual['sha256'] == BINDING_SHA, 'Full-gradient binding changed or unregistered')
    bench.immutable_source(PRODUCER, PRODUCER_GIT)
    b = load(path)
    require(b['producer']['root'] == str(PRODUCER) and b['producer']['git_sha'] == PRODUCER_GIT
            and b['producer']['files_sha256'] == {REL: PRODUCER_SHA}, 'Wrong full-gradient producer')
    require(b['forward_source_binding'] == original_ref, 'Original44 forward context changed')
    original = load(original_ref['path'])
    require(ref(original_ref['path']) == original_ref and original_ref['sha256'] == forward.BINDING_SHA, 'Original44 changed')
    require(b['schema'] == 'execution_p0_fullgrad_source_binding_v1' and b['state'] == 'frozen_before_probe'
            and b['contract'] == json.loads(json.dumps(full.CONTRACT)) and b['preset'] == forward.PRESET
            and b['roles'] == original['roles'] and b['registration_sha256'] == forward.REGISTRATION_SHA
            and b['candidate_mode'] == 'engineering_fast' and b['model'] == original['model'], 'Full-gradient scientific contract differs')
    frozen = datetime.fromisoformat(b['frozen_UTC'].replace('Z', '+00:00'))
    require(frozen.tzinfo is not None and frozen <= datetime.now(timezone.utc), 'Binding must precede probe')
    p0 = full.upstream(Path(b['forward_producer']['root']))
    bench.immutable_source(b['model']['root'], forward.MODEL_GIT)
    require(p0.inventory(b['model']['root']) == b['model']['files_sha256'], 'Frozen model file inventory changed')
    native_ref = original['original_acceptance_bindings']
    require(ref(native_ref['path']) == native_ref and b['original_acceptance_bindings'] == native_ref
            and b['cases'] == load(native_ref['path'])['cases'] and b['comparator_sha256'] == full.COMPARATOR_SHA
            and b['legacy_sha256'] == forward.LEGACY_SHA, 'Original native cases/comparator changed')
    return b, actual  # No torch import; actual producer preflight independently rechecks its full contract.


def sources(manifest):
    original = forward.sources(manifest)
    b, new_ref = checked_binding(manifest['fullgradient_source_binding'], ref(manifest['source_binding']))
    require(b['model'] == original['model'], 'be5 scientific model context changed')
    return b, new_ref


def phase_budget(manifest, jobs):
    b = native.validate_gpu_budget({'budget': manifest['native_budget']}, jobs)
    require(b['phase_limit'] == b['total_internal_limit'] == 30 and b['user_GPU_hours_limit'] is None
            and b['already_spent_external'] == b['reserved_elsewhere'] == 0
            and b['per_job_reserves'] == {'fullgradient': 2.5}
            and b['accounting_scope'] == 'fresh_fullgradient_phase_only'
            and b['old_zt3_cost_state'] == 'UNKNOWN_NOT_ZERO', 'Explicit30GPUh phase-only registration required')
    return b


def build_plan(manifest, output):
    require(manifest['schema'] == 'p0_fullgradient_coordinator_manifest_v1', 'Full-gradient manifest required')
    b, new_ref = sources(manifest)
    old_plan = forward.build_plan({**manifest, 'schema': 'p0_forward_coordinator_manifest_v1'}, output)
    jobs = []
    for old in old_plan['jobs']:
        job = {**old, 'stage': 'fullgradient', 'trainer': str(PRODUCER/REL), 'fullgradient_source_binding': new_ref}
        args = ['probe', '--model-source-root', b['model']['root'], '--forward-source-root', b['forward_producer']['root'],
            '--source-binding', new_ref['path'], '--source-binding-sha256', new_ref['sha256'],
            '--run', job['origin']['run'], '--tag', job['origin']['tag'], '--cache', manifest['cache'],
            '--dino-root', manifest['dino_root'], '--output', job['observed'], '--microbatch', '4']
        job['command'] = [manifest['python'], '-B', '-u', '-m', 'torch.distributed.run', '--standalone',
            '--nnodes=1', '--nproc_per_node=8', '--max_restarts=0', '--', str(PRODUCER/REL)]+args
        job['CPU_preflight_command'] = [manifest['python'], '-B', str(PRODUCER/REL)]+args+['--preflight']
        jobs.append(job)
    prefix = observed_prefix(manifest, jobs, output) if 'observed_prefix' in manifest else None
    if prefix: jobs[0] = prefix['job']
    budget = phase_budget(manifest, jobs)
    return {'schema': 'p0_fullgradient_coordinator_plan_v1', 'jobs': jobs, 'forward_slots': 12,
        'observed_prefix': prefix, 'new_GPU_jobs': 11 if prefix else 12,
        'fullgradient_source_binding': new_ref, 'original_forward_source_binding': ref(manifest['source_binding']),
        'producer_git_SHA': PRODUCER_GIT, 'producer_sha256': PRODUCER_SHA, 'native_budget': budget,
        'driver_sha256': bench.digest(__file__), 'GPU_jobs_started': 0, 'passed': False, 'formal_authorized': False,
        'limits': ['New direct full-parameter AA_B capture; old forward44/native819/routingd8 are not a new acceptance envelope',
            'Old strict failure and forward report remain unchanged; native500/E1/dev/recovery and new E2 binding require separate review']}


def classify_forward_context(job, b, binding):
    """Validate the real 746->6d16 sidecar in its actual869 context.

    cold68's old44 classifier cannot consume this delegated emission. Keep the
    bytes unchanged; reuse its exact-stat checker and the same coverage rules.
    """
    folder = Path(job['observed'])
    path, strict_path = folder/'P0_FORWARD_COMPARISON.json', folder/'strict/COMPLETE.json'
    process_path = Path(job['run'])/'PROCESS_EXIT.json'
    report, strict, process = load(path), load(strict_path), load(process_path)
    proto_path = Path(job['folder'])/'PROBE.json'; proto = load(proto_path)
    require(proto['binding'] == job['source_binding'] and proto['fullgradient_binding'] == binding
            and proto['job']['command'] == job['command'], 'Prospective original44/new869 contexts or argv differ')
    argv = job['command']; require('--' in argv and argv[argv.index('--')+1] == str(PRODUCER/REL)
            and '--nproc_per_node=8' in argv and '--max_restarts=0' in argv, 'Actual fullgradient entrypoint/full8/no-restart argv required')
    for flag, value in {'--source-binding': binding['path'], '--source-binding-sha256': binding['sha256'],
            '--model-source-root': b['model']['root'], '--forward-source-root': b['forward_producer']['root'],
            '--run': job['origin']['run'], '--tag': job['origin']['tag'], '--output': job['observed'], '--microbatch': '4'}.items():
        require(flag in argv and argv[argv.index(flag)+1] == value, 'Actual fullgradient argv/context differs: '+flag)
    require(process['returncode'] in (0, 1) and process['monitor_error'] is None, 'Unexpected worker/monitor failure')
    require(type(strict['passed']) is bool and (process['returncode'] == 0) == strict['passed'], 'Legacy exit/strict mismatch')
    require(report['schema'] == 'execution_p0_exact_forward_stats_v1' and report['preset'] == forward.PRESET
            and report['producer_sha256'] == forward.PRODUCER_SHA
            and report['registration_sha256'] == forward.REGISTRATION_SHA, 'Wrong actual delegated forward producer/preset')
    require(report['producer_source_binding'] == binding, 'Actual delegated forward binding must be869; do not relabel old44')
    require(report['model_source'] == {k: b['model'][k] for k in ('root', 'git_sha')}
            and report['original_acceptance_bindings'] == b['original_acceptance_bindings'], 'Delegated forward model/native context differs')
    require(report['checkpoint'] == {'run': str(Path(job['origin']['run']).resolve()), 'tag': job['origin']['tag']}, 'Wrong actual forward origin')
    original = report['original_strict']
    require(original == {**ref(strict_path), 'passed': strict['passed'], 'tool_sha256': forward.LEGACY_SHA,
            'unchanged_original_json': True}, 'Original strict JSON/hash changed')
    require(strict['training_identity'] == b['cases'][job['case']]['origin']['identity']
            and strict['checkpoint'] == job['origin']['tag'] and strict['optimized_mode'] == 'engineering_fast'
            and strict['attention_diagnostic'] == strict['precision_diagnostic'] == 'native', 'Strict origin/operator differs')
    require(report['ranks'] == strict['ranks'] == 8 and report['scene_count'] == strict['scene_count'] == 32
            and report['full_tensor_rank_coverage'] is True and report['advisory_forward_passed'] is True, 'Incomplete actual full8 forward')
    records = report['per_rank']; require(len(records) == 8 and {r['rank'] for r in records} == set(range(8)), 'Missing/duplicate forward ranks')
    require(len(strict['per_rank']) == 8 and {r['rank'] for r in strict['per_rank']} == set(range(8)), 'Missing strict ranks')
    tokens = [t for r in records for t in r['tokens']]
    require(len(tokens) == len(set(tokens)) == 32 and all(len(r['tokens']) == 4 for r in records), 'Missing global32 forward scene coverage')
    require({r['rank']: r['tokens'] for r in records} == {r['rank']: r['tokens'] for r in strict['per_rank']}, 'Strict/forward scene order differs')
    names = set(report['logical_forward_module_inventory'])
    require(forward.CORE <= names and any(n.startswith('task_loss/') for n in names)
            and any(n.startswith('raw_loss/') for n in names), 'Missing full total/task/raw loss modules')
    expected = names | {f'rank{r}/{n}' for r in range(8) for n in names}
    require(set(report['modules']) == set(report['forward_module_inventory']) == expected, 'Full module/rank coverage missing')
    for checks in report['modules'].values():
        require(set(checks) == forward.PAIRS, 'Three actual forward pairs required')
        for row in checks.values(): forward.stat_exact(row)
    exact = forward.CORE | {'RNG', 'BN', 'BN_finite', 'BN_float', 'BN_counter', 'query_and_counts',
                           'all_buffers', 'gradient_inventory', 'buffer_inventory'}
    require(all(report['forward_exact'].get(n) is True for n in exact), 'Actual forward exact flags failed/missing')
    require(set(report['forward_exact_by_pair']) == forward.PAIRS
            and all(v is True for checks in report['forward_exact_by_pair'].values() for v in checks.values()), 'Forward role-pair exact flags failed')
    frozen = datetime.fromisoformat(b['frozen_UTC'].replace('Z', '+00:00'))
    prepared = datetime.fromisoformat(proto['created_UTC'].replace('Z', '+00:00'))
    for r in records:
        require(set(r['roles']) == {'A', 'AA', 'B'} and set(r['comparisons']) == forward.PAIRS, 'Incomplete role capture')
        for checks in r['comparisons'].values():
            require(set(checks['modules']) == names and all(v is True for v in checks['exact'].values()), 'Per-rank forward coverage/exact flags missing')
            for row in checks['modules'].values(): forward.stat_exact(row)
        started = datetime.fromisoformat(r['roles']['A']['started_UTC'].replace('Z', '+00:00'))
        require(started.tzinfo is not None and frozen <= prepared <= started, 'New binding/PROBE must precede first A')
    physical = report['physical_q_final']; require(physical['full_tensor_rank_coverage'] is True, 'Physical full tensor coverage missing')
    for pair in forward.PAIRS:
        row = physical[pair]
        require(row['finite'] is True and row['max_position_error_m'] == row['max_heading_error_rad'] == 0, 'Nonfinite/nonexact physical trajectory')
    require(report['full_parameter_gradient_statistics_captured'] is False
            and all(report[k] is False for k in ('passed', 'formal_authorized', 'trainer_consumable')), 'Do not relabel forward sidecar as fullgrad/acceptance')
    wall = process['wall_seconds']; require(type(wall) in (float, int) and math.isfinite(wall) and wall >= 0, 'Actual process cost missing')
    return {'schema': 'p0_fullgradient_delegated_forward_context_v2', 'case': job['case'], 'repeat': job['repeat'],
        'original_strict_passed': strict['passed'], 'actual_returncode': process['returncode'],
        'expected_legacy_strict_failure': process['returncode'] == 1, 'PROCESS_EXIT': ref(process_path),
        'P0_FORWARD_COMPARISON': ref(path), 'original_strict': ref(strict_path), 'prospective_PROBE': ref(proto_path),
        'actual_delegated_forward_source_binding': binding, 'preserved_original44_context': job['source_binding'],
        'forward_receipt_bytes_rewritten': False, 'wall_seconds': wall, 'GPU_hours': wall*len(job['gpu_uuids'])/3600,
        'forward_exact_observed': True, 'passed': False, 'formal_authorized': False}


def combine_in_inventory_order(records, expected_hash):
    """Restore producer insertion order after sort_keys JSON; never edit raw rows.

    746 builds tensors in optimizer-group order (expected_trainable_inventory).
    Its stable worst_tensors sort preserves that order for ties, including zero
    gradients. Numeric statistics and the final comparison remain exact.
    """
    ordered = []
    for record in records:
        value = record['fullgrad']; expected = value['expected_trainable_inventory']
        names = [row[0] for row in expected]
        require(full.metadata_hash(expected) == expected_hash
                and names == [n for g in value['optimizer_groups'] for n in g['names']]
                and len(names) == len(set(names)) and set(names) == set(value['tensors']), 'Producer tensor order/inventory differs')
        ordered.append({**record, 'fullgrad': {**value, 'tensors': {n: value['tensors'][n] for n in names}}})
    return full.combine_gradients(ordered, expected_hash)


def classify(job):
    b, binding = checked_binding(job['fullgradient_source_binding'], job['source_binding'])
    observed = classify_forward_context(job, b, binding)
    path = Path(job['observed'])/'P0_FULL_GRADIENT_COMPARISON.json'
    report, proto = load(path), load(Path(job['folder'])/'PROBE.json')
    require(proto['fullgradient_binding'] == binding and proto['binding'] == job['source_binding'], 'Prospective contexts missing/mixed')
    require(report['schema'] == 'execution_acceptance_gradient_stats_v1'
            and report['producer_sha256'] == PRODUCER_SHA and report['producer_source_binding'] == binding
            and report['registration_sha256'] == forward.REGISTRATION_SHA and report['contract'] == b['contract'], 'Wrong full-gradient source/contract')
    require(report['candidate_mode'] == 'engineering_fast' and report['model_source'] == b['model']
            and report['original_forward_source_binding'] == job['source_binding']
            and report['original_acceptance_bindings'] == b['original_acceptance_bindings'], 'Wrong gradient model/original binding')
    require(report['checkpoint'] == {'run': str(Path(job['origin']['run']).resolve()), 'tag': job['origin']['tag'], 'case': job['case']}, 'Gradient origin differs')
    sidecar = Path(job['observed'])/'P0_FORWARD_COMPARISON.json'
    f = load(sidecar)
    require(report['forward_observation'] == ref(sidecar) and report['original_strict'] == f['original_strict'], 'Original forward/strict byte hashes differ')
    require(all(report[k] is True for k in ('full_parameter_rank_coverage', 'full_parameter_gradient_statistics_captured',
            'AA_B_directly_measured', 'numeric_thresholds_unchanged')) and report['ranks'] == 8
            and all(report[k] is False for k in ('passed', 'formal_authorized', 'trainer_consumable')), 'Incomplete coverage or forged acceptance')
    records = report['per_rank']
    require(len(records) == 8 and {r['rank'] for r in records} == set(range(8)), 'Missing/duplicate gradient ranks')
    require({r['rank']: r['tokens'] for r in records} == {r['rank']: r['tokens'] for r in f['per_rank']}, 'Gradient scene/rank order differs')
    case = b['cases'][job['case']]
    group_path = Path(job['origin']['run'])/'optimizer_groups.json'
    require(bench.digest(group_path) == case['optimizer_groups_sha256'], 'Original optimizer groups changed')
    groups = load(group_path)
    group_of = {n: i for i, g in enumerate(groups) for n in g['names']}
    require(len(group_of) == sum(len(g['names']) for g in groups), 'Duplicate optimizer parameter')
    for r in records:
        value = r['fullgrad']; expected = value['expected_trainable_inventory']
        require(value['optimizer_groups'] == groups and value['captured_roles'] == ['A', 'AA', 'B']
                and value['all_trainable_gradients_present'] is True
                and value['gradient_inventory_sha256'] == full.metadata_hash(expected) == case['gradient_inventory_sha256'], 'Gradient/group inventory differs')
        shapes = {n: shape for n, shape, dtype in expected if dtype == 'torch.float32'}
        require(len(shapes) == len(expected) == len(group_of) and set(shapes) == set(group_of) == set(value['tensors']), 'Full parameter inventory missing')
        inventory = value['inventory']
        require(set(inventory) == {'A', 'AA', 'B'} and inventory['A'] == inventory['AA'] == inventory['B'], 'Parameter/None inventory drift')
        named = {row[0]: row for row in inventory['A']}
        require(len(named) == len(inventory['A']) and {n for n, row in named.items() if row[3]} == set(shapes), 'Trainable inventory drift')
        for n, shape in shapes.items():
            require(named[n][1:] == [shape, 'torch.float32', True, [shape, 'torch.float32']], 'Missing/foreign actual gradient: '+n)
            tensor = value['tensors'][n]
            require(tensor['component'] == n.split('.')[0] and tensor['optimizer_group'] == str(group_of[n])
                    and set(tensor['pairs']) == forward.PAIRS, 'Missing gradient group/three direct pairs')
            for row in tensor['pairs'].values():
                count = math.prod(shape)
                require(row['elements'] == count and row['nonfinite_elements'] == 0
                        and all(type(row[k]) in (int, float) and math.isfinite(row[k]) for k in
                            ('reference_square', 'candidate_square', 'difference_square', 'dot', 'max_absolute'))
                        and all(row[k] >= 0 for k in ('reference_square', 'candidate_square', 'difference_square', 'max_absolute'))
                        and all(type(row[k]) is int and 0 <= row[k] <= count for k in ('bit_different_elements', 'fixed_numeric_violations')), 'Nonfinite/incomplete raw gradient statistics')
        require(all(sum(math.prod(shapes[n]) for n in g['names']) == g['elements'] for g in groups), 'Optimizer group elements drift')
    recomputed = combine_in_inventory_order(records, case['gradient_inventory_sha256'])
    require(report['comparisons'] == recomputed['comparisons'] and set(report['comparisons']) == forward.PAIRS, 'Aggregate full-gradient statistics differ')
    frozen = datetime.fromisoformat(b['frozen_UTC'].replace('Z', '+00:00'))
    prepared = datetime.fromisoformat(proto['created_UTC'].replace('Z', '+00:00'))
    require(frozen.tzinfo is not None and frozen <= prepared
            and all(prepared <= datetime.fromisoformat(r['roles']['A']['started_UTC'].replace('Z', '+00:00')) for r in f['per_rank']), 'New binding/PROBE must precede first A')
    return {**observed, 'schema': 'p0_fullgradient_attempt_observation_v1', 'status': 'completed_scientific_observation_no_retry',
        'P0_FULL_GRADIENT_COMPARISON': ref(path), 'fullgradient_source_binding': binding,
        'full_parameter_AA_B_gradients_captured': True, 'gradient_numeric_acceptance': 'PENDING independent unchanged V1 review',
        'old_bindings_reinterpreted': False, 'passed': False, 'formal_authorized': False}


def observed_prefix(manifest, jobs, output):
    """Explicit reuse of just the first actual869 slot; no worker retry or launch.

    All raw receipts, old failure/source and runtime qualification are pinned.
    The old full phase must contain no second started/partial slot. Reclassify
    the complete direct three-pair report on CPU and retain its actual charge.
    """
    request = manifest['observed_prefix']; root = Path(request['output']).resolve()
    require(root != Path(output).resolve() and not root.is_relative_to(Path(output).resolve())
            and not Path(output).resolve().is_relative_to(root), 'Prefix requires separate NEW output')
    files = request['files']
    first = 'G1_initial_repeat0/'
    paths = {'plan': 'PLAN.json', 'manifest_binding': 'MANIFEST_BINDING.json', 'registration': 'REGISTRATION.json',
        'failure': 'FAILURE.json', 'probe': first+'PROBE.json', 'incomplete': first+'INCOMPLETE.json',
        'process': first+'process/PROCESS.json', 'process_exit': first+'process/PROCESS_EXIT.json',
        'cpu_preflight': first+'CPU_PREFLIGHT.json', 'resource_gate': first+'RESOURCE_GATE.json',
        'fullgradient': first+'observed/P0_FULL_GRADIENT_COMPARISON.json',
        'forward': first+'observed/P0_FORWARD_COMPARISON.json', 'strict': first+'observed/strict/COMPLETE.json'}
    require(set(files) == set(paths) | {'manifest', 'old_coordinator'}, 'Explicit complete prefix receipt hashes required')
    for name, relative in paths.items():
        require(files[name] == ref(root/relative), 'Preserved prefix receipt changed: '+name)
    original_ref = load(files['manifest_binding']['path'])
    require(files['manifest'] == original_ref == ref(original_ref['path']), 'Original manifest changed')
    old_manifest, old_plan, registration = (load(files[n]['path']) for n in ('manifest', 'plan', 'registration'))
    fields = ('host', 'python', 'cache', 'dino_root', 'artifact_root', 'gpu_uuids', 'qualification', 'origins',
        'source_binding', 'fullgradient_source_binding', 'native_budget', 'previous_forward_completion', 'minimum_free_MiB')
    require(all(manifest[k] == old_manifest[k] for k in fields), 'Prefix source/environment/data/budget routing changed')
    old_jobs = old_plan['jobs']
    require(len(old_jobs) == len(jobs) == 12 and old_jobs[0]['key'] == jobs[0]['key'] == 'G1_initial_repeat0'
            and old_plan['manifest_sha256'] == original_ref['sha256'], 'Only exact first-slot prefix is supported')
    def science(job):
        value = {k: v for k, v in job.items() if k not in ('folder', 'run', 'observed')}
        for key in ('command', 'CPU_preflight_command'):
            argv = list(value[key]); require(argv.count('--output') == 1, 'Ambiguous prefix output argv')
            argv[argv.index('--output')+1] = '<owned-observation-output>'; value[key] = argv
        return value
    require(all(science(a) == science(b) for a, b in zip(old_jobs, jobs)), 'Prefix scientific job argv/order changed')
    require(all(not Path(j['folder']).exists() for j in old_jobs[1:])
            and not (root/'COMPLETE.json').exists(), 'Prefix contains another completed/unfinished slot')
    proto = load(files['probe']['path']); job = old_jobs[0]
    require(proto['job'] == job and Path(job['folder']).resolve() == root/'G1_initial_repeat0'
            and Path(job['observed']).resolve() == root/'G1_initial_repeat0/observed'
            and Path(job['run']).resolve() == root/'G1_initial_repeat0/process', 'Actual first-slot paths/job differ')
    driver = files['old_coordinator']
    require(driver == ref(driver['path']) == registration['driver'] == proto['coordinator']
            and old_plan['driver_sha256'] == driver['sha256'], 'Original coordinator identity changed')
    bench.immutable_source(Path(driver['path']).resolve().parents[2], registration['driver_git'])
    qref = manifest['qualification']; q = load(qref['path'])
    require(qref == ref(qref['path']) == registration['qualification'] and qref['sha256'] == QUALIFICATION_SHA
            and q['passed'] is True and q['host'] == manifest['host']
            and set(q['physical_gpu_UUIDs']) == set(manifest['gpu_uuids']), 'Prefix qualification/environment/UUID differs')
    require(registration['native_budget'] == manifest['native_budget']
            and registration['fullgradient_binding'] == job['fullgradient_source_binding']
            and registration['original_forward_binding'] == job['source_binding'], 'Prefix original E1/E2/budget context differs')
    preflight, gate = (load(files[n]['path']) for n in ('cpu_preflight', 'resource_gate'))
    require(preflight['preflight_passed'] is True and preflight['case'] == job['case']
            and preflight['source_binding_sha256'] == BINDING_SHA
            and preflight['model_source_sha'] == forward.MODEL_GIT
            and preflight['forward_source_sha'] == full.FORWARD_COMMIT, 'Prefix actual CPU source preflight differs')
    require(gate['decision'] == 'ready' and gate['compute_apps'] == []
            and {r['uuid'] for r in gate['gpus']} == set(job['gpu_uuids']), 'Prefix physical qualification gate differs')
    process = load(files['process']['path'])
    require(process['command'] == job['command'], 'Actual prefix process argv differs')
    failure, incomplete = (load(files[n]['path']) for n in ('failure', 'incomplete'))
    require(failure['error'] == incomplete['error'] == "ValueError('Aggregate full-gradient statistics differ')"
            and incomplete['PROCESS_EXIT'] == files['process_exit'], 'Only preserved aggregate-classification failure may be reused')
    observation = classify(job)  # Full inventory, exact aggregates, all8/three pairs and original forward/strict.
    charges = native.actual_process_charges([job])
    require(charges == failure['costs'] and observation['GPU_hours'] == charges['actual_phase_GPU_hours'], 'Prefix actual cost differs')
    for name, reference in files.items(): require(ref(reference['path']) == reference, 'Prefix receipt changed during classification: '+name)
    return {'schema': 'p0_fullgradient_first_observed_prefix_v1', 'job': job, 'observation': observation,
        'preserved_files': files, 'actual_process_costs': charges, 'CPU_reclassified_only': True,
        'original_failure_preserved': True, 'GPU_jobs_started': 0, 'passed': False, 'formal_authorized': False}


def run_attempt(job, locks):
    require(not Path(job['run']).exists() and not Path(job['observed']).exists(), 'NEW process/observer output required; no retry')
    try:
        try: bench.run_job(job, locks)
        except ValueError as exc:
            require(str(exc) == 'Real worker stage failed; logs retained, no retry', 'Unexpected process helper error')
        observation = classify(job)
    except BaseException as exc:
        process = Path(job['run'])/'PROCESS_EXIT.json'
        write(Path(job['folder'])/'INCOMPLETE.json', {'status': 'incomplete_no_retry', 'error': repr(exc),
            'PROCESS_EXIT': ref(process) if process.is_file() else None, 'passed': False, 'formal_authorized': False})
        raise
    write(Path(job['folder'])/'OBSERVATION.json', observation)
    return observation


def cost_record(jobs):
    try: return native.actual_process_charges(jobs)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return {'actual_phase_GPU_hours': None, 'cost_unknown': True, 'error': repr(exc)}


def execute(manifest, plan, output):
    git_sha = bench.checked(['git', 'rev-parse', 'HEAD'], ROOT)
    bench.immutable_source(ROOT, git_sha)
    require(bench.checked(['git', 'ls-files', '--', str(Path(__file__).relative_to(ROOT))], ROOT), 'Freeze coordinator before execute')
    previous = Path(manifest['previous_forward_completion'])
    done = load(previous)
    require(done['completed_forward_slots'] == len(done['observations']) == 12 and done['original_strict_relabelled'] is False
            and {(r['case'], r['repeat']) for r in done['observations']} == {(j['case'], j['repeat']) for j in plan['jobs']}
            and previous.parent.resolve() != Path(output).resolve(), 'Previous distinct forward12 phase must complete first')
    qref = manifest['qualification']; q = load(qref['path'])
    require(qref['sha256'] == QUALIFICATION_SHA and ref(qref['path']) == qref and q['passed'] is True and q['host'] == manifest['host']
            and set(q['physical_gpu_UUIDs']) == set(manifest['gpu_uuids']), 'Qualification extension/UUID mismatch')
    versions = q['environment']['packages']
    require(set(bench.PACKAGES) <= versions.keys() and 'flash_attn' in versions, 'Qualified runtime extras missing')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    code = 'import json,importlib.metadata as m;print(json.dumps({n:m.version(n)for n in '+repr(sorted(versions))+'}))'
    require(json.loads(bench.checked([manifest['python'], '-B', '-c', code], env=env, timeout=30)) == versions, 'Private runtime differs from extension')
    budget, jobs = plan['native_budget'], plan['jobs']
    prefix = observed_prefix(manifest, jobs, output) if 'observed_prefix' in manifest else None
    require(prefix == plan['observed_prefix'], 'Observed prefix changed after CPU planning')
    if prefix: write(Path(output)/'REUSED_PREFIX.json', prefix)
    admission = native.gpu_admission(budget, jobs, native.actual_process_charges(jobs), whole_plan=True)
    write(Path(output)/'REGISTRATION.json', {'driver_git': git_sha, 'driver': ref(__file__), 'native_budget': budget,
        'phase_scope': 'ONLY this fullgradient12 phase including explicit first-slot reuse; prior/zt3 cost UNKNOWN_NOT_ZERO',
        'admission': admission, 'prospective_plan': ref(Path(output)/'PLAN.json'),
        'reused_prefix': ref(Path(output)/'REUSED_PREFIX.json') if prefix else None,
        'new_GPU_jobs': plan['new_GPU_jobs'],
        'previous_forward_completion': ref(previous), 'qualification': qref, 'fullgradient_binding': plan['fullgradient_source_binding'],
        'original_forward_binding': plan['original_forward_source_binding'], 'passed': False, 'formal_authorized': False})
    stop = [False]; observations = [prefix['observation']] if prefix else []
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    for job in jobs[1:] if prefix else jobs:
        if stop[0] or (Path(output)/'STOP_BEFORE_NEXT_JOB').exists(): break
        _, binding = sources(manifest)
        require(binding == plan['fullgradient_source_binding'] and bench.digest(__file__) == plan['driver_sha256'], 'Bound source changed')
        if prefix:
            for reference in prefix['preserved_files'].values():
                require(ref(reference['path']) == reference, 'Observed prefix evidence changed before next job')
        folder = Path(job['folder']); folder.mkdir(mode=0o700)
        gate = native.gpu_admission(budget, jobs, native.actual_process_charges(jobs), next_job=job)
        write(folder/'GPU_BUDGET_BEFORE.json', gate)
        write(folder/'PROBE.json', {'schema': 'p0_fullgradient_prospective_probe_v1', 'created_UTC': datetime.now(timezone.utc).isoformat(),
            'binding': job['source_binding'], 'fullgradient_binding': binding, 'job': job, 'coordinator': ref(__file__),
            'context': 'Original44 sidecar context preserved; new fullgrad binding is separate, not an819/44 acceptance envelope', 'GPU_process_started': False})
        preflight = json.loads(bench.checked(job['CPU_preflight_command'], env=env, timeout=60))
        require(preflight['preflight_passed'] is True and preflight['case'] == job['case']
                and preflight['source_binding_sha256'] == binding['sha256'], 'Actual full-gradient source/origin CPU preflight failed')
        write(folder/'CPU_PREFLIGHT.json', preflight)
        with ExitStack() as stack:
            key = hashlib.sha256(json.dumps(sorted(manifest['gpu_uuids'])).encode()).hexdigest()
            locks = [bench.acquire(stack, Path(manifest['artifact_root'])/'physical_gpu_locks'/(key+'.lock'))]
            write(folder/'RESOURCE_GATE.json', bench.readiness(manifest, record_sample=lambda x: write(folder/f"RESOURCE_SAMPLE_{x['attempt']}.json", x)))
            try: observations.append(run_attempt(job, locks))
            finally: write(folder/'GPU_COST_AFTER.json', cost_record(jobs))
    write(Path(output)/'COMPLETE.json', {'schema': 'p0_fullgradient_coordinator_completion_v1', 'completed_fullgradient_slots': len(observations),
        'all12_observed': len(observations) == 12, 'observations': observations, 'actual_process_costs': native.actual_process_charges(jobs),
        'original_strict_relabelled': False, 'passed': False, 'formal_authorized': False, 'trainer_consumable': False})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path); p.add_argument('--output', required=True, type=Path)
    p.add_argument('--execute', action='store_true'); args = p.parse_args()
    require(not args.output.exists(), 'NEW output required; existing/partial attempts refused')
    manifest = load(args.manifest); plan = build_plan(manifest, args.output)
    plan['manifest_sha256'] = bench.digest(args.manifest)
    args.output.mkdir(parents=True, mode=0o700)
    write(args.output/'PLAN.json', plan); write(args.output/'MANIFEST_BINDING.json', ref(args.manifest))
    if not args.execute:
        print(json.dumps({'status': 'CPU_PLANNED_ONLY', 'fullgradient_slots': 12,
            'new_GPU_jobs': plan['new_GPU_jobs'], 'reused_slots': 1 if plan['observed_prefix'] else 0, 'GPU_jobs_started': 0})); return
    try: execute(manifest, plan, args.output)
    except BaseException as exc:
        write(args.output/'FAILURE.json', {'error': repr(exc), 'costs': cost_record(plan['jobs']),
            'no_retry_or_cleanup': True, 'passed': False, 'formal_authorized': False})
        raise


if __name__ == '__main__': main()
