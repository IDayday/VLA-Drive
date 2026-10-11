"""CPU sidecar for actual V1 evidence; never dispatches training or authorizes it.

contract emits requirements, compare runs the unchanged CPU comparator, assemble
joins hash-bound observations to real artifacts and calls the unchanged reviewer.
Missing observations are pending. No expected binding fact becomes an observed
fact, and no old strict or incomplete comparison is relabeled as a pass.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = ROOT/'reports/structured_world/execution_20261011/INDEPENDENT_ACCEPTANCE_PREREGISTRATION_V1.json'
REGISTRATION_SHA = '8bc0cc1dec82c88cc73e8cde4e5e9eac1d4d88633339c6fcd6670a1d984a8007'
REVIEWER = ROOT/'tools/structured_world/evaluate_execution_acceptance.py'
REVIEWER_SHA = 'a4effc9b1e56ea6787de9e37bb57cffb2c83b87f7543e78c04569e01b2e34c55'
COMPARATOR = ROOT/'tools/structured_world/compare_execution_checkpoints.py'
COMPARATOR_SHA = '9dede8df91b9def8623c608e9581cb5246647c4870c8e52cdca4f6a8c76cb703'
ROLES = ('A', 'AA', 'B')
COUNT_FIELDS = ('updates', 'epoch', 'offset_after_update', 'global_counts', 'schedule_horizon',
                'schedule_warmup', 'continuation_endpoint', 'source_population_scenes',
                'rank', 'global_scene_indices', 'local_scene_indices', 'loss_horizons')
HEADER_FIELDS = ('registration_sha256', 'bindings_sha256', 'started_utc', 'candidate_mode',
                 'tier', 'kind', 'group', 'stage', 'repeat', 'updates', 'producer_sha256',
                 'roles', 'exact_checks')


class Incomplete(ValueError):
    pass


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def reviewer():
    require(digest(REVIEWER) == REVIEWER_SHA, 'Existing reviewer source changed; audit required')
    spec = importlib.util.spec_from_file_location('_execution_evidence_reviewer', REVIEWER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def policy(path):
    require(digest(path) == REGISTRATION_SHA, 'Actual V1 preregistration hash required; no revised tolerances')
    return load(path)


def reference(ref, base, seen):
    require(isinstance(ref, dict) and isinstance(ref.get('sha256'), str) and len(ref['sha256']) == 64,
            'Hash-bound actual artifact required')
    path = Path(ref['path'])
    path = (path if path.is_absolute() else Path(base)/path).resolve()
    if not path.is_file():
        raise Incomplete('Missing actual artifact: '+str(path))
    actual = digest(path)
    require(actual == ref['sha256'], 'Artifact hash changed: '+str(path))
    seen[str(path)] = actual
    return {'path': str(path), 'sha256': actual}


def document(ref, base, seen):
    actual = reference(ref, base, seen)
    return load(actual['path']), actual


def contract(registration=REGISTRATION):
    p, gate = policy(registration), reviewer()
    return {
        'schema': 'execution_evidence_assembly_contract_v1', 'executed_trial_facts': False,
        'registration_sha256': REGISTRATION_SHA, 'adapter_sha256': digest(__file__),
        'reviewer_sha256': REVIEWER_SHA, 'comparator_sha256': COMPARATOR_SHA,
        'required_bindings': gate.binding_template(p, REGISTRATION_SHA),
        'required_receipt_keys': [list(k) for k in sorted(gate.required_keys(p))],
        'manifest_schema': 'execution_evidence_assembly_manifest_v1',
        'entry_contract': {
            'observation': 'hash-ref JSON execution_evidence_observation_v1, containing every HEADER_FIELD below; actual observed role facts, not copied from expected bindings',
            'header_fields': list(HEADER_FIELDS),
            'forward': ['comparison: hash-ref actual execution_acceptance_gradient_stats_v1 (all AA/AB/AA_B full gradient norms)',
                        'original_strict: hash-ref unmodified legacy COMPLETE',
                        'forward_observation: hash-ref JSON with actual forward_exact checks; no derived total-loss or invented query check'],
            'native': ['comparison: hash-ref actual native_execution_checkpoint_comparison_v1',
                       'counts: {triplets: hash-ref COUNT_SCENE_HORIZON_TRIPLETS.json, case: actual case name, captures: A/AA/B hash-refs of native_execution_capture_receipt_v1}'],
            'recovery': ['native fields plus load_boundary_receipt: hash-ref actual V1 boundary observation',
                         'load_boundary_source: hash-ref original RESUME_BOUNDARY_VERIFIED.json'],
            'dev': ['dev_payload: hash-ref actual {protocol,index,banks} JSON, hash-ref CSV/summary per role and q0/q_final; all +500 checkpoint facts must match native comparator'],
        },
        'prospective_binding': [
            'Freeze populated adopted execution_acceptance_bindings_v1 before candidate B execution. Pin actual forward/native/recovery/dev producer hashes; the assembly adapter hash is recorded separately.',
            'Actual observation records must bind that exact registration/bindings SHA, B started_utc, observed source/environment/science/ancestry and E1 checks. Never use assembly time as B start.',
            'All source inventories, execution flag hashes, environment fields, four origin/case inventories and canonical dev protocol must be real and populated; the template is not evidence.',
            'sources.files_sha256 must reproduce the native trainer source_hashes() inventory exactly, not an arbitrary larger repository manifest; its canonical tree digest is rechecked against identity.json.',
            'Preserve historical protected-file checks through --protected-source-root pointing to the preregistered original source, not a capture-modified trainer tree.',
        ],
        'actual_mapping_limits': [
            'Old legacy forward COMPLETE lacks full three-pair gradient norms and standalone total loss/query hashes; cannot manufacture V1 forward stats from maxima/relative-L2 summaries.',
            'Capture/count top-level passed=false remains false. Actual captures_complete and independently checked JSONL equality support only count/scene/horizon evidence.',
            'Real split/resume load boundary is +16; V1 evaluator requires origin+32. Missing actual +32 boundary stays pending, never relabel +16.',
            'Current recovery runner compares A/AA IO with B fast; that is not by itself a same-mode continuous-fast versus split-fast comparison. Preserve actual mode labels; parent reviews additional same-mode evidence separately.',
            'Old comparator strict/fixed false is copied unchanged. Incomplete, metadata-only or skipped-payload-hash comparison is not eligible.',
            'Evidence_complete_for_review can be true with old strict false under the existing prospective criteria. It is not trainer authorization.',
        ],
    }


def check_counts(spec, comparison, header, base, seen):
    triplets, triplet_ref = document(spec['triplets'], base, seen)
    rows = [r for r in triplets['discrete_evidence'] if r['case'] == spec['case']]
    require(len(rows) == 1 and rows[0]['all8_count_scene_horizon_exact'] is True, 'Missing/failed actual count triplet')
    require(set(spec['captures']) == set(ROLES), 'Three actual capture receipts required')
    origin = comparison['provenance']['origin']['completed']
    end = origin+header['updates']
    prefixes, refs = {}, {}
    for role in ROLES:
        capture, ref = document(spec['captures'][role], base, seen)
        if capture.get('captures_complete') is not True:
            raise Incomplete('Actual capture incomplete: '+role)
        require(capture['schema'] == 'native_execution_capture_receipt_v1' and capture['role'] == role,
                'Wrong actual capture role/schema')
        provenance = comparison['provenance'][role]
        require(capture['run_identity'] == provenance['identity'] and capture['training_source_sha'] == provenance['source']
                and capture['mode'] == provenance['execution_mode'], 'Capture/comparator source or identity mismatch')
        require({r['rank'] for r in capture['count_evidence']} == set(range(8)) and len(capture['count_evidence']) == 8,
                'All eight actual count files required')
        by_rank = {}; refs[role] = {'capture': ref, 'counts': []}
        for record in capture['count_evidence']:
            actual = reference(record, base, seen)
            lines = [json.loads(line) for line in Path(actual['path']).read_text().splitlines() if line.strip()]
            require(len(lines) == record['updates'] and [r['updates'] for r in lines] == list(range(origin+1, origin+1+len(lines))),
                    'Count JSONL update coverage is not one continuous chain')
            require(len(lines) >= header['updates'], 'Missing actual prefix count updates')
            rank = record['rank']; selected = lines[:header['updates']]
            for row in selected:
                indices, counts = row['global_scene_indices'], row['global_counts']
                require(row['rank'] == rank and row['local_scene_indices'] == indices[rank::8]
                        and len(indices) == len(set(indices)) and all(type(v) is int and 0 <= v < 101592 for v in indices),
                        'Actual scene order/coverage invalid')
                require(counts.get('ego_scenes') == len(indices) and all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in counts.values()),
                        'Invalid actual global denominators')
                science = header['roles'][role]['science']
                require(row['schedule_horizon'] == science['schedule_horizon'] and row['schedule_warmup'] == science['warmup']
                        and row['source_population_scenes'] == 101592 and row['continuation_endpoint'] >= end,
                        'Actual count schedule/population/endpoint differs')
                horizons = ({'dino_horizon'} if 'future_dino' in counts else set()) | ({'visual_horizon'} if 'visual' in counts else set())
                require(set(row['loss_horizons']) == horizons and all(len(v) == len(row['local_scene_indices']) and
                        all(type(h) is int and 0 <= h < 3 for h in v) for v in row['loss_horizons'].values()), 'Actual loss horizons missing/invalid')
            by_rank[rank] = [{k: r[k] for k in COUNT_FIELDS} for r in selected]
            refs[role]['counts'].append(actual)
        prefixes[role] = by_rank
    require(canonical(prefixes['A']) == canonical(prefixes['AA']) == canonical(prefixes['B']), 'Actual count/scene/horizon triplet drift')
    for update in range(header['updates']):
        rows = [prefixes['A'][r][update] for r in range(8)]
        require(len({canonical({k: v for k, v in r.items() if k not in ('rank', 'local_scene_indices', 'loss_horizons')}) for r in rows}) == 1,
                'Ranks disagree on actual global count/scene/update facts')
    return {'triplets': triplet_ref, 'captures': refs, 'actual_prefix_updates': header['updates'], 'all8_count_scene_horizon_exact': True}


def check_boundary(entry, header, comparison, base, seen):
    normalized, norm_ref = document(entry['load_boundary_receipt'], base, seen)
    actual, actual_ref = document(entry['load_boundary_source'], base, seen)
    expected = header['roles']['B']['origin']['completed']+header['updates']
    if actual.get('completed') != expected:
        raise Incomplete(f'Actual load boundary is {actual.get("completed")}; existing V1 requires {expected}; cannot relabel split16')
    require(actual['identity'] == comparison['provenance']['B']['identity'], 'Native boundary belongs to a different candidate')
    require(actual.get('passed') is True and actual.get('FP32_model_and_buffers') == 'bitwise'
            and actual.get('FP32_masters') == 'bitwise all ranks, actual parameters'
            and actual.get('Adam_moments_steps_and_LR') == 'bitwise all ranks'
            and actual.get('all_task_RNG') == 'exact all ranks'
            and actual.get('timing') == 'before next forward/backward/update', 'Actual native load boundary is not exact')
    require(normalized['registration_sha256'] == header['registration_sha256'] and normalized['bindings_sha256'] == header['bindings_sha256']
            and normalized['completed'] == actual['completed'] and normalized.get('native_payload_hashes_verified') is True,
            'V1 load boundary provenance differs')
    require(all(normalized.get(k) is True for k in ('model_FP32', 'master_FP32', 'Adam', 'BN_float', 'RNG', 'discrete_state')),
            'V1 boundary state checks missing')
    return norm_ref, actual_ref


def build_receipt(entry, bindings, p, base, seen, gate):
    header, header_ref = document(entry['observation'], base, seen)
    require(header['schema'] == 'execution_evidence_observation_v1', 'Actual observation header schema required')
    require(set(HEADER_FIELDS) <= header.keys(), 'Incomplete actual observation fields')
    require(all(type(header[k]) is int and header[k] >= 0 for k in ('repeat', 'updates')), 'Actual repeat/update indices must be nonnegative integers')
    key = (header['kind'], header['group'], header['stage'], header['repeat'], header['updates'])
    require(key in gate.required_keys(p), 'Unexpected registered receipt key')
    require(header['registration_sha256'] == REGISTRATION_SHA and header['bindings_sha256'] == bindings['_sha256'], 'Observation refers to different frozen bindings')
    require(header['producer_sha256'] == bindings['producer_sha256'][header['kind']], 'Observed producer differs from prospectively bound producer')
    require(gate.time_value(header['started_utc']) >= gate.time_value(bindings['frozen_utc']), 'Candidate started before binding freeze; no retroactive qualification')
    require(header['candidate_mode'] == bindings['candidate_mode'] and header['tier'] == bindings['tier'], 'Observation mode/tier differs')
    case = bindings['cases'][header['group']+'/'+header['stage']]
    gate.exact(header['roles'], gate.expected_roles(bindings, case), 'actual observed role facts')
    require(set(p['E1']['exact_checks']) <= header['exact_checks'].keys() and all(header['exact_checks'][k] is True for k in p['E1']['exact_checks']),
            'Actual E1 observations failed or missing')
    receipt = {k: copy.deepcopy(header[k]) for k in HEADER_FIELDS}
    receipt.update(schema='execution_acceptance_receipt_v1', assembly_observation=header_ref)
    kind = header['kind']; payload = {}
    if kind == 'dev':
        payload, ref = document(entry['dev_payload'], base, seen)
        require(set(payload) == {'protocol', 'index', 'banks'}, 'Actual dev protocol/index/banks required')
        payload = copy.deepcopy(payload); payload['index'] = reference(payload['index'], base, seen)
        require(set(payload['banks']) == set(ROLES), 'Actual three-role dev banks required')
        for banks in payload['banks'].values():
            require(set(banks) == set(p['dev']['variants']), 'Actual q0/q_final dev banks required')
            for bank in banks.values():
                for name in ('csv', 'summary'):
                    bank[name] = reference(bank[name], base, seen)
        receipt['assembly_dev_payload'] = ref
    else:
        comp, comp_ref = document(entry['comparison'], base, seen)
        payload['comparison'] = comp_ref
        if kind == 'forward':
            if comp.get('schema') != 'execution_acceptance_gradient_stats_v1':
                raise Incomplete('Legacy summaries cannot supply full AA/AB/AA_B gradient statistics')
            require(comp.get('full_parameter_rank_coverage') is True, 'Incomplete full forward gradient rank coverage')
            observation, ref = document(entry['forward_observation'], base, seen)
            require(isinstance(observation.get('forward_exact'), dict) and all(type(v) is bool for v in observation['forward_exact'].values()),
                    'Actual forward exact observations must be boolean checks, never arrays or guessed scalar facts')
            payload['forward_exact'] = observation['forward_exact']; receipt['assembly_forward_observation'] = ref
            payload['original_strict'] = reference(entry['original_strict'], base, seen)
            if header['candidate_mode'] != 'engineering_fast':
                raise ValueError('This bounded adapter supports selected P0 engineering_fast only')
        else:
            if comp.get('comparison_performed') is not True or comp.get('valid_native_inputs') is not True:
                raise Incomplete('Native comparison incomplete; original false remains false')
            require(comp['schema'] == 'native_execution_checkpoint_comparison_v1' and comp.get('integrity_verified') is True
                    and comp['tool_sha256'] == COMPARATOR_SHA and comp['coverage']['all_rank_slice_membership_complete'] is True,
                    'Native comparison metadata-only/unchecked/foreign/partial')
            require(all(comp['E1'][pair]['passed'] is True and comp['E1'][pair]['failure_count'] == 0 for pair in gate.PAIRS), 'Actual native E1 failed')
            payload['count_scene_horizon'] = check_counts(entry['counts'], comp, header, base, seen)
            if kind == 'recovery':
                ref, actual_ref = check_boundary(entry, header, comp, base, seen)
                payload.update(load_boundary_receipt=ref, load_boundary_exact=True)
                receipt['assembly_native_boundary'] = actual_ref
    receipt['payload'] = payload
    return key, receipt


def assemble(manifest_path, bindings_path, bindings_sha256, output, *, registration=REGISTRATION, protected_source_root=ROOT):
    output = Path(output)
    require(not output.exists(), 'NEW output directory required; no overwrite/retry')
    p, gate = policy(registration), reviewer()
    require(digest(bindings_path) == bindings_sha256, 'Frozen bindings SHA mismatch')
    bindings, manifest = load(bindings_path), load(manifest_path)
    gate.validate_bindings(p, bindings, REGISTRATION_SHA)
    require(bindings['adopted_for_pilot'] is True and bindings['candidate_mode'] == 'engineering_fast' and bindings['tier'] == 'E2', 'Adopted actual P0 bindings required')
    require(manifest['schema'] == 'execution_evidence_assembly_manifest_v1' and isinstance(manifest['entries'], list), 'Actual assembly manifest schema required')
    bindings['_sha256'] = bindings_sha256
    seen = {str(Path(bindings_path).resolve()): bindings_sha256, str(Path(manifest_path).resolve()): digest(manifest_path),
            str(Path(registration).resolve()): REGISTRATION_SHA}
    pending, failures, collected, keys = [], [], [], set()
    for number, entry in enumerate(manifest['entries']):
        try:
            key, receipt = build_receipt(entry, bindings, p, Path(manifest_path).parent, seen, gate)
            require(key not in keys, 'Duplicate actual receipt key')
            keys.add(key); collected.append((key, receipt))
        except (Incomplete, FileNotFoundError) as error:
            pending.append(f'entry {number}: {error}')
        except (ValueError, KeyError, TypeError, OSError) as error:
            failures.append(f'entry {number}: {type(error).__name__}: {error}')
    # Reject files that drifted while their actual records were being read.
    for name, expected in seen.items():
        require(digest(name) == expected, 'Artifact changed during assembly: '+name)
    output.mkdir(parents=True, exist_ok=False)
    receipts = []
    for key, receipt in collected:
        path = output/('receipt_'+'_'.join(map(str, key))+'.json')
        new_json(path, receipt); receipts.append(path)
    review = gate.evaluate(registration, bindings_path, receipts, source_root=protected_source_root)
    new_json(output/'INDEPENDENT_REVIEW.json', review)
    missing = [list(k) for k in sorted(gate.required_keys(p)-keys)]
    ready = not failures and not pending and not missing and review['advisory_decision'] == 'evidence_complete_for_review'
    result = {'schema': 'execution_evidence_assembly_result_v1', 'independent_evidence_satisfied': ready,
              'status': 'evidence_complete_for_review' if ready else 'not_supported' if failures or review['failures'] else 'pending',
              'passed': False, 'accepted_mode': None, 'formal_authorized': False, 'trainer_consumable': False,
              'adapter_sha256': digest(__file__), 'GPU_jobs_started': 0, 'registration_sha256': REGISTRATION_SHA,
              'bindings_sha256': bindings_sha256, 'missing_receipt_keys': missing, 'pending': pending,
              'failures': failures, 'input_artifacts': seen, 'receipt_paths': list(map(str, receipts)),
              'independent_review': {'path': str((output/'INDEPENDENT_REVIEW.json').resolve()), 'sha256': digest(output/'INDEPENDENT_REVIEW.json')},
              'old_strict_results_preserved': review['old_strict_results_preserved'],
              'limits': ['V1 reviewer decision unchanged; independent_evidence_satisfied is usable evidence readiness, never trainer authorization',
                         'Adapter reads compact metadata/statistics; native tensor integrity belongs to actual protected CPU comparator',
                         'No expected binding fact or placeholder was substituted for an observation']}
    new_json(output/'ASSEMBLY.json', result)
    return result


def run_comparison(reference_path, replay, candidate, output, *, origin=None, threads=2, chunk_elements=1 << 18):
    require(digest(COMPARATOR) == COMPARATOR_SHA, 'Protected actual comparator changed')
    require(not Path(output).exists() and not Path(str(output)+'.tmp').exists(), 'NEW comparator output required')
    require(type(threads) is int and threads > 0 and type(chunk_elements) is int and chunk_elements > 0, 'Positive CPU bounds required')
    args = [sys.executable, '-B', str(COMPARATOR), '--reference', str(reference_path), '--reference-replay', str(replay),
            '--candidate', str(candidate), '--output', str(output), '--threads', str(threads), '--chunk-elements', str(chunk_elements)]
    if origin is not None:
        args += ['--origin', str(origin)]
    process = subprocess.run(args, env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'}, check=False)
    report = load(output)
    return {'actual_exit_code': process.returncode, 'comparison': {'path': str(Path(output).resolve()), 'sha256': digest(output)},
            'comparison_performed': report.get('comparison_performed', False), 'original_passed': report.get('passed', False),
            'integrity_verified': report.get('integrity_verified', False), 'GPU_jobs_started': 0}


def metadata_hash(value):
    """Native trainer JSON recipe, including its default separators."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_inventory(root):
    root = Path(root).resolve()
    paths = []
    for directory in ('starVLA/model', 'starVLA/dataloader', 'tools/ddpolicy_vehicle', 'tools/foresight', 'third_party'):
        paths.extend(sorted((root/directory).rglob('*.py')))
    paths.extend(root/'tools/structured_world'/name for name in (
        'train_vla.py', 'recovery_boundary.py', 'training_assets.py', 'stage_timing.py',
        'prefetch.py', 'resume_origin.py', 'execution_log.py', 'native_capture.py'))
    return {str(p.relative_to(root)): digest(p) for p in paths}


def inspect_origin(origin):
    """Read trusted native pickle metadata on CPU; never read tensor values.

    The COMPLETE payload manifest is bound, not rehashed here. Actual comparison
    must independently verify all payload hashes before evidence can qualify.
    """
    require(digest(COMPARATOR) == COMPARATOR_SHA, 'Protected comparator changed')
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    require(not torch.cuda.is_initialized(), 'Origin metadata inspection requires CPU-only process')
    spec = importlib.util.spec_from_file_location('_execution_origin_comparator', COMPARATOR)
    cmp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cmp)
    bank = cmp.Bank(Path(origin).resolve(), verify_hashes=False)
    with FakeTensorMode():
        model = bank.model()
        layout = cmp.model_layout(model)
    group_by_name = {n: g for g, shapes in enumerate(model['param_shapes']) for n in shapes}
    trainable = [[n, list(s), 'torch.float32'] for shapes in model['param_shapes'] for n, s in shapes.items()]
    categories = {}
    def add(category, name, elements, group):
        scopes = categories.setdefault(category, {'all': 0, 'by_component': {}, 'by_optimizer_group': {}})
        scopes['all'] += elements
        component = name.split('.')[0]
        scopes['by_component'][component] = scopes['by_component'].get(component, 0)+elements
        if group is not None:
            key = str(group)
            scopes['by_optimizer_group'][key] = scopes['by_optimizer_group'].get(key, 0)+elements
    buffers, aliases = set(model['buffer_names']), set(model['shared_params'])
    for name, tensor in model['module'].items():
        if not tensor.is_floating_point():
            continue
        category = ('BN_float' if name.endswith(('running_mean', 'running_var')) else
                    'other_float_buffer' if name in buffers else 'model_alias' if name in aliases else 'model_FP32')
        add(category, name, tensor.numel(), group_by_name.get(name))
    for name, tensor in model.get('frozen_param_fragments', {}).items():
        add('frozen_fragment', name, tensor.numel(), None)
    for name, shape, _ in trainable:
        for category in ('master_FP32', 'Adam/exp_avg', 'Adam/exp_avg_sq'):
            add(category, name, math.prod(shape), group_by_name[name])
    # The reviewer flattens present scopes; empty group maps have no entries.
    inventory = {k: {'all': v['all'], **{'by_component/'+n: c for n, c in v['by_component'].items()},
                     **{'by_optimizer_group/'+n: c for n, c in v['by_optimizer_group'].items()}}
                 for k, v in categories.items()}
    return {'schema': 'execution_origin_metadata_v1', 'origin_path': str(bank.folder),
            'identity': bank.identity, 'COMPLETE': bank.complete, 'provenance': bank.provenance(),
            'model_layout': layout, 'expected_gradient_inventory': trainable, 'category_inventory': inventory,
            'parameter_inventory_sha256': metadata_hash(layout), 'gradient_inventory_sha256': metadata_hash(trainable),
            'gradient_inventory_semantics': 'Prospective all trainable parameter names/shapes/FP32 coverage; actual non-None gradients must be proved by future receipt',
            'tensor_values_read': False, 'payload_hashes_verified': False, 'GPU_jobs_started': 0}


def build_bindings(spec_path, output, *, registration=REGISTRATION):
    """Freeze prospective facts from hash-bound metadata; not an execution receipt.

    spec contains source_root, host_qualification, CPU_qualification, four origins
    (execution_origin_metadata_v1 hash refs), train_index, dev_protocol and four
    producers (file hash refs). Extra bound_files pins drivers/backend manifests.
    Hash recipes and expected inventories are explicit preregistered contracts.
    """
    p, gate = policy(registration), reviewer()
    spec, base, seen = load(spec_path), Path(spec_path).parent, {}
    require(spec['schema'] == 'execution_binding_build_spec_v1', 'Foreign build spec')
    host, host_ref = document(spec['host_qualification'], base, seen)
    cpu, cpu_ref = document(spec['CPU_qualification'], base, seen)
    require(host['passed'] is True and host['FP32_master_update_verified'] is True and cpu['passed_CPU'] is True,
            'Actual qualified host/CPU evidence required')
    root = Path(spec['source_root']).resolve()
    git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    require(git_sha == host['source']['git_SHA'], 'Frozen training source differs from qualified source')
    require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip(), 'Immutable clean training root required')
    files = source_inventory(root)
    require(metadata_hash(files) == host['source']['training_source_tree_sha256'], 'Native source inventory differs')
    b = gate.binding_template(p, REGISTRATION_SHA)
    b.update(adopted_for_pilot=True, candidate_mode='engineering_fast', tier='E2')
    flags = {}
    for role in ROLES:
        mode = 'engineering_fast' if role == 'B' else 'io_preserving_v1'
        flags[role] = {'mode': mode, 'configure_execution_sha256': digest(root/'starVLA/model/modules/structured_world/execution.py'),
                       'Qwen_language_activation_checkpointing': True, 'DiT_activation_checkpointing': True,
                       'geometry_checkpointing_and_BN': 'unchanged', 'TF32': False,
                       'parameter_master_optimizer_communication': 'FP32', 'time_vectorization': False,
                       'geometry_convolution_BF16_autocast': False, 'engineering_execution_reuse': role == 'B'}
        b['sources'][role] = {'git_sha': git_sha, 'files_sha256': files, 'execution_mode': mode,
                              'execution_mode_flags_sha256': metadata_hash(flags[role])}
    packages, env = host['environment']['packages'], host['environment']
    gpus = cpu['physical_gpu_inventory']
    require(len(gpus) == 8 and {g['uuid'] for g in gpus} == set(host['physical_gpu_UUIDs']), 'Qualified eight GPU UUIDs required')
    require(len({g['name'] for g in gpus}) == len({g['driver'] for g in gpus}) == 1, 'Ambiguous qualified GPU environment')
    b['environment'] = {'python': cpu['python_version'], 'torch': packages['torch'], 'cuda_runtime': env['torch_cuda_runtime'],
        'cuda_driver': gpus[0]['driver'], 'cudnn': env['cudnn_version'], 'nccl': env['NCCL_version'],
        **{k: packages[k] for k in ('deepspeed', 'transformers', 'flash_attn', 'numpy')},
        'gpu_model': gpus[0]['name'], 'world_size': 8, 'TF32': False, 'sync_BN': True,
        'activation_checkpoint_policy': {'Qwen_language': True, 'DiT': True, 'geometry': 'unchanged'},
        'model_dtype': 'FP32', 'master_dtype': 'FP32', 'optimizer_dtype': 'FP32'}
    train_index = reference(spec['train_index'], base, seen)
    details = {}
    loss_files = {name: digest(root/name) for name in (
        'starVLA/model/framework/vla_structured_fgtr.py', 'starVLA/model/modules/structured_world/losses.py',
        'starVLA/model/modules/structured_world/queries.py', 'starVLA/model/modules/structured_world/grid.py')}
    require(set(spec['origins']) == set(b['cases']), 'Exactly four registered origins required')
    for name, origin_ref in spec['origins'].items():
        m, ref = document(origin_ref, base, seen)
        require(m['schema'] == 'execution_origin_metadata_v1' and m['tensor_values_read'] is False, 'CPU native metadata snapshot required')
        i, c, provenance = m['identity'], m['COMPLETE'], m['provenance']
        require(i['identity'] == c['identity'] == provenance['identity'] and i['world_size'] == 8 and i['microbatch'] == 4,
                'Native origin identity/rank/microbatch mismatch')
        group, stage = name.split('/')
        require(i['config']['structured_world']['group'].startswith(group+'_') and c['completed'] == (2 if stage == 'initial' else 25000 if group == 'G1' else 50000), 'Wrong origin assignment')
        calibration = {'grid': i['config']['structured_world']['grid'], 'geometry_image_size': i['config']['structured_world']['geometry_image_size'],
                       'current_and_GT_identity': i['current_and_GT_identity'], 'shared_geometry_identity': i['shared_geometry_identity']}
        order = {'epoch_batches_source_sha256': digest(root/'tools/ddpolicy_vehicle/training_state.py'),
                 'seed': i['training_seed'], 'world_size': i['world_size'], 'global_batch': i['effective_global_batch'],
                 'microbatch': i['microbatch'], 'train_index_sha256': train_index['sha256'], 'origin_completed': c['completed']}
        schedule = {k: i['config']['trainer'][k] for k in ('lr_scheduler_type', 'scheduler_specific_kwargs', 'learning_rate')}
        schedule.update(horizon=i['schedule_horizon'], warmup=i['schedule_warmup'])
        definitions = {'source_files': loss_files, 'structured_world_config': i['config']['structured_world']}
        physical = {'grid_source_sha256': loss_files['starVLA/model/modules/structured_world/grid.py'],
                    'wrapper_source_sha256': loss_files['starVLA/model/framework/vla_structured_fgtr.py'], 'grid': calibration['grid']}
        science = {'training_source_sha': p['scientific_training_source'], 'config_sha256': metadata_hash(i['config']),
            'training_population_sha256': train_index['sha256'], 'current_and_GT_identity': i['current_and_GT_identity'],
            'DINO_identity': i['DINO_identity'], 'calibration_axis_sha256': metadata_hash(calibration), 'batch_order_sha256': metadata_hash(order),
            'training_seed': i['training_seed'], 'global_batch': i['effective_global_batch'], 'FM_repeat': i['FM_repeat'],
            'schedule_sha256': metadata_hash(schedule), 'schedule_horizon': i['schedule_horizon'], 'warmup': i['schedule_warmup'],
            'loss_definition_sha256': metadata_hash(definitions), 'physical_conversion_sha256': metadata_hash(physical)}
        b['cases'][name] = {'science': science,
            'origin': {'identity': c['identity'], 'completed': c['completed'], 'COMPLETE_sha256': provenance['COMPLETE_sha256'],
                       'payload_manifest_sha256': metadata_hash(c['file_sha256'])},
            'optimizer_groups_sha256': provenance['optimizer_groups_sha256'],
            'parameter_inventory_sha256': m['parameter_inventory_sha256'], 'gradient_inventory_sha256': m['gradient_inventory_sha256'],
            'forward_module_inventory': ['H_A', 'B0', 'Bt', 'q0', 'q_final', 'loss'], 'category_inventory': m['category_inventory']}
        require(m['parameter_inventory_sha256'] == metadata_hash(m['model_layout']) and
                m['gradient_inventory_sha256'] == metadata_hash(m['expected_gradient_inventory']), 'Metadata inventory hash mismatch')
        details[name] = {'metadata': ref, 'calibration_axis': calibration, 'batch_order_contract': order,
                         'schedule': schedule, 'loss_definition': definitions, 'physical_conversion': physical,
                         'gradient_inventory_semantics': m['gradient_inventory_semantics']}
        if 'determinism' not in b['environment']:
            b['environment']['determinism'] = i['determinism']
        gate.exact(b['environment']['determinism'], i['determinism'], 'origin determinism')
    protocol, protocol_ref = document(spec['dev_protocol'], base, seen)
    b['dev_protocol'].update(protocol)
    for kind in b['producer_sha256']:
        b['producer_sha256'][kind] = reference(spec['producers'][kind], base, seen)['sha256']
    for ref in spec.get('bound_files', []):
        reference(ref, base, seen)
    for path, sha in seen.items():
        require(digest(path) == sha, 'Input changed during binding build: '+path)
    require(source_inventory(root) == files, 'Training source changed during binding build')
    b['frozen_utc'] = datetime.now(timezone.utc).isoformat()
    gate.validate_bindings(p, b, REGISTRATION_SHA)
    out = Path(output); require(not out.exists(), 'NEW binding output directory required')
    out.mkdir(parents=True)
    new_json(out/'BINDINGS.json', b)
    record = {'schema': 'execution_binding_build_receipt_v1', 'bindings': {'path': str((out/'BINDINGS.json').resolve()), 'sha256': digest(out/'BINDINGS.json')},
        'frozen_utc': b['frozen_utc'], 'spec_sha256': digest(spec_path), 'builder_sha256': digest(__file__),
        'registration_sha256': REGISTRATION_SHA, 'host_qualification': host_ref, 'CPU_qualification': cpu_ref,
        'host': host['host'], 'physical_gpu_UUIDs': host['physical_gpu_UUIDs'], 'source_root': str(root),
        'source_tree_sha256': metadata_hash(files), 'dev_protocol': protocol_ref, 'execution_flag_descriptors': flags,
        'hash_recipe': 'SHA256(json.dumps(value, sort_keys=True, allow_nan=False).encode()); file hashes use exact bytes',
        'case_derivations': details, 'input_files_sha256': seen, 'executed_candidate_facts': False,
        'payload_hashes_verified': False, 'GPU_jobs_started': 0, 'formal_authorized': False,
        'limitations': ['Metadata-only origin inspection does not prove numerical values or payload integrity',
                        'Gradient inventory is prospective coverage, not an observed gradient pass',
                        'Future observed role facts and start times must independently match bindings; no retroactive qualification']}
    new_json(out/'BUILD_RECEIPT.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('contract'); c.add_argument('--output', type=Path, required=True)
    i = sub.add_parser('inspect-origin'); i.add_argument('--origin', type=Path, required=True); i.add_argument('--output', type=Path, required=True)
    f = sub.add_parser('build-bindings'); f.add_argument('--spec', type=Path, required=True); f.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('assemble')
    for name in ('manifest', 'bindings', 'output'):
        a.add_argument('--'+name, type=Path, required=True)
    a.add_argument('--bindings-sha256', required=True)
    a.add_argument('--protected-source-root', type=Path, required=True)
    b = sub.add_parser('compare')
    for name in ('reference', 'reference-replay', 'candidate', 'output'):
        b.add_argument('--'+name, type=Path, required=True)
    b.add_argument('--origin', type=Path); b.add_argument('--threads', type=int, default=2)
    b.add_argument('--chunk-elements', type=int, default=1 << 18)
    args = parser.parse_args()
    if args.command == 'contract':
        new_json(args.output, contract()); return 0
    if args.command == 'inspect-origin':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        new_json(args.output, inspect_origin(args.origin)); return 0
    if args.command == 'build-bindings':
        result = build_bindings(args.spec, args.output)
        print(json.dumps({'bindings': result['bindings'], 'frozen_utc': result['frozen_utc'], 'GPU_jobs_started': 0})); return 0
    if args.command == 'compare':
        result = run_comparison(args.reference, args.reference_replay, args.candidate, args.output,
                                origin=args.origin, threads=args.threads, chunk_elements=args.chunk_elements)
        print(json.dumps(result)); return result['actual_exit_code']
    result = assemble(args.manifest, args.bindings, args.bindings_sha256, args.output, protected_source_root=args.protected_source_root)
    print(json.dumps({k: result[k] for k in ('status', 'independent_evidence_satisfied', 'passed', 'GPU_jobs_started')}))
    return 2 if result['status'] == 'not_supported' else 0


if __name__ == '__main__':
    raise SystemExit(main())
