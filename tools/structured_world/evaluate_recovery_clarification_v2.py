"""CPU-only, prospective recovery clarification; never grants acceptance.

Freeze before any new B16 execution:
  python -B this_file.py --freeze --source-root FROZEN_BE5 --v1-bindings BINDINGS \
    --plan G1_SAME_MODE_MANIFEST --plan G3_SAME_MODE_MANIFEST --output NEW_DIRECTORY

Review actual receipts on the host containing the native banks:
  CUDA_VISIBLE_DEVICES='' python -B this_file.py --clarification CLARIFICATION.json \
    --receipt G1_RECEIPT --receipt G3_RECEIPT --output NEW_REVIEW.json

Review independently hashes all 17 payloads in each bank. It uses the unchanged
V1 numerical checker and frozen native comparator, not a metadata-only check.
V1 native500/forward/dev and old unsupported recovery records are unaffected.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

SOURCE_SHA = 'be5e72248a46afd43e1bf279c21134c391c5edad'
V1_SOURCE_ROOT = Path('/mnt/project/VLA-Drive-structured-world-fgtr-perf-v3-frozen-20261011')
V1_SOURCE_SHA = '1e5b10956c0af603400779d1cbcbc81b4b6882de'
SCHEMA = 'execution_recovery_only_clarification_v2'
RECEIPT_SCHEMA = 'execution_recovery_only_receipt_v2'
GROUPS = ('G1', 'G3')
KEYS = ('A32', 'AA32', 'B16', 'B_resume32')
V1_SECTIONS = ('numeric', 'dev', 'E1', 'trial_plan', 'tiers')
BOUNDARY = {
    'passed': True, 'FP32_model_and_buffers': 'bitwise',
    'FP32_masters': 'bitwise all ranks, actual parameters',
    'Adam_moments_steps_and_LR': 'bitwise all ranks',
    'all_task_RNG': 'exact all ranks',
    'alignment_padding': 'excluded using DeepSpeed original _get_groups_without_padding',
    'timing': 'before next forward/backward/update',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(), parse_constant=invalid)


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': digest(path)}


def checked_ref(value):
    path = Path(value['path'])
    require(path.is_absolute() and path.is_file() and digest(path) == value['sha256'],
            'Evidence hash mismatch: '+str(path))
    return path


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def helpers(root):
    base = Path(root)/'tools/structured_world'
    return (module(base/'evaluate_execution_acceptance.py', 'recovery_v1_numeric'),
            module(base/'compare_execution_checkpoints.py', 'recovery_native_compare'))


def source_check(root, expected=SOURCE_SHA):
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    require(git('rev-parse', 'HEAD') == expected and not git('status', '--porcelain'),
            'Clean immutable source required: '+expected)


def freeze_contract(root, bindings_path, plan_paths):
    source_check(root)
    gate, _ = helpers(root)
    policy_path = Path(root)/'reports/structured_world/execution_20261011/INDEPENDENT_ACCEPTANCE_PREREGISTRATION_V1.json'
    policy, bindings = load(policy_path), load(bindings_path)
    source_check(V1_SOURCE_ROOT, V1_SOURCE_SHA)
    for item in (*policy['protected_files'], *policy['background_only']):
        gate.hash_ref(item, V1_SOURCE_ROOT)
    gate.validate_bindings(policy, bindings, digest(policy_path))
    require(bindings['candidate_mode'] == 'engineering_fast' and bindings['tier'] == 'E2', 'P0 E2 only')
    source = bindings['sources']['B']
    require(source['git_sha'] == SOURCE_SHA, 'Preserved P0 source pin differs')
    for name, sha in source['files_sha256'].items():
        require(digest(Path(root)/name) == sha, 'Source inventory changed: '+name)
    plans = {}
    for path in plan_paths:
        p = load(path)
        group = p['group']
        require(group in GROUPS and group not in plans, 'Exactly one plan per recovery group')
        case = bindings['cases'][group+'/mid']
        jobs = p['jobs']
        require(p['schema'] == 'native_same_mode_recovery_commands_v1' and
                p['source_sha'] == SOURCE_SHA and p['execution_mode'] == 'engineering_fast', 'Foreign recovery plan')
        require(Path(p['source_root']).resolve() == Path(root).resolve(), 'Plan source root changed')
        require([j['key'] for j in jobs] == list(KEYS), 'Four ordered recovery stages required')
        require(len(p['gpu_uuids']) == 8 and len(set(p['gpu_uuids'])) == 8, 'Eight unique physical devices required')
        for job in jobs:
            require(job['mode'] == 'engineering_fast' and job['training_source_sha'] == SOURCE_SHA and
                    job['gpu_uuids'] == p['gpu_uuids'], 'All stages must use identical P0 source/devices')
            start = case['origin']['completed']
            exit_step = start+(16 if job['key'] == 'B16' else 32)
            load_step = start+(16 if job['key'] == 'B_resume32' else 0)
            require(job['expected_exit_completed'] == exit_step and job['expected_resume_completed'] == load_step and
                    job['capture_plan']['end_update'] == start+32 and job['capture_plan']['absolute_updates'] == [start+32],
                    'Load16 and endpoint32 cannot be mixed')
            require(job['origin']['complete_sha256'] == case['origin']['COMPLETE_sha256'], 'Original checkpoint pin changed')
            command = job['command']
            require(command[command.index('--')+1] == str(Path(root)/'tools/structured_world/train_vla.py') and
                    command[command.index('--debug-updates')+1] == '32' and command[command.index('--capture-updates')+1] == '32',
                    'Frozen trainer and immutable32 pilot required')
            require(('--resume' in command) == (job['key'] == 'B_resume32') and
                    ('--stop-after' in command) == (job['key'] == 'B16'), 'Unexpected stop/resume stage')
            if job['key'] == 'B16':
                require(command[command.index('--stop-after')+1] == str(start+16), 'Actual CLI must stop at16')
        # Scientific CLI equality, allowing only output and the split/resume controls.
        def args(job):
            command = list(job['command'])
            command[command.index('--output')+1] = '<owned-output>'
            if '--stop-after' in command:
                i = command.index('--stop-after'); del command[i:i+2]
            if '--resume' in command:
                command.remove('--resume')
            return command
        require(all(args(j) == args(jobs[0]) for j in jobs), 'Same-mode controls must retain identical scientific commands')
        require('--resume' in jobs[-1]['command'] and '--stop-after' in jobs[2]['command'], 'Actual split/reload commands required')
        require(jobs[2]['run'] == jobs[3]['run'] and len({j['run'] for j in jobs}) == 3, 'B must reload its own run; A/AA must be fresh')
        plans[group] = ref(path)
    require(set(plans) == set(GROUPS), 'G1 and G3 recovery plans required')
    protected = {name: ref(Path(root)/'tools/structured_world'/name) for name in (
        'evaluate_execution_acceptance.py', 'compare_execution_checkpoints.py', 'recovery_boundary.py', 'train_vla.py', 'training_assets.py')}
    return {
        'schema': SCHEMA, 'scope': 'PROSPECTIVE_RECOVERY_ONLY',
        'frozen_utc': datetime.now(timezone.utc).isoformat(), 'source_root': str(Path(root).resolve()),
        'source_sha': SOURCE_SHA, 'validator': ref(__file__), 'protected_helpers': protected,
        'V1_registration': ref(policy_path), 'V1_bindings': ref(bindings_path), 'plans': plans,
        'V1_protected_source_root': str(V1_SOURCE_ROOT), 'V1_protected_source_sha': V1_SOURCE_SHA,
        'unchanged_V1_sections_sha256': {k: canonical(policy[k]) for k in V1_SECTIONS},
        'candidate_mode': 'engineering_fast', 'tier': 'E2', 'groups': list(GROUPS), 'stage': 'mid',
        'load_relative_update': 16, 'endpoint_relative_update': 32,
        'role_assignment': {'A': 'P0 continuous32', 'AA': 'fresh P0 continuous32', 'B': 'P0 split16 then native resume to32'},
        'sources': {r: copy.deepcopy(source) for r in gate.ROLES},
        'V1_schema_mismatch': {'status': 'not_supported', 'preserved': True,
            'actual_load_relative_update': 16, 'V1_evaluator_expected_load_relative_update': 32,
            'original_A_AA_modes': 'io_preserving_v1', 'V2_A_AA_modes': 'engineering_fast',
            'V1_protected_trainer_sha256': next(r['sha256'] for r in policy['protected_files'] if r['path'].endswith('/train_vla.py')),
            'reason': 'V1 recovery evaluator conflates endpoint32 with load16; V1 bindings describe IO controls and retain their original protected trainer hash. No V1 hash/record is replaced by the separately pinned be5 fork.'},
        'quality_rule': 'Unchanged V1 E2 absolute limits AND fresh same-device P0 A_A envelope for A_B and AA_B; strict/fixed results retained verbatim.',
        'limits': [
            'Freeze precedes every new stage, especially B16. Existing forward/native500/dev results retain V1 interpretation.',
            'Frozen boundary helper checks all eight master/Adam/RNG partitions, and the shared model/BN payload on rank0 before an all-rank barrier. No per-rank model/BN receipt is invented.',
            'This recovery-only review cannot establish complete numerical/dev quality, authorize source adoption, or turn an original strict failure into a pass.',
            'Full payload rehashing is mandatory during actual receipt review; freeze and synthetic CPU tests do not read large training banks.'
        ],
        'passed': False, 'accepted_mode': None, 'formal_authorized': False, 'trainer_consumable': False,
        'GPU_jobs_started': 0, 'actual_receipts': 'PENDING',
    }


def review(clarification_path, receipt_paths):
    result = {'schema': 'execution_recovery_only_review_v2', 'passed': False, 'accepted_mode': None,
              'formal_authorized': False, 'trainer_consumable': False, 'GPU_jobs_started': 0,
              'advisory_decision': 'pending', 'failures': [], 'pending': [], 'groups': {},
              'original_strict_results_preserved': [], 'clarification': ref(clarification_path)}
    try:
        c = load(clarification_path)
        require(c['schema'] == SCHEMA and c['scope'] == 'PROSPECTIVE_RECOVERY_ONLY', 'Recovery-only clarification required')
        require(c['source_sha'] == SOURCE_SHA and c['sources']['B']['git_sha'] == SOURCE_SHA, 'Immutable source changed')
        require(c['passed'] is False and c['formal_authorized'] is False and c['trainer_consumable'] is False, 'Advisory cannot grant acceptance')
        checked_ref(c['validator']); require(c['validator']['sha256'] == digest(__file__), 'Validator pin changed')
        for item in c['protected_helpers'].values(): checked_ref(item)
        source_check(c['source_root'])
        gate, native = helpers(c['source_root'])
        policy = load(checked_ref(c['V1_registration']))
        gate.exact(c['V1_registration']['sha256'], digest(gate.REGISTRATION), 'Unchanged frozen V1 registration')
        gate.exact(c['V1_protected_source_sha'], V1_SOURCE_SHA, 'Original V1 protected source')
        source_check(c['V1_protected_source_root'], V1_SOURCE_SHA)
        for item in (*policy['protected_files'], *policy['background_only']):
            gate.hash_ref(item, Path(c['V1_protected_source_root']))
        bindings = load(checked_ref(c['V1_bindings']))
        gate.validate_bindings(policy, bindings, c['V1_registration']['sha256'])
        gate.exact(c['unchanged_V1_sections_sha256'], {k: canonical(policy[k]) for k in V1_SECTIONS}, 'V1 numeric/dev/scientific criteria unchanged')
        gate.exact(c['sources'], {r: bindings['sources']['B'] for r in gate.ROLES}, 'Same-mode P0 source facts')
        gate.exact((c['load_relative_update'], c['endpoint_relative_update']), (16, 32), 'Recovery load vs endpoint')
        seen = set()
        for path in receipt_paths:
            receipt = load(path)
            require(receipt['schema'] == RECEIPT_SCHEMA and receipt['kind'] == 'recovery' and receipt['stage'] == 'mid', 'Foreign receipt; V1/native500/forward cannot be reinterpreted')
            gate.exact(receipt['clarification_sha256'], digest(clarification_path), 'Prospective recovery binding')
            group = receipt['group']
            require(group in GROUPS and group not in seen, 'Duplicate/foreign recovery group')
            seen.add(group)
            p = load(checked_ref(c['plans'][group])); case = bindings['cases'][group+'/mid']
            gate.exact(receipt['plan_sha256'], c['plans'][group]['sha256'], 'Recovery command plan')
            facts = dict(bindings, sources=c['sources'])
            gate.exact(receipt['roles'], gate.expected_roles(facts, case), 'Frozen source/environment/science role facts')
            qualification = load(checked_ref(p['qualification']))
            require(qualification['passed'] is True and qualification['host'] == p['host'] and
                    len(qualification['physical_gpu_UUIDs']) == 8 and
                    set(qualification['physical_gpu_UUIDs']) == set(p['gpu_uuids']), 'Bound same-host full8 qualification changed')
            require(set(receipt['jobs']) == set(KEYS), 'All four actual stages required')
            split, endpoint = case['origin']['completed']+16, case['origin']['completed']+32
            banks = {}
            comp = load(checked_ref(receipt['comparison']))
            require(comp['schema'] == 'native_execution_checkpoint_comparison_v1', 'Frozen native comparator required')
            gate.exact(comp['tool_sha256'], c['protected_helpers']['compare_execution_checkpoints.py']['sha256'], 'Comparator source pin')
            require(all(comp.get(k) is True for k in ('integrity_verified', 'valid_native_inputs', 'comparison_performed')) and
                    comp['coverage']['all_rank_slice_membership_complete'] is True, 'Unverified/partial native comparison')
            require([r['rank'] for r in comp['rank_checks']] == list(range(8)), 'Missing/duplicate native comparison ranks')
            require(all(comp['E1'][pair]['passed'] is True and comp['E1'][pair]['failure_count'] == 0 and
                        comp['E1'][pair]['failure_paths'] == [] for pair in gate.PAIRS), 'Native RNG/groups/data/scheduler E1 mismatch')
            for role in (*gate.ROLES, 'origin'):
                prov = comp['provenance'][role]
                bank = native.Bank(prov['path'], verify_hashes=True)
                gate.exact(prov, bank.provenance(), 'Complete independently hashed native provenance/'+role)
                require(bank.ranks == 8, 'All eight native partitions required')
                banks[role] = bank
                if role != 'origin':
                    require(prov['completed'] == endpoint and prov['execution_mode'] == 'engineering_fast', 'All endpoints must be same-mode +32')
                    gate.native_identity_checks(prov, c['sources'][role], case['science'], case['origin'], Path(path).parent)
                else:
                    gate.exact(prov['identity'], case['origin']['identity'], 'Original native identity')
                    gate.exact(prov['COMPLETE_sha256'], case['origin']['COMPLETE_sha256'], 'Original native payload pin')
                    gate.exact(prov['completed'], case['origin']['completed'], 'Original completed')
            origin_science = {k: banks['origin'].identity[k] for k in native.ORIGIN_SCIENCE_FIELDS}
            for role in gate.ROLES:
                bank = banks[role]
                job = next(j for j in p['jobs'] if j['role'] == role)
                require(bank.run == Path(job['run']).resolve(), 'Paired endpoint must belong to the bound own run')
                gate.exact({k: bank.identity[k] for k in native.ORIGIN_SCIENCE_FIELDS}, origin_science, 'Original data/config/schedule/'+role)
                gate.exact(bank.groups, banks['origin'].groups, 'Original optimizer group membership/'+role)
                require(bank.identity['benchmark_no_end_checkpoint'] is False, 'Native saves cannot be suppressed')
                gate.exact({k: bank.identity[k] for k in native.SCIENCE_FIELDS},
                           {k: banks['A'].identity[k] for k in native.SCIENCE_FIELDS}, 'Same-mode scientific population/'+role)
            for job in p['jobs']:
                proof = receipt['jobs'][job['key']]
                process, exit_record, stage, resource = [load(checked_ref(proof[k])) for k in ('process', 'exit', 'stage', 'resource_gate')]
                gate.exact(process['command'], job['command'], 'Actual recovery command')
                require(process['started_unix'] >= gate.time_value(c['frozen_utc']).timestamp(), 'Actual stage preceded V2 freeze')
                require(exit_record['returncode'] == 0 and exit_record.get('monitor_error') is None and
                        exit_record['pid'] == process['pid'], 'Incomplete/failed actual stage')
                require(resource['decision'] == 'ready' and resource['compute_apps'] == [] and
                        len(resource['gpus']) == 8 and {g['uuid'] for g in resource['gpus']} == set(p['gpu_uuids']), 'Actual same-device resource receipt missing')
                gate.exact(proof['host'], p['host'], 'Actual host pairing')
                require(stage['key'] == job['key'] and stage['run_identity'] == banks[job['role']].identity['identity'] and
                        stage['identity_sha256'] == digest(banks[job['role']].run/'identity.json'), 'Stage switched native identity')
                capture = banks[job['role']].identity['engineering_native_capture']
                gate.exact({k: capture[k] for k in job['capture_plan']}, job['capture_plan'], 'Immutable capture endpoint')
                gate.exact(capture['storage_budget_sha256'], job['storage_budget']['sha256'], 'Owned storage pin')
                boundary = stage['origin_boundary']
                for k, v in BOUNDARY.items(): gate.exact(boundary[k], v, 'Exact native load/'+k)
                gate.exact(boundary['identity'], stage['run_identity'], 'Boundary run identity')
                gate.exact(boundary['completed'], job['expected_resume_completed'], 'Actual native load16 (not endpoint32)')
                require([r['rank'] for r in stage['logging_cost_by_rank']] == list(range(8)) and
                        all(r['completed'] == job['expected_exit_completed'] for r in stage['logging_cost_by_rank']), 'Missing/mixed all-rank stage completion')
                checkpoint = stage['checkpoint']
                bank = native.Bank(checkpoint['path'], verify_hashes=True) if job['key'] == 'B16' else banks[job['role']]
                require(str(bank.folder) == str(Path(checkpoint['path']).resolve()), 'Stage endpoint path differs')
                gate.exact(checkpoint['complete'], bank.complete, 'Preserved stage COMPLETE')
                gate.exact(checkpoint['COMPLETE_sha256'], digest(bank.folder/'COMPLETE.json'), 'Stage checkpoint hash')
                require(bank.ranks == 8 and bank.complete['completed'] == job['expected_exit_completed'] and
                        bank.complete['status'] == ('PAUSED' if job['key'] == 'B16' else 'COMPLETE'), 'Split16/endpoint32 native state mixed')
                require(bank.folder.name == bank.complete['tag'] == job['expected_native_tag'] and
                        bank.run == Path(job['run']).resolve(), 'Actual native tag/run differs from bound recovery stage')
                if job['key'] == 'B16':
                    require(bank.complete['completed'] == split and bank.identity == banks['B'].identity, 'B16 must be the same native B identity')
                    split_bank = bank
                elif job['key'] == 'B_resume32':
                    model = split_bank.model()
                    for k in ('completed', 'epoch', 'offset', 'exposure'):
                        gate.exact(boundary[k], model[k], 'Restored split client/'+k)
                    gate.exact(model['scheduler_horizon'], case['science']['schedule_horizon'], 'Restored original scheduler')
            result['original_strict_results_preserved'].append({'group': group, 'comparison': receipt['comparison'],
                'original_strict': comp['original_strict'], 'fixed_numeric_results': comp['fixed_numeric_results']})
            failures, statistics = gate.numerical_checks(policy, 'E2', comp, native=True, inventory=case['category_inventory'])
            result['failures'].extend(group+': '+v for v in failures)
            result['groups'][group] = {'actual_load_completed': split, 'endpoint_completed': endpoint,
                'complete_payload_hashes_reverified': True, 'same_mode': 'engineering_fast', 'numeric_statistics': statistics,
                'boundary_coverage': c['limits'][1], 'receipt': ref(path)}
        result['pending'] = ['Missing actual '+g+' recovery receipt' for g in GROUPS if g not in seen]
        if not result['failures'] and not result['pending']:
            result['advisory_decision'] = 'recovery_evidence_complete_for_review'
        result['V1_schema_mismatch_preserved'] = c['V1_schema_mismatch']
        result['other_quality_gates'] = 'V1 forward/native500/dev and formal source guard remain separate; no acceptance is granted.'
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        result['failures'].append(type(error).__name__+': '+str(error))
    if result['failures']: result['advisory_decision'] = 'not_supported'
    return result


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--v1-bindings', type=Path)
    parser.add_argument('--plan', type=Path, action='append', default=[])
    parser.add_argument('--clarification', type=Path)
    parser.add_argument('--receipt', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Fresh output required; never overwrite old evidence')
    if args.freeze:
        require(args.source_root and args.v1_bindings and len(args.plan) == 2 and not args.receipt, 'Freeze needs immutable source, original bindings and two plans')
        c = freeze_contract(args.source_root, args.v1_bindings, args.plan)
        args.output.mkdir(parents=True, exist_ok=False)
        write_new(args.output/'CLARIFICATION.json', c)
        write_new(args.output/'V1_SCHEMA_MISMATCH_PRESERVED.json', c['V1_schema_mismatch'])
        write_new(args.output/'RECEIPT_TEMPLATE.json', {'schema': RECEIPT_SCHEMA, 'kind': 'recovery', 'stage': 'mid',
            'clarification_sha256': digest(args.output/'CLARIFICATION.json'), 'group': None, 'plan_sha256': None,
            'roles': None,
            'comparison': {'path': None, 'sha256': None},
            'jobs': {k: {'host': None, **{name: {'path': None, 'sha256': None} for name in ('process', 'exit', 'stage', 'resource_gate')}} for k in KEYS},
            'notes': 'Bind real preserved V8 stage snapshots (B16 before resume), process/exit and resource receipts. Review on bank-owning host with unchanged complete payloads.'})
        print(json.dumps({'status': 'PROSPECTIVE_RECOVERY_ONLY_FROZEN', 'path': str(args.output), 'passed': False, 'GPU_jobs_started': 0}))
        return 0
    require(args.clarification, '--clarification required for review')
    report = review(args.clarification, args.receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_new(args.output, report)
    print(json.dumps({k: report[k] for k in ('advisory_decision', 'passed', 'GPU_jobs_started')}))
    return 2 if report['failures'] else 0


if __name__ == '__main__':
    sys.exit(main())
