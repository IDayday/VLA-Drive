"""CPU-only advisory review of prospectively bound E1/E2/E3 pilot receipts.

This cannot grant trainer acceptance: passed is ALWAYS false, accepted_mode is
ALWAYS null. It never dispatches jobs or rewrites old strict reports. First freeze
the proposed registration and a populated binding template; future producers
must bind both hashes. Run --write-binding-template to see the receipt contract.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REGISTRATION = ROOT/'reports/structured_world/execution_20261011/INDEPENDENT_ACCEPTANCE_PREREGISTRATION_V1.json'
ROLES = ('A', 'AA', 'B')
PAIRS = {'A_A': ('A', 'AA'), 'A_B': ('A', 'B'), 'AA_B': ('AA', 'B')}


class InvalidEvidence(ValueError):
    pass


def require(value, message):
    if not value:
        raise InvalidEvidence(message)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            sha.update(block)
    return sha.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(InvalidEvidence('Nonfinite JSON: '+x)))


def time_value(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(dt.tzinfo is not None, 'Timezone required')
    return dt


def populated(value):
    if value is None or value == '' or value == {} or value == []:
        return False
    if isinstance(value, dict):
        return all(populated(v) for v in value.values())
    if isinstance(value, list):
        return all(populated(v) for v in value)
    return True  # Explicit false flags and zero counters ARE populated.


def sha_value(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def exact(left, right, label):
    # JSON canonicalization preserves type distinctions (true vs 1, 1 vs 1.0).
    require(json.dumps(left, sort_keys=True, allow_nan=False) ==
            json.dumps(right, sort_keys=True, allow_nan=False), 'Exact E1 mismatch: '+label)


def hash_ref(ref, base):
    require(isinstance(ref, dict) and sha_value(ref.get('sha256')), 'Malformed evidence hash reference')
    path = Path(ref['path'])
    if not path.is_absolute():
        path = base/path
    require(path.is_file() and digest(path) == ref['sha256'], 'Evidence hash mismatch: '+str(path))
    return path


def binding_template(policy, registration_hash):
    e1 = policy['E1']
    source = {k: None for k in e1['required_source_fields']}
    source.update(execution_mode=None, execution_mode_flags_sha256=None)
    cases = {}
    for group in policy['trial_plan']['groups']:
        for stage in policy['trial_plan']['stages']:
            cases[group+'/'+stage] = {
                'science': {k: None for k in e1['required_science_fields'] if k != 'execution_mode_flags_sha256'},
                'origin': {k: None for k in e1['required_origin_fields']},
                **{k: None for k in e1['required_case_fields'] if k != 'origin'},
                'category_inventory': None,
            }
    protocol = {k: policy['dev'][k] for k in ('population', 'scenes', 'logs', 'precision', 'sampling_seed', 'candidate_count', 'proposal_steps', 'canonical_inference_source')}
    protocol.update({k: None for k in policy['dev']['protocol_bindings']})
    return {'schema': 'execution_acceptance_bindings_v1', 'registration_sha256': registration_hash,
            'adopted_for_pilot': False, 'formal_authorized': False, 'frozen_utc': None,
            'candidate_mode': None, 'tier': None, 'sources': {r: dict(source) for r in ROLES},
            'environment': {k: None for k in e1['required_environment_fields']},
            'cases': cases, 'dev_protocol': protocol,
            'producer_sha256': {k: None for k in ('forward', 'native', 'recovery', 'dev')},
            'E3_declaration': None}


def validate_bindings(policy, bindings, registration_hash):
    require(bindings['schema'] == 'execution_acceptance_bindings_v1', 'Foreign binding schema')
    exact(bindings['registration_sha256'], registration_hash, 'registration hash')
    require(bindings.get('formal_authorized') is False, 'This advisory binding cannot authorize formal training')
    mode, tier = bindings.get('candidate_mode'), bindings.get('tier')
    require(mode not in policy['forbidden_modes'], 'No-recompute remains rejected')
    require(tier in ('E2', 'E3') and mode in policy['tiers'][tier]['modes'], 'Wrong candidate tier/mode')
    require(time_value(bindings['frozen_utc']) >= time_value(policy['last_criteria_update_utc']), 'Binding predates current criteria')
    for role in ROLES:
        source = bindings['sources'][role]
        require(set(policy['E1']['required_source_fields']) <= set(source) and populated(source), 'Incomplete immutable source facts')
        require(len(source['git_sha']) == 40 and all(c in '0123456789abcdef' for c in source['git_sha']), 'Immutable source commit required')
        require(isinstance(source['files_sha256'], dict) and source['files_sha256'] and
                all(sha_value(v) for v in source['files_sha256'].values()), 'Source file hashes required')
        require(sha_value(source['execution_mode_flags_sha256']), 'Execution flag hash required')
        exact(source['execution_mode'], mode if role == 'B' else 'io_preserving_v1', 'role execution mode')
    require(set(policy['E1']['required_environment_fields']) <= set(bindings['environment']) and populated(bindings['environment']), 'Incomplete environment facts')
    exact(bindings['environment']['world_size'], policy['trial_plan']['world_size'], 'world size')
    for key in ('model_dtype', 'master_dtype', 'optimizer_dtype'):
        exact(bindings['environment'][key], 'FP32', key)
    expected_cases = {g+'/'+s for g in policy['trial_plan']['groups'] for s in policy['trial_plan']['stages']}
    require(set(bindings['cases']) == expected_cases, 'All registered initial/mid origins required')
    for name, case in bindings['cases'].items():
        science = dict(case['science'], execution_mode_flags_sha256=bindings['sources']['B']['execution_mode_flags_sha256'])
        require(set(policy['E1']['required_science_fields']) <= set(science) and populated(case), 'Incomplete case binding: '+name)
        require(set(policy['E1']['required_origin_fields']) <= set(case['origin']), 'Incomplete origin')
        require(all(sha_value(case[k]) for k in ('optimizer_groups_sha256', 'parameter_inventory_sha256', 'gradient_inventory_sha256')), 'Malformed inventory hash')
        require(sha_value(case['origin']['COMPLETE_sha256']) and sha_value(case['origin']['payload_manifest_sha256']), 'Origin payload binding required')
        for key, value in (('training_seed', 42), ('global_batch', 32), ('FM_repeat', 8), ('schedule_horizon', 100000)):
            exact(science[key], value, name+'/'+key)
        exact(science['training_source_sha'], policy['scientific_training_source'], 'scientific source')
        require(set(policy['numeric']['required_native_categories']) <= set(case['category_inventory']), 'Required full native categories missing')
    for key in ('population', 'scenes', 'logs', 'precision', 'sampling_seed', 'candidate_count', 'proposal_steps', 'canonical_inference_source'):
        exact(bindings['dev_protocol'][key], policy['dev'][key], 'dev protocol/'+key)
    require(set(policy['dev']['protocol_bindings']) <= set(bindings['dev_protocol']) and populated(bindings['dev_protocol']), 'Unbound canonical protocol')
    require(set(bindings['producer_sha256']) == {'forward', 'native', 'recovery', 'dev'} and
            all(sha_value(v) for v in bindings['producer_sha256'].values()), 'Frozen receipt producer hashes required')
    if tier == 'E3':
        exact(bindings['E3_declaration'], policy['tiers']['E3']['must_declare'], 'E3 precision/BN declaration')


def expected_roles(bindings, case):
    facts = {}
    for role in ROLES:
        science = dict(case['science'], execution_mode_flags_sha256=bindings['sources'][role]['execution_mode_flags_sha256'])
        facts[role] = {'source': bindings['sources'][role], 'environment': bindings['environment'], 'science': science,
                       **{k: case[k] for k in ('origin', 'optimizer_groups_sha256', 'parameter_inventory_sha256', 'gradient_inventory_sha256')}}
    return facts


def native_identity_checks(provenance, source, science, origin, base):
    """Re-read only small native metadata, never multi-GB tensor payloads.

    File-inventory and config digests use json.dumps(..., sort_keys=True),
    matching native source_hashes(); payload reductions belong to the frozen
    comparator. A copied bank can use its hash-identical local metadata path.
    """
    folder = Path(provenance['path'])
    if not folder.is_absolute():
        folder = base/folder
    run = folder.parent.parent
    identity_path = hash_ref({'path': str(run/'identity.json'), 'sha256': provenance['identity_sha256']}, base)
    identity = load(identity_path)
    exact(identity['identity'], provenance['identity'], 'native run identity')
    exact(identity['training_source_sha'], source['git_sha'], 'native immutable source commit')
    tree_hash = hashlib.sha256(json.dumps(source['files_sha256'], sort_keys=True).encode()).hexdigest()
    exact(identity['training_source_tree_sha256'], tree_hash, 'native complete source-file inventory hash')
    config_hash = hashlib.sha256(json.dumps(identity['config'], sort_keys=True).encode()).hexdigest()
    exact(config_hash, science['config_sha256'], 'native scientific config hash')
    for native_key, binding_key in (('current_and_GT_identity', 'current_and_GT_identity'), ('DINO_identity', 'DINO_identity'),
                                    ('training_seed', 'training_seed'), ('effective_global_batch', 'global_batch'),
                                    ('FM_repeat', 'FM_repeat'), ('schedule_horizon', 'schedule_horizon'), ('schedule_warmup', 'warmup')):
        exact(identity[native_key], science[binding_key], 'actual native science/'+native_key)
    ancestry = identity['engineering_continue_origin']
    for native_key, binding_key in (('identity', 'identity'), ('completed', 'completed'), ('complete_sha256', 'COMPLETE_sha256')):
        exact(ancestry[native_key], origin[binding_key], 'actual native ancestry/'+native_key)
    complete = load(hash_ref({'path': str(folder/'COMPLETE.json'), 'sha256': provenance['COMPLETE_sha256']}, base))
    exact(complete['identity'], provenance['identity'], 'native COMPLETE identity')
    exact(complete['completed'], provenance['completed'], 'native COMPLETE step')
    hash_ref({'path': str(run/'optimizer_groups.json'), 'sha256': provenance['optimizer_groups_sha256']}, base)


def scalar(value, label):
    require(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value), 'Missing/nonfinite number: '+label)
    return float(value)


def measures(stat, pair, native):
    require(stat['nonfinite_elements'] == 0 and stat['elements'] > 0, 'Nonfinite/empty complete tensor coverage')
    a, b, d = [scalar(stat[k], k) for k in ('reference_norm', 'candidate_norm', 'difference_norm')]
    require(min(a, b, d) >= 0, 'Negative norms')
    if a == 0 or b == 0:
        require(a == b == d == 0, 'Zero reference/candidate norm requires exact zero values')
        result = dict(relative_L2=0., norm_relative=0., cosine_error=0.)
    else:
        result = {'relative_L2': d/a, 'norm_relative': abs(b/a-1), 'cosine_error': max(0., 1-scalar(stat['cosine'], 'cosine'))}
        require(-1.000000001 <= stat['cosine'] <= 1.000000001, 'Invalid cosine')
        require(math.isclose(result['relative_L2'], scalar(stat['relative_L2'], 'relative_L2'), rel_tol=1e-7, abs_tol=1e-12), 'Inconsistent relL2/norm receipt')
    if native:
        updates = stat['origin_relative_updates']['update_norms']
        u, v = (scalar(updates[k], 'update norm') for k in PAIRS[pair])
        require(min(u, v) >= 0, 'Negative update norm')
        if u == 0 or v == 0:
            require(u == v == d == 0, 'Zero update requires exact unchanged values')
            result.update(update_error=0., update_norm_relative=0., update_cosine_error=0.)
        else:
            cos = (u*u+v*v-d*d)/(2*u*v)
            require(-1.00000001 <= cos <= 1.00000001, 'Inconsistent update norms/difference')
            result.update(update_error=d/u, update_norm_relative=abs(v/u-1), update_cosine_error=max(0., 1-cos))
    return result


def flatten_category(category):
    require(set(category) == {'all', 'by_component', 'by_optimizer_group'}, 'Incomplete numerical scopes')
    yield 'all', category['all']
    for scope in ('by_component', 'by_optimizer_group'):
        # BN/frozen buffers have no optimizer-group membership.
        if scope == 'by_component':
            require(category[scope], 'Empty named component scope')
        yield from ((scope+'/'+name, value) for name, value in category[scope].items())


def numerical_checks(policy, tier, comparison, *, native, inventory=None):
    comparisons = comparison['comparisons']
    require(set(comparisons) == set(PAIRS), 'All AA/AB/AA_B comparisons required')
    categories = set(comparisons['A_A'])
    require(all(set(c) == categories for c in comparisons.values()), 'Pair category coverage mismatch')
    require(categories and (native or categories == {'gradient'}), 'Wrong numerical receipt categories')
    if native:
        require(set(policy['numeric']['required_native_categories']) <= categories, 'Incomplete native tensor categories')
        require(set(inventory) == categories, 'Native category inventory differs from preregistered full model')
    failures, stats = [], {}
    limits = policy['numeric'][tier+'_absolute']
    envelope = policy['numeric']['E2_AA_envelope']
    for category in sorted(categories):
        rows = {p: dict(flatten_category(comparisons[p][category])) for p in PAIRS}
        require(all(set(v) == set(rows['A_A']) for v in rows.values()), 'Named scope coverage mismatch')
        for pair in PAIRS:
            components = [v['elements'] for k, v in rows[pair].items() if k.startswith('by_component/')]
            require(sum(components) == rows[pair]['all']['elements'], 'Incomplete named component coverage')
            if category == 'gradient' or category == 'master_FP32' or category.startswith('Adam/'):
                groups = [v['elements'] for k, v in rows[pair].items() if k.startswith('by_optimizer_group/')]
                require(sum(groups) == rows[pair]['all']['elements'], 'Incomplete optimizer-group coverage')
        if native:
            exact({name: row['elements'] for name, row in rows['A_A'].items()}, inventory[category], 'Full category/component/group element inventory')
        ceiling = limits['gradient' if not native else 'moment' if category.startswith('Adam/') else 'BN' if category == 'BN_float' else 'state']
        for name in rows['A_A']:
            require(len({rows[p][name]['elements'] for p in PAIRS}) == 1, 'Element coverage mismatch')
            numbers = {p: measures(rows[p][name], p, native) for p in PAIRS}
            for pair, values in numbers.items():
                for metric, value in values.items():
                    if value > ceiling[metric]:
                        failures.append(f'{category}/{name}/{pair}/{metric}: absolute ceiling')
                    if tier == 'E2' and pair != 'A_A' and value > envelope['factor']*numbers['A_A'][metric]+envelope['floors'][metric]:
                        failures.append(f'{category}/{name}/{pair}/{metric}: fresh AA envelope')
            stats[category+'/'+name] = {'measures': numbers, 'AB_over_AA_relL2_plus_floor': numbers['A_B']['relative_L2']/(numbers['A_A']['relative_L2']+envelope['floors']['relative_L2']),
                                      'raw_max_absolute_and_quantiles': {p: {k: rows[p][name].get(k) for k in ('max_absolute', 'logical_tensor_max_error_quantiles', 'worst_tensors')} for p in PAIRS}}
    return failures, stats


def forward_checks(policy, mode, payload, base, case):
    criteria = policy['tiers']['E2']['forward_by_mode'][mode]
    require(all(payload['forward_exact'].get(k) is True for k in criteria['required_exact']), 'Mode-specific exact forward checks failed')
    bounds = criteria['module_tolerances']
    if bounds is None:
        return []
    report = load(hash_ref(payload['forward_comparison'], base))
    require(report['schema'] == 'execution_acceptance_forward_stats_v1' and report['full_tensor_rank_coverage'] is True, 'Complete timevec forward receipt required')
    require(set(report['modules']) == set(case['forward_module_inventory']) and
            {'Bt', 'q_final', 'loss'} <= set(report['modules']), 'Forward module inventory mismatch')
    failures = []
    for module, pairs in report['modules'].items():
        require(set(pairs) == {'A_B', 'AA_B'}, 'Both candidate forward comparisons required')
        for pair, stat in pairs.items():
            exact(stat['atol'], bounds['atol'], 'Fixed module atol')
            exact(stat['rtol'], bounds['rtol'], 'Fixed module rtol')
            require(stat['elements'] > 0 and stat['nonfinite_elements'] == 0, 'Nonfinite/empty forward module')
            rel = scalar(stat['relative_L2'], 'normalized forward L2')
            maximum = scalar(stat['max_absolute'], 'maximum forward error')
            require(rel >= 0 and maximum >= 0, 'Negative forward error')
            require(isinstance(stat['violating_elements'], int) and 0 <= stat['violating_elements'] <= stat['elements'], 'Invalid elementwise violation count')
            if stat['zero_reference_norm']:
                require(maximum == 0 and rel == 0, 'Zero forward reference requires exact zero')
            if stat['violating_elements'] != 0 or rel > bounds['relative_L2_max']:
                failures.append(module+'/'+pair+': registered module FP32 tolerance')
    require(report['physical_q_final']['unit'] == 'm' and report['physical_q_final']['full_tensor_rank_coverage'] is True, 'Physical q_final receipt must be complete and meter-valued')
    exact(report['physical_q_final']['conversion_source_sha256'], case['science']['physical_conversion_sha256'], 'Physical coordinate conversion source')
    for pair in ('A_B', 'AA_B'):
        error = scalar(report['physical_q_final'][pair]['max_position_error_m'], 'Physical trajectory error')
        require(error >= 0, 'Negative physical error')
        if error > bounds['physical_q_final_max_position_error_m']:
            failures.append(pair+': physical q_final position error above 1e-4m')
    return failures


def dev_bank(bank, base, membership, metrics):
    path = hash_ref(bank['csv'], base)
    summary_path = hash_ref(bank['summary'], base)
    summary = load(summary_path)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == len(membership) and len({r['token'] for r in rows}) == len(membership), 'Partial/duplicate canonical CSV')
    require({r['token']: r['log'] for r in rows} == membership, 'Foreign scene/log population')
    require(all(r['status'] == 'ok' for r in rows), 'Failed canonical score rows')
    table = {r['token']: r for r in rows}
    for key in metrics.values():
        vals = [scalar(float(r[key]), key) for r in rows]
        require(all(0 <= v <= 1 for v in vals), 'Out-of-range canonical metric')
        require(math.isclose(sum(vals)/len(vals), scalar(summary['metrics'][key], key), abs_tol=1e-10, rel_tol=0), 'CSV/summary mean mismatch')
    require(summary['scenes'] == len(membership) and summary['failed'] == 0 and summary['valid'] is True, 'Invalid canonical summary')
    return table


def dev_checks(policy, payload, base, bindings, native_provenance):
    exact(payload['protocol'], bindings['dev_protocol'], 'canonical scoring protocol')
    rows = load(hash_ref(payload['index'], base))
    require(isinstance(rows, list), 'Token/log index list required')
    membership = {r['token']: r['log'] for r in rows}
    require(len(rows) == len(membership) == policy['dev']['scenes'] and len(set(membership.values())) == policy['dev']['logs'], 'Canonical dev population incomplete')
    require(all(membership) and all(membership.values()), 'Empty population identifiers')
    require(payload['index']['sha256'] == bindings['dev_protocol']['metric_index_sha256'], 'Population index is not protocol-bound')
    require(set(payload['banks']) == set(ROLES), 'Three paired dev roles required')
    tables = {}
    for role in ROLES:
        require(set(payload['banks'][role]) == set(policy['dev']['variants']), 'Both q0/final required')
        for variant in policy['dev']['variants']:
            bank = payload['banks'][role][variant]
            exact(bank['checkpoint'], native_provenance[role], 'dev/native +500 checkpoint ancestry')
            tables[role, variant] = dev_bank(bank, base, membership, policy['dev']['metrics'])
    logs = sorted(set(membership.values()))
    tokens = sorted(membership)
    sizes = np.array([sum(membership[t] == log for t in tokens) for log in logs])
    rng = np.random.default_rng(policy['dev']['bootstrap']['seed'])
    draws = rng.integers(0, len(logs), size=(policy['dev']['bootstrap']['replicates'], len(logs)))
    failures, pending, stats = [], [], {}
    for variant in policy['dev']['variants']:
        for reference in ('A', 'AA'):
            for metric, field in policy['dev']['metrics'].items():
                delta = {t: float(tables['B', variant][t][field])-float(tables[reference, variant][t][field]) for t in tokens}
                sums = np.array([sum(delta[t] for t in tokens if membership[t] == log) for log in logs])
                boot = sums[draws].sum(1)/sizes[draws].sum(1)*100
                mean = sum(delta.values())/len(tokens)*100
                lower = float(np.quantile(boot, policy['dev']['bootstrap']['lower_quantile']))
                key = variant+'/B-'+reference+'/'+metric
                stats[key] = {'delta_points': mean, 'simultaneous_one_sided_lower_points': lower}
                if metric in policy['dev']['mean_guard_metrics']:
                    margin = policy['dev']['noninferiority_margins_points'][metric]
                    if mean < -margin:
                        failures.append(key+': mean regression beyond registered margin')
                    elif lower < -margin:
                        pending.append(key+': insufficient noninferiority precision; do not widen margin')
    return failures, pending, stats


def required_keys(policy):
    plan = policy['trial_plan']
    keys = set()
    for group in plan['groups']:
        for stage in plan['stages']:
            keys |= {('forward', group, stage, r, 0) for r in range(plan['forward_repeats'])}
            keys |= {('native', group, stage, 0, n) for n in plan[stage+'_native_updates']}
        keys.add(('recovery', group, 'mid', 0, plan['recovery_updates']))
        keys.add(('dev', group, 'mid', 0, plan['dev_after_updates']))
    return keys


def evaluate(registration_path, bindings_path=None, receipt_paths=(), *, source_root=ROOT):
    policy = load(registration_path)
    registration_hash = digest(registration_path)
    result = {'schema': 'execution_acceptance_advisory_v1', 'registration_sha256': registration_hash,
              'passed': False, 'accepted_mode': None, 'formal_authorized': False, 'trainer_consumable': False,
              'advisory_decision': 'pending', 'pending': [], 'failures': [], 'receipts': [],
              'old_strict_results_preserved': [], 'source_sha256': digest(__file__), 'GPU_jobs_started': 0}
    try:
        require(policy['schema'] == 'execution_acceptance_preregistration_v1', 'Foreign registration')
        require(policy['passed'] is False and policy['formal_authorized'] is False and policy['trainer_consumable'] is False, 'Registration grants unauthorized acceptance')
        for ref in (*policy['protected_files'], *policy['background_only']):
            hash_ref(ref, Path(source_root))
        if bindings_path is None:
            result['pending'].append('Exact prospective binding manifest and actual GPU/native/dev receipts absent')
            return result
        bindings = load(bindings_path)
        mode = bindings.get('candidate_mode')
        require(mode not in policy['forbidden_modes'], 'No-recompute remains rejected')
        if not bindings.get('adopted_for_pilot') or not populated({k: v for k, v in bindings.items() if k != 'E3_declaration'}):
            result['pending'].append('Unadopted or incomplete prospective bindings; no candidate passes')
            return result
        validate_bindings(policy, bindings, registration_hash)
        binding_hash = digest(bindings_path)
        result.update(bindings_sha256=binding_hash, candidate_mode=mode, tier=bindings['tier'])
        seen, native_by_key, drift, chain_identities = set(), {}, {}, {}
        staged = []
        for path in receipt_paths:
            path = Path(path)
            receipt = load(path)
            key = (receipt['kind'], receipt['group'], receipt['stage'], receipt['repeat'], receipt['updates'])
            require(key in required_keys(policy) and key not in seen, 'Unexpected/duplicate registered receipt key')
            seen.add(key)
            require(receipt['schema'] == 'execution_acceptance_receipt_v1', 'Foreign receipt schema')
            exact(receipt['registration_sha256'], registration_hash, 'receipt registration')
            exact(receipt['bindings_sha256'], binding_hash, 'receipt bindings')
            require(time_value(receipt['started_utc']) >= time_value(bindings['frozen_utc']), 'Retrospective receipt is not an eligible trial')
            exact(receipt['candidate_mode'], mode, 'receipt mode')
            exact(receipt['tier'], bindings['tier'], 'receipt tier')
            exact(receipt['producer_sha256'], bindings['producer_sha256'][key[0]], 'receipt producer')
            case = bindings['cases'][key[1]+'/'+key[2]]
            exact(receipt['roles'], expected_roles(bindings, case), 'all role source/environment/science/ancestry facts')
            require(set(policy['E1']['exact_checks']) <= set(receipt['exact_checks']) and
                    all(receipt['exact_checks'][k] is True for k in policy['E1']['exact_checks']), 'E1 exact/discrete checks failed or missing')
            if bindings['tier'] == 'E3':
                exact(receipt['E3_declaration'], policy['tiers']['E3']['must_declare'], 'actual E3 declaration')
                require(receipt['native_precision_exercised'] is True, 'BF16 bypass does not qualify E3')
            result['receipts'].append({'key': key, 'sha256': digest(path), 'path': str(path)})
            staged.append((key, receipt, path.parent, case))
        # Native receipts first: dev must independently bind the +500 checkpoints.
        staged.sort(key=lambda x: x[0][0] == 'dev')
        for key, receipt, base, case in staged:
            kind, group, stage, repeat, updates = key
            payload = receipt['payload']
            if kind == 'dev':
                native = native_by_key.get(('native', group, stage, 0, updates))
                require(native is not None, 'Dev lacks actual matching +500 native comparison')
                failures, pending, stats = dev_checks(policy, payload, base, bindings, native['provenance'])
                result['failures'].extend(failures)
                result['pending'].extend(pending)
                result.setdefault('dev', {})[group] = stats
                continue
            comparison_path = hash_ref(payload['comparison'], base)
            comparison = load(comparison_path)
            if kind == 'forward':
                require(comparison['schema'] == 'execution_acceptance_gradient_stats_v1', 'Full gradient-stat receipt needed')
                require(comparison['gradient_inventory_sha256'] == case['gradient_inventory_sha256'] and
                        comparison['full_parameter_rank_coverage'] is True, 'Full gradient inventory required')
                require(all(payload['forward_exact'].get(k) is True for k in policy['tiers'][bindings['tier']]['same_input_forward_exact']), 'Same-input exact forward/RNG/BN checks missing or failed')
                if bindings['tier'] == 'E2':
                    result['failures'].extend(str(key)+': '+f for f in forward_checks(policy, mode, payload, base, case))
                strict = load(hash_ref(payload['original_strict'], base))
                if mode == 'engineering_fast' and strict.get('q0_qfinal_bitwise_passed') is False:
                    require(False, 'P0 exact checks contradict original q0/q_final bitwise failure')
                result['old_strict_results_preserved'].append({'kind': kind, 'key': key, 'sha256': payload['original_strict']['sha256'], 'original': strict})
                failures, stats = numerical_checks(policy, bindings['tier'], comparison, native=False)
            else:
                require(comparison['schema'] == 'native_execution_checkpoint_comparison_v1', 'Frozen native comparator schema required')
                exact(comparison['tool_sha256'], next(r['sha256'] for r in policy['protected_files'] if r['path'].endswith('/compare_execution_checkpoints.py')), 'native comparator source')
                require(comparison.get('integrity_verified') is True and comparison.get('valid_native_inputs') is True and
                        comparison.get('comparison_performed') is True and comparison['coverage']['all_rank_slice_membership_complete'] is True, 'Unverified/partial native comparison')
                require(all(comparison['E1'][p]['passed'] is True and comparison['E1'][p]['failure_count'] == 0 for p in PAIRS), 'Native E1 mismatch')
                for role in ROLES:
                    provenance = comparison['provenance'][role]
                    exact(provenance['source'], bindings['sources'][role]['git_sha'], 'actual checkpoint source')
                    exact(provenance['execution_mode'], bindings['sources'][role]['execution_mode'], 'actual checkpoint mode')
                    exact(provenance['optimizer_groups_sha256'], case['optimizer_groups_sha256'], 'native optimizer groups')
                    exact(provenance['ranks'], policy['trial_plan']['world_size'], 'native rank coverage')
                    exact(provenance['completed'], case['origin']['completed']+updates, 'native update/ancestry offset')
                    require(provenance['payload_hashes_verified'] is True, 'Payload integrity must include every native rank')
                    native_identity_checks(provenance, bindings['sources'][role], case['science'], case['origin'], base)
                origin = comparison['provenance']['origin']
                exact(origin['identity'], case['origin']['identity'], 'actual native origin identity')
                exact(origin['COMPLETE_sha256'], case['origin']['COMPLETE_sha256'], 'actual native origin hash')
                exact(origin['completed'], case['origin']['completed'], 'actual native origin step')
                require(origin['payload_hashes_verified'] is True, 'Native origin payload not verified')
                result['old_strict_results_preserved'].append({'kind': kind, 'key': key, 'sha256': payload['comparison']['sha256'],
                    'original_strict': comparison['original_strict'], 'fixed_numeric_results': comparison['fixed_numeric_results']})
                if kind == 'recovery':
                    boundary = load(hash_ref(payload['load_boundary_receipt'], base))
                    require(payload.get('load_boundary_exact') is True, 'Native restore boundary must be bitwise exact')
                    require(boundary.get('native_payload_hashes_verified') is True and boundary.get('completed') == case['origin']['completed']+updates and
                            all(boundary.get(k) is True for k in ('model_FP32', 'master_FP32', 'Adam', 'BN_float', 'RNG', 'discrete_state')), 'Incomplete/nonexact actual restore boundary')
                    exact(boundary['registration_sha256'], registration_hash, 'boundary registration')
                    exact(boundary['bindings_sha256'], binding_hash, 'boundary binding')
                failures, stats = numerical_checks(policy, bindings['tier'], comparison, native=True, inventory=case['category_inventory'])
                native_by_key[key] = comparison
                if kind == 'native':
                    chain = {r: comparison['provenance'][r]['identity'] for r in ROLES}
                    prior = chain_identities.setdefault((group, stage), chain)
                    exact(chain, prior, 'one continuous native A/AA/B chain per stage')
                if kind == 'native' and stage == 'mid' and updates in policy['numeric']['stage_drift']['updates']:
                    for scope, stat in stats.items():
                        drift.setdefault(group+'/'+scope, {})[updates] = stat['AB_over_AA_relL2_plus_floor']
            result['failures'].extend(str(key)+': '+f for f in failures)
            result.setdefault('numerical', []).append({'key': key, 'stats': stats})
        result['stage_drift'] = {}
        drift_steps = policy['numeric']['stage_drift']['updates']
        for scope, values in drift.items():
            if set(values) != set(drift_steps):
                continue
            x = np.array(drift_steps, dtype=float)
            x = (x-x.min())/(x.max()-x.min())
            y = np.array([values[n] for n in drift_steps])
            slope = float(np.dot(x-x.mean(), y-y.mean())/np.dot(x-x.mean(), x-x.mean()))
            result['stage_drift'][scope] = {'AB_over_AA_plus_floor': values, 'OLS_slope': slope}
            if bindings['tier'] == 'E2' and slope > policy['numeric']['stage_drift']['E2_max_positive_OLS_slope_per_468_updates']:
                result['failures'].append(scope+': increasing stage excess beyond registered slope')
        result['pending'].extend('Missing actual registered receipt: '+str(k) for k in sorted(required_keys(policy)-seen))
        if not result['failures'] and not result['pending']:
            result['advisory_decision'] = 'evidence_complete_for_review'
    except (InvalidEvidence, OSError, KeyError, TypeError, ValueError, StopIteration) as error:
        result['failures'].append(type(error).__name__+': '+str(error))
    if result['failures']:
        result['advisory_decision'] = 'not_supported'
    return result


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, default=REGISTRATION)
    parser.add_argument('--bindings', type=Path)
    parser.add_argument('--receipt', type=Path, action='append', default=[])
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--write-binding-template', type=Path)
    args = parser.parse_args()
    if args.write_binding_template:
        write_new(args.write_binding_template, binding_template(load(args.registration), digest(args.registration)))
        print(json.dumps({'template': str(args.write_binding_template), 'adopted_for_pilot': False, 'GPU_jobs_started': 0}))
        return
    parser.error('--output is required') if args.output is None else None
    report = evaluate(args.registration, args.bindings, args.receipt, source_root=args.source_root)
    write_new(args.output, report)
    print(json.dumps({k: report[k] for k in ('advisory_decision', 'passed', 'formal_authorized', 'GPU_jobs_started')}))
    sys.exit(2 if report['advisory_decision'] == 'not_supported' else 0)


if __name__ == '__main__':
    main()
