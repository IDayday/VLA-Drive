"""Complete fixed-population metrics and paired physical-log bootstrap."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from starVLA.rl.qwen_world_postcot.contracts import fingerprint
from starVLA.rl.qwen_world_postcot.data import POPULATION
from starVLA.rl.qwen_world_postcot.checkpoint import atomic_json
from tools.qwen_world_postcot.evaluate import validate_scene_row


METRICS = ('score', 'no_at_fault_collisions', 'drivable_area_compliance',
           'time_to_collision_within_bound', 'ego_progress', 'comfort',
           'driving_direction_compliance')


def physical_log(log):
    return re.sub(r'_\d+_\d+$', '', log)


def load(root, *, population=None):
    root = Path(root); identity = json.loads((root/'identity.json').read_text())
    body = dict(identity); saved = body.pop('sha256', None)
    if (fingerprint(body) != saved or body.get('schema') != 'world_postcot_evaluation_v1'
            or type(body.get('qualification')) is not bool):
        raise ValueError('Evaluation identity schema or seal changed')
    population = json.loads(POPULATION.read_text()) if population is None else population
    expected = population['dev_tokens' if identity['cohort'] == 'DEV' else 'navtest_tokens']
    tokens = identity['tokens']
    target = expected[:len(tokens)] if identity['qualification'] else expected
    if (identity['cohort'] not in ('DEV', 'NAVTEST') or not tokens or tokens != target
            or len(tokens) != len(set(tokens)) or identity['count'] != len(tokens)
            or identity['token_sha256'] != fingerprint(tokens)):
        raise ValueError('A complete result must retain its fixed full population or explicit diagnostic prefix')
    existing = {path.stem for path in (root/'scenes').glob('*.json')}
    if existing != set(tokens):
        raise ValueError('Missing or foreign scene receipts: no partial/filtered population summary')
    rows = {}
    for token in tokens:
        row = validate_scene_row(json.loads((root/'scenes'/f'{token}.json').read_text()),
            token=token, identity_sha256=saved, protocol_sha256=identity['protocol']['sha256'],
            expected_log=population['token_logs'][token])
        rows[token] = row
    return identity, rows


def compare(identity, rows, base_identity, base, *, bootstrap=2000, seed=42):
    for key in ('cohort', 'token_sha256', 'data_sha256', 'seed', 'inference', 'qualification', 'numerics'):
        if identity.get(key) != base_identity.get(key):
            raise ValueError('Paired evaluation contract differs: '+key)
    if identity['protocol']['sha256'] != base_identity['protocol']['sha256'] or set(base) != set(rows):
        raise ValueError('Paired analysis needs the exact same official protocol and population')
    return paired_statistics(rows,base,bootstrap=bootstrap,seed=seed)


def paired_statistics(rows,base,*,bootstrap=2000,seed=42):
    """Statistics after a caller's explicit same-method/legacy protocol audit."""
    if set(rows)!=set(base)or not rows:raise ValueError('Paired scene denominator must be exactly matched')
    groups = {}; repaired = new_failure = high_to_zero = base_zero = base_high = 0
    safety = {name: dict(new_failure=0, repaired_failure=0, baseline_pass=0, baseline_failure=0)
              for name in METRICS[1:4]}
    for token, row in rows.items():
        original = base[token]
        if (row['log'] != original['log'] or row['feedback']['cache']['sha256'] != original['feedback']['cache']['sha256']):
            raise ValueError('Paired scene/log/official reference context changed')
        metrics, reference = row['feedback']['metrics'], original['feedback']['metrics']
        value, previous = metrics['score'], reference['score']
        groups.setdefault(physical_log(row['log']), []).append(value-previous)
        base_zero += previous <= 1e-8; base_high += previous >= .8
        repaired += previous <= 1e-8 and value > 1e-8
        new_failure += previous > 1e-8 and value <= 1e-8
        high_to_zero += previous >= .8 and value <= 1e-8
        for name, counts in safety.items():
            old_pass, new_pass = reference[name] >= 1-1e-8, metrics[name] >= 1-1e-8
            counts['baseline_pass'] += int(old_pass); counts['baseline_failure'] += int(not old_pass)
            counts['new_failure'] += int(old_pass and not new_pass)
            counts['repaired_failure'] += int(not old_pass and new_pass)
    keys = list(groups); rng = np.random.default_rng(seed); estimates = []
    for _ in range(bootstrap):
        sampled = rng.integers(0, len(keys), len(keys))
        estimates.append(float(np.mean(np.concatenate([groups[keys[index]] for index in sampled]))))
    total = len(rows)
    return dict(mean_delta=float(np.mean([value for values in groups.values() for value in values])),
        CI95=np.quantile(estimates, [.025, .975]).tolist(), bootstrap=bootstrap, bootstrap_seed=seed,
        physical_groups=len(keys), repaired_zero=int(repaired), new_zero=int(new_failure),
        high_to_zero=int(high_to_zero), scene_denominator=total, baseline_zero_denominator=int(base_zero),
        baseline_nonzero_denominator=int(total-base_zero), baseline_high_denominator=int(base_high),
        repaired_zero_rate=None if not base_zero else repaired/base_zero,
        new_zero_rate=None if total == base_zero else new_failure/(total-base_zero),
        high_to_zero_rate=None if not base_high else high_to_zero/base_high,
        safety_transitions=safety, all_scores_units='0..1; PDMS_display_explicitly_times100')


def summarize(evaluation, baseline=None, *, population=None,legacy_binding=None):
    identity, rows = load(evaluation, population=population)
    result = dict(status='COMPLETE', cohort=identity['cohort'], scenes=len(rows),
        physical_logs=len({physical_log(row['log']) for row in rows.values()}),
        seed=identity['seed'], qualification=identity['qualification'],
        evaluation_identity_sha256=identity['sha256'], model_sha256=identity['model_sha256'],
        protocol_sha256=identity['protocol']['sha256'], token_sha256=identity['token_sha256'],
        means={name: float(np.mean([row['feedback']['metrics'][name] for row in rows.values()])) for name in METRICS},
        evaluation=str(Path(evaluation).resolve()), invalid_scenes=0, scene_denominator=len(rows))
    result['PDMS'] = 100*result['means']['score']
    if baseline:
        base_identity, base = load(baseline, population=population)
        result['paired'] = compare(identity, rows, base_identity, base)
    if legacy_binding:
        from tools.qwen_world_postcot.legacy_baseline import compare_registered
        result['legacy_native_comparisons']=compare_registered(identity,rows,legacy_binding,population=population)
    result['sha256'] = fingerprint(result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--evaluation', required=True); parser.add_argument('--baseline')
    parser.add_argument('--legacy-baseline-binding',help='Explicit pinned native-v1 adapter; not a general mismatch override')
    parser.add_argument('--output', required=True); args = parser.parse_args()
    result = summarize(args.evaluation, args.baseline,legacy_binding=args.legacy_baseline_binding)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if json.loads(output.read_text()) != result:
            raise FileExistsError('Keep previous result immutable; choose a new version')
    else:
        atomic_json(output, result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
