"""Reuse and validate the existing whole-log RL holdout; no score-based selection.

Full token manifests stay in the private runtime directory. The public receipt
contains hashes, counts and the historical exposure caveats only.
"""
import argparse
import csv
import json
from pathlib import Path

from starVLA.rl.safe_flow_improvement.contracts import PopulationContract, file_sha, fingerprint
from starVLA.rl.flow_grpo.transactions import atomic_json


def select(tokens, namespace, count):
    return sorted(tokens, key=lambda token: fingerprint([namespace, token]))[:count]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--navtest-csv', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = json.loads(Path(args.source).read_text())
    train, dev = source['train_tokens'], source['dev_tokens']
    assert len(train) == len(set(train)) and len(dev) == len(set(dev))
    assert set(source['dev_logs']) == {source['token_logs'][t] for t in dev}
    rows = list(csv.DictReader(Path(args.navtest_csv).open()))
    # The canonical score CSV may have an aggregate row. It is never a scene.
    rows = [row for row in rows if row['token'] != 'average']
    navtest = [row['token'] for row in rows]
    assert navtest and len(navtest) == len(set(navtest))
    token_logs = dict(source['token_logs'])
    token_logs.update({row['token']: row['log_name'] for row in rows})
    population = PopulationContract(frozenset(train), frozenset(dev), frozenset(navtest), token_logs)
    screen = select(train, 'flow_safe_reference_screen_v1', 8192)
    random_learning = select(screen, 'flow_safe_learning_random_v1', 2048)
    random_diagnostic = select(train, 'flow_safe_diagnostic_random_v1', 256)
    calibration = select(screen, 'flow_safe_eta_calibration_v1', 256)
    for tokens in (screen, random_learning, random_diagnostic, calibration):
        population.require_training(tokens, purpose='pre-registered training cohort selection')
    manifest = {
        'schema': 'flow_safe_population_v1', 'train_tokens': train, 'dev_tokens': dev,
        'navtest_tokens': navtest, 'token_logs': token_logs,
        'reference_screen_tokens': screen, 'learning_random_tokens': random_learning,
        'diagnostic_random_tokens': random_diagnostic, 'calibration_tokens': calibration,
        'dev_audit_tokens': select(dev, 'flow_safe_dev_audit_v1', 512),
        'targeted_cohorts_status': 'PENDING_FIXED_SFT_REFERENCE_FEEDBACK',
        'source_sha256': file_sha(args.source), 'navtest_csv_sha256': file_sha(args.navtest_csv),
        'selection': 'Existing legal log holdout reused. Hash-ordered uniform screening and random cohorts; targeted cohorts registered separately before search.',
        'reference_seeds': [1101, 1102, 1103, 1104],
        'evaluation_seeds': [2101, 2102, 2103, 2104, 2105],
        'diagnostic_root_seeds': list(range(3101, 3117)),
        'train_seeds': [42, 43],
        'sft_unseen': False, 'old_full_navtrain_rl_unseen': False,
        'new_rl_updates_unseen_dev': True, 'navtest_historically_observed': True,
    }
    manifest['identity_sha256'] = fingerprint(manifest)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'manifest.json'
    if target.exists() and json.loads(target.read_text()) != manifest:
        raise ValueError('Refusing to replace a different population manifest')
    atomic_json(target, manifest)
    receipt = {key: manifest[key] for key in (
        'schema', 'source_sha256', 'navtest_csv_sha256', 'identity_sha256', 'selection',
        'reference_seeds', 'evaluation_seeds', 'diagnostic_root_seeds', 'train_seeds',
        'sft_unseen', 'old_full_navtrain_rl_unseen', 'new_rl_updates_unseen_dev',
        'navtest_historically_observed', 'targeted_cohorts_status')}
    receipt['populations'] = {name: {'scenes': len(tokens), 'logs': len({token_logs[t] for t in tokens}),
                                     'token_list_sha256': fingerprint(tokens)}
                              for name, tokens in (('RL_TRAIN', train), ('RL_DEV', dev), ('NAVTEST', navtest))}
    receipt['cohort_counts'] = {name: len(manifest[name]) for name in (
        'reference_screen_tokens', 'learning_random_tokens', 'diagnostic_random_tokens',
        'calibration_tokens', 'dev_audit_tokens')}
    receipt['dev_log_fraction'] = len(source['dev_logs']) / len(set(source['token_logs'].values()))
    receipt['limited_dev_logs_caveat'] = '16 whole logs retained by the existing legal split; uncertainty is clustered by log. This is not a newly drawn 10-15% holdout.'
    atomic_json(output / 'receipt.json', receipt)
    print(json.dumps(receipt, sort_keys=True))


if __name__ == '__main__':
    main()
