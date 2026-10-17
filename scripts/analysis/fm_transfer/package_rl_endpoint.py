"""Package existing, audited 20k RL evidence for review; no model computation.

This supplements the existing analysis controllers. It never starts, retries,
selects or evaluates a policy. A fresh output directory is required; COMPLETE
is written last. Partial packages remain evidence, not a published result.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import time


ROOT = Path('/mnt/project/fm-transfer-20261016')
VARIANTS = {'A_DDP_TIME': ('A', 20000, 96),
            'B_SOURCE_ACTION_DIT': ('B', 100000, 240)}
MODES = ['E_NATIVE', 'E_SHIFT5']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(a, b):
    require(abs(a - b) <= 1e-12, 'Independent numerical summaries disagree')


def collect(root, variant):
    """Validate actual registered evidence before creating any output."""
    root = Path(root)
    label, sft_update, projections = VARIANTS[variant]
    analysis = root / 'analysis'
    stem = f'{variant}_RL_update020000'
    audit_path = root / f'qualification/{label}_RL20k_analysis/completion.json'
    audit = read(audit_path)
    require(audit['status'] == f'PASS_COMPLETE_{label}_RL_NOT_FINAL_NAVTEST'
            and audit['all_eight_training_results_complete'], 'Endpoint exit audit missing')
    require(not audit['training_exit']['run_processes']
            and not any(x['same_process'] for x in audit['training_exit']['pinned']),
            'Training exit was not verified')
    train_path = analysis / f'{variant}_RL_complete_training.json'
    population_path = analysis / f'{stem}_population.json'
    gradient_path = analysis / f'{stem}_saved_batch.json'
    train, population, gradient = map(read, (train_path, population_path, gradient_path))
    run = train['run_identity']
    require(train['status'] == 'TRAINING_COMPLETE', 'Only the fixed 20k endpoint is eligible')
    expected = dict(actual_global_optimizer_updates=20000, fresh_global_rollouts=5000,
                    scene_exposures=40000, new_candidates=320000)
    require(all(train['cumulative'][k] == v for k, v in expected.items()), 'Wrong effective budget')
    ledger = train['cost']['invocation_ledger']
    require(ledger['incomplete_invocations'] == 0
            and ledger['unclosed_invocation_worker_wall_seconds'] == 0,
            'Training cost ledger remains open')
    checkpoint = root / f'runs/{variant}_R_SOURCE_seed42_v1/checkpoints/update020000_inner0'
    seal = read(checkpoint / 'COMPLETE')
    require(all(seal[k] == v for k, v in dict(phase='COMPLETE', world=8, updates=20000,
                                            next_inner=0).items()), 'Wrong checkpoint phase')
    require(seal['files_sha256'] == sha(checkpoint / 'files.json')
            == population['checkpoint']['files_sha256']
            == audit['optimizer']['checkpoint_files_sha256'], 'Checkpoint evidence disagrees')
    require(audit['optimizer']['adam_states'] == 2 * projections
            and audit['optimizer']['adam_steps'] == [20000.0]
            and audit['optimizer']['scheduler_last_epoch'] == 20000, 'Wrong optimizer endpoint')
    require(population['run_identity'] == gradient['run_identity'] == run
            and population['checkpoint']['phase'] == 'COMPLETE'
            and population['population']['fresh_candidates'] == 320000, 'Mixed run or population')
    require(gradient['optimizer_updates'] == 20000 and gradient['collection'] == 5000
            and gradient['adapters_unchanged']
            and gradient['new_candidates'] == gradient['new_scoring_calls'] == 0,
            'Not the registered read-only saved batch')
    export_path = root / f'exports/{variant}/RL_update020000/export.json'
    export = read(export_path)
    require(export == read(export_path.parent / 'COMPLETE')
            and export['stage'] == 'RL' and export['updates'] == 20000
            and export['run_identity'] == run and not export['qualification']
            and export['new_SFT']['updates'] == sft_update
            and export['new_SFT']['config']['variant'] == variant
            and export['frozen_upstream_exact']
            and len(export['changed_attention_from_own_SFT']) == projections,
            'Export does not belong to the fixed route and own SFT')
    copies, checks, evaluations = [], [], {}
    for cohort, scenes in [('DEV', 1696), ('TRAIN_DIAGNOSTIC', 512)]:
        base = f'{stem}_{cohort}'
        evaluation = root / 'evaluations' / base
        reference = root / 'evaluations' / f'{variant}_SFT_update{sft_update:06d}_{cohort}'
        paired = read(analysis / f'{base}_paired_own_SFT.json')
        safety = read(analysis / f'{base}_safety_transitions.json')
        previous = read(analysis / f'{base}_vs15k.json')
        plain = read(analysis / f'{base}.json')
        identity = read(evaluation / 'manifest.json')
        require(paired['identity'] == previous['identity'] == plain['identity'] == identity
                and identity['export_receipt_sha256'] == sha(export_path)
                and identity['weight_sha256'] == export['weight_sha256']
                and identity['seed'] == 2101 and identity['modes'] == MODES
                and not identity['qualification_only'], 'Mixed evaluation identity')
        require(all(x['status'] == 'COMPLETE_POPULATION' and x['scenes'] == scenes
                    and x['cost']['complete_cost_accounting'] for x in (plain, paired, previous)),
                'Incomplete evaluation or costs')
        require(paired['reference'] == safety['reference'] == str(reference)
                and previous['reference'] == str(root / 'evaluations' /
                    f'{variant}_RL_update015000_{cohort}')
                and safety['scenes'] == scenes
                and safety['evaluation_manifest_sha256'] == sha(evaluation / 'manifest.json'),
                'Wrong paired reference or safety population')
        worker = audit['cohorts'][cohort]
        require(worker['all_pinned_workers_exited'] and worker['closed_worker_invocations'] == 8
                and worker['raw_original_checks_passed'] == 8
                and worker['actual_scene_count'] == scenes,
                'Missing complete raw-observation worker audit')
        recorded_manifest_sha = worker.get('manifest_sha256')
        require((label == 'A' and recorded_manifest_sha is None)
                or recorded_manifest_sha == sha(evaluation / 'manifest.json'),
                'Worker audit manifest differs')
        ranks = [read(evaluation / f'rank{rank}.json') for rank in range(8)]
        require(all(row['status'] == 'ASSIGNED_COMPLETE' and row['direct_original_check']
                    and row['identity_sha256'] == identity['sha256'] for row in ranks)
                and sum(row['new'] + row['reused'] for row in ranks) == scenes,
                'Actual rank evidence differs from the evaluation identity')
        for mode in MODES:
            row = paired['modes'][mode]
            subsets = safety['modes'][mode]['subsets']
            all_scenes = subsets['all_scenes']
            for metric, value in row['mean'].items():
                equal(value, all_scenes['metrics'][metric]['current']['mean'])
                equal(value, plain['modes'][mode]['mean'][metric])
            comparison = row['paired_comparison']
            equal(comparison['PDMS_log_cluster_bootstrap']['mean'],
                  all_scenes['metrics']['score']['delta']['mean'])
            equal(comparison['positive_gain_sum'], all_scenes['canonical_positive_gain_sum_raw'])
            equal(comparison['negative_degradation_sum'], all_scenes['canonical_negative_degradation_sum_raw'])
            require(comparison['reference_safe_new_joint_violation']
                    == subsets['SFT_safe_new_joint_violation']['scenes'], 'Fixed-safe count differs')
            for metric, name in [('time_to_collision_within_bound', 'TTC'),
                                 ('drivable_area_compliance', 'DAC')]:
                require(comparison['safety_transitions'][metric]['new_violation']
                        == subsets[f'new_{name}']['scenes'], 'New safety count differs')
            require(comparison['safety_transitions']['time_to_collision_within_bound']['recovery']
                    == subsets['recovered_TTC']['scenes'], 'Safety recovery count differs')
            checks.append(dict(cohort=cohort, mode=mode, independent_analysis_matches=True,
                               tolerance=1e-12))
        evaluations[cohort] = dict(cost=paired['cost'], worker_audit=worker)
        for source_suffix, target_suffix in [('', ''), ('_paired_own_SFT', '_vs_own_SFT'),
                                            ('_vs15k', '_vs15k'),
                                            ('_safety_transitions', '_safety_transitions')]:
            copies.append((analysis / f'{base}{source_suffix}.json',
                           f'{label}_RL_update020000_{cohort}{target_suffix}.json'))
    public_population, public_gradient = copy.deepcopy(population), copy.deepcopy(gradient)
    public_population['checked_behavior_payload_count'] = len(public_population.pop('checked_behavior_payloads'))
    evidence = public_gradient['transfer_diagnostic_identity']
    evidence['verified_scene_score_count'] = len(evidence.pop('verified_scene_scores'))
    derived = []
    for path, data, name in [(population_path, public_population, 'population'),
                             (gradient_path, public_gradient, 'saved_batch')]:
        data['private_original_sha256'] = sha(path)
        data['public_omission'] = 'Individual private scene payload/score identities omitted; original retained locally.'
        derived.append((f'{label}_RL_update020000_{name}.json', data))
    copies.extend([(train_path, f'{label}_RL_complete_training.json'),
                   (analysis / f'{label}_RL20k_DEV_native_minus_shift5.json',
                    f'{label}_RL20k_DEV_native_minus_shift5.json'),
                   (root / f'resources/verify_{label}_RL20k_completion.py',
                    f'reproduce_{label}_RL20k_completion.py')])
    # A's already reviewed v2 changed plot spacing only; B uses the queued filename.
    plot = f'{label}_RL_DEV_through_20k' + ('_v2' if label == 'A' else '')
    copies.extend((analysis / f'{plot}.{ext}', f'{label}_RL_DEV_through_20k.{ext}')
                  for ext in ('png', 'pdf', 'json'))
    require(all(p.is_file() for p, _ in copies), 'An endpoint artifact is still missing')
    verifier_sha = sha(root / f'resources/verify_{label}_RL20k_completion.py')
    recorded_verifier_sha = audit.get('evidence_script_sha256')
    # The already published A audit predates this field. Do not manufacture a
    # historical hash for it; B's registered audit must supply the exact hash.
    require((label == 'A' and recorded_verifier_sha is None)
            or verifier_sha == recorded_verifier_sha,
            'Completion verifier differs from executed source')
    receipt = dict(schema='fm_transfer_RL_endpoint_package_v1', label=label, variant=variant,
        status=f'COMPLETE_{label}_RL_EVIDENCE_NOT_FINAL_NAVTEST', run_identity=run,
        captured_unix=time.time(), packager_sha256=sha(__file__), checkpoint=population['checkpoint'],
        export=export, completion_audit=audit, completion_audit_sha256=sha(audit_path),
        completion_verifier_current_sha256=verifier_sha,
        completion_verifier_historical_hash_available=recorded_verifier_sha is not None,
        evaluations=evaluations, checks=checks, population_summary=population['population'],
        training_cost=train['cost'], new_candidates=0, new_scores=0, optimizer_updates=0,
        source_scope='Packaging only; numerical training/evaluation source identities are preserved in each input.',
        copied_analysis=[])
    return copies, derived, receipt


def package(root, variant, output):
    copies, derived, receipt = collect(root, variant)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for source, name in copies:
        shutil.copyfile(source, output / name)
        require(sha(source) == sha(output / name), 'Copied evidence differs')
        receipt['copied_analysis'].append(dict(path=name, sha256=sha(output / name)))
    for name, data in derived:
        with (output / name).open('x') as f:
            json.dump(data, f, indent=2); f.write('\n')
        receipt['copied_analysis'].append(dict(path=name, sha256=sha(output / name),
                                              private_original_sha256=data['private_original_sha256']))
    with (output / 'execution.json').open('x') as f:
        json.dump(receipt, f, indent=2); f.write('\n')
    with (output / 'COMPLETE.json').open('x') as f:
        json.dump(dict(status=receipt['status'], execution_sha256=sha(output / 'execution.json')), f)
        f.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.validate_only:
        copies, derived, receipt = collect(args.root, args.variant)
    else:
        if args.output is None:
            parser.error('--output must be a new review directory')
        receipt = package(args.root, args.variant, args.output)
    print(json.dumps(dict(status=receipt['status'], mode='VALIDATE_ONLY' if args.validate_only else 'PACKAGED',
                          crosschecks=len(receipt['checks']), new_candidates=0, optimizer_updates=0)))


if __name__ == '__main__':
    main()
