"""User-requested, matched-exposure Navtest observation; never selects a model.

The unchanged locked sampler calls its protocol field FINAL_NAVTEST. This
adapter retains that compatibility token, but records an explicit interim
schema, purpose and incomplete formal budget. It cannot make a formal endpoint
lock or alter the trainer's selection schedule.
"""
import argparse
import datetime
from pathlib import Path

from scripts.analysis.flow_safe_improvement.sample_locked_navtest import locked_contract
from starVLA.rl.constrained_flow_noise.contracts import SFT_SHA256, VARIANTS, file_sha, fingerprint, require_dispatch
from starVLA.rl.flow_grpo.transactions import atomic_json
from .evaluate_final import evaluate_one
from .lock_final import SFT, read, sealed

SCHEMA = 'cfn_user_requested_interim_navtest_v1'
RAW = 25398


def validate_boundary(boundary, identity, variant, population, budget_sha):
    if (boundary['phase'] != 'ready' or boundary['raw_exposure'] != RAW
            or boundary['next_inner'] != 0 or boundary['world'] != 8):
        raise ValueError('A complete, matched-exposure native boundary is required')
    if (identity['qualification_only'] or identity['variant'] != variant
            or identity['SFT_sha256'] != SFT_SHA256 or identity['training_seed'] != 42
            or identity['evaluation_seed'] != 2101 or identity['budget_sha256'] != budget_sha
            or identity['population_identity'] != population['identity_sha256']):
        raise ValueError('Qualification or unmatched training lineage')


def make_interim_lock(root, population, c_run, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError('The observation lock is immutable')
    output.parent.mkdir(parents=True, exist_ok=True)
    budget = sealed(root/'formal_budget.json')
    if RAW not in budget['milestones'] or budget['raw_exposures_per_variant'] != 203184:
        raise ValueError('Registered exposure schedule changed')
    baseline = sealed(root/'evaluation/SFT_DEV_original/scores/manifest.json')
    sampler = Path(__file__).parents[1]/'flow_safe_improvement/sample_locked_navtest.py'
    models = {}
    for name in ('SFT', *VARIANTS):
        if name == 'SFT':
            export, evidence = SFT, dict(raw_exposure=0, role='permanent_SFT_baseline')
        else:
            run = Path(c_run) if name == 'C_SDE' else root/f'formal_{name}_seed42_v1'
            choices = list((run/'checkpoints').glob(f'update*_raw{RAW:06d}'))
            if len(choices) != 1 or not (choices[0]/'COMPLETE').is_file():
                raise ValueError('Exactly one complete registered milestone checkpoint required')
            checkpoint = choices[0]
            checksums = read(checkpoint/'files.json')
            for filename in ('boundary.json', 'identity.json'):
                if file_sha(checkpoint/filename) != checksums[filename]:
                    raise ValueError('Native checkpoint metadata checksum mismatch')
            boundary, identity = read(checkpoint/'boundary.json'), sealed(checkpoint/'identity.json')
            validate_boundary(boundary, identity, name, population, file_sha(root/'formal_budget.json'))
            if sealed(run/'run_identity.json')['sha256'] != identity['sha256']:
                raise ValueError('Checkpoint belongs to another run')
            export = run/f'export_raw{RAW:06d}'
            if not (export/'COMPLETE').is_file():
                raise ValueError('Wait for the trainer\'s complete original ODE export')
            exported = read(export/'cfn_export_identity.json')
            if (exported['run_identity']['sha256'] != identity['sha256']
                    or exported['noise_head_in_export']
                    or not exported['upstream_preserved_bytewise_tensor_values']):
                raise ValueError('Not the unchanged-upstream original ODE export')
            evidence = dict(raw_exposure=RAW, update=boundary['update'], checkpoint=str(checkpoint),
                            checkpoint_files_sha256=file_sha(checkpoint/'files.json'),
                            run_identity=identity['sha256'], role='user_requested_matched_interim_observation')
            if (run/'execution_identity.json').exists():
                evidence['numerical_execution_sha256'] = read(run/'execution_identity.json')['sha256']
        weight = file_sha(export/'pytorch_model.pt')
        if name == 'SFT' and weight != SFT_SHA256:
            raise ValueError('SFT changed')
        if name != 'SFT' and weight != exported['export_sha256']:
            raise ValueError('ODE export payload changed')
        models[name] = dict(export=str(export), weight_sha256=weight,
                            config_sha256=file_sha(export/'config.yaml'),
                            sampler_source_sha256=file_sha(sampler), **evidence)
    datalist = Path('/mnt/project/DriveDreamer-Policy/test_meta.json')
    tokens = read(datalist)
    if len(tokens) != 12146 or len(set(tokens)) != 12146 or set(tokens) != set(population['navtest_tokens']):
        raise ValueError('Complete Navtest input membership required')
    lock = dict(schema=SCHEMA, status='LOCKED_BEFORE_FINAL_NAVTEST',
                legacy_status_semantics='Frozen sampler compatibility token; this is an INTERIM observation, not a final endpoint',
                purpose='Explicit user request: 测一下测试集结果', formal_budget_complete=False,
                utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                checkpoint_rule='Both RL variants at the predetermined first 25398 raw-exposure milestone; no score-based selection',
                model_selection_performed=False, training_or_selection_rule_changed=False,
                models=models, unique_export_count=3, inference_seeds=[2101], training_seeds=[42],
                population_identity=population['identity_sha256'], navtest_tokens_sha256=fingerprint(population['navtest_tokens']),
                deployment_execution_profile=baseline['sample_manifest']['execution_profile'],
                v1_protocol_sha256=baseline['protocol_sha256'],
                data_root='/mnt/project/DriveDreamer-Policy-paired/runs/paired_full_assets_v1/dataset',
                navtest_datalist=str(datalist), navtest_datalist_sha256=file_sha(datalist),
                v1_cache='/mnt/project/DriveDreamer-Policy-full-navtrain/runs/full_navtrain_epoch1/metric_cache_navtest_v1',
                historically_observed_Navtest_not_fully_blind=True,
                this_observation_must_be_disclosed_in_final_report=True, source_sha256=file_sha(__file__))
    lock['sha256'] = fingerprint(lock)
    atomic_json(output, lock)
    for name in models:
        locked_contract(output, name, population)
    return lock


def main():
    parser = argparse.ArgumentParser(__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    lock = sub.add_parser('lock')
    for name in ('run-root', 'population', 'c-run', 'output'):
        lock.add_argument('--'+name, required=True)
    run = sub.add_parser('evaluate')
    for name in ('lock', 'model', 'population', 'output', 'baseline-scores'):
        run.add_argument('--'+name, required=True)
    run.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    require_dispatch()
    if args.command == 'lock':
        result = make_interim_lock(args.run_root, read(args.population), args.c_run, args.output)
        print(result['sha256'], flush=True)
    else:
        observation = sealed(args.lock)
        if (observation['schema'] != SCHEMA or observation['formal_budget_complete']
                or observation['model_selection_performed'] or args.model not in ('SFT', *VARIANTS)
                or not 1 <= args.workers <= 32):
            raise ValueError('Not this locked user-requested observation')
        context = dict(stage='user_requested_interim', lock_sha256=observation['sha256'],
                       matched_RL_raw_exposure=RAW, formal_budget_fraction=RAW/203184,
                       final_confirmation=False, checkpoint_selection_uses_this_report=False)
        result = evaluate_one(args.lock, args.model, args.population, args.output,
                              args.baseline_scores, args.workers, baseline_wait_seconds=3600,
                              observation_context=context)
        print(dict(model=args.model, scenes=result['scenes'], PDMS=100*result['metrics']['score']['current']), flush=True)


if __name__ == '__main__':
    main()
