"""Lock both completed transfer routes before any final Navtest inference.

Reuse the existing asset preparation, own-base exports and milestone analyses.
This module never launches a model, selects a score maximum or changes a trainer.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path

from scripts.analysis.fm_transfer.milestone_controller import jobs, initial_A_reused
from scripts.analysis.simwam_recipe.final_navtest import prepare as prepare_assets
from starVLA.rl.fm_transfer.contracts import ROOT, CONFIGS, config as sft_config, require_dispatch
from starVLA.rl.simwam_recipe.contracts import POPULATION, file_sha, fingerprint, load_recipe
from starVLA.rl.simwam_recipe.recovery_state import validate_ranks
from starVLA.rl.flow_grpo.transactions import atomic_json

EVALUATION_SOURCE = ROOT / 'source/45d2318d2dec265583860a26f891d8b79bb6068a'


def authenticated(path):
    value = json.loads(Path(path).read_text())
    body = dict(value)
    sha = body.pop('sha256')
    if fingerprint(body) != sha:
        raise ValueError('Changed identity: ' + str(path))
    return value


def required_files(root):
    root = Path(root)
    for variant, cfg in CONFIGS.items():
        for stage, update in [('SFT', cfg['max_updates']), ('RL', 20000)]:
            family = 'SFT' if stage == 'SFT' else 'R_SOURCE'
            run = root / 'runs' / f'{variant}_{family}_seed42_v1'
            checkpoint = run / 'checkpoints' / (f'update{update:06d}' + ('_inner0' if stage == 'RL' else ''))
            yield run / 'identity.json'
            yield checkpoint / 'COMPLETE'
            for rank in range(8):
                yield run / f'result_rank{rank}.json'
            yield root / 'exports' / variant / f'{stage}_update{update:06d}' / 'COMPLETE'
        yield root / 'analysis' / f'{variant}_RL_complete_training.json'
    for job in jobs(root, include_rl=True):
        if job['variant'] == 'A_DDP_TIME' and job['stage'] == 'SFT' and job['update'] == 0:
            yield root / 'qualification/A_DDP_TIME/initial_baseline_reuse.json'
            continue
        name = Path(job['output']).name
        yield root / 'analysis' / (name + '.json')
        if job['update']:
            suffix = '_paired_own_SFT.json' if job['stage'] == 'RL' else '_paired_initial.json'
            yield root / 'analysis' / (name + suffix)


def readiness(root):
    missing = sorted({str(p) for p in required_files(root) if not p.is_file()})
    return dict(status='FILES_PRESENT_REQUIRES_VALIDATION' if not missing else 'WAITING_FOR_BOTH_ENDPOINTS_AND_DEV',
                missing=missing, Navtest_opened=False, new_candidates=0)


def checkpoint_files(checkpoint, identity, update, *, rl):
    checkpoint = Path(checkpoint)
    seal = json.loads((checkpoint / 'COMPLETE').read_text())
    files = json.loads((checkpoint / 'files.json').read_text())
    phase = 'COMPLETE' if rl else 'READY'
    required = {'identity.json','optimizer.pt',*[f'rank{r}.pt' for r in range(8)]}
    required.add('transaction.json' if rl else 'model.pt')
    if ('.incomplete' in checkpoint.name or seal['phase'] != phase or seal['world'] != 8 or
            seal['updates'] != update or (rl and seal['next_inner'] != 0) or
            file_sha(checkpoint / 'files.json') != seal['files_sha256'] or
            json.loads((checkpoint / 'identity.json').read_text()) != identity or not required <= files.keys()):
        raise ValueError('Invalid fixed endpoint: ' + str(checkpoint))
    for name, sha in files.items():
        if Path(name).name != name or file_sha(checkpoint / name) != sha:
            raise ValueError('Changed endpoint payload: ' + name)
    if rl:
        transaction = json.loads((checkpoint / 'transaction.json').read_text())
        ranks = validate_ranks(transaction['ranks'])
        expected = dict(phase='COMPLETE',updates=update,collections=update//4,scene_cursor=update*2,
                        next_inner=0,run=identity['sha256'],pending=False)
        if (len(ranks) != 8 or any(ranks[0][k] != v for k,v in expected.items()) or
                transaction['execution_identity'] != identity['execution_identity']):
            raise ValueError('Unmatched terminal checkpoint transaction')
    return seal, files


def validate_sft_results(run, identity, update, checkpoint):
    for rank in range(8):
        result = json.loads((Path(run) / f'result_rank{rank}.json').read_text())
        if (result['status'] != 'COMPLETE' or result['end_update'] != update or
                result['rank'] != rank or result['identity'] != identity['sha256'] or
                result['cursor'] != update * 32 or result['qualification'] or
                Path(result['checkpoint']).resolve() != Path(checkpoint).resolve()):
            raise ValueError('SFT endpoint did not complete on every rank')


def validate_rl_summary(summary, identity):
    cumulative = summary['cumulative']
    if (summary['status'] != 'TRAINING_COMPLETE' or summary['run_identity'] != identity['sha256'] or
            summary['matched_completed_updates'] != 20000 or summary['observed_updates_by_rank'] != [20000] * 8 or
            cumulative['fresh_global_rollouts'] != 5000 or cumulative['scene_exposures'] != 40000 or
            cumulative['new_candidates'] != 320000 or summary['cost']['checkpoint_updates'] != 20000):
        raise ValueError('Incomplete or unmatched formal RL budget')


def validate_export(path, checkpoint, identity, stage, update, own_sft):
    path = Path(path)
    info = authenticated(path / 'export.json')
    if (json.loads((path / 'COMPLETE').read_text()) != info or
            info['schema'] != 'fm_transfer_original_entry_export_v1' or info['qualification'] or
            not info['frozen_upstream_exact'] or info['stage'] != stage or info['updates'] != update or
            Path(info['checkpoint']).resolve() != Path(checkpoint).resolve() or
            info['run_identity'] != identity['sha256'] or
            info['new_SFT'] != own_sft or info['architecture'] != own_sft['config']['architecture'] or
            file_sha(path / 'pytorch_model.pt') != info['weight_sha256'] or
            file_sha(path / 'config.yaml') != info['config_sha256']):
        raise ValueError('Wrong own-SFT endpoint export: ' + str(path))
    return dict(export=str(path),weight_sha256=info['weight_sha256'],config_sha256=info['config_sha256'],
                export_receipt_sha256=file_sha(path / 'export.json'),checkpoint=str(checkpoint),
                run_identity=identity['sha256'],stage=stage,updates=update,new_SFT=own_sft)


def validate_milestones(root, selected_jobs=None, *, snapshot=None):
    evidence = {}
    for job in (jobs(root, include_rl=True) if selected_jobs is None else selected_jobs):
        if initial_A_reused(root, job):
            path = root / 'qualification/A_DDP_TIME/initial_baseline_reuse.json'
            evidence[str(path)] = file_sha(path)
            continue
        output = Path(job['output'])
        manifest = authenticated(output / 'manifest.json')
        exported = authenticated(Path(job['export']) / 'export.json')
        if snapshot is not None:
            for key, relative in [('source_sha256','scripts/analysis/simwam_recipe/evaluate_policy.py'),
                                  ('transfer_evaluation_sha256','scripts/analysis/fm_transfer/evaluate_policy.py')]:
                if manifest[key] != file_sha(Path(snapshot) / relative):
                    raise ValueError('Final worker differs from completed milestone protocol')
        if (manifest['qualification_only'] or manifest['cohort'] != job['cohort'] or
                manifest['stage'] != job['stage'] or manifest['seed'] != 2101 or
                manifest['modes'] != ['E_NATIVE','E_SHIFT5'] or
                manifest['new_SFT']['config']['variant'] != job['variant'] or
                exported['updates'] != job['update'] or exported['stage'] != job['stage'] or
                exported['weight_sha256'] != manifest['weight_sha256'] or
                manifest['export_receipt_sha256'] != file_sha(Path(job['export']) / 'export.json')):
            raise ValueError('Changed registered milestone: ' + str(output))
        for rank in range(manifest['shards']):
            result = json.loads((output / f'rank{rank}.json').read_text())
            if result['status'] != 'ASSIGNED_COMPLETE' or result['identity_sha256'] != manifest['sha256']:
                raise ValueError('Incomplete milestone worker: ' + str(output))
        paths = [root / 'analysis' / (output.name + '.json')]
        if job['update']:
            suffix = '_paired_own_SFT.json' if job['stage'] == 'RL' else '_paired_initial.json'
            paths.append(root / 'analysis' / (output.name + suffix))
        for path in paths:
            result = json.loads(path.read_text())
            if result['status'] != 'COMPLETE_POPULATION' or result['identity'] != manifest:
                raise ValueError('Milestone analysis has not verified its full population')
            evidence[str(path)] = file_sha(path)
    return evidence


def make_lock(root, snapshot, output):
    root, snapshot, output = Path(root), Path(snapshot), Path(output)
    if output.exists():
        raise FileExistsError('Preserve the original final lock')
    ready = readiness(root)
    if ready['missing']:
        raise ValueError('Navtest remains closed: ' + json.dumps(ready))
    if snapshot.resolve() != EVALUATION_SOURCE.resolve():
        raise ValueError('Use the qualified immutable evaluation source; new code needs explicit qualification')
    if not json.loads((snapshot / 'SOURCE_COMPLETE.json').read_text())['git_archive_extraction_complete']:
        raise ValueError('Evaluation source is not published')
    milestones = validate_milestones(root,snapshot=snapshot)
    reference_recipe = load_recipe(snapshot / 'configs/flow_rl/simwam_source_ddp.yaml')
    models, training_evidence, training_sources = {}, {}, {}
    for variant, cfg in CONFIGS.items():
        sft_run = root / 'runs' / f'{variant}_SFT_seed42_v1'
        sft_checkpoint = sft_run / 'checkpoints' / f"update{cfg['max_updates']:06d}"
        sft_identity = authenticated(sft_run / 'identity.json')
        if sft_identity['qualification'] or sft_identity['config'] != sft_config(variant):
            raise ValueError('Wrong formal SFT parent')
        _, sft_files = checkpoint_files(sft_checkpoint,sft_identity,cfg['max_updates'],rl=False)
        own_sft = dict(checkpoint=str(sft_checkpoint),model_sha256=sft_files['model.pt'],
            identity_sha256=sft_identity['sha256'],config=sft_identity['config'],updates=cfg['max_updates'],qualification=False)
        validate_sft_results(sft_run,sft_identity,cfg['max_updates'],sft_checkpoint)
        models[variant + '_SFT'] = validate_export(root / 'exports' / variant / f"SFT_update{cfg['max_updates']:06d}",
            sft_checkpoint,sft_identity,'SFT',cfg['max_updates'],own_sft)
        run = root / 'runs' / f'{variant}_R_SOURCE_seed42_v1'
        identity = authenticated(run / 'identity.json')
        training_sources[variant] = identity['source_files']
        recipe = {k:v for k,v in identity['recipe'].items() if k != 'supervised_transfer'}
        if (identity['purpose'] != 'formal' or identity['world'] != 8 or recipe != reference_recipe or
                identity['recipe']['variant'] != 'R_SOURCE' or
                identity['recipe']['supervised_transfer']['SFT_identity'] != own_sft):
            raise ValueError('RL endpoint uses a different supervised parent')
        for rank in range(8):
            result = json.loads((run / f'result_rank{rank}.json').read_text())
            if (result['status'] != 'COMPLETE' or result['rank'] != rank or result['qualification'] or
                    result['updates'] != 20000 or
                    result['collections'] != 5000 or result['raw_scene_exposures'] != 40000 or
                    result['new_candidates'] != 320000):
                raise ValueError('RL endpoint did not complete its registered exposure')
        path = root / 'analysis' / f'{variant}_RL_complete_training.json'
        validate_rl_summary(json.loads(path.read_text()),identity)
        training_evidence[str(path)] = file_sha(path)
        checkpoint = run / 'checkpoints/update020000_inner0'
        checkpoint_files(checkpoint,identity,20000,rl=True)
        models[variant + '_RL'] = validate_export(root / 'exports' / variant / 'RL_update020000',
            checkpoint,identity,'RL',20000,own_sft)
    if len({fingerprint(v) for v in training_sources.values()}) != 1:
        raise ValueError('Routes used different training code; explicit compatibility evidence is required')
    # This reads input/cache membership, never Navtest scores or trajectories.
    assets = prepare_assets(output.with_suffix('.assets.json'))
    pop = json.loads(Path(POPULATION).read_text())
    value = dict(status='LOCKED_BEFORE_FINAL_NAVTEST',schema='fm_transfer_final_four_models_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),formal_budget_complete=True,
        selection_rule='Both complete own SFTs and fixed RL update20000; no score or seed maximum',
        models=models,weight_sha256=[x['weight_sha256'] for x in models.values()],
        population_identity=pop['identity_sha256'],inference_seeds=[2101],training_seeds=[42],
        modes=['E_NATIVE','E_SHIFT5'],sampler_sha256=file_sha(snapshot / 'scripts/analysis/simwam_recipe/evaluate_policy.py'),
        transfer_evaluation_sha256=file_sha(snapshot / 'scripts/analysis/fm_transfer/evaluate_policy.py'),
        evaluation_source=str(snapshot),asset_identity=assets['sha256'],
        data_root=assets['data_root'],navtest_datalist=assets['navtest_datalist'],v1_cache=assets['v1_cache'],
        milestone_evidence=milestones,training_evidence=training_evidence,training_source_files=training_sources,
        historically_observed_Navtest_not_fully_blind=True,Navtest_scores_read_for_selection=False,
        lock_implementation_sha256=file_sha(__file__))
    value['sha256'] = fingerprint(value)
    atomic_json(output,value)
    return value


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('command',choices=['check','prepare','lock'])
    parser.add_argument('--root',default=str(ROOT))
    parser.add_argument('--source',default=str(EVALUATION_SOURCE))
    parser.add_argument('--output')
    args = parser.parse_args()
    require_dispatch()
    root = Path(args.root)
    if args.command == 'check':
        print(json.dumps(readiness(root)))
        return
    queue = root / 'final_navtest'
    queue.mkdir(exist_ok=True)
    with (queue / 'owner.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX | fcntl.LOCK_NB)
        output = Path(args.output) if args.output else queue / ('assets.json' if args.command == 'prepare' else 'LOCK.json')
        if output.exists():
            raise FileExistsError('Preserve the previous final evidence')
        result = prepare_assets(output) if args.command == 'prepare' else make_lock(root,args.source,output)
        print(json.dumps({k:v for k,v in result.items() if k not in ['models','milestone_evidence','training_evidence']}))


if __name__ == '__main__':
    main()
