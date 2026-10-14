"""Seal this two-line DEV selection before any new Navtest prediction or score.

This is an evaluation adapter, not a training controller. It cannot resume a
safety-paused actor or substitute a qualification export for a formal run.
"""
import argparse
import datetime
import json
from pathlib import Path

from scripts.analysis.flow_safe_improvement.sample_locked_navtest import locked_contract
from starVLA.rl.constrained_flow_noise.contracts import (
    SFT_SHA256, VARIANTS, file_sha, fingerprint, require_dispatch,
)
from starVLA.rl.flow_grpo.transactions import atomic_json

SFT = Path('/mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-checkpoints-v1/frozen_visual')
SELECTION = ('Among predeclared DEV milestones, require each NC/DAC/TTC violation count no higher '
             'than matched SFT and no higher zero count; highest PDMS, earlier raw exposure '
             'breaks tie. SFT fallback. Endpoints always reported.')


def read(path):
    return json.loads(Path(path).read_text())


def sealed(path):
    value = read(path)
    if fingerprint({k: v for k, v in value.items() if k != 'sha256'}) != value['sha256']:
        raise ValueError('Changed sealed JSON: ' + str(path))
    return value


def choose(rows):
    """Exactly the pre-training rule: safe milestones only, then earlier tie."""
    eligible = [x for x in rows if x['safe_DEV_selection_eligible']]
    return min(eligible, key=lambda x: (-x['PDMS'], x['raw_exposure'])) if eligible else None


def formal_run_paths(root, variant):
    """Follow an explicit, same-scientific-lineage native continuation only."""
    root = Path(root)
    parent = root/f'formal_{variant}_seed42_v1'
    pointer = root/f'formal_{variant}_active_continuation.json'
    if not pointer.exists():
        return [parent]
    route = read(pointer)
    child, checkpoint = Path(route['output']), Path(route['checkpoint'])
    if (Path(route['parent']).resolve() != parent.resolve() or child.resolve() == parent.resolve()
            or child.resolve().parent != root.resolve() or not (checkpoint/'COMPLETE').is_file()):
        raise ValueError('Invalid formal continuation route')
    identities = [sealed(p) for p in (parent/'run_identity.json', child/'run_identity.json', checkpoint/'identity.json')]
    if (len({x['sha256'] for x in identities}) != 1
            or any(x['qualification_only'] or x['variant'] != variant for x in identities)):
        raise ValueError('Continuation is not the same formal scientific lineage')
    origin = read(child/'native_continuation_origin.json')
    if (origin['qualification_only'] or Path(origin['checkpoint']).resolve() != checkpoint.resolve()
            or origin['scientific_identity'] != identities[0]['sha256']
            or origin['numerical_execution_identity'] != read(child/'execution_identity.json')['sha256']):
        raise ValueError('Continuation origin/execution mismatch')
    return [parent, child]


def collect(root, population):
    root = Path(root)
    budget = sealed(root/'formal_budget.json')
    registration = read(root/'precalibration_registration.json')
    if registration['checkpoint_selection'] != SELECTION:
        raise ValueError('Pre-training selection rule changed')
    if (budget['variants'] != list(VARIANTS) or budget['training_seed'] != 42
            or budget['evaluation_seed'] != 2101 or budget['population_identity'] != population['identity_sha256']):
        raise ValueError('Not this matched two-line formal budget')
    baseline = sealed(root/'evaluation/SFT_DEV_original/report.json')
    baseline_manifest = sealed(root/'evaluation/SFT_DEV_original/scores/manifest.json')
    if baseline['behavior_sha256'] != SFT_SHA256 or baseline['scenes'] != len(population['dev_tokens']):
        raise ValueError('Full matched original SFT DEV baseline required')
    pending, runs = [], {}
    for variant in VARIANTS:
        lineage = formal_run_paths(root, variant)
        run = lineage[-1]
        result_path = run/'result_rank0.json'
        if not result_path.exists():
            pending.append(dict(variant=variant, reason='formal endpoint not yet saved'))
            continue
        result, identity = read(result_path), sealed(run/'run_identity.json')
        if (result['qualification_only'] or identity['qualification_only'] or identity['variant'] != variant
                or identity['SFT_sha256'] != SFT_SHA256 or identity['budget_sha256'] != file_sha(root/'formal_budget.json')
                or identity['population_identity'] != population['identity_sha256']
                or identity['total_raw_exposures'] != budget['raw_exposures_per_variant']):
            raise ValueError('Qualification or unmatched run cannot enter final comparison')
        complete = result['status'] == 'COMPLETE_FIXED_RAW_BUDGET'
        pause = result.get('safety_pause')
        if not complete and not pause:
            pending.append(dict(variant=variant, reason='formal budget incomplete; saved result may precede live progress', run=str(run)))
            continue
        if complete and result['raw_cursor'] != budget['raw_exposures_per_variant']:
            raise ValueError('Completion label does not match registered raw exposure')
        for rank in range(8):
            other = read(run/f'result_rank{rank}.json')
            for key in ('identity_sha256', 'status', 'raw_cursor', 'updates', 'completed_evaluations', 'safety_pause'):
                if other[key] != result[key]:
                    raise ValueError(f'Rank endpoint mismatch: {variant} rank{rank} {key}')
        expected = [x for x in budget['milestones'] if x <= result['raw_cursor']]
        if result['completed_evaluations'] != expected:
            raise ValueError('A fixed DEV observation is unfinished')
        rows = []
        for raw in expected:
            report_path = run/'DEV'/f'raw{raw:06d}'/'report.json'
            report = sealed(report_path)
            if (report['scenes'] != len(population['dev_tokens']) or report['seed'] != 2101
                    or report['protocol_sha256'] != baseline['protocol_sha256']
                    or report['baseline_payloads_sha256'] != baseline['baseline_payloads_sha256']
                    or report['population_identity'] != population['identity_sha256']):
                raise ValueError('Unmatched DEV report')
            rows.append(dict(raw_exposure=raw, PDMS=100*report['metrics']['score']['current'],
                             safe_DEV_selection_eligible=report['safe_DEV_selection_eligible'],
                             behavior_sha256=report['behavior_sha256'], report=str(report_path),
                             report_sha256=report['sha256'], export=str(run/f'export_raw{raw:06d}')))
        # A numerical pause can occur between milestones. It must be exported
        # and evaluated explicitly; do not relabel the preceding milestone as
        # the actual endpoint or make the aborted budget look complete.
        endpoint = next((x for x in rows if x['raw_exposure'] == result['raw_cursor']), None)
        if endpoint is None:
            pending.append(dict(variant=variant, reason='registered pause between milestones requires actual endpoint export/evaluation',
                                raw_exposure=result['raw_cursor'], safety_pause=pause))
            continue
        runs[variant] = dict(run=str(run), run_identity=identity['sha256'], endpoint=endpoint,
                             native_run_lineage=[str(p) for p in lineage],
                             selection=choose(rows), DEV=rows, actual_raw_exposure=result['raw_cursor'],
                             complete_fixed_budget=complete, safety_pause=pause, result_sha256=file_sha(result_path))
    return dict(status='READY_TO_LOCK' if not pending else 'WAITING_FOR_FORMAL_ENDPOINTS',
                pending=pending, variants=runs, budget_sha256=file_sha(root/'formal_budget.json'),
                registration_sha256=file_sha(root/'precalibration_registration.json'),
                selection_rule=SELECTION, baseline_manifest=baseline_manifest,
                baseline_report_sha256=baseline['sha256'])


def make_lock(root, population, evidence, output):
    if evidence['status'] != 'READY_TO_LOCK':
        raise ValueError('Both registered formal lines must reach their recorded endpoints')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Final selection is immutable: ' + str(output))
    requests = [('SFT', SFT, None, 'permanent_SFT_baseline')]
    for variant, run in evidence['variants'].items():
        requests.append((variant+'_ENDPOINT', Path(run['endpoint']['export']), run['endpoint'],
                         'fixed_budget_endpoint' if run['complete_fixed_budget'] else 'registered_safety_pause_endpoint'))
        chosen = run['selection']
        requests.append((variant+'_DEV_SELECTED', Path(chosen['export']) if chosen else SFT,
                         chosen, 'predeclared_safe_DEV_selection' if chosen else 'SFT_fallback'))
    sampler = Path(__file__).parents[1]/'flow_safe_improvement/sample_locked_navtest.py'
    models, seen, cache = {}, {}, {}
    for name, export, row, role in requests:
        if export not in cache:
            cache[export] = file_sha(export/'pytorch_model.pt'), file_sha(export/'config.yaml')
        weight, config = cache[export]
        if export == SFT:
            if weight != SFT_SHA256:
                raise ValueError('SFT payload changed')
        else:
            saved = read(export/'cfn_export_identity.json')
            if (not (export/'COMPLETE').is_file() or saved['noise_head_in_export']
                    or saved['export_sha256'] != weight or row['behavior_sha256'] != weight):
                raise ValueError('Unsealed or mismatched original ODE export')
        model = dict(export=str(export), weight_sha256=weight, config_sha256=config, role=role,
                     sampler_source_sha256=file_sha(sampler), DEV_report_sha256=row['report_sha256'] if row else None,
                     raw_exposure=row['raw_exposure'] if row else 0)
        if (weight, config) in seen:
            model['reuse_identical_model'] = seen[(weight, config)]
        else:
            seen[(weight, config)] = name
        models[name] = model
    datalist = Path('/mnt/project/DriveDreamer-Policy/test_meta.json')
    tokens = read(datalist)
    if len(tokens) != 12146 or len(set(tokens)) != 12146 or set(tokens) != set(population['navtest_tokens']):
        raise ValueError('Final original-input Navtest population changed')
    baseline = evidence['baseline_manifest']
    lock = dict(status='LOCKED_BEFORE_FINAL_NAVTEST', schema='cfn_two_line_final_confirmation_v1',
                utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                population_identity=population['identity_sha256'],
                navtest_tokens_sha256=fingerprint(population['navtest_tokens']), inference_seeds=[2101], training_seeds=[42],
                models=models, unique_export_count=len(seen), maximum_unique_exports=5,
                formal_evidence={k: v for k, v in evidence.items() if k != 'baseline_manifest'},
                deployment_execution_profile=baseline['sample_manifest']['execution_profile'],
                v1_protocol_sha256=baseline['protocol_sha256'],
                data_root='/mnt/project/DriveDreamer-Policy-paired/runs/paired_full_assets_v1/dataset',
                navtest_datalist=str(datalist), navtest_datalist_sha256=file_sha(datalist),
                v1_cache='/mnt/project/DriveDreamer-Policy-full-navtrain/runs/full_navtrain_epoch1/metric_cache_navtest_v1',
                selection_did_not_read_Navtest_predictions_or_scores=True,
                historically_observed_Navtest_not_fully_blind=True, source_sha256=file_sha(__file__))
    lock['sha256'] = fingerprint(lock)
    atomic_json(output, lock)
    for name in models:
        locked_contract(output, name, population)
    return lock


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--population', required=True)
    parser.add_argument('--output', help='Omit for a read-only readiness check; a lock is never fabricated')
    args = parser.parse_args()
    require_dispatch()
    population = read(args.population)
    evidence = collect(args.run_root, population)
    if args.output:
        lock = make_lock(args.run_root, population, evidence, args.output)
        print(json.dumps(dict(status=lock['status'], unique_exports=lock['unique_export_count'], sha256=lock['sha256'])))
    else:
        print(json.dumps({k: v for k, v in evidence.items() if k != 'baseline_manifest'}))


if __name__ == '__main__':
    main()
