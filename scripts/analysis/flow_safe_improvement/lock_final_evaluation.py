"""Lock a finite final confirmation only after every registered DEV window.

This reads DEV evidence and immutable export identities. It never reads Navtest
predictions or scores, and cannot expand the declared model or seed budget.
"""
import argparse
import datetime
import json
from pathlib import Path
import re

from scripts.analysis.flow_safe_improvement.run_original_evaluation_queue import authorized
from scripts.analysis.flow_safe_improvement.sample_locked_navtest import locked_contract
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import SFT_SHA256, file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch


def read(path):
    return json.loads(Path(path).read_text())


def checked_summary(path):
    report = read(path); digest = report.pop('sha256')
    if fingerprint(report) != digest:
        raise ValueError('DEV summary hash mismatch')
    if (not report['all_registered_DEV_windows_complete'] or not report['fixed_budget_DEV_endpoints_complete']
            or report['training_seeds'] != [42] or report['inference_seeds'] != [2101]
            or len(report['evaluations']) != 24 or report['missing']):
        raise ValueError('All 24 single-seed DEV windows must be complete before final selection')
    return report, digest


def export_for(root, name):
    match = re.fullmatch(r'(P[012]_[A-Z_]+)_epoch(1|2|4|8)', name)
    if match:
        return root/'training'/(match[1]+'_seed42')/'train'/f'export_update{int(match[2])*256:06d}'
    match = re.fullmatch(r'(ODE_REWARD_FM|ODE_SAFE_FM|EXPLORE_SAFE_FM|ODE_NFT)_round([123])', name)
    if match:
        return root/'online'/(match[1]+'_seed42')/('round'+match[2])/'learning/train/export_update000512'
    raise ValueError('Unknown registered checkpoint')


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('run-root', 'summary', 'registration', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args(); authorized(require_dispatch()); root = Path(a.run_root)
    registration = read(a.registration)
    variants = ['ODE_REWARD_FM', 'ODE_SAFE_FM', 'EXPLORE_SAFE_FM', 'ODE_NFT']
    if (registration['schema'] != 'flow_safe_finite_final_confirmation_v1'
            or registration['fixed_online_variants'] != variants
            or registration['maximum_unique_exports'] != 6 or registration['fixed_online_endpoint_round'] != 3
            or registration['training_seeds'] != [42] or registration['inference_seeds'] != [2101]):
        raise ValueError('Final confirmation budget differs from the finite registration')
    if Path(a.output).exists():
        raise ValueError('Final selection locks are immutable; never overwrite an existing one')
    summary, summary_sha = checked_summary(a.summary)
    population = read(root/'population/manifest.json')
    rows = {x['name']: x for x in summary['evaluations']}
    if len(rows) != 24:
        raise ValueError('Duplicate DEV checkpoint names')
    for row in rows.values():
        original = read(row['report'])
        digest = original.pop('sha256')
        if digest != row['report_sha256'] or fingerprint(original) != digest:
            raise ValueError('A completed DEV report changed after summarization')
    for variant in variants:
        state = read(root/'online'/(variant+'_seed42')/'single_seed_controller_state.json')
        if state['status'] != 'COMPLETE_THREE_FRESH_POLICY_ITERATIONS':
            raise ValueError('A registered online controller has not completed three rounds')
    eligible = sorted((x for x in rows.values() if x['finite_DEV_selection_eligible']),
                      key=lambda x: (-x['PDMS'], x['name']))
    winner = eligible[0]['name'] if eligible else 'SFT'
    sft_export = Path('/mnt/project/DriveDreamer-Policy-flow-grpo/artifacts/action-only-checkpoints-v1/frozen_visual')
    requests = [('SFT', 'SFT', sft_export, 'permanent_matched_baseline')]
    requests += [(v, v+'_round3', export_for(root, v+'_round3'), 'fixed_budget_online_endpoint') for v in variants]
    requests += [('DEV_SELECTED', winner, sft_export if winner == 'SFT' else export_for(root, winner),
                  'finite_DEV_qualified_selection_separate_from_budget_endpoint')]
    sampler = Path(__file__).with_name('sample_locked_navtest.py')
    models, hashes, aliases = {}, {}, {}
    for name, checkpoint, export, role in requests:
        if export not in hashes:
            hashes[export] = (file_sha(export/'pytorch_model.pt'), file_sha(export/'config.yaml'))
        weight, config = hashes[export]
        if checkpoint == 'SFT':
            if weight != SFT_SHA256:
                raise ValueError('Permanent SFT identity changed')
        else:
            if not (export/'COMPLETE').exists():
                raise ValueError('Unsealed checkpoint cannot enter final evaluation')
            sealed = read(export/'export_state.json')['files']['pytorch_model.pt']
            if weight != (sealed['sha256'] if isinstance(sealed, dict) else sealed):
                raise ValueError('Sealed model payload changed')
        models[name] = dict(checkpoint_name=checkpoint, export=str(export), weight_sha256=weight,
                            config_sha256=config, role=role, sampler_source_sha256=file_sha(sampler),
                            DEV_report_sha256=rows[checkpoint]['report_sha256'] if checkpoint != 'SFT' else None)
        if (weight, config) in aliases:
            models[name]['reuse_identical_model'] = aliases[(weight, config)]
        else:
            aliases[(weight, config)] = name
    baseline = read(root/'evaluations/original_export/SFT_DEV5_v1/manifest.json')
    datalist = Path('/mnt/project/DriveDreamer-Policy/test_meta.json')
    tokens = read(datalist)
    if len(tokens) != 12146 or len(set(tokens)) != 12146 or set(tokens) != set(population['navtest_tokens']):
        raise ValueError('Final current-input population is not the exact registered Navtest')
    lock = dict(status='LOCKED_BEFORE_FINAL_NAVTEST', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        population_identity=population['identity_sha256'], navtest_tokens_sha256=fingerprint(population['navtest_tokens']),
        inference_seeds=[2101], training_seeds=[42], models=models, unique_export_count=len(aliases),
        registration_sha256=file_sha(a.registration), completed_DEV_summary_sha256=summary_sha,
        selection_rule_sha256=file_sha(root/'evaluations/original_export/SELECTION_RULE.json'),
        single_seed_override_sha256=file_sha(root/'resources/single_seed_override/REGISTRATION.json'),
        eligible_DEV_checkpoints=[dict(name=x['name'],PDMS=x['PDMS']) for x in eligible], selected_checkpoint=winner,
        deployment_execution_profile=baseline['sample_manifest']['execution_profile'],
        v1_protocol_sha256=baseline['protocol_sha256'],
        data_root='/mnt/project/DriveDreamer-Policy-paired/runs/paired_full_assets_v1/dataset',
        navtest_datalist=str(datalist), navtest_datalist_sha256=file_sha(datalist),
        v1_cache='/mnt/project/DriveDreamer-Policy-full-navtrain/runs/full_navtrain_epoch1/metric_cache_navtest_v1',
        v2_cache='/mnt/project/DriveDreamer-Policy-paired/runs/metric_cache_navtest_v2',
        selection_did_not_read_Navtest_predictions_or_scores=True,
        historically_observed_Navtest_not_fully_blind=True,
        source_sha256=file_sha(__file__))
    lock['sha256'] = fingerprint(lock); Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    atomic_json(a.output, lock)
    for name in models:
        locked_contract(a.output, name, population)
    print(json.dumps(dict(status=lock['status'],selected=winner,unique_exports=len(aliases),sha256=lock['sha256'])))


if __name__ == '__main__':
    main()
