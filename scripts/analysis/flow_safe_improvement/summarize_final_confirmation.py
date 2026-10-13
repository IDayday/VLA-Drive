"""Aggregate the already locked final models and their predeclared contrasts."""
import argparse
import csv
import json
import math
from pathlib import Path

from scripts.analysis.flow_safe_improvement.run_original_evaluation_queue import authorized
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import SAFETY_METRICS, file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch


def read(path):
    return json.loads(Path(path).read_text())


def verified(path):
    value = read(path)
    if fingerprint({k: v for k, v in value.items() if k != 'sha256'}) != value['sha256']:
        raise ValueError('Sealed report changed: '+str(path))
    return value


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('run-root', 'final-lock', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args(); authorized(require_dispatch())
    root = Path(a.run_root); base = root/'evaluations/final_navtest'
    lock = verified(a.final_lock)
    if lock['status'] != 'LOCKED_BEFORE_FINAL_NAVTEST' or lock['inference_seeds'] != [2101]:
        raise ValueError('An unchanged single-seed final lock is required')
    population = read(root/'population/manifest.json'); expected = set(population['navtest_tokens'])
    reports, auxiliary, scenes, identities, aliases = {}, {}, {}, {}, {}
    for name, model in lock['models'].items():
        source = model.get('reuse_identical_model', name)
        path = base/source
        if not (path/'COMPLETE.json').exists():
            raise ValueError('Final confirmation is incomplete: '+source)
        complete = read(path/'COMPLETE.json'); report = verified(path/'summary/report.json')
        v2 = read(path/'v2/report.json')
        if (report['status'] != 'COMPLETE_LOCKED_FINAL_NAVTEST' or report['scenes'] != 12146
                or report['final_selection_lock_sha256'] != lock['sha256']
                or report['inference_seeds'] != [2101] or report['training_seeds'] != [42]
                or report['deployment_execution_profile'] != lock['deployment_execution_profile']
                or complete['report_sha256'] != report['sha256']
                or complete['auxiliary_report_sha256'] != file_sha(path/'v2/report.json')
                or v2['status'] != 'COMPLETE_OFFICIAL_V2_ONE_STAGE_AUXILIARY'
                or v2['scenes'] != 12146 or v2['seeds'] != [2101]):
            raise ValueError('Unmatched final v1/v2/seed contract')
        records = read(path/'summary/per_scene.json')
        table = {r['token']: r for r in records}
        if len(records) != 12146 or set(table) != expected:
            raise ValueError('Final scene denominator changed')
        if any(table[t]['log'] != population['token_logs'][t] for t in expected):
            raise ValueError('Final log identity changed')
        for metric, aggregate in report['means'].items():
            values = [table[t]['means'][metric] for t in expected]
            if not all(math.isfinite(x) for x in values) or abs(math.fsum(values)/12146-aggregate['policy']) > 1e-12:
                raise ValueError('Per-scene values differ from the sealed final means')
        reports[name], auxiliary[name], scenes[name] = report, v2, table
        identities[name] = dict(report_sha256=report['sha256'],
            per_scene_file_sha256=file_sha(path/'summary/per_scene.json'),
            v2_report_file_sha256=file_sha(path/'v2/report.json'), model=lock['models'][name])
        if source != name:
            aliases[name] = dict(source=source, extra_model_calls=0, extra_scoring_calls=0)
    if len({r['baseline_identity'] for r in reports.values()}) != 1:
        raise ValueError('Final methods must share one matched baseline')
    for name, table in scenes.items():
        if any(abs(table[t]['delta_score']-(table[t]['means']['score']-scenes['SFT'][t]['means']['score'])) > 1e-12
               for t in expected):
            raise ValueError('Per-scene baseline subtraction changed: '+name)
    pairs = [('ODE_safe_minus_reward', 'ODE_SAFE_FM', 'ODE_REWARD_FM'),
             ('search_minus_ODE_safe', 'EXPLORE_SAFE_FM', 'ODE_SAFE_FM'),
             ('NFT_minus_reward_FM', 'ODE_NFT', 'ODE_REWARD_FM')]
    tokens = sorted(expected); logs = [population['token_logs'][t] for t in tokens]
    contrasts = {}
    for name, left, right in pairs:
        metrics = sorted(scenes[left][tokens[0]]['means'])
        paired = {m: log_bootstrap_difference(
            [scenes[left][t]['means'][m]-scenes[right][t]['means'][m] for t in tokens], logs) for m in metrics}
        transitions = {}
        for metric in (*SAFETY_METRICS, 'joint', 'nonzero_score'):
            keys = SAFETY_METRICS if metric == 'joint' else (metric,)
            def passes(row):
                return row['means']['score'] > 0 if metric == 'nonzero_score' else all(row['means'][k] == 1 for k in keys)
            now, old = [[passes(scenes[model][t]) for t in tokens] for model in (left, right)]
            transitions[metric] = dict(new_in_left=sum(b and not n for n, b in zip(now, old)),
                recovered_in_left=sum(n and not b for n, b in zip(now, old)),
                failed_in_both=sum(not n and not b for n, b in zip(now, old)))
        contrasts[name] = dict(left=left, right=right, paired_changes=paired, paired_transitions=transitions)
    result = dict(status='COMPLETE_LOCKED_FINITE_CONFIRMATION', final_lock_sha256=lock['sha256'],
        selected_before_Navtest=lock['selected_checkpoint'], no_reselection_from_final_scores=True,
        unique_exports=lock['unique_export_count'], scenes_per_model=12146, logs=136,
        inference_seeds=[2101], training_seeds=[42], units='raw_0_to_1',
        reports=reports, auxiliary_v2=auxiliary, model_identities=identities, identical_export_aliases=aliases,
        predeclared_contrasts=contrasts, source_sha256=file_sha(__file__),
        limitations='Single training/inference seed by user request. Log-cluster CI excludes seed uncertainty. Navtest was historically observed. Final SFT-safe membership is single-output, not stable four-seed safety.')
    result['sha256'] = fingerprint(result)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False); atomic_json(out/'report.json', result)
    with (out/'policy_scores.csv').open('w') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['model', 'PDMS_100', 'delta_100', 'CI_low_100', 'CI_high_100',
                         'NC_100', 'DAC_100', 'TTC_100', 'EP_100', 'Comfort_100', 'DDC_100', 'v2_EPDMS_100'])
        for name, r in reports.items():
            delta = r['paired_changes']['score']
            metrics = ('score', *SAFETY_METRICS, 'ego_progress', 'comfort', 'driving_direction_compliance')
            values = [100*r['means'][m]['policy'] for m in metrics]
            writer.writerow([name, values[0], 100*delta['mean'], *[100*x for x in delta['CI95']],
                             *values[1:], 100*auxiliary[name]['means']['score']])
    print(json.dumps(dict(status=result['status'], sha256=result['sha256'], selected=result['selected_before_Navtest'])))


if __name__ == '__main__':
    main()
