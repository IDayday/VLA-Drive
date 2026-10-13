"""Collect observed research endpoints, safety transfers and measured work only.

Incomplete windows remain incomplete. The collector never chooses a seed, fits
hyperparameters, treats candidate maxima as policy scores, or reads Navtest.
"""
import argparse
import csv
import datetime
import json
from pathlib import Path

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import SAFETY_METRICS, file_sha, fingerprint


def read(path):
    return json.loads(Path(path).read_text())


def training_cost(root):
    path = root/'training_rank0.jsonl'
    if not path.exists():
        return dict(updates=0, status='NOT_STARTED')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    updates = [x['update'] for x in rows]
    if len(updates) != len(set(updates)) or updates != sorted(updates):
        raise ValueError('Duplicate or reordered optimizer updates; resolve before cost aggregation')
    exposures = len(rows)*16
    return dict(updates=max(updates), observed_optimizer_updates=len(rows), scene_exposures=exposures,
                unique_scene_population=4096, training_seed=42,
                velocity_forward_scene_calls=sum(16*x['velocity_forward_calls_per_scene'] for x in rows),
                velocity_backward_scene_calls=sum(16*x['velocity_backward_calls_per_scene'] for x in rows),
                FM_noise_repeats=8, rank0_optimizer_loop_seconds=sum(x['seconds'] for x in rows),
                allocated_rank_hours_optimizer_loop_only=8*sum(x['seconds'] for x in rows)/3600,
                allocated_rank_hours_are_not_unique_physical_GPU_hours=True,
                peak_gpu_allocated_GiB=max(x['peak_gpu_allocated_GiB'] for x in rows),
                log_sha256=file_sha(path))


def scored_row(name, path, endpoint):
    if not path.exists():
        return dict(name=name, status='NOT_YET_COMPLETE', fixed_budget_endpoint=endpoint)
    report = read(path); checksum = report.pop('sha256')
    if fingerprint(report) != checksum:
        raise ValueError('Evaluation report checksum mismatch')
    if (report['inference_seeds'] != [2101] or report['scenes'] != 1696
            or report['status'] != 'COMPLETE_MATCHED_EVALUATION' or not report['no_best_of_N']):
        raise ValueError('Only the complete matched single-seed DEV contract belongs in this table')
    safe_means = all(report['paired_changes'][m]['mean'] >= -1e-12 for m in SAFETY_METRICS)
    safe_counts = all(v['current_any_seed_violation_in_fixed_safe'] <= v['baseline_any_eval_seed_violation_in_fixed_safe']
                      for v in report['safety'].values())
    return dict(name=name, status=report['status'], fixed_budget_endpoint=endpoint, report=str(path),
                report_sha256=checksum, PDMS=report['means']['score']['policy'],
                SFT_PDMS=report['means']['score']['SFT'], paired_changes=report['paired_changes'],
                safety=report['safety'], zero_transitions=report['zero_transitions'],
                high_score_decline=report['high_score_decline'],
                finite_DEV_selection_eligible=(safe_means and safe_counts and report['paired_changes']['score']['mean'] > 0),
                eligibility_is_not_policy_safety_guarantee=True,
                inference_seeds=report['inference_seeds'], training_seed_repetitions=1)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--run-root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args(); root = Path(a.run_root); out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    evaluations, learning, banks = [], {}, {}
    for variant in ('P0_REWARD_FM', 'P1_SAFE_FM', 'P2_SEARCH_SAFE_FM'):
        run = root/'training'/(variant+'_seed42')
        learning[variant] = training_cost(run/'train')
        for epoch in (1, 2, 4, 8):
            evaluations.append(scored_row(variant+'_epoch'+str(epoch),
                run/f'DEV_seed2101_original_epoch{epoch}/summary/report.json', epoch == 8))
    for variant in ('ODE_REWARD_FM', 'ODE_SAFE_FM', 'EXPLORE_SAFE_FM', 'ODE_NFT'):
        for iteration in (1, 2, 3):
            run = root/'online'/(variant+'_seed42')/f'round{iteration}'
            key = variant+'_round'+str(iteration)
            learning[key] = training_cost(run/'learning/train')
            evaluations.append(scored_row(key, run/'learning/DEV_seed2101_original_epoch2/summary/report.json', iteration == 3))
            bank = run/'bank'
            if not (bank/'manifest.json').exists():
                continue
            ranks = [read(p) for p in sorted(bank.glob('rank*.json'))]
            banks[key] = dict(identity=read(bank/'manifest.json')['sha256'],
                complete=len(ranks) == 8 and all(x['status'] == 'ASSIGNED_POPULATION_COMPLETE' for x in ranks),
                counters={k: sum(x[k] for x in ranks) for k in ('new_scenes', 'logical_proposals', 'velocity_forward_calls',
                    'actual_new_scoring_calls', 'invalid_physical_proposals', 'GPU_generation_seconds', 'wall_seconds')},
                wall_seconds_is_sum_of_rank_allocations_not_unique_physical_time=True,
                reference_historical_scoring_cost_separate=True)
    complete = all(x['status'] == 'COMPLETE_MATCHED_EVALUATION' for x in evaluations)
    all_endpoints = [x for x in evaluations if x['fixed_budget_endpoint']]
    endpoints_complete = all(x['status'] == 'COMPLETE_MATCHED_EVALUATION' for x in all_endpoints)
    observations = []
    for x in evaluations:
        if x['status'] != 'COMPLETE_MATCHED_EVALUATION':
            continue
        if x['PDMS'] > x['SFT_PDMS'] and not x['finite_DEV_selection_eligible']:
            observations.append(dict(name=x['name'], evidence='MEAN_GAIN_WITH_UNRESOLVED_SAFETY_REGRESSION',
                                     supported_case='CASE_C_projection_or_retention_candidate', report_sha256=x['report_sha256']))
    report = dict(schema='flow_safe_observed_direction_status_v1',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='COMPLETE_DEV_WINDOWS_FINAL_CONFIRMATION_SEPARATE' if complete else 'RESEARCH_IN_PROGRESS',
        all_registered_DEV_windows_complete=complete, fixed_budget_DEV_endpoints_complete=endpoints_complete,
        multi_seed_experiments='CANCELLED_BY_USER', inference_seeds=[2101], training_seeds=[42],
        units='raw_0_to_1', evaluations=evaluations, learning_costs=learning, online_candidate_banks=banks,
        observed_interpretation=observations,
        missing=[x['name'] for x in evaluations if x['status'] != 'COMPLETE_MATCHED_EVALUATION'],
        no_Navtest_read_or_selection=True,
        limitations=['Single training/inference seed does not measure seed stability.',
                    'Cost allocations overlap when jobs share GPUs; do not sum as unique hardware GPU-hours.',
                    'Candidates, targets, training rewards and DEV policy scores have separate meanings.',
                    'CASE labels here are diagnostic evidence, not a final method ranking.'])
    report['sha256'] = fingerprint(report); atomic_json(out/'report.json', report)
    with (out/'policy_scores.csv').open('w') as stream:
        writer = csv.DictWriter(stream, lineterminator='\n', fieldnames=['name', 'status', 'endpoint', 'PDMS_100', 'delta_100', 'CI95_low_100',
            'CI95_high_100', 'NC_delta_100', 'DAC_delta_100', 'TTC_delta_100', 'fixed_safe_joint_violations', 'SFT_fixed_safe_joint_violations', 'eligible'])
        writer.writeheader()
        for x in evaluations:
            row = dict(name=x['name'], status=x['status'], endpoint=x['fixed_budget_endpoint'])
            if x['status'] == 'COMPLETE_MATCHED_EVALUATION':
                d=x['paired_changes']; j=x['safety']['joint']
                row.update(PDMS_100=100*x['PDMS'], delta_100=100*d['score']['mean'], CI95_low_100=100*d['score']['CI95'][0],
                    CI95_high_100=100*d['score']['CI95'][1], NC_delta_100=100*d[SAFETY_METRICS[0]]['mean'],
                    DAC_delta_100=100*d[SAFETY_METRICS[1]]['mean'], TTC_delta_100=100*d[SAFETY_METRICS[2]]['mean'],
                    fixed_safe_joint_violations=j['current_any_seed_violation_in_fixed_safe'],
                    SFT_fixed_safe_joint_violations=j['baseline_any_eval_seed_violation_in_fixed_safe'], eligible=x['finite_DEV_selection_eligible'])
            writer.writerow(row)
    print(json.dumps(dict(status=report['status'], complete_evaluations=len(evaluations)-len(report['missing']),
                          pending_evaluations=len(report['missing']))))


if __name__ == '__main__':
    main()
