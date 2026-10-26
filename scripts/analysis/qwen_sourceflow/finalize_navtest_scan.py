"""Finish a sealed finite scan using existing strict analyses; launches no inference.

May wait for the one registered owner. Publication is a separate, scoped operator
action. Never interprets a partial scene mean as a full Navtest score.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

# Support the documented direct script entry as well as python -m. This imports
# only the stdlib finite-plan contract, without loading a model on the operator.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.analysis.qwen_sourceflow.finite_navtest_scan import read, seal, validate_plan


MODEL_PYTHON = '/mnt/project/flow-safe-improvement-20261013/venv/bin/python'
NATIVE = Path('/mnt/project/qwen-sourceflow-native-20261018')
SAFE = Path('/mnt/project/qwen-sourceflow-safe-rl-20261024')
HISTORICAL = {
    1300: NATIVE / 'latest_NAVTEST_20261024T152508Z/analysis/scadv3_summary.json',
    12900: NATIVE / 'latest_NAVTEST_20261024T234038Z/analysis/scadv3_summary.json',
    26900: NATIVE / 'latest_NAVTEST_20261025T084551Z/analysis/scadv3_summary.json',
    32000: NATIVE / 'latest_NAVTEST_20261025T120536Z/analysis/scadv3_summary.json',
    50000: SAFE / 'analysis/RL_scadv3_update050000_NAVTEST.json',
}


def verified(path):
    row = read(path); body = dict(row)
    if body.pop('sha256') != seal(body):
        raise ValueError('Changed analysis seal: ' + str(path))
    if row.get('status') != 'COMPLETE' or row.get('scenes') != 12146 or row.get('logs') != 136:
        raise ValueError('Incomplete full Navtest analysis: ' + str(path))
    return row


def wait_for_owner(plan, wait):
    root = Path(plan['root'])
    while not ((root / 'COMPLETE.json').exists() and (root / 'lease_closed.json').exists()):
        if (root / 'lease_closed.json').exists() and (root / 'failure.json').exists():
            raise RuntimeError('Owner failed; preserve evidence and do not automatically retry')
        owner = read(root / 'owner.json'); stat = Path('/proc') / str(owner['pid']) / 'stat'
        alive = stat.exists() and int(stat.read_text().split()[21]) == owner['start_ticks']
        if not alive:
            raise RuntimeError('Registered owner exited without a complete, released scan')
        if not wait:
            raise RuntimeError('Scan is still running; use --wait or return after owner completion')
        time.sleep(30)
    done = read(root / 'COMPLETE.json')
    if (done['status'] != 'COMPLETE' or not done['all_owned_workers_exited'] or
            done['registration_sha256'] != plan['sha256']):
        raise RuntimeError('Not all registered models completed; no incomplete main results')
    return done


def run_analysis(root, name, module, current, reference):
    destination = root / 'analysis' / (name + '.json')
    if not destination.exists():
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with (root / 'analysis' / (name + '.log')).open('x') as log:
            subprocess.run([MODEL_PYTHON, '-m', module, '--evaluation', str(current),
                            '--reference', str(reference), '--output', str(destination)],
                           cwd=Path(__file__).resolve().parents[3], env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    row = verified(destination)
    if row['evaluation'] != str(current) or row['reference'] != str(reference):
        raise ValueError('Analysis belongs to a different exact checkpoint pair')
    return destination, row


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--registration', required=True)
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    os.nice(10)
    plan = read(args.registration); validate_plan(plan)
    root = Path(plan['root']); report = Path(args.report_dir)
    done = wait_for_owner(plan, args.wait)
    summary = {label: verified(root / 'analysis' / (label + '_summary.json')) for label in plan['models']}
    for label, row in summary.items():
        if (row['evaluation'] != plan['models'][label]['output'] or row['seed'] != plan['seed'] or
                row['metric'] != plan['metric'] or row['inference'] != plan['inference']):
            raise ValueError('Summary differs from sealed registration: ' + label)
    historical = {step: verified(path) for step, path in HISTORICAL.items()}
    curve = {step: row for step, row in historical.items()}
    for label, row in summary.items():
        if label.startswith('SCADV3_'):
            curve[plan['models'][label]['updates']] = row
    best_step = max(curve, key=lambda step: curve[step]['PDMS'])
    best = curve[best_step]
    compare = 'scripts.analysis.qwen_sourceflow.compare_rl_checkpoints'
    paired = 'scripts.analysis.qwen_sourceflow.analyze_paired'
    il = 'scripts.analysis.qwen_sourceflow.analyze_il_milestones'
    analyses = {}
    jobs = [
        ('best_SCADV3_vs_CANONICAL50k', compare, best['evaluation'], plan['baseline_CANONICAL50k']),
        ('best_SCADV3_vs_own_IL53k', paired, best['evaluation'], plan['SCADV3_parent_baseline']),
        ('RL77_latest_vs_exact_IL77_control', paired, summary['RL77_LATEST']['evaluation'], summary['IL77_CONTROL']['evaluation']),
        ('IL86k_vs_IL77k', il, summary['IL_LATEST']['evaluation'], NATIVE / 'evaluations/IL_update077000_NAVTEST_user_20261025T120536Z'),
        ('DINO14k_vs_DINO5k', il, summary['DINO_FRESH_LATEST']['evaluation'], NATIVE / 'evaluations/DINO_FRESH_update005000_NAVTEST_user_20261025T120536Z'),
    ]
    if best_step != 50000:
        jobs.insert(0, ('best_SCADV3_vs_fixed50k', compare, best['evaluation'], historical[50000]['evaluation']))
    for label in ['IL_080000', 'IL_087000']:
        if label in summary:
            jobs.append((label + '_vs_IL77k', il, summary[label]['evaluation'],
                         NATIVE / 'evaluations/IL_update077000_NAVTEST_user_20261025T120536Z'))
    for name, module, current, reference in jobs:
        path, row = run_analysis(root, name, module, current, reference)
        analyses[name] = dict(path=str(path), sha256=row['sha256'])
    cost = dict(logical_single_candidates=sum(r['logical_single_candidates'] for r in summary.values()),
                physical_inference_including_retries=sum(r['physical_inference_including_retries'] for r in summary.values()),
                worker_wall_seconds=sum(r['worker_wall_seconds'] for r in summary.values()),
                model_loading_worker_seconds=sum(r['model_loading_seconds'] for r in summary.values()),
                forward_worker_seconds=sum(r['forward_worker_seconds'] for r in summary.values()),
                scoring_wait_worker_seconds=sum(r['scoring_wait_worker_seconds'] for r in summary.values()),
                elapsed_owner_seconds=done['at'] - read(root / 'owner.json')['at'],
                physical_cost_is_lower_bound=any(r.get('physical_cost_is_lower_bound', False) for r in summary.values()),
                transport_lost_invocations=[x for r in summary.values() for x in r.get('transport_lost_invocations', [])],
                semantics='Shared worker occupancy; neither exclusive GPU compute nor sum of workers equals job wall.')
    if cost['logical_single_candidates'] != plan['total_new_single_candidate_inferences']:
        raise ValueError('Changed logical evaluation budget')
    placements = [read(path) for path in (root / 'jobs').glob('*/meta.json')]
    cost['physical_GPU_UUIDs'] = len({u for j in placements for u in j['spec']['gpu_uuids']})
    cost['job_invocations'] = len(placements)
    cost['export_and_parity_costs'] = 'Separate native job receipts in registration root; four historic SCADV exports/parity proofs reused, not free data.'
    if plan.get('previous_control_root'):
        previous = Path(plan['previous_control_root'])
        start = read(previous / 'owner.json')['at']
        cost['elapsed_campaign_span_seconds'] = done['at'] - start
        cost['elapsed_prior_failed_owner_seconds'] = read(previous / 'COMPLETE.json')['at'] - start
        cost['elapsed_owner_scope'] = 'Continuation only; campaign span includes the failed transport attempt and handoff.'
    result = dict(schema='finite_navtest_scan_report_v1', status='COMPLETE',
                  completed_utc=datetime.now(timezone.utc).isoformat(), registration_sha256=plan['sha256'],
                  summaries=summary, SCADV3_curve={str(k): v for k, v in sorted(curve.items())},
                  highest_observed_SCADV3_update=best_step, highest_observed_SCADV3_PDMS=best['PDMS'],
                  fixed_SCADV3_50000_PDMS=historical[50000]['PDMS'],
                  checkpoint_selection='Exploratory Test scan; bootstrap does not correct for checkpoint selection.',
                  analyses=analyses, costs=cost, lease_closed=read(root / 'lease_closed.json'))
    result['sha256'] = seal(result)
    report.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.registration, report / 'registration.json')
    (report / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, entry in analyses.items():
        shutil.copyfile(entry['path'], report / (name + '.json'))
    lines = ['# SCADV3 intermediate and latest-checkpoint Navtest', '',
             'Full12146 scenes/136 logs; canonical NAVSIM v1.1; single-candidate shift5 ODE K10; seed2101. '
             'All continuation workers exited and exact scene membership/seals passed. The former vla-zt3 admission is closed. '
             'No training, checkpoint choice or reward configuration was changed.', '',
             '| SCADV3 optimizer update | Full Navtest PDMS |', '|---:|---:|']
    lines += [f'| {step:,} | {row["PDMS"]:.6f} |' for step, row in sorted(curve.items())]
    lines += ['', f'Highest observed SCADV3: **{best_step:,}, {best["PDMS"]:.6f}**. '
              f'Fixed50k endpoint: **{historical[50000]["PDMS"]:.6f}**. '
              'This is exploratory Test checkpoint scanning; a higher intermediate score is not an independent confirmation.', '',
              '| Additional checkpoint | Updates | Full Navtest PDMS |', '|---|---:|---:|']
    lines += [f'| {label} | {plan["models"][label]["updates"]:,} | {row["PDMS"]:.6f} |'
              for label, row in summary.items() if not label.startswith('SCADV3_')]
    lines += ['', 'Paired analyses preserve strict source, protocol, population, precision and parent checks. '
              'IL77-parent RL is compared with its newly reevaluated exact-source IL77 control, not with IL53. '
              'DINO14k is compared with fresh DINO5k, not the paused warm-start experiment.', '',
              '| Paired comparison | Net PDMS points | 95% log-bootstrap interval | G+ | G− | Nonzero→zero / zero→nonzero |',
              '|---|---:|---|---:|---:|---:|']
    for name, entry in analyses.items():
        a = read(entry['path']); b = a['PDMS_log_cluster_bootstrap']
        plus = a.get('G_plus_PDMS_points', 100 * a.get('G_plus_raw_score_sum', 0) / a['scenes'])
        minus = a.get('G_minus_PDMS_points', 100 * a.get('G_minus_raw_score_sum', 0) / a['scenes'])
        lo, hi = [100 * x for x in b['CI95']]
        lines.append(f'| {name} | {100*b["mean"]:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {plus:.6f} | {minus:.6f} | {a["nonzero_to_zero"]} / {a["zero_to_nonzero"]} |')
    lines += ['', 'NC/DAC/TTC new violations and recoveries are recorded separately in each linked paired JSON; '
              'net zero-score counts do not establish safety improvement. Uncertainty is paired log-cluster resampling only, '
              'with one training and one inference seed; it does not include seed or checkpoint-selection uncertainty.', '',
              f'New logical single candidates: {cost["logical_single_candidates"]:,}; physical including retries: '
              f'{cost["physical_inference_including_retries"]:,}. This owner elapsed: {cost["elapsed_owner_seconds"]/3600:.3f} h; '
              f'summed shared worker occupancy: {cost["worker_wall_seconds"]/3600:.3f} h. '
              'Export/parity and historic reuse costs remain separate in native receipts.', '',
              'Machine-readable details: [results.json](results.json); frozen scope: [registration.json](registration.json). '
              'All scores use the same official metric; no oracle, best-of-N, seed selection or protocol switch.', '']
    if cost['physical_cost_is_lower_bound']:
        lines += ['The previous vla-zt3 transport stopped with released actor/worker locks. Original invocation records '
                  'remain unmodified and their exact process exits cannot be remotely confirmed. Only missing scenes '
                  'were resumed on fixed hosts under the same rank locks/source/seed. Recorded physical costs are '
                  'lower bounds because lost-host tail time is unknown; it is not filled with zero. The prior attempt '
                  'is retained as COMPLETE_WITH_FAILURES, not relabeled as successful.', '']
    if 'elapsed_campaign_span_seconds' in cost:
        lines += [f'Full elapsed campaign span, including original failed attempt, handoff and continuation: '
                  f'{cost["elapsed_campaign_span_seconds"]/3600:.3f} h. This wall time includes idle/lost-host delay, '
                  'and is not GPU compute time.', '']
    (report / 'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='COMPLETE', report=str(report), best_SCADV3_update=best_step, PDMS=best['PDMS'])))


if __name__ == '__main__':
    main()
