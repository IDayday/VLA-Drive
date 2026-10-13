"""Paired original-deployment contrasts, retaining log clusters and scene weights."""
import argparse
import json
from pathlib import Path

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import SAFETY_METRICS, file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('left-summary', 'right-summary', 'contrast', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--paired-single-seed-transitions', action='store_true',
                   help='Also count paired violations/zeros; never infer these events from multi-seed means')
    a = p.parse_args(); require_dispatch()
    paths = [Path(x) for x in (a.left_summary, a.right_summary)]
    reports = [json.loads((x/'report.json').read_text()) for x in paths]
    left, right = reports
    for report in reports:
        if fingerprint({k: v for k, v in report.items() if k != 'sha256'}) != report['sha256']:
            raise ValueError('Published policy report hash changed')
    for key in ('scenes', 'inference_seeds', 'cohort', 'baseline_identity', 'reference_identity', 'deployment_execution_profile'):
        if key not in left or left[key] != right[key]:
            raise ValueError('Unmatched original-profile comparison: '+key)
    if any(x['status'] != 'COMPLETE_MATCHED_EVALUATION' or not x['no_best_of_N'] for x in reports):
        raise ValueError('Only complete fresh single-candidate evaluations are policy results')
    rows = [{x['token']: x for x in json.loads((root/'per_scene.json').read_text())} for root in paths]
    if set(rows[0]) != set(rows[1]) or len(rows[0]) != left['scenes']:
        raise ValueError('Scene denominator changed')
    tokens = sorted(rows[0])
    if any(rows[0][t]['log'] != rows[1][t]['log'] for t in tokens):
        raise ValueError('Log identity changed')
    metrics = sorted(rows[0][tokens[0]]['means'])
    paired = {metric: log_bootstrap_difference(
        [rows[0][t]['means'][metric]-rows[1][t]['means'][metric] for t in tokens],
        [rows[0][t]['log'] for t in tokens]) for metric in metrics}
    report = dict(status='COMPLETE_MATCHED_POLICY_CONTRAST', contrast=a.contrast,
                  direction='left_minus_right', left_report_sha256=left['sha256'], right_report_sha256=right['sha256'],
                  source_sha256=file_sha(__file__), scenes=len(tokens), units='raw_0_to_1',
                  paired_changes=paired, deployment_execution_profile=left['deployment_execution_profile'],
                  both_share_fixed_SFT_baseline_and_reference=True, no_best_of_N=True,
                  training_seed_repetitions=1, seed_uncertainty_not_in_cluster_CI=True,
                  counts_are_in_parent_reports_not_subtracted_as_independent_samples=True)
    if a.paired_single_seed_transitions:
        if left['inference_seeds'] != [2101]:
            raise ValueError('Paired event counts require the registered single output, not averaged seeds')
        transitions = {}
        for metric in (*SAFETY_METRICS, 'joint', 'nonzero_score'):
            keys = SAFETY_METRICS if metric == 'joint' else (metric,)

            def passes(row):
                return row['means']['score'] > 0 if metric == 'nonzero_score' else all(row['means'][k] == 1 for k in keys)

            now, old = [[passes(rows[index][token]) for token in tokens] for index in (0, 1)]
            transitions[metric] = dict(left_violations=sum(not x for x in now), right_violations=sum(not x for x in old),
                new_in_left=sum(y and not x for x, y in zip(now, old)),
                recovered_in_left=sum(not y and x for x, y in zip(now, old)),
                failed_in_both=sum(not x and not y for x, y in zip(now, old)))
        report.update(paired_single_seed_transitions=transitions,
            transitions_population='Full matched evaluation population; fixed-SFT safe-set membership remains in parent reports.',
            per_scene_files_sha256=[file_sha(path/'per_scene.json') for path in paths],
            single_seed_any_all_mean_zero_events_coincide=True)
    report['sha256'] = fingerprint(report)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
