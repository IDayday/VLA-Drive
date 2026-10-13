"""Paired original-deployment contrasts, retaining log clusters and scene weights."""
import argparse
import json
from pathlib import Path

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.safe_flow_improvement.contracts import file_sha, fingerprint
from starVLA.rl.safe_flow_improvement.diagnostics import log_bootstrap_difference
from starVLA.rl.safe_flow_improvement.iteration import require_dispatch


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('left-summary', 'right-summary', 'contrast', 'output'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args(); require_dispatch()
    paths = [Path(x) for x in (a.left_summary, a.right_summary)]
    reports = [json.loads((x/'report.json').read_text()) for x in paths]
    left, right = reports
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
    report['sha256'] = fingerprint(report)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
    atomic_json(out/'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
