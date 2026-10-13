"""Plot sealed single-seed DEV summaries without filling unfinished windows."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from starVLA.rl.safe_flow_improvement.contracts import fingerprint


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--summary', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    report = json.loads(Path(a.summary).read_text())
    checksum = report.pop('sha256')
    if fingerprint(report) != checksum or report['inference_seeds'] != [2101] or report['training_seeds'] != [42]:
        raise ValueError('Only the sealed current single-seed research contract is supported')
    rows = {x['name']: x for x in report['evaluations'] if x['status'] == 'COMPLETE_MATCHED_EVALUATION'}
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    groups = [
        ([1, 2, 4, 8], 'epoch', [('P0_REWARD_FM', 'P0 reward FM', '#0072B2'),
            ('P1_SAFE_FM', 'P1 safe FM', '#56B4E9'), ('P2_SEARCH_SAFE_FM', 'P2 search safe FM', '#D55E00')]),
        ([1, 2, 3], 'round', [('ODE_REWARD_FM', 'ODE reward FM', '#0072B2'),
            ('ODE_SAFE_FM', 'ODE safe FM', '#009E73'), ('EXPLORE_SAFE_FM', 'Search safe FM', '#D55E00'),
            ('ODE_NFT', 'ODE NFT', '#CC79A7')]),
    ]
    for column, (points, unit, variants) in enumerate(groups):
        for variant, label, color in variants:
            available = [(i, rows[f'{variant}_{unit}{step}']) for i, step in enumerate(points)
                         if f'{variant}_{unit}{step}' in rows]
            if not available:
                continue
            x, values = zip(*available)
            y = [100*v['paired_changes']['score']['mean'] for v in values]
            low = [100*v['paired_changes']['score']['CI95'][0] for v in values]
            high = [100*v['paired_changes']['score']['CI95'][1] for v in values]
            style = '--' if variant == 'P1_SAFE_FM' else '-'
            axes[0, column].errorbar(x, y, yerr=[[max(0, value-lo) for value, lo in zip(y, low)],
                                               [max(0, hi-value) for value, hi in zip(y, high)]],
                                     fmt=style+'o', color=color, label=label, capsize=2, elinewidth=.8)
            axes[0, column].fill_between(x, low, high, color=color, alpha=.10)
            axes[1, column].plot(x, [v['safety']['joint']['current_any_seed_violation_in_fixed_safe'] for v in values],
                                 style, marker='o', color=color)
        axes[0, column].axhline(0, color='black', linewidth=.7)
        baseline_counts = {v['safety']['joint']['baseline_any_eval_seed_violation_in_fixed_safe'] for v in rows.values()}
        if len(baseline_counts) != 1:
            raise ValueError('Fixed SFT baseline differs between methods')
        axes[1, column].axhline(next(iter(baseline_counts)), color='black', linestyle=':', label='Matched SFT')
        axes[1, column].yaxis.set_major_locator(MaxNLocator(integer=True))
        axes[0, column].legend(fontsize=8, loc='upper left')
        for row in range(2):
            axes[row, column].set_xticks(range(len(points)), points)
            axes[row, column].set_xlim(-.2, len(points)-.8)
            axes[row, column].set_xlabel('Fixed-bank epoch' if unit == 'epoch' else 'Online round')
            axes[row, column].grid(alpha=.2)
    axes[0, 0].set_ylabel('PDMS change vs SFT (0–100 points)')
    axes[1, 0].set_ylabel('Violating scenes in fixed SFT-safe set')
    axes[1, 0].legend(fontsize=8)
    axes[0, 0].set_title('Fixed-bank policy extraction')
    axes[0, 1].set_title('Fresh-sample policy iteration')
    pending = ' | Online endpoints pending' if not report['fixed_budget_DEV_endpoints_complete'] else ''
    fig.suptitle('Original 10-step ODE | 1,696 DEV scenes | train42 / inference2101'+pending, fontsize=12)
    fig.supxlabel('Bands: log-cluster 95% intervals, not seed uncertainty. Missing results are not extrapolated.\n'
                   'P0/P1 overlap; finite candidate safety does not guarantee fitted-policy safety.', fontsize=9)
    for suffix in ('svg', 'png'):
        fig.savefig(out/('policy_learning_curves.'+suffix), dpi=150)
    plt.close(fig)
    (out/'source.json').write_text(json.dumps(dict(summary_sha256=checksum, utc=report['utc'],
        complete_evaluations=len(rows), no_candidate_maxima_in_policy_plots=True), indent=2)+'\n')
    print(json.dumps(dict(status='PLOTTED_COMPLETE_WINDOWS_ONLY', complete_evaluations=len(rows))))


if __name__ == '__main__':
    main()
