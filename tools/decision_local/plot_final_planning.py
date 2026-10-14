"""Plot fixed final Navtest comparisons and both directions of failure transfer."""
import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, digest, read, source_sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed42', type=Path, required=True)
    p.add_argument('--seed43', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    rows, pairs = [], []
    for seed, directory in [(42, a.seed42), (43, a.seed43)]:
        data = read(directory/'summary.json')
        if data['population'] != 'navtest' or data['tables']['oracle_probe']:
            raise ValueError('Final camera-only Navtest required')
        for row in data['tables']['camera_only']:
            if row['scenes'] != 12146 or not row['valid']:
                raise ValueError('Incomplete population')
            paired = read(directory/(row['run']+'_versus_q0.json'))['paired']
            rows.append({'seed': seed, 'name': row['run'],
                'delta_pp': paired['delta_percentage_points'],
                'CI95_pp': [v*100 for v in paired['metrics']['score']['log_cluster_95_interval']],
                'repairs': row['zero_to_nonzero'], 'new_failures': row['nonzero_to_zero']})
        for name, pair in data['comparisons'].items():
            if pair['primary_preregistered_comparison']:
                pairs.append({'seed': seed, 'name': name, 'delta_pp': pair['paired']['delta_percentage_points'],
                    'CI95_pp': [v*100 for v in pair['paired']['metrics']['score']['log_cluster_95_interval']]})
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw={'width_ratios': [1.1, .9, 1.4]})
    def label(row):
        return row['name'].replace('R_FUTURE_', 'Future ').replace('R_CURRENT_PRED', 'Current prediction').replace('E_NONE', 'No relations').replace('_minus_', ' - ')+' / '+str(row['seed'])
    for ax, values in [(axes[0], rows), (axes[2], pairs)]:
        for i, row in enumerate(values):
            lo, hi = row['CI95_pp']; value = row['delta_pp']
            ax.errorbar(value, i, xerr=[[value-lo], [hi-value]], fmt='o', capsize=3,
                        color='#2166ac' if row['seed'] == 42 else '#b35806')
        ax.set_yticks(np.arange(len(values)), [label(x) for x in values]); ax.invert_yaxis()
        ax.axvline(0, color='black', linestyle='--', linewidth=1); ax.grid(axis='x', alpha=.2)
        ax.set_xlabel('PDMS difference (percentage points), 95% log-cluster interval')
    y = np.arange(len(rows))
    axes[1].barh(y-.18, [r['repairs'] for r in rows], .34, label='Zero -> nonzero', color='#4393c3')
    axes[1].barh(y+.18, [r['new_failures'] for r in rows], .34, label='Nonzero -> zero', color='#d6604d')
    axes[1].set_yticks(y, [label(x) for x in rows]); axes[1].invert_yaxis()
    axes[1].set_xlabel('Scenes relative to the same S0 q0'); axes[1].grid(axis='x', alpha=.2)
    axes[0].set_title('All final heads relative to S0')
    axes[1].set_title('Zero -> nonzero (blue); nonzero -> zero (red)')
    axes[2].set_title('Registered comparisons')
    fig.suptitle('Fixed 24-epoch endpoints; 12,146 scenes / 136 logs; one shared S0 training seed')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(a.output/('final_planning.'+suffix), dpi=180, bbox_inches='tight')
    atomic_json(a.output/'MANIFEST.json', {'source_sha': source_sha(ROOT), 'tool_sha256': digest(__file__),
        'summary_sha256': [digest(x/'summary.json') for x in (a.seed42, a.seed43)], 'rows': rows, 'pairs': pairs,
        'limits': 'Head seeds42/43 do not replicate the S0 foundation; pointwise log intervals do not correct post-hoc multiple comparisons or historical test reuse.'})


if __name__ == '__main__':
    main()
