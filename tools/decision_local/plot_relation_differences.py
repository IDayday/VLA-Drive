"""Export the completed ABS/DIFF paired-effect evidence as a research figure."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = {seed: json.loads((args.analysis_root/f'seed{seed}_DIFF_minus_ABS_dev.json').read_text())
            for seed in (42, 43)}
    if any(x['scenes'] != 1696 or x['logs'] != 16 for x in rows.values()):
        raise ValueError('The figure requires both completed common development populations')
    keys = [
        ('road_margin_m_absolute_MAE', 'Road absolute error'),
        ('road_margin_m_pair_MAE', 'Road pair error'),
        ('future_clearance_m_absolute_MAE', 'Future clearance absolute error'),
        ('future_clearance_m_pair_MAE', 'Future clearance pair error'),
        ('near_sensitive_future_clearance_m_pair_MAE', 'Near-sensitive clearance pair error')]
    categorical = [('future_clearance_m_direction_accuracy', 'Clearance difference direction'),
                   ('future_overlap_recall', 'Future overlap recall'),
                   ('future_overlap_precision', 'Future overlap precision')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout='constrained', gridspec_kw={'width_ratios': [1.15, 1]})
    for ax, entries, scale, xlabel in [(axes[0], keys, 1, 'DIFF minus ABS / m (lower is better)'),
                                      (axes[1], categorical, 100, 'DIFF minus ABS / percentage points')]:
        ax.axvline(0, color='0.4', linewidth=.9, linestyle='--')
        for seed, color, marker, offset in [(42, '#0072B2', 'o', -.12), (43, '#D55E00', 's', .12)]:
            values = [rows[seed]['metrics'][key] for key, _ in entries]
            centers = np.array([x['delta'] for x in values])*scale
            bounds = np.array([x['paired_log_cluster_95_interval'] for x in values])*scale
            error = np.vstack((centers-bounds[:, 0], bounds[:, 1]-centers))
            if np.any(error < 0):
                raise ValueError('Use explicit interval segments if a point is outside its percentile interval')
            ax.errorbar(centers, np.arange(len(entries))+offset, xerr=error, fmt=marker,
                        color=color, capsize=3, markersize=5, linewidth=1.2, label=f'Head training seed {seed}')
        ax.set_yticks(np.arange(len(entries)), [label for _, label in entries])
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.grid(axis='x', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, -.01), ncol=2, fontsize=9)
    axes[0].set_title('Continuous consequences')
    axes[1].set_title('Direction and conflict judgments')
    fig.suptitle('Frozen S0; matched 24-epoch readouts on 1696 development scenes\n'
                 'Intervals: paired log-cluster bootstrap (16 logs); one shared S0 training seed', fontsize=11)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output/'ABS_DIFF_DEVELOPMENT.png', dpi=180, bbox_inches='tight')
    fig.savefig(args.output/'ABS_DIFF_DEVELOPMENT.pdf', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
