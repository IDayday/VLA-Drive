"""Aggregate correction demand and fixed-refiner dependence, without scene images."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, digest, source_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data, receipts = {}, {}
    for split in ('train4096', 'dev'):
        root = args.diagnostics/('seed42_E_FUTURE_GT_'+split)
        rows = [json.loads(line) for line in (root/'scenes.jsonl').read_text().splitlines()]
        expected = 4096 if split == 'train4096' else 1696
        if len(rows) != expected or len({row['token'] for row in rows}) != expected:
            raise ValueError('Use the complete fixed diagnostic population')
        if any(row['valid_future_points'] != 8 for row in rows):
            raise ValueError('This plot requires the registered full-horizon scenes')
        data[split] = rows
        receipts[split] = {'summary_sha256': digest(root/'summary.json'),
                           'scene_table_sha256': digest(root/'scenes.jsonl'), 'scenes': expected}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), layout='constrained')
    colors = ['#3569a8', '#cc6225']
    names = ['Train diagnostic (4096)', 'Development (1696)']
    for split, color, name in zip(data, colors, names):
        values = np.sort([row['q0_ADE_m'] for row in data[split]])
        axes[0].plot(values, np.arange(1, len(values)+1)/len(values), color=color, label=name)
    axes[0].set(xscale='log', xlabel='Original S0 q0 ADE / m', ylabel='Empirical cumulative fraction',
                title='Different correction demand', ylim=(0, 1.01))
    axes[0].legend(fontsize=8, loc='lower right')
    for ax, fields, labels, scale, title, ylabel in [
            (axes[1], ['q0_ADE_m', 'residual_mean_m'], ['q0 error', 'Learned residual'], 1.,
             'E_FUTURE_GT at common epoch24', 'Mean distance / m'),
            (axes[2], ['relation_zeroing_trajectory_change_m'], ['Relation values zeroed'], 1000.,
             'Fixed-refiner input dependence', 'Mean trajectory change / mm')]:
        x = np.arange(len(fields))
        for i, (split, color, name) in enumerate(zip(data, colors, names)):
            values = [np.mean([row[field] for row in data[split]])*scale for field in fields]
            bars = ax.bar(x+(i-.5)*.34, values, width=.34, color=color, label=name)
            ax.bar_label(bars, fmt='%.3f' if scale == 1 else '%.2f', fontsize=8, padding=3)
        ax.set(xticks=x, xticklabels=labels, title=title, ylabel=ylabel)
        ax.margins(y=.15)
    fig.suptitle('Frozen mature S0: diagnostic evidence, not an oracle upper bound', fontsize=12)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('.png', '.pdf'):
        path = args.output.with_suffix(extension)
        if path.exists():
            raise FileExistsError('Keep previous figure evidence')
        fig.savefig(path, dpi=170)
    plt.close(fig)
    atomic_json(args.output.with_suffix('.json'), {'source_sha': source_sha(ROOT),
        'tool_sha256': digest(__file__), 'inputs': receipts,
        'limits': ['Training diagnostic is in-sample for both S0 and the refiner',
                   'Relation zeroing preserves the same GT support and queries; it is not a trained E_NONE control',
                   'Both panels describe a privileged probe, not deployable planning gains',
                   'No example or checkpoint selection; no private image or spatial map exported']})


if __name__ == '__main__':
    main()
