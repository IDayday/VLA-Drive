"""Plot an existing sealed-input loss analysis; no model or score rerun."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--analysis', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = json.loads(Path(args.analysis).read_text())
    rows = result['points_by_window'][str(result['primary_window'])]
    stats = result['correlations'][str(result['primary_window'])]
    steps = [r['update']/1000 for r in rows]; pdms = [r['PDMS'] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    axes[0, 0].plot(steps, pdms, 'o-', color='#1b7837')
    axes[0, 0].set(xlabel='IL optimizer updates (thousands)', ylabel='Navtest PDMS (0–100)', title='Measured checkpoint scores')
    axes[0, 1].plot(steps, [r['total'] for r in rows], 'o-', color='#2166ac')
    axes[0, 1].set(xlabel='IL optimizer updates (thousands)', ylabel='Weighted online TRAIN loss', title='Trailing500 updates; already global across8 ranks')
    offsets = [(6, 8), (6, -9), (6, -19), (6, 6), (6, -10), (6, 6), (6, -11), (6, 6)]
    for ax, component in zip(axes[1], ['total', 'x']):
        x = [r[component] for r in rows]; metric = stats[component]['levels']
        ax.scatter(x, pdms, c=steps, cmap='viridis', s=46, zorder=3)
        for row, xx, yy, offset in zip(rows, x, pdms, offsets):
            ax.annotate(f'{row["update"]//1000}k', (xx, yy), xytext=offset, textcoords='offset points', fontsize=8)
        ax.set(xlabel='Weighted total TRAIN loss' if component == 'total' else 'Normalized clean-x TRAIN L1',
               ylabel='Navtest PDMS (0–100)',
               title=f'n=8; Pearson {metric["pearson"]:.3f}; Spearman {metric["spearman"]:.3f}')
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.margins(x=.12, y=.15)
    fig.suptitle('Qwen-World IL: overall loss association does not ensure adjacent score gains', fontsize=12)
    fig.supxlabel('One training/inference seed. Lines connect observed checkpoints only. No Test loss was computed.', fontsize=9)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/'loss_pdms.png', dpi=180)
    fig.savefig(out/'loss_pdms.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
