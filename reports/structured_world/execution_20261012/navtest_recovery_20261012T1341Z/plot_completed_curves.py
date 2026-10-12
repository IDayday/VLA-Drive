"""Plot only complete registered Navtest observations; never infer missing scores."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
with (root / 'LEARNING_CURVES.csv').open() as stream:
    rows = [r for r in csv.DictReader(stream) if r['variant'] == 'q_final']
fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
for group, label, color in [
    ('G0_NO_FUTURE_LABEL', 'G0: no future labels', '#666666'),
    ('G1_FULL_UNIFORM', 'G1: full occupancy, uniform', '#0072B2'),
    ('G2_EVENT_UNIFORM', 'G2: events, uniform', '#D55E00'),
    ('G3_EVENT_LOCAL', 'G3: events, planning-local', '#009E73'),
]:
    selected = sorted([r for r in rows if r['group'] == group], key=lambda r: int(r['updates']))
    assert selected and all(int(r['scenes']) == 12146 and int(r['failed']) == 0 for r in selected)
    ax.plot([int(r['updates']) / 1000 for r in selected], [float(r['score']) for r in selected],
            marker='o', markersize=5, linewidth=1.6, color=color, label=label)
ax.set(xlabel='Optimizer updates (thousands)', ylabel='Canonical Navtest PDMS',
       title='G-series: complete 12,146-scene observations, final trajectory')
ax.grid(alpha=0.2)
ax.legend(loc='lower right', fontsize=9)
fig.text(0.015, -0.025, 'One training seed. Lines connect measured checkpoints; missing observations are not estimates.', fontsize=8)
fig.savefig(root / 'G_NAVTEST_CURVES.svg', bbox_inches='tight')
fig.savefig(root / 'G_NAVTEST_CURVES.png', dpi=150, bbox_inches='tight')
print(root / 'G_NAVTEST_CURVES.svg')
