"""Common-endpoint learning curves from actual run logs and canonical scores."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.decision_local.common import atomic_json, digest, read, source_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous complete learning curves')
    root = args.artifacts
    expected = {42: ['E_NONE', 'E_CURRENT_GT', 'E_CV_GT', 'E_FUTURE_GT', 'E_MASK_ONLY',
                     'P_ABS', 'P_DIFF', 'P_QUERY_ONLY', 'R_CURRENT_PRED', 'R_FUTURE_ABS', 'R_FUTURE_DIFF'],
                43: ['E_NONE', 'E_FUTURE_GT', 'P_ABS', 'P_DIFF', 'R_FUTURE_ABS', 'R_FUTURE_DIFF']}
    epochs, metadata = [], {}
    for seed, kinds in expected.items():
        for kind in kinds:
            run = root/f'formal_seed{seed}'/kind
            done = read(run/'COMPLETE.json')
            if done['epoch'] != 24 or done['exposure'] != 2438208 or done['completed'] != 19056:
                raise ValueError('Do not plot a partial run as the common endpoint')
            rows = [json.loads(line) for line in (run/'train.jsonl').read_text().splitlines()]
            if [row['update'] for row in rows] != list(range(1, 19057)):
                raise ValueError('Training log contains missing or repeated optimizer updates')
            metadata[f'{seed}_{kind}'] = {'training_log_sha256': digest(run/'train.jsonl'),
                                        'complete_sha256': digest(run/'COMPLETE.json')}
            for epoch in range(1, 25):
                chosen = [row for row in rows if row['epoch'] == epoch-1]
                row = {'seed': seed, 'kind': kind, 'epoch': epoch, 'updates': len(chosen),
                       'train_total': float(np.mean([x['total'] for x in chosen])),
                       'update_seconds_median': float(np.median([x['seconds'] for x in chosen]))}
                for key in ('absolute', 'difference', 'trajectory'):
                    if key in chosen[0]:
                        row['train_'+key] = float(np.mean([x[key] for x in chosen]))
                if epoch in (6, 12, 18, 24):
                    val = read(run/f'dev_epoch_{epoch:02d}.json')
                    if val['scenes'] != 1696:
                        raise ValueError('Common development population changed')
                    for key in ('total', 'absolute', 'difference', 'trajectory', 'ADE', 'FDE'):
                        if key in val:
                            row['dev_'+key] = val[key]
                epochs.append(row)
    planning = []
    for registry_path in ('development_observations_v1/REGISTRY.json', 'information_seed43_v1/EVALUATIONS.json'):
        for job in read(root/registry_path)['jobs']:
            result = read(Path(job['scores'])/'COMPLETE.json')
            if not result['q_final']['valid'] or result['q_final']['scenes'] != 1696:
                raise ValueError('Incomplete/invalid planning observation')
            planning.append({'seed': job['seed'], 'kind': job['kind'], 'epoch': job['epoch'],
                             'PDMS': result['q_final']['metrics']['score']*100,
                             'q0_PDMS': result['q0']['metrics']['score']*100})
    if len(planning) != 48 or len({(x['seed'], x['kind'], x['epoch']) for x in planning}) != 48:
        raise ValueError('Expected the registered forty primary and eight extra E observations')
    baseline = {x['q0_PDMS'] for x in planning}
    if max(baseline)-min(baseline) > 1e-10:
        raise ValueError('All learning observations must share the same q0')
    args.output.mkdir(parents=True)
    for name, rows in [('optimization', epochs), ('planning', planning)]:
        with (args.output/(name+'.csv')).open('w') as stream:
            writer = csv.DictWriter(stream, sorted(set().union(*(row.keys() for row in rows))))
            writer.writeheader()
            writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained', sharex=True, sharey=True)
    for ax, seed, prefix, title in zip(axes.flat, (42, 43, 42, 43), ('E_', 'E_', 'R_', 'R_'),
                                      ('Information probes, seed42', 'Key information repeats, seed43',
                                       'Camera-only refiners, seed42', 'Key camera-only repeats, seed43')):
        kinds = sorted({row['kind'] for row in planning if row['seed'] == seed and row['kind'].startswith(prefix)})
        for kind in kinds:
            data = sorted((row for row in planning if row['seed'] == seed and row['kind'] == kind), key=lambda x: x['epoch'])
            ax.plot([x['epoch'] for x in data], [x['PDMS'] for x in data], 'o-', label=kind, markersize=4)
        ax.axhline(next(iter(baseline)), color='black', ls='--', linewidth=1, label='Shared frozen S0 q0')
        ax.set(title=title, xlabel='Epoch', ylabel='Development PDMS', xticks=[6, 12, 18, 24])
        ax.grid(alpha=.15)
        ax.legend(fontsize=7, loc='lower left')
    fig.suptitle('All registered observations; main comparison uses epoch24, not the peak')
    for extension in ('.png', '.pdf'):
        fig.savefig(args.output/('planning_learning'+extension), dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), layout='constrained')
    for ax, kinds, key, title in [(axes[0], ['P_ABS', 'P_DIFF'], 'absolute', 'Common absolute relationship loss'),
                                  (axes[1], ['E_NONE', 'E_FUTURE_GT', 'R_FUTURE_ABS', 'R_FUTURE_DIFF'],
                                   'trajectory', 'Original trajectory correction loss')]:
        for kind in kinds:
            values = [row for row in epochs if row['seed'] == 42 and row['kind'] == kind]
            line = ax.plot([x['epoch'] for x in values], [x['train_'+key] for x in values], label=kind+' train')[0]
            val = [x for x in values if 'dev_'+key in x]
            ax.plot([x['epoch'] for x in val], [x['dev_'+key] for x in val], 'o--', color=line.get_color(), label=kind+' dev', markersize=4)
        ax.set(title=title, xlabel='Epoch', ylabel='Registered normalized loss', yscale='log')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=.15)
    fig.suptitle('Seed42 small heads; train and development receive different correction demands')
    for extension in ('.png', '.pdf'):
        fig.savefig(args.output/('optimization_learning'+extension), dpi=170)
    plt.close(fig)
    atomic_json(args.output/'MANIFEST.json', {'source_sha': source_sha(ROOT), 'tool_sha256': digest(__file__),
        'training_logs': metadata, 'planning_observations': 48, 'formal_small_heads': 17,
        'main_endpoint_epoch': 24, 'chosen_by_best_development_score': False,
        'limits': ['Losses retain the registered scene normalization and units, not PDMS surrogates',
                   'Update timings exclude separately recorded checkpoint overhead; whole-run cost is reported separately',
                   'Head seed43 repeats do not replicate the frozen S0 training']})


if __name__ == '__main__':
    main()
