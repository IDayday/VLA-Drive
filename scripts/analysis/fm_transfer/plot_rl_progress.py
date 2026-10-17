"""Plot completed own-SFT RL observations and paired safety transitions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.analysis.fm_transfer.analyze_evaluation import match_transfer
from starVLA.rl.fm_transfer.contracts import CONFIGS, ROOT
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.simwam_recipe.contracts import file_sha


def load_evidence(root, variant):
    endpoint = CONFIGS[variant]['max_updates']
    reference_path = root / 'analysis' / f'{variant}_SFT_update{endpoint:06d}_DEV.json'
    reference = json.loads(reference_path.read_text())
    identity = reference['identity']
    if (reference['status'] != 'COMPLETE_POPULATION' or identity['stage'] != 'SFT'
            or identity['seed'] != 2101 or identity['cohort'] != 'DEV'
            or identity['qualification_only'] or identity['new_SFT']['updates'] != endpoint
            or identity['new_SFT']['config']['variant'] != variant):
        raise ValueError('Not the complete registered own-SFT deployment reference')
    rows, missing = [], []
    for update in (5000, 10000, 15000, 20000):
        path = root / 'analysis' / f'{variant}_RL_update{update:06d}_DEV_paired_own_SFT.json'
        if not path.exists():
            missing.append(update)
            continue
        result = json.loads(path.read_text())
        export_path = root / 'exports' / variant / f'RL_update{update:06d}' / 'export.json'
        export = json.loads(export_path.read_text())
        if (result['status'] != 'COMPLETE_POPULATION' or result['identity']['stage'] != 'RL'
                or result['identity']['qualification_only'] or export['updates'] != update
                or export['weight_sha256'] != result['identity']['weight_sha256']
                or export['new_SFT'] != identity['new_SFT']
                or Path(result['reference']) != root / 'evaluations' / f'{variant}_SFT_update{endpoint:06d}_DEV'):
            raise ValueError('Not a complete matched registered RL milestone')
        match_transfer(result['identity'], identity)
        for mode in ('E_NATIVE', 'E_SHIFT5'):
            paired = result['modes'][mode]['paired_comparison']
            gain = result['modes'][mode]['mean']['score'] - reference['modes'][mode]['mean']['score']
            if abs(gain - paired['PDMS_log_cluster_bootstrap']['mean']) > 1e-12:
                raise ValueError('Paired gain does not match the own-SFT reference')
            if paired['reference_safe_population'] != reference['modes'][mode]['fully_safe_scenes']:
                raise ValueError('Reference-safe population changed')
        rows.append(dict(update=update, path=str(path), sha256=file_sha(path),
                         export_sha256=file_sha(export_path), modes=result['modes']))
    if not rows:
        raise ValueError('No completed own-SFT RL observations')
    return reference_path, reference, rows, missing


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--variant', choices=CONFIGS, required=True)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != '.png':
        raise ValueError('Output must be PNG; PDF and evidence JSON are saved alongside')
    paths = [args.output, args.output.with_suffix('.pdf'), args.output.with_suffix('.json')]
    if any(path.exists() for path in paths):
        raise FileExistsError('Preserve prior figures and evidence')
    ref_path, reference, rows, missing = load_evidence(args.root, args.variant)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.7))
    x = [0] + [row['update'] / 1000 for row in rows]
    styles = [('E_NATIVE', '#2563eb', 'Native uniform 10-step ODE'),
              ('E_SHIFT5', '#d97706', 'Shift5 10-step ODE')]
    for index, (mode, color, label) in enumerate(styles):
        means = [reference['modes'][mode]['mean_percent']['score']] + [r['modes'][mode]['mean_percent']['score'] for r in rows]
        axes[0, 0].plot(x, means, marker='o', color=color, label=label, linewidth=1.5)
        gains = [0.] + [100 * r['modes'][mode]['paired_comparison']['PDMS_log_cluster_bootstrap']['mean'] for r in rows]
        axes[0, 1].plot(x, gains, marker='o', color=color, linewidth=1.5)
        for row in rows:
            low, high = [100 * v for v in row['modes'][mode]['paired_comparison']['PDMS_log_cluster_bootstrap']['CI95']]
            xp = row['update'] / 1000 + (-.08 if index == 0 else .08)
            axes[0, 1].vlines(xp, low, high, color=color, linewidth=1.3)
            axes[0, 1].hlines([low, high], xp - .12, xp + .12, color=color, linewidth=1.3)
        ax = axes[1, index]
        largest_count = 0
        for metric, metric_label, metric_color in [
                ('no_at_fault_collisions', 'NC', '#7c3aed'),
                ('drivable_area_compliance', 'DAC', '#059669'),
                ('time_to_collision_within_bound', 'TTC', '#dc2626')]:
            transitions = [r['modes'][mode]['paired_comparison']['safety_transitions'][metric] for r in rows]
            largest_count = max(largest_count, *(t[key] for t in transitions for key in ('new_violation', 'recovery')))
            ax.plot(x, [0] + [t['new_violation'] for t in transitions], 'o-', color=metric_color, label=metric_label + ' new', linewidth=1.4, markersize=4)
            ax.plot(x, [0] + [t['recovery'] for t in transitions], 's--', color=metric_color, label=metric_label + ' recovered', linewidth=1.2, markersize=3)
        joint = [r['modes'][mode]['paired_comparison']['reference_safe_new_joint_violation'] for r in rows]
        safe_n = reference['modes'][mode]['fully_safe_scenes']
        ax.set_title(label + '\nNew joint failures in fixed SFT-safe set: ' + '/'.join(map(str, joint)) + f' (N={safe_n})', fontsize=9)
        ax.set_ylabel('Scenes, paired with own SFT')
        ax.legend(fontsize=7.5, ncol=3, loc='upper right')
        ax.set_ylim(-.7, largest_count * 1.25 + 1)
    axes[0, 0].set_title('Actual single-candidate deployment score')
    axes[0, 0].set_ylabel('Canonical v1.1 PDMS (0-100)')
    low, high = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(low - .25 * (high - low), high)
    axes[0, 0].legend(fontsize=8, loc='lower right')
    axes[0, 1].set_title('RL gain vs each grid\'s own SFT + log-cluster 95% CI')
    axes[0, 1].set_ylabel('PDMS change (percentage points)')
    axes[0, 1].axhline(0, color='#666666', linewidth=.7, linestyle='--')
    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xlabel('RL optimizer updates (thousands); 0 = own SFT')
        ax.grid(axis='y', alpha=.18)
    fig.suptitle(args.variant + ' — completed full DEV observations only', fontsize=13)
    fig.subplots_adjust(hspace=.46, wspace=.25, bottom=.12, top=.9)
    fig.text(.5, .028, f"DEV {reference['scenes']} scenes / {reference['logs']} logs; seed2101; no candidate selection. "
             'Training/inference seed uncertainty excluded.\n'
             'NC/DAC/TTC require full pass; violations are not all collisions. '
             'Missing RL milestones (not extrapolated): ' + (', '.join(str(x) for x in missing) or 'none'),
             ha='center', fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths[0], dpi=180)
    fig.savefig(paths[1])
    plt.close(fig)
    atomic_json(paths[2], dict(schema='fm_transfer_observed_RL_curve_v1', variant=args.variant,
        source_sha256=file_sha(__file__), reference=dict(path=str(ref_path), sha256=file_sha(ref_path)),
        observed=rows, missing_registered_updates=missing, new_inference=0, new_scores=0, optimizer_updates=0,
        SFT_zero_point='Actual own-SFT score; zero paired difference and transitions to itself.',
        not_Navtest=True, image_sha256=file_sha(paths[0]), pdf_sha256=file_sha(paths[1])))
    print(json.dumps(dict(output=str(paths[0]), updates=[r['update'] for r in rows], missing=missing)))


if __name__ == '__main__':
    main()
