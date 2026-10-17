"""Plot completed paired SFT gains; no interpolation into missing milestones."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from starVLA.rl.fm_transfer.contracts import ROOT,CONFIGS
from starVLA.rl.simwam_recipe.contracts import file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


def plot_learning(args):
    """Read complete fixed SFT observations; keep TRAIN and DEV separate."""
    from scripts.analysis.fm_transfer.analyze_evaluation import match_transfer

    root, output = Path(args.root), Path(args.output)
    if output.suffix != '.png':
        raise ValueError('Learning view writes PNG, PDF and evidence JSON')
    products = [output, output.with_suffix('.pdf'), output.with_suffix('.json')]
    if any(path.exists() for path in products):
        raise FileExistsError('Preserve previous figures and evidence')
    cohorts = ('DEV', 'TRAIN_DIAGNOSTIC')
    references, rows, missing = {}, [], []
    for update in CONFIGS[args.variant]['milestones']:
        paths = {c: root / 'analysis' / f'{args.variant}_SFT_update{update:06d}_{c}.json' for c in cohorts}
        if not all(path.exists() for path in paths.values()):
            if update == 0 or any(path.exists() for path in paths.values()):
                raise ValueError('Both complete fixed populations are required at each plotted milestone')
            missing.append(update)
            continue
        export_path = root / 'exports' / args.variant / f'SFT_update{update:06d}' / 'export.json'
        export = json.loads(export_path.read_text())
        row = dict(update=update, populations={}, export_sha256=file_sha(export_path))
        for cohort, path in paths.items():
            result = json.loads(path.read_text())
            identity = result['identity']
            if (result['status'] != 'COMPLETE_POPULATION' or identity['stage'] != 'SFT'
                    or identity['cohort'] != cohort or identity['seed'] != 2101
                    or identity['qualification_only'] or identity['new_SFT']['qualification']
                    or identity['new_SFT']['config']['variant'] != args.variant
                    or identity['new_SFT']['updates'] != update
                    or export['stage'] != 'SFT' or export['updates'] != update
                    or export['weight_sha256'] != identity['weight_sha256']
                    or export['new_SFT'] != identity['new_SFT']
                    or identity['export_receipt_sha256'] != file_sha(export_path)):
                raise ValueError('Not the registered complete SFT observation')
            if update == 0:
                references[cohort] = result
            else:
                reference = references[cohort]
                match_transfer(identity, reference['identity'])
                for key in ('identity_sha256', 'config'):
                    if identity['new_SFT'][key] != reference['identity']['new_SFT'][key]:
                        raise ValueError('Mixed SFT runs or configurations')
                if (result['scenes'], result['logs']) != (reference['scenes'], reference['logs']):
                    raise ValueError('Changed fixed population')
            row['populations'][cohort] = dict(path=str(path), sha256=file_sha(path),
                scenes=result['scenes'], logs=result['logs'], modes=result['modes'])
        rows.append(row)
    learned = [row for row in rows if row['update'] > 0]
    if not learned:
        raise ValueError('No completed post-initialization observation')
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5))
    styles = [('E_NATIVE', '#2563eb', 'Native uniform'), ('E_SHIFT5', '#d97706', 'Shift5')]
    x = [row['update'] / 1000 for row in learned]
    for column, cohort in enumerate(cohorts):
        reference = references[cohort]
        initial = []
        for mode, color, label in styles:
            values = [row['populations'][cohort]['modes'][mode] for row in learned]
            short_label = 'Native' if mode == 'E_NATIVE' else 'Shift5'
            axes[0, column].plot(x, [v['mean_percent']['score'] for v in values], 'o-',
                                color=color, label=label, linewidth=1.6)
            axes[1, column].plot(x, [reference['scenes'] - v['fully_safe_scenes'] for v in values],
                                'o-', color=color, label=short_label + ': any safety fail', linewidth=1.4)
            axes[1, column].plot(x, [v['zero_scenes'] for v in values], 's--',
                                's--', color=color, label=short_label + ': zero score', linewidth=1.3)
            initial.append(f"{reference['modes'][mode]['mean_percent']['score']:.3f}")
        axes[0, column].set_title(f"{cohort}: {reference['scenes']} scenes / {reference['logs']} logs\n"
                                  f"Initial native/shift5 PDMS: {' / '.join(initial)}", fontsize=10)
        axes[0, column].set_ylabel('Canonical v1.1 PDMS (0-100)')
        axes[0, column].legend(fontsize=8)
        axes[1, column].set_title('Full-population failure counts, not paired new violations', fontsize=9)
        axes[1, column].set_ylabel('Scenes')
        axes[1, column].legend(fontsize=7.5, ncol=2)
        axes[1, column].set_ylim(bottom=0)
    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xlabel('SFT optimizer updates (thousands)')
        ax.grid(axis='y', alpha=.18)
    fig.suptitle(args.variant + ' — completed SFT learning observations; no RL results', fontsize=12)
    fig.subplots_adjust(hspace=.42, wspace=.24, bottom=.16, top=.87)
    fig.text(.5, .025, 'Single-candidate ODE; seed2101; separate grids and populations. '
             'TRAIN diagnostic is deliberately enriched for difficult scenes.\n'
             'Point means only; paired log-cluster intervals are in milestone reports. '
             'Initial point labelled above, omitted from axes for zoom.\n'
             'Safety fail = any NC/DAC/TTC not fully passing; not all are collisions. '
             'Not Navtest. Missing milestones: ' + (', '.join(map(str, missing)) or 'none'),
             ha='center', fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(products[0], dpi=180)
    fig.savefig(products[1])
    plt.close(fig)
    atomic_json(products[2], dict(schema='fm_transfer_observed_SFT_learning_curve_v1',
        variant=args.variant, source_sha256=file_sha(__file__), observed=rows,
        not_available_updates=missing, initial_point='Recorded and labelled, omitted from axes for zoom',
        new_inference=0, new_scores=0, optimizer_updates=0, not_RL_or_Navtest=True,
        image_sha256=file_sha(products[0]), pdf_sha256=file_sha(products[1])))
    print(json.dumps(dict(output=str(output), updates=[row['update'] for row in rows], not_available=missing)))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--variant',choices=CONFIGS,required=True)
    p.add_argument('--root',default=str(ROOT));p.add_argument('--output',required=True)
    p.add_argument('--view', choices=('paired_gain', 'learning'), default='paired_gain');a=p.parse_args()
    if a.view == 'learning':
        return plot_learning(a)
    output=Path(a.output)
    if output.exists() or output.with_suffix('.json').exists():raise FileExistsError('Preserve previous figure and evidence')
    rows=[];missing=[]
    for update in CONFIGS[a.variant]['milestones']:
        if update==0:continue
        path=Path(a.root)/'analysis'/f'{a.variant}_SFT_update{update:06d}_DEV_paired_initial.json'
        if not path.exists():missing.append(update);continue
        d=json.loads(path.read_text());identity=d['identity']
        if (d['status']!='COMPLETE_POPULATION' or identity['stage']!='SFT' or identity['cohort']!='DEV' or
                identity['seed']!=2101 or identity['qualification_only'] or
                identity['new_SFT']['config']['variant']!=a.variant or identity['new_SFT']['updates']!=update):
            raise ValueError('Not a complete registered SFT observation')
        rows.append(dict(update=update,path=str(path),sha256=file_sha(path),
            modes={m:d['modes'][m]['paired_comparison']['PDMS_log_cluster_bootstrap'] for m in ('E_NATIVE','E_SHIFT5')}))
    if not rows:raise ValueError('No completed paired SFT observation; do not plot a fabricated zero result')
    fig,ax=plt.subplots(figsize=(7.2,4.2),constrained_layout=True)
    for mode,color,offset,label in [('E_NATIVE','#2563eb',-80,'Native uniform 10-step ODE'),
            ('E_SHIFT5','#d97706',80,'Shift5 10-step ODE')]:
        x=[0]+[r['update'] for r in rows];y=[0.]+[100*r['modes'][mode]['mean'] for r in rows]
        ax.plot(x,y,color=color,marker='o',label=label,linewidth=1.6,markersize=4)
        for row in rows:
            low,high=[100*v for v in row['modes'][mode]['CI95']];xp=row['update']+offset
            ax.vlines(xp,low,high,color=color,linewidth=1.5)
            ax.hlines([low,high],xp-50,xp+50,color=color,linewidth=1.5)
    ax.axhline(0,color='#666666',linewidth=.8,linestyle='--')
    ax.set_xticks([0]+[r['update'] for r in rows]);ax.set_xlabel('Supervised optimizer updates')
    ax.set_ylabel('PDMS gain vs each grid\'s initial SFT (percentage points)')
    ax.set_title(a.variant+' — SFT stage, not RL',fontsize=12)
    ax.grid(axis='y',alpha=.18);ax.legend(fontsize=9,loc='lower left')
    fig.text(.5,-.02,'Fixed seed2101; log-cluster 95% intervals. Training/inference seed uncertainty excluded.',ha='center',fontsize=8)
    output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=170,bbox_inches='tight');plt.close(fig)
    atomic_json(output.with_suffix('.json'),dict(schema='fm_transfer_observed_SFT_curve_v1',variant=a.variant,
        source_sha256=file_sha(__file__),observed=rows,not_available_updates=missing,
        no_extra_rollouts=True,zero_point='zero paired difference for each own initial reference; no new evaluation',
        not_RL_or_Navtest=True,image_sha256=file_sha(output)))
    print(json.dumps(dict(output=str(output),updates=[r['update'] for r in rows],not_available=missing)))


if __name__=='__main__':main()
