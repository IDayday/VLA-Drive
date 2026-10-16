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


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--variant',choices=CONFIGS,required=True)
    p.add_argument('--root',default=str(ROOT));p.add_argument('--output',required=True);a=p.parse_args()
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
