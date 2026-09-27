"""Publish complete fixed-budget planner loss/exposure curves without a convergence verdict."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.structured_world_v1p1.reaudit_metrics import write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts',required=True);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.artifacts);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    ledger=json.loads((root/'budget_ledger.json').read_text());runs={r['id']:r for r in ledger['runs']}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4));records=[];costs=[]
    for run in a.runs:
        if runs[run]['status']!='complete':raise ValueError('Only terminal completed training traces can be published here')
        source=root/run;manifest=json.loads((source/'manifest.json').read_text())
        rows=[json.loads(line) for line in (source/'train.jsonl').read_text().splitlines()]
        if rows[-1]['step']!=manifest['planned_steps'] or len(rows)!=manifest['planned_steps']:raise ValueError('Incomplete training trace')
        for name in ['manifest.json','train.jsonl']:
            dest=out/run;dest.mkdir(exist_ok=True);(dest/name).write_text((source/name).read_text())
        window=min(100,len(rows));x=[r['effective_epochs'] for r in rows]
        loss=np.array([r['ego_FM_loss'] for r in rows]);label=run.replace('planner_','').replace('7284','')
        axes[0].plot(x[window-1:],np.convolve(loss,np.ones(window)/window,'valid'),label=label)
        values=[]
        for path in sorted(source.glob('eval_*.json'),key=lambda p:int(p.stem.split('_')[-1])):
            d=json.loads(path.read_text());epoch=min(d['step']*manifest['arguments']['batch'],manifest['planned_presentations'])/(manifest['planned_presentations']/manifest['arguments']['epochs'])
            row={'run':run,'step':d['step'],'effective_epochs':epoch,'ego_ADE':d['ego_ADE'],'ego_FDE':d['ego_FDE'],'scenes':d['scenes'],'failed':d['failed']}
            records.append(row);values.append(row)
            for ext in ['json','csv']:
                name=path.with_suffix('.'+ext).name;(out/run/name).write_text((source/name).read_text())
        axes[1].plot([v['effective_epochs'] for v in values],[v['ego_ADE'] for v in values],marker='o',label=label)
        axes[2].plot(x,[r['gate'] for r in rows],label=label+' action')
        if 'bev_gate' in rows[0] and manifest.get('bev_enabled'):
            axes[2].plot(x,[r['bev_gate'] for r in rows],linestyle='--',label=label+' BEV')
        costs.append({'run':run,'optimizer_steps':rows[-1]['step'],'presentations':rows[-1]['presentations'],
                      'passes':rows[-1]['effective_epochs'],'trainable_parameters':manifest['trainable_parameters'],
                      'peak_gpu_bytes':max(r['peak_gpu_bytes'] for r in rows),'allocated_gpu_hours':runs[run]['gpu_hours'],
                      'samples_per_second_including_setup_and_diagnostics':rows[-1]['presentations']/(runs[run]['gpu_hours']*3600),
                      'mean_ego_FM_loss_first100':float(loss[:100].mean()),'mean_ego_FM_loss_last100':float(loss[-100:].mean()),
                      'last_lr':rows[-1]['lr'],'training_data_limit':'ADE diagnostics use the fixed first64 training scenes; not heldout planning or fulltrain mean.'})
    for ax in axes:ax.set_xlabel('Training corpus passes');ax.grid(alpha=.2);ax.legend(fontsize=7)
    axes[0].set_title('100-step mean original DiT ego FM loss');axes[0].set_ylabel('Loss')
    axes[1].set_title('Fixed train64 original DiT trajectory ADE');axes[1].set_ylabel('Metres')
    axes[2].set_title('Learned gates, actual values');axes[2].set_ylabel('Gate')
    fig.suptitle('Fixed exposure — reaching the budget does not establish convergence')
    fig.tight_layout();fig.savefig(out/'transfer_curves.png',dpi=160);fig.savefig(out/'transfer_curves.pdf');plt.close(fig)
    write_csv(out/'learning_curves.csv',records)
    path=out/'learning_curves.csv';path.write_text(path.read_text())
    (out/'COSTS_AND_LIMITS.json').write_text(json.dumps(costs,indent=2)+'\n')
    print(json.dumps({'completed_runs':len(costs),'output':str(out)}))


if __name__=='__main__':main()
