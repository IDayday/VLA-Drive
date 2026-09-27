"""Rebuild graph/bridge learning curves and supervision exposure from complete run logs."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import numpy as np


def summarize_training(rows, phase):
    if not rows or [row['step'] for row in rows] != list(range(1, len(rows)+1)):
        raise ValueError('Missing or duplicated optimizer updates')
    epochs={}; previous=0
    for row in rows:
        batch=row['presentations']-previous; previous=row['presentations']
        if batch<=0:raise ValueError('Non-increasing data exposure')
        item=epochs.setdefault(row['epoch'],{'epoch':row['epoch']+1,'updates':0,'presentations':0,
            'step_seconds':0.,'peak_gpu_bytes':0,'loss_weighted_sum':0.,'component_sums':{},'coordinates':{},
            'nominal':[0,0,0],'actual':[0,0,0],'fallbacks':0,'empty_tasks':0,'per_task':{}})
        item['updates']+=1;item['presentations']+=batch;item['step_seconds']+=row['seconds']
        item['peak_gpu_bytes']=max(item['peak_gpu_bytes'],row['peak_gpu_bytes'])
        item['loss_weighted_sum']+=batch*row['loss' if phase=='graph' else 'ego_FM_loss']
        if phase=='graph':
            for name,count in row['supervised_coordinates'].items():
                item['component_sums'][name]=item['component_sums'].get(name,0.)+row['loss_components'][name]*count
                item['coordinates'][name]=item['coordinates'].get(name,0)+count
            for dest,source in [('nominal','task_nominal'),('actual','task_actual')]:
                item[dest]=[x+y for x,y in zip(item[dest],row[source])]
            item['fallbacks']+=row['fallbacks'];item['empty_tasks']+=row['tasks_without_valid_hidden_target']
            for name,counts in row.get('supervision_by_actual_task',{}).items():
                task=item['per_task'].setdefault(name,{})
                for key,value in counts.items():task[key]=task.get(key,0)+value
    for item in epochs.values():
        item['sample_weighted_training_loss']=item.pop('loss_weighted_sum')/item['presentations']
        item['coordinate_weighted_components']={name:item['component_sums'][name]/count if count else None
            for name,count in item['coordinates'].items()}
        item.pop('component_sums')
    return list(epochs.values())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['graph','planner'],required=True)
    parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    reports=[];csv_rows=[]
    for path in map(Path,args.runs):
        manifest=json.loads((path/'manifest.json').read_text());state=json.loads((path/'status.json').read_text())
        if state['status']!='complete':raise ValueError('Report completed finite runs only; retain paused artifacts separately')
        rows=[json.loads(line) for line in (path/'train.jsonl').read_text().splitlines()]
        epochs=summarize_training(rows,args.phase)
        if rows[-1]['step']!=state['step'] or rows[-1]['presentations']!=state['presentations']:
            raise ValueError('Terminal checkpoint status disagrees with training log')
        if len(epochs)!=state['epochs'] or len({item['presentations'] for item in epochs})!=1:
            raise ValueError('Incomplete or unequal full epoch exposure')
        holdout={int(p.stem.split('_')[1]):json.loads(p.read_text()) for p in path.glob('holdout_*.json')}
        report={'run':str(path),'mode':manifest['arguments']['mode'],'state':state,'manifest':manifest,
            'unique_scenes':rows[-1]['unique_scenes'],'epochs':epochs,'holdout':holdout,
            'cost_scope':'step_seconds excludes loader wait, process startup and holdout; budget ledger separately charges full GPU walltime',
            'graph_neighbor_empty_tasks_visible':any(item['per_task'] for item in epochs),
            'interpretation':'Finite schedule and training-domain diagnostics, not convergence or planning effectiveness'}
        reports.append(report)
        for epoch in epochs:
            item={key:epoch[key] for key in ('epoch','updates','presentations','step_seconds','peak_gpu_bytes','sample_weighted_training_loss')}
            item.update(run=path.name,mode=manifest['arguments']['mode'])
            if args.phase=='graph':
                for name,count in epoch['coordinates'].items():
                    item['supervised_'+name+'_coordinates']=count
                    item['coordinate_weighted_'+name+'_loss']=epoch['coordinate_weighted_components'][name]
                item['empty_hidden_task_scenes']=epoch['empty_tasks'];item['fallback_scenes']=epoch['fallbacks']
                for task,counts in epoch['per_task'].items():
                    for key,value in counts.items():item[task+'_'+key]=value
            item.update({'holdout_'+key:value for key,value in holdout.get(epoch['epoch'],{}).items()
                if value is None or isinstance(value,(int,float))})
            csv_rows.append(item)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    keys=['deploy_ego_ADE_m','deploy_agents_ADE_m'] if args.phase=='graph' else ['ego_ADE_m','ego_FDE_m']
    for report in reports:
        name=Path(report['run']).name
        axes[0].plot([x['epoch'] for x in report['epochs']],
            [x['sample_weighted_training_loss'] for x in report['epochs']],label=name)
        for axis,key in zip(axes[1:],keys):
            points=sorted((epoch,row[key]) for epoch,row in report['holdout'].items()
                if row.get(key) is not None and np.isfinite(row[key]))
            if points:axis.plot(*zip(*points),marker='o',label=name)
    for axis,title in zip(axes,['Training loss (different tasks)' if args.phase=='graph' else 'Training ego FM loss',*keys]):
        axis.set(xlabel='Complete training passes',title=title);axis.grid(alpha=.2);axis.legend()
    fig.tight_layout();fig.savefig(output/'curves.png',dpi=160);plt.close(fig)
    with (output/'epochs.csv').open('w') as handle:
        writer=csv.DictWriter(handle,sorted(set().union(*(row.keys() for row in csv_rows))))
        writer.writeheader();writer.writerows(csv_rows)
    (output/'summary.json').write_text(json.dumps({'phase':args.phase,
        'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'runs':reports,
        'warning':'ALL and MASK have different effective supervised coordinates; training losses are not equal-task comparative scores.'},indent=2)+'\n')


if __name__=='__main__':main()
