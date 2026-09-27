"""Export every fixed checkpoint, loss/exposure curves and paired log-cluster uncertainty."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np


def bootstrap_difference(masked, control, log_lookup, seed=20260927, repeats=2000):
    if {r['token'] for r in masked}!={r['token'] for r in control}:raise ValueError('Unpaired scene sets')
    m={r['token']:r for r in masked};c={r['token']:r for r in control}
    if any(r['status']!='ok' for r in masked+control):raise ValueError('Keep failures; no valid scientific comparison yet')
    groups={}
    for token in m:groups.setdefault(log_lookup[token],[]).append(token)
    logs=sorted(groups);rng=np.random.default_rng(seed)
    def difference(tokens,key):
        if key=='ego_ADE':return float(np.mean([float(m[t][key])-float(c[t][key]) for t in tokens]))
        points=sum(int(m[t]['agent_points']) for t in tokens)
        control_points=sum(int(c[t]['agent_points']) for t in tokens)
        if points!=control_points:raise ValueError('Current detection coverage differs in graph-only pair')
        return sum(float(m[t]['agent_error_sum'])-float(c[t]['agent_error_sum']) for t in tokens)/points if points else None
    indices=rng.integers(len(logs),size=(repeats,len(logs)));result={}
    for metric in ['ego_ADE','agent_ADE']:
        values=[]
        for selected in indices:
            delta=difference([t for i in selected for t in groups[logs[i]]],metric)
            if delta is not None:values.append(delta)
        result[metric]={'masked_minus_control':difference(list(m),metric),
                        'log_cluster_percentile_95':np.percentile(values,[2.5,97.5]).tolist() if values else None,
                        'resamples_with_valid_denominator':len(values)}
    return {'scenes':len(m),'logs':len(logs),'bootstrap_repeats':repeats,'seed':seed,'metrics':result,
            'limitation':'One training seed; scene/log uncertainty is not cross-training-seed stability.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['artifacts','target-audit','output']:p.add_argument('--'+key,required=True)
    a=p.parse_args();root=Path(a.artifacts);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    logs={r['token']:r['log'] for r in json.loads(Path(a.target_audit).read_text())['records']}
    ledger=json.loads((root/'budget_ledger.json').read_text());runs={r['id']:r for r in ledger['runs']}
    summary={'research_status':'INCONCLUSIVE','planning_status':'NOT_EVALUATED_HERE','variants':{},'paired_holdout':{}}
    curves=[];training={};evals={};scene_rows={}
    for variant in ['randommask','allmask']:
        name=f'extended_{variant}7284';run=root/name
        training[variant]=[json.loads(line) for line in (run/'train.jsonl').read_text().splitlines()]
        info=json.loads((run/'manifest.json').read_text());last=training[variant][-1]
        summary['variants'][variant]={'run_status':runs[name]['status'],'optimizer_steps':runs[name]['optimizer_steps'],
            'planned_steps':info['max_steps'],'sample_presentations':last['sample_presentations'],
            'unique_scenes_seen':last['unique_scenes_seen'],'effective_epochs':last['effective_epochs'],
            'source_sha':info['code_sha'],'peak_gpu_bytes':max(r['gpu_peak_bytes'] for r in training[variant]),
            'note':'Train prediction diagnostics use fixed first64 training scenes, not a whole-training-set mean.'}
        evals[variant]={}
        for split,paths in [('train64',run.glob('eval_*.json')),('holdout64',root.glob(f'extended_{variant}_holdout_step*/eval_*.json'))]:
            values=[]
            for path in sorted(paths,key=lambda p:int(p.stem.split('_')[-1])):
                d=json.loads(path.read_text());step=d['step'];values.append(d)
                curves.append({'variant':variant,'split':split,'step':step,'effective_epochs':min(step*info['arguments']['batch'],info['planned_presentations'])/info['unique_scenes'],
                    **{k:d[k] for k in ['scenes','failed','ego_ADE','agent_ADE','stationary_ADE','motion_point_coverage']}})
                destination=out/variant/split;destination.mkdir(parents=True,exist_ok=True)
                for filename in [f'eval_{step}.json',f'eval_{step}.csv',f'objects_{step}.csv']:
                    shutil.copyfile(path.parent/filename,destination/filename)
                if split=='holdout64':
                    with (path.parent/f'eval_{step}.csv').open() as f:scene_rows[(variant,step)]=list(csv.DictReader(f))
            evals[variant][split]=values
    for step in sorted({s for v,s in scene_rows}):
        if ('randommask',step) in scene_rows and ('allmask',step) in scene_rows:
            summary['paired_holdout'][step]=bootstrap_difference(scene_rows['randommask',step],scene_rows['allmask',step],logs)
    with (out/'learning_curves.csv').open('w') as f:
        w=csv.DictWriter(f,list(curves[0]));w.writeheader();w.writerows(curves)
    (out/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for variant,color in [('randommask','#1865ab'),('allmask','#dc7d27')]:
        rows=training[variant];window=min(100,len(rows));loss=np.array([r['loss'] for r in rows])
        axes[0].plot([r['effective_epochs'] for r in rows][window-1:],np.convolve(loss,np.ones(window)/window,'valid'),label=variant,color=color)
        for j,metric in [(1,'ego_ADE'),(2,'agent_ADE')]:
            rows=[r for r in curves if r['variant']==variant and r['split']=='holdout64']
            axes[j].plot([r['effective_epochs'] for r in rows],[r[metric] for r in rows],marker='o',label=variant,color=color)
    for ax in axes:ax.set_xlabel('Training corpus passes');ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('100-step mean training FM loss');axes[0].set_title('Within-run optimization trend')
    axes[1].set_ylabel('Metres');axes[1].set_title('Heldout joint-graph ego ADE')
    heldout=[r for r in curves if r['split']=='holdout64']
    coverage='; coverage '+format(100*heldout[-1]['motion_point_coverage'],'.2f')+'%' if heldout else ''
    axes[2].set_ylabel('Metres');axes[2].set_title('Heldout matched-agent ADE'+coverage)
    fig.suptitle('Fixed checkpoint curves — not DiT planning or PDMS; one training seed')
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=160);fig.savefig(out/'learning_curves.pdf');plt.close(fig)
    print(json.dumps(summary))


if __name__=='__main__':main()
