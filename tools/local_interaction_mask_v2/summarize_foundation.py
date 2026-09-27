"""Summarize the complete public foundation lineage without loading or selecting model weights."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',nargs='+',required=True)
    parser.add_argument('--frozen-manifest',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    runs=list(map(Path,args.runs));frozen=json.loads(Path(args.frozen_manifest).read_text())
    manifests=[json.loads((run/'manifest.json').read_text()) for run in runs]
    for identity in manifests:
        if identity['private_driving_weights_loaded']:raise ValueError('This report is only for a public-origin lineage')
    populations=[json.loads(Path(identity['arguments']['tokens']).read_text()) for identity in manifests]
    if any(population!=populations[0] for population in populations):raise ValueError('Foundation population changed')
    count=len(populations[0]);steps={};holdout={}
    for run in runs:
        for line in (run/'train.jsonl').read_text().splitlines():
            row=json.loads(line)
            if row['step'] in steps:raise ValueError('Repeated optimizer update across continuation segments')
            steps[row['step']]=row
        for path in run.glob('holdout_*.json'):
            suffix=path.stem.split('_')[-1]
            if suffix.isdigit():holdout[int(suffix)]=json.loads(path.read_text())
    state=frozen['training_status'];last=state['epochs_completed']
    if set(steps)!=set(range(1,state['step']+1)):raise ValueError('Incomplete foundation update trace')
    rows=[]
    for epoch in range(1,last+1):
        selected=[row for row in steps.values() if (row['presentations']-1)//count+1==epoch]
        if sum(row['global_batch_actual'] for row in selected)!=count:raise ValueError('Incomplete epoch exposure')
        denominators=np.array([row['global_denominators'] for row in selected])
        values=np.array([row['loss_ego_cls_box'] for row in selected])
        means=(values*denominators).sum(0)/denominators.sum(0)
        measured=holdout[epoch]
        row={'epoch':epoch,'updates':len(selected),'presentations':count,
             'train_ego_FM':float(means[0]),'train_classification':float(means[1]),'train_current_box':float(means[2]),
             'holdout_ego_ADE_m':measured['ego_ADE_m'],'holdout_current_class_recall':measured['class_filtered_recall'],
             'lr_end':selected[-1]['lr'],'peak_gpu_bytes':max(item['peak_gpu_bytes'] for item in selected)}
        rows.append(row)
    with (out/'epochs.csv').open('w') as handle:
        writer=csv.DictWriter(handle,list(rows[0]));writer.writeheader();writer.writerows(rows)
    older=rows[-3];latest=rows[-1]
    changes={key:(latest[key]-older[key])/max(abs(older[key]),1e-8) for key in
             ('train_ego_FM','train_classification','train_current_box','holdout_ego_ADE_m','holdout_current_class_recall')}
    summary={'foundation':frozen,'training_code_shas':[item['code_sha'] for item in manifests],
        'report_code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'input_files_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                             [Path(args.frozen_manifest)]+[run/name for run in runs for name in ('manifest.json','train.jsonl')]},
        'unique_scenes':count,'optimizer_updates':len(steps),'presentations':sum(row['presentations'] for row in rows),
        'epochs':last,'seed':manifests[0]['arguments']['seed'],'initial_holdout':{key:holdout[0][key] for key in ('ego_ADE_m','class_filtered_recall')},
        'final':latest,'relative_changes_last_two_epochs':changes,
        'public_origin':manifests[0]['public_provenance'],
        'private_driving_weights_loaded':False,'termination':f'Registered{last}pass finite budget boundary, not a convergence certificate',
        'scope':'Training-domain curves under the original native-BF16 LoRA training arithmetic; formal frozen inference uses the declared shared FP32-LoRA repair. No planning test metric consulted.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7));x=[row['epoch'] for row in rows]
    panels=[('train_ego_FM','Training ego FM loss'),('train_classification','Training current classification loss'),
            ('holdout_ego_ADE_m','Holdout actual DiT ADE (m)'),('holdout_current_class_recall','Holdout current class-filtered recall')]
    for axis,(key,title) in zip(axes.flat,panels):
        axis.plot(x,[row[key] for row in rows],marker='.',linewidth=1.5);axis.set(title=title,xlabel='Complete training passes')
        axis.axvline(8,color='gray',linestyle='--',alpha=.5);axis.axvline(16,color='gray',linestyle='--',alpha=.5);axis.grid(alpha=.25)
    fig.suptitle(f"Public Qwen + fresh original DiT, shared current supervision | seed{summary['seed']} | {count:,} scenes")
    fig.tight_layout();fig.savefig(out/'curves.png',dpi=140);plt.close(fig)


if __name__=='__main__':main()
