"""Complete frozen benchmark tables and vehicle coverage without GPU work."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from .prepare_data import atomic_json
from .paired_results import log_bootstrap
from .summarize_training import summarize
from starVLA.model.modules.vehicle_joint.initialization import file_sha256


def run(module, *args):
    subprocess.run([sys.executable, '-m', 'tools.ddpolicy_vehicle.'+module, *map(str,args)],check=True)


def common_vehicle_comparison(left, right, output):
    """Errors on a fixed intersection, beside the full GT coverage denominators."""
    if len(left)!=len(right) or not left:raise ValueError('Unequal vehicle sampling protocol')
    groups=[]
    for files in (left,right):
        banks=[]
        for path in files:
            with Path(path).open() as stream:rows=list(csv.DictReader(stream))
            bank={(r['token'],r['track_id']):r for r in rows}
            if len(bank)!=len(rows) or any(r['export_failure'] for r in rows):
                raise ValueError('Duplicate/failed vehicle result cannot be filtered out')
            banks.append(bank)
        groups.append(banks)
    population=set(groups[0][0])
    if any(set(bank)!=population for banks in groups for bank in banks):
        raise ValueError('Vehicle GT populations differ')
    metrics=('head_ADE','head_FDE','joint_ADE','joint_FDE','joint_relative_vector_ADE')
    summaries={};paired=[]
    for metric in metrics:
        common=[key for key in sorted(population) if all(bank[key].get(metric,'') not in ('',None) for banks in groups for bank in banks)]
        rows=[]
        for key in common:
            ref=groups[0][0][key]
            if any(bank[key]['log']!=ref['log'] or bank[key]['motion_group']!=ref['motion_group'] for banks in groups for bank in banks):
                raise ValueError('Target identity changed across predictions')
            lv=float(np.mean([float(bank[key][metric]) for bank in groups[0]]))
            rv=float(np.mean([float(bank[key][metric]) for bank in groups[1]]))
            if not np.isfinite([lv,rv]).all():raise ValueError('Nonfinite common-target result')
            row={'token':key[0],'track_id':key[1],'log':ref['log'],'motion_group':ref['motion_group'],
                 'metric':metric,'left':lv,'right':rv,'delta':rv-lv}
            rows.append(row);paired.append(row)
        summaries[metric]={}
        for group in ('all','stationary','moving'):
            values=rows if group=='all' else [r for r in rows if r['motion_group']==group]
            summaries[metric][group]={'common_targets_all_inference_seeds':len(values),
                'left':float(np.mean([r['left'] for r in values])) if values else None,
                'right':float(np.mean([r['right'] for r in values])) if values else None,
                'right_minus_left':log_bootstrap([r['delta'] for r in values],[r['log'] for r in values]) if values else None}
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    with (out/'common_vehicles.csv').open('w') as stream:
        writer=csv.DictWriter(stream,['token','track_id','log','motion_group','metric','left','right','delta'])
        writer.writeheader();writer.writerows(paired)
    coverage={}
    for side,banks in zip(('left','right'),groups):
        coverage[side]={field:sum(all(bank[key][field]=='True' for bank in banks) for key in population)
                        for field in ('detected','selected')}
    atomic_json(out/'summary.json',{'full_supervised_vehicle_population':len(population),
        'coverage_all_inference_seeds':coverage,'metrics':summaries,
        'scope':'Fixed shared GT population; error intersection is selection-conditioned, not full-scene prediction accuracy',
        'joint_consistency':'Relative-vector error uses ego and neighbor from the SAME sample; center distances are not official collision scores',
        'sources':{str(p):file_sha256(p) for p in [*left,*right]}})


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',required=True);p.add_argument('--campaign-directory',required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text());directory=Path(a.campaign_directory)
    lock=json.loads((directory/'final_lock.json').read_text());selection=json.loads((directory/'selection.json').read_text())
    if file_sha256(directory/'selection.json')!=lock['selection_sha256']:raise ValueError('Frozen selection changed')
    selected={Path(item['run']).name:item['tag'] for item in selection}
    models=[m for m in plan['models'] if m['run_id'] in selected];root=Path(plan['campaign_root'])
    out=directory/'analysis';out.mkdir(exist_ok=True)
    for split in ('dev','navtest'):
        labels=root/('evaluation_vehicle_targets_'+split+'_v1')
        if not (labels/'audit.json').exists():
            args=['--current-root',plan['data']['current_'+split],'--raw-log-root',plan['data']['raw_'+split],
                  '--output',labels]
            if labels.exists():args+=['--resume']
            if split=='navtest':args+=['--final-lock',directory/'final_lock.json']
            run('prepare_evaluation_targets',*args)
        audit=json.loads((labels/'audit.json').read_text())
        expected=1696 if split=='dev' else 12146
        if audit['failures'] or audit['completed']!=expected:raise ValueError('Incomplete offline evaluation population')
        index=Path(plan['data']['current_'+split])/'index.json'
        for model in models:
            bankroot=root/'formal_evaluation'/model['run_id']/selected[model['run_id']]/split
            for seed in plan['sampling_seeds']:
                result=bankroot/f'vehicles_seed{seed}'
                if not (result/'summary.json').exists():
                    run('evaluate_vehicles','--predictions',bankroot/f'predictions_seed{seed}',
                        '--vehicle-root',labels,'--index',index,'--output',result)
                if json.loads((result/'summary.json').read_text())['failed_scenes']:
                    raise ValueError('Failed vehicle predictions retained')
        comparisons=[('A','B',42),('B','C',42)]+([('B','C',43)] if ('B',43) in {(m['arm'],m['seed']) for m in models} else [])
        for left,right,seed in comparisons:
            pair=[next(m for m in models if m['arm']==arm and m['seed']==seed) for arm in (left,right)]
            roots=[root/'formal_evaluation'/m['run_id']/selected[m['run_id']]/split for m in pair]
            name=f'{split}_{left}_vs_{right}_training_seed{seed}'
            result=out/name
            if not (result/'summary.json').exists():
                run('paired_results','--left',*[r for s in plan['sampling_seeds'] for r in [roots[0]/f'scores_seed{s}'/'scenes.csv']],
                    '--right',*[roots[1]/f'scores_seed{s}'/'scenes.csv' for s in plan['sampling_seeds']],
                    '--index',index,'--training-seed',seed,'--output',result)
            if left=='B':
                result=out/(name+'_vehicles')
                if not (result/'summary.json').exists():
                    common_vehicle_comparison(*[[folder/f'vehicles_seed{s}'/'vehicles.csv' for s in plan['sampling_seeds']] for folder in roots],result)
    curves=out/'training_curves'
    if not curves.exists():run('summarize_training','--runs',*[root/'training'/m['run_id'] for m in models],'--output',curves)
    summaries=[summarize(root/'training'/m['run_id'])[0] for m in models]
    atomic_json(out/'COMPLETE.json',{'final_lock_sha256':file_sha256(directory/'final_lock.json'),
        'models':summaries,'second_training_seed':lock['second_training_seed'],
        'A_second_training_seed':lock['A_second_training_seed'],
        'claim':'Camera-only v1 benchmark and paired vehicle tables completed; inspect actual paired intervals before declaring mask benefit'})


if __name__=='__main__':main()
