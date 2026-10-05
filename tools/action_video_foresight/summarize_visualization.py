"""Publish numeric auxiliary audit data without private scene pictures/IDs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .visualization_metrics import relative_gain
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import file_sha256


def task_score(row,task,reference,group=None):
    scores=[v for k,v in row['scores'].items() if k.startswith(task+'/') and k.endswith('/'+reference)
            and ((group is None and len(k.split('/'))==(3 if task=='current' else 4)) or
                 (group is not None and '/'+group+'/' in k))]
    n=sum(v['patches'] for v in scores)
    return sum(v['squared_channel_mean_sum'] for v in scores)/n if n else None


def clustered_difference(rows,task,reference,iterations=3000):
    groups={}
    for r in rows:
        p,t=task_score(r,task,'model'),task_score(r,task,reference)
        if p is None or t is None:continue
        groups.setdefault(r['log'],[]).append(p-t)
    logs=sorted(groups);sums=np.array([sum(groups[l]) for l in logs]);n=np.array([len(groups[l]) for l in logs])
    if not logs:return None
    ids=np.random.default_rng(742).integers(0,len(logs),size=(iterations,len(logs)))
    boot=sums[ids].sum(-1)/n[ids].sum(-1)
    return {'model_minus_reference':float(sums.sum()/n.sum()),'log_cluster_95':np.quantile(boot,[.025,.975]).tolist(),'logs':len(logs),'scenes':int(n.sum()),'iterations':iterations,'sampling_seed':742}


def summarize(artifacts,output):
    root=Path(artifacts);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    galleries=json.loads((root/'representatives.json').read_text());gallery_alias={r['token']:f'scene_{i:02d}' for i,r in enumerate(galleries)}
    scope=json.loads((root/'C1/SUMMARY.json').read_text())['scope']
    population=json.loads((root/('representatives.json' if scope=='representatives' else 'population.json')).read_text());aliases={r['token']:f'scene_{i:04d}' for i,r in enumerate(population)}
    result={'scope':'frozen current/future auxiliary diagnostics; no new PDMS or optimizer updates',
        'identity':json.loads((root/'identity.json').read_text()),'models':{},'representatives':[]}
    # The public projection training identifiers are hashes/counts, not raw logs.
    result['identity']['projection_training_scenes_count']=len(result['identity'].pop('projection_training_scenes'))
    public_rows=[]
    for arm in ('C1','S0','S1','S2','S3','S4'):
        summary=json.loads((root/arm/'SUMMARY.json').read_text())
        rows=[json.loads(l) for l in (root/arm/'scenes.jsonl').read_text().splitlines()]
        if summary['failures'] or len(rows)!=len(population) or [r['token'] for r in rows]!=[r['token'] for r in population]:raise ValueError('Incomplete audit or changed denominator')
        model={'checkpoint':summary['checkpoint'],'kind':summary['kind'],'precision':summary['precision'],
            'scenes':len(rows),'logs':summary['logs'],'failures':summary['failures'],'time_intervals_s':summary['time_intervals_s'],
            'allocation_gpu_hours':summary['allocation_gpu_hours'],'evaluation_source':summary['source'],
            'full_scores':summary['scores'],'vehicle_anchors':summary['vehicle_anchors'],
            'fixed_position_variance':summary['fixed_position_variance'],'tasks':{},'subgroups':{}}
        for task in ('current','future'):
            references=['model','train_mean','shuffled_W']+(['static'] if task=='future' else [])
            stats={}
            for name in references:
                values=[task_score(r,task,name) for r in rows];values=[x for x in values if x is not None]
                stats[name]={'scene_mean_mse':float(np.mean(values)) if values else None,'valid_scenes':len(values)}
            stats['gains']={name:relative_gain(stats['model']['scene_mean_mse'],stats[name]['scene_mean_mse']) for name in references[1:]}
            stats['paired_intervals']={name:clustered_difference(rows,task,name) for name in references[1:]} if scope=='full_dev' else {'status':'NOT_ESTIMATED:10 representative cases are illustrative, not a statistical evaluation sample'}
            model['tasks'][task]=stats
            for group in ('high_change','low_change'):
                values={name:[task_score(r,task,name,group) for r in rows] for name in ('model','static')}
                if any(v is not None for v in values['model']):
                    means={k:float(np.mean([v for v in vs if v is not None])) for k,vs in values.items()}
                    model['tasks'][task][group]={**means,'gain':relative_gain(means['model'],means['static'])}
            for attribute in ('navigation','ego_motion','peer_motion','clip_valid'):
                for value in sorted({r[attribute] for r in rows},key=str):
                    selected=[r for r in rows if r[attribute]==value]
                    model['subgroups'][f'{task}/{attribute}/{value}']={
                        name:float(np.mean(vs)) if (vs:=[task_score(r,task,name) for r in selected if task_score(r,task,name) is not None]) else None for name in references}
        for row in rows:
            pub={'arm':arm,'scene':aliases[row['token']],'navigation':row['navigation'],'ego_motion':row['ego_motion'],
                 'peer_motion':row['peer_motion'],'clip_valid':row['clip_valid'],'failure':row['failure']}
            for task in ('current','future'):
                for name in ('model','train_mean','shuffled_W','static'):pub[task+'_'+name]=task_score(row,task,name)
            public_rows.append(pub)
            if row['token'] in gallery_alias:
                result['representatives'].append({'gallery':gallery_alias[row['token']],**pub,
                    'vehicle_anchor_count':len(row.get('vehicle_anchor_diagnostics',[])),
                    'current_vehicle_anchor_similarity':float(np.mean([x['target_anchor_similarity'] for x in row['vehicle_anchor_diagnostics']])) if row.get('vehicle_anchor_diagnostics') else None})
        result['models'][arm]=model
    result['total_allocation_gpu_hours']=sum(r['allocation_gpu_hours'] for r in result['models'].values())
    atomic_json(out/'SUMMARY.json',result)
    with (out/'ANONYMIZED_SCENES.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(public_rows[0]));writer.writeheader();writer.writerows(public_rows)
    with (out/'REPRESENTATIVE_SCENES.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(result['representatives'][0]));writer.writeheader();writer.writerows(result['representatives'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,task,arms,title in zip(axes,('current','future','future'),(('C1','S0','S1','S2','S3','S4'),('S0','S1'),('S2','S3','S4')),
                                ('Current DINO, raw feature MSE','Future DINO sequence, LN MSE','Future video clip, LN MSE')):
        refs=('model','train_mean','shuffled_W') if task=='current' else ('model','train_mean','static')
        x=np.arange(len(arms))
        for i,ref in enumerate(refs):ax.bar(x+(i-1)*.23,[result['models'][a]['tasks'][task][ref]['scene_mean_mse'] for a in arms],.23,label=ref)
        ax.set_xticks(x,arms);ax.set_title(title,fontsize=10);ax.legend(fontsize=8);ax.set_ylabel('MSE (same target only)')
    fig.tight_layout();fig.savefig(out/'REFERENCE_COMPARISONS.svg');plt.close(fig)
    atomic_json(out/'PROVENANCE.json',{'artifact_identity_sha256':file_sha256(root/'identity.json'),'real_optimizer_updates':0,
        'private_images_uploaded':False,'private_feature_tensors_uploaded':False,
        'feature_projection':'diagnostic only; shared train-fit colours, not an image decoder',
        'mean_current':'all 101592 training scenes','mean_sequence_and_video':'existing fixed 4096-scene training-only reference subset',
        'statistics':f'{len(population)} fixed representative cases; not a full-dev statistical conclusion; missing clips kept in current denominator',
        'semantic_limit':'GT selected-vehicle centre patches can mix vehicle/background; no dense class labels or segmentation accuracy'})
    print(json.dumps({a:m['tasks'] for a,m in result['models'].items()},ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--artifacts',required=True);p.add_argument('--output',required=True);a=p.parse_args();summarize(a.artifacts,a.output)


if __name__=='__main__':main()
