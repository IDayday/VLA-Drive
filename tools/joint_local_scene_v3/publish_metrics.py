"""Export reviewable metrics only; raw identities, trajectories, checkpoints and scene images stay private."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from tools.joint_local_scene_v3.analyze_campaign import read_queries,csv_out
from tools.joint_local_scene_v3.budget import atomic_json


def pseudonym(kind,value):return hashlib.sha256((kind+':'+str(value)).encode()).hexdigest()[:20]


def main():
    p=argparse.ArgumentParser()
    for key in ('campaign','analysis','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--endpoint-updates',type=int,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[42]);a=p.parse_args()
    root=Path(a.campaign);analysis=Path(a.analysis);out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('New review export required')
    out.mkdir(parents=True,exist_ok=True);queries=[];dependencies=[];identities={}
    for seed in a.seeds:
        for mode in ('all','mask'):
            name=f'formal{seed}_{mode}';run=root/name;status=json.loads((run/'status.json').read_text())
            if status['step']!=a.endpoint_updates or status['status'] not in ('paused','complete') or status['presentations']!=7284*(a.endpoint_updates//228):raise ValueError('Not at the common complete-epoch endpoint')
            table=run/f'holdout_{a.endpoint_updates}/conditional_queries.csv';summary=json.loads((table.parent/'summary.json').read_text())
            if not summary['aggregate_valid']:raise ValueError('Cannot publish failed evaluation as complete')
            identities[name]={'checkpoint':summary['evaluation_identity'],'query_identity':summary['query_manifest_sha256'],'status':status,'local_csv_sha256':hashlib.sha256(table.read_bytes()).hexdigest()}
            for r in read_queries(table):
                row={k:v for k,v in r.items() if k not in ('token','log','query_id','valid_timesteps','scene_index','source_index','current_distance_m','graph_origin','scope')}
                row.update(seed=seed,variant='J_'+mode.upper(),step=a.endpoint_updates,scene_id=pseudonym('scene',r['token']),log_id=pseudonym('log',r['log']),query_id=pseudonym('query',r['query_id']),current_distance_bin='0_10' if r['current_distance_m']<=10 else '10_20' if r['current_distance_m']<=20 else '20_50')
                queries.append(row)
            diagnostic=root/f'diag{seed}_{mode}'
            if diagnostic.exists():
                ds=json.loads((diagnostic/'summary.json').read_text())
                if not ds['complete'] or ds['failed']:raise ValueError('Incomplete diagnostic export')
                for r in read_queries(diagnostic/'condition_queries.csv'):
                    row={k:v for k,v in r.items() if k not in ('token','log','query_id','valid_timesteps','scene_index','source_index','current_distance_m','strong_distance_m','weak_distance_m','graph_origin','scope')}
                    row.update(seed=seed,variant='J_'+mode.upper(),scene_id=pseudonym('scene',r['token']),log_id=pseudonym('log',r['log']),query_id=pseudonym('query',r['query_id']));dependencies.append(row)
                if (diagnostic/'analysis.json').exists():shutil.copyfile(diagnostic/'analysis.json',out/(f'diag{seed}_{mode}_summary.json'))
    csv_out(out/'endpoint_queries_anonymized.csv',queries);csv_out(out/'condition_ablation_queries_anonymized.csv',dependencies);atomic_json(out/'identities.json',identities)
    for name in ('run_statistics.json','milestone_summaries.json','completion_effects.json','milestone_curves.csv','paired_results.json','pairing_checks.json','gradient_update_probes.csv'):
        if (analysis/name).exists():shutil.copyfile(analysis/name,out/name)
    for path in analysis.glob('*.png'):shutil.copyfile(path,out/path.name)
    with (analysis/'scene_log_paired_deltas.csv').open() as f:rows=list(csv.DictReader(f))
    for row in rows:row['id']=pseudonym('scene' if row['level']=='token' else 'log',row['id'])
    csv_out(out/'scene_log_paired_deltas_anonymized.csv',rows)
    atomic_json(out/'EXPORT_MANIFEST.json',{'source_population_filtered':True,'identifiers':'SHA256(kind:original) first20hex, consistent across model/training seeds','raw_trajectories_uploaded':False,'scene_images_uploaded':False,'checkpoints_uploaded':False,'endpoint_queries':len(queries),'condition_queries':len(dependencies),'seeds':a.seeds,'endpoint_updates':a.endpoint_updates,'files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}})


if __name__=='__main__':main()
