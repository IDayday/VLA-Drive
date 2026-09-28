"""Full-split timestamp/file availability audit before any VAE encoding; no scene is removed."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from tools.ddpolicy_vehicle.prepare_data import load_trusted,CAMERAS,camera_path
from tools.ddpolicy_vehicle.run_meter import metered_run
from starVLA.model.modules.vehicle_joint.initialization import file_sha256,identity_hash
from .prepare_vehicle_trajectories import timed_frames
from .score_pdms import atomic_json


def check_log(task):
    log,scenes,config=task;rows=[]
    try:
        frames=load_trusted(Path(config['raw_log_root'])/(log+'.pkl'))
        where={frame['token']:i for i,frame in enumerate(frames)}
        if len(where)!=len(frames):raise ValueError('Duplicate raw scene tokens')
    except Exception as error:
        return [{'token':token,'log':log,'split':split,'failure':repr(error)} for token,split in scenes]
    # Identical images recur in overlapping scene horizons. Cache only file
    # availability within this one immutable raw log, not any model feature.
    availability={}
    def exists(relative):
        if relative not in availability:
            try:camera_path(relative,config);availability[relative]=True
            except FileNotFoundError:availability[relative]=False
        return availability[relative]
    for token,split in scenes:
        row={'token':token,'log':log,'split':split,'failure':None}
        try:
            at=where[token];current=frames[at];future=timed_frames(frames,at,[1.,2.,4.],.05)
            for view,cam in enumerate(CAMERAS):row['current_'+str(view)]=exists(current['cams'][cam]['data_path'])
            for h,frame in enumerate(future):
                row[f'timestamp_valid_{h}']=frame is not None
                row[f'timestamp_error_us_{h}']=abs(int(frame['timestamp'])-int(current['timestamp'])-(1,2,4)[h]*1000000) if frame is not None else None
                for v,cam in enumerate(CAMERAS):row[f'future_{h}_{v}']=frame is not None and exists(frame['cams'][cam]['data_path'])
        except Exception as error:row['failure']=repr(error)
        rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('split-manifest','raw-log-root','sensor-root','output','campaign-root','run-id'):p.add_argument('--'+key,required=True)
    p.add_argument('--fallback-sensor-root',action='append',default=[]);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if not 1<=a.workers<=16:raise ValueError('Bounded CPU workers must be1..16')
    with metered_run(a.campaign_root,a.run_id,0,{'kind':'future_timestamp_file_availability','real_optimizer_updates':0}) as (meter,_,save):
        out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
        partition=json.loads(Path(a.split_manifest).read_text());mapping=partition['token_logs'];groups=defaultdict(list)
        train_logs={mapping[t] for t in partition['train_tokens']};dev_logs={mapping[t] for t in partition['dev_tokens']}
        if train_logs&dev_logs or set(partition['train_tokens'])&set(partition['dev_tokens']):raise ValueError('Train/development leakage')
        expected=[]
        for split in ('train','dev'):
            for token in partition[split+'_tokens']:groups[mapping[token]].append((token,split));expected.append((token,split,mapping[token]))
        if len(set(expected))!=len(expected):raise ValueError('Duplicate manifest scene')
        rows=[];begin=time.time()
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for batch in pool.map(check_log,[(log,scenes,vars(a)) for log,scenes in groups.items()]):
                rows.extend(batch);meter['inference_scenes']=len(rows);save()
                atomic_json(out/'progress.json',{'checked':len(rows),'requested':len(expected)})
        lookup={(r['token'],r['split'],r['log']):r for r in rows}
        if set(lookup)!=set(expected):raise ValueError('Audit silently changed scene population')
        rows=[lookup[key] for key in expected];keys=sorted(set().union(*(r.keys() for r in rows)))
        with (out/'scenes.csv').open('w') as stream:
            writer=csv.DictWriter(stream,keys);writer.writeheader();writer.writerows(rows)
        summary={'schema':'foresight_future_frame_availability_v1','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 'partition_sha256':file_sha256(a.split_manifest),'writer_sha256':file_sha256(__file__),'scope':'file availability and real timestamps only; VAE/image decode NOT_RUN',
                 'horizons_s':[1.,2.,4.],'timestamp_tolerance_s':.05,'views':CAMERAS,'splits':{},'elapsed_seconds':time.time()-begin,
                 'scene_csv_sha256':file_sha256(out/'scenes.csv'),'scene_removal_policy':'none; missing future targets only mask auxiliary loss'}
        for split in ('train','dev'):
            part=[r for r in rows if r['split']==split];failed=sum(bool(r['failure']) for r in part)
            stats={'scenes':len(part),'logs':len({r['log'] for r in part}),'failed':failed,
                   'index_hash':identity_hash([{'token':r['token'],'log':r['log']} for r in part])}
            if not failed:
                mask=np.asarray([[[r[f'future_{h}_{v}'] for v in range(3)] for h in range(3)] for r in part],dtype=bool)
                stats.update(current_complete_scenes=sum(all(r['current_'+str(v)] for v in range(3)) for r in part),
                    valid_horizon_view_counts=mask.sum(0).tolist(),scenes_with_any_future=int(mask.any((1,2)).sum()),
                    scenes_with_all_nine_future_views=int(mask.all((1,2)).sum()),
                    valid_timestamp_scenes=[sum(r[f'timestamp_valid_{h}'] for r in part) for h in range(3)],
                    max_abs_timestamp_error_us=[max((r[f'timestamp_error_us_{h}'] for r in part if r[f'timestamp_valid_{h}']),default=None) for h in range(3)])
            summary['splits'][split]=stats
        summary['valid']=not any(p['failed'] for p in summary['splits'].values());atomic_json(out/'summary.json',summary)
        if not summary['valid']:raise RuntimeError('Invalid audit rows retained; do not infer valid target coverage')


if __name__=='__main__':main()
