"""Explain missing target support without relaxing it or changing any labels."""
import argparse,json,os,sys,multiprocessing,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import shapely
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import read,atomic_json,check_host,digest
from starVLA.dataloader.minimal_repair_scene import load_annotated_scene
from starVLA.model.modules.decision_local.recovery_plans import PlanCurve
from starVLA.model.modules.decision_local.local_constraint_checker import body_polygons
from starVLA.dataloader.foresight_dataset import decode_ego


def one(task):
    row,anchor,root,velocity_boundary=task;start=time.monotonic();cpu=time.process_time()
    scene=load_annotated_scene(anchor,'train',row);rec=read(Path(root)/'records'/(row['token']+'.json'))
    with np.load(Path(root)/'scenes'/(row['token']+'.npz')) as data:plan=data['plan_in_physical'][0,0].copy()
    with safe_open(rec['source_feature_chunk'],framework='pt',device='cpu') as f:
        state=f.get_slice('state')[rec['source_feature_offset']]
    velocity=decode_ego(state[None])[0,:2].numpy()/.5;times=np.linspace(0,4,81)
    bodies=body_polygons(PlanCurve(plan,velocity,velocity_boundary=velocity_boundary).poses(times),scene.body)
    counts={key:0 for key in ('missing_frame','unreliable_height_or_original_support','disappeared_track_tube','appearance_interval')}
    ages=[]
    for t,body in zip(times,bodies):
        pos=t/.5;lo=int(np.floor(pos+1e-7));hi=int(np.ceil(pos-1e-7));a,b=scene.frames[lo],scene.frames[hi]
        if a is None or b is None:counts['missing_frame']+=1;continue
        hit={key:False for key in counts}
        for track,history in scene.track_history.items():
            polygon=kind=None;age=None
            if track in a and track in b:
                if not (a[track]['reliable'] and b[track]['reliable']):
                    polygon=scene.interpolate_rectangle(a[track]['corners'],b[track]['corners'],pos-lo)
                    kind='unreliable_height_or_original_support'
            else:
                previous=[(i,obj) for i,obj in history if i<=pos]
                if previous:
                    at,obj=previous[-1];age=t-at*.5;polygon=scene.unknown_reachable(obj,age);kind='disappeared_track_tube'
                elif track in b and hi!=lo:
                    polygon=scene.unknown_reachable(b[track],t-hi*.5);kind='appearance_interval'
            if polygon is not None and body.distance(polygon)<=.05:
                hit[kind]=True
                if age is not None:ages.append(float(age))
        for key,yes in hit.items():counts[key]+=int(yes)
    return {'token':row['token'],'log':row['log'],'road_height_known':bool(scene.road_height_known),
        'original_native_status':rec['cases'][0]['status_name'],'blocking_dense_time_counts':counts,
        'disappearance_ages_s':ages,'CPU_process_seconds':time.process_time()-cpu,'worker_wall_seconds':time.monotonic()-start}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('labels','anchor','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--limit',type=int,default=256);p.add_argument('--workers',type=int,default=8);a=p.parse_args();resource=check_host()
    meta=read(a.labels/'identity.json');anchor=read(a.anchor)
    if meta['split']!='train' or not 1<=a.workers<=16 or not 1<=a.limit<=4096:raise ValueError('Bounded train-only support audit')
    started=time.monotonic();tasks=[(r,anchor,str(a.labels),meta['constraints'].get('velocity_boundary','body_aligned_speed_v1')) for r in meta['index'][:a.limit]]
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:rows=list(pool.map(one,tasks,chunksize=2))
    ages=[age for row in rows for age in row['disappearance_ages_s']]
    result={'scenes':len(rows),'labels':meta['identity'],'tool_sha256':digest(__file__),
        'road_height_unknown_scenes':sum(not r['road_height_known'] for r in rows),
        'scenes_with_each_blocker':{k:sum(r['blocking_dense_time_counts'][k]>0 for r in rows) for k in rows[0]['blocking_dense_time_counts']},
        'disappearance_age_distribution_s':{'count':len(ages),'median':float(np.median(ages)) if ages else None,'p90':float(np.quantile(ages,.9)) if ages else None},
        'CPU_process_seconds':sum(r['CPU_process_seconds'] for r in rows),'elapsed_seconds':time.monotonic()-started,
        'no_label_or_validity_changes':True,'scope':'fixed original native seed42 training plans; categories can overlap; not a counterfactual policy experiment',
        'resource_check':resource,'rows':rows}
    atomic_json(a.output,result);print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
