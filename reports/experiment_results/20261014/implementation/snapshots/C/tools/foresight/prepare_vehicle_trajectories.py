"""Independent raw-GT vehicle MAE records; timestamps and graph are current-only."""
import argparse
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from tools.ddpolicy_vehicle.prepare_data import load_trusted,atomic_json,CAMERAS
from starVLA.model.modules.structured_world.geometry import geometric_fov,future_track_positions,transform_points
from starVLA.model.modules.vehicle_joint.graphs import VehicleGraphConfig,select_vehicles
from starVLA.model.modules.vehicle_joint.initialization import file_sha256,identity_hash

SCHEMA='foresight_gt_vehicle_mae_v1'


def timed_frames(frames,at,horizons,tolerance=.05):
    timestamp=frames[at]['timestamp'];times=np.array([f['timestamp'] for f in frames],dtype=np.int64)
    selected=[]
    for h in horizons:
        expected=timestamp+int(round(h*1e6));loc=int(np.argmin(np.abs(times-expected)))
        selected.append(frames[loc] if times[loc]>timestamp and abs(times[loc]-expected)<=tolerance*1e6 else None)
    return selected


def make_record(current,future,calibration,graph_config):
    if not np.allclose(current['lidar2ego'],np.eye(4),atol=1e-6):raise ValueError('Current nonidentity annotation transform needs adapter')
    anns=current.get('anns')
    names=np.asarray(anns['gt_names'] if anns else [],dtype=str)
    raw=np.asarray(anns['gt_boxes'] if anns else np.empty((0,7)),dtype=np.float32)
    tracks=np.asarray(anns['track_tokens'] if anns else [],dtype=str)
    if len(names)!=len(raw) or len(tracks)!=len(names) or len(set(tracks))!=len(tracks):raise ValueError('Annotation correspondence')
    # Vehicle filtering precedes ANY spatial selection/capacity operation.
    vehicle=names=='vehicle';raw,tracks=raw[vehicle],tracks[vehicle]
    if not np.isfinite(raw).all() or (raw[:,3:6]<=0).any():raise ValueError('Invalid real current vehicle annotation')
    boxes=np.concatenate((raw[:,:6],np.sin(raw[:,6:7]),np.cos(raw[:,6:7])),-1)
    support=geometric_fov(raw[:,:3],*[np.asarray(calibration[k]) for k in ('intrinsics','extrinsics','distortion')])
    navigation=int(np.asarray(current['driving_command']).argmax())
    speed=float(np.linalg.norm(current['ego_dynamic_state'][:2]))
    chosen,_,audit=select_vehicles(torch.from_numpy(boxes),torch.ones(len(boxes)),torch.from_numpy(support),speed,navigation,graph_config)
    # Selection has finished BEFORE future tracks/validity are inspected.
    chosen_tracks=tuple(str(tracks[i]) for i in chosen)
    xy,valid=future_track_positions(chosen_tracks,future,current['ego2global'],8)
    n=1+graph_config.max_vehicles
    state=np.zeros((n,8),dtype=np.float32);state[0]=[0,0,0,4.9,2.,1.6,0,1]
    active=np.zeros(n,dtype=bool);active[:len(chosen)+1]=True
    state[1:len(chosen)+1]=boxes[chosen]
    trajectories=np.zeros((n,8,2),dtype=np.float32);point_valid=np.zeros((n,8),dtype=bool)
    trajectories[1:len(chosen)+1]=xy;point_valid[1:len(chosen)+1]=valid
    inv=np.linalg.inv(current['ego2global'])
    for t,frame in enumerate(future):
        if frame is not None:
            pose=np.asarray(frame['ego2global']);trajectories[0,t]=(inv@pose)[:2,3];point_valid[0,t]=True
    # Current ego velocity is legal; no neighbor velocity is invented.
    dynamic=np.asarray(current['ego_dynamic_state'],dtype=np.float32).reshape(-1)
    ego_state=np.zeros(4,dtype=np.float32);ego_state[:min(4,len(dynamic))]=dynamic[:4]
    if not np.isfinite(ego_state).all():raise ValueError('Invalid current ego state')
    return {'current':torch.from_numpy(state),'active':torch.from_numpy(active),
            'future':torch.from_numpy(trajectories),'point_valid':torch.from_numpy(point_valid),
            'navigation':torch.tensor(navigation),'ego_state':torch.from_numpy(ego_state)}, {
        'source_current_objects':len(names),'source_vehicles':len(raw),'selected_vehicles':len(chosen),
        'trajectory_candidates_with_support':int(support.sum()),'selected_valid_future_vehicles':int(valid.any(-1).sum()),
        'ego_only':len(chosen)==0,'valid_ego_points':int(point_valid[0].sum()),
        'vehicle_future_points':int(valid.sum()),'annotation_present':anns is not None,
        'raw_log_population_available':True,'geometric_support_not_occlusion':True,
        'graph_audit':audit,'track_ids_for_audit_only':chosen_tracks}


def process_log(task):
    log,tokens,cfg=task;out=Path(cfg['output']);report_path=out/'logs'/(log+'.json')
    if report_path.exists():
        r=json.loads(report_path.read_text())
        if r['identity']!=cfg['identity']:raise ValueError('Existing log identity mismatch')
        if any(not (out/'records'/(t+'.pt')).exists() for t in tokens):raise ValueError('Missing completed record')
        return r
    frames=load_trusted(Path(cfg['raw_log_root'])/(log+'.pkl'));where={f['token']:i for i,f in enumerate(frames)}
    rows=[]
    for token in tokens:
        at=where[token];current=frames[at]
        with np.load(Path(cfg['observation_root'])/'observations'/(token+'.npz')) as obs:
            if int(obs['timestamp'])!=current['timestamp']:raise ValueError('Current calibration timestamp mismatch')
            calibration={k:obs[k] for k in ('intrinsics','extrinsics','distortion')}
        future=timed_frames(frames,at,[.5*(i+1) for i in range(8)],cfg['time_tolerance_s'])
        record,audit=make_record(current,future,calibration,VehicleGraphConfig(**cfg['graph']))
        payload={'schema':SCHEMA,'identity':cfg['identity'],'token':token,'log':log,
                 'timestamp':int(current['timestamp']),'record':record,'audit':audit}
        dst=out/'records'/(token+'.pt');tmp=dst.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(dst)
        rows.append({'token':token,'log':log,**{k:v for k,v in audit.items() if k not in ('graph_audit','track_ids_for_audit_only')}})
    report={'identity':cfg['identity'],'log':log,'raw_log_sha256':file_sha256(Path(cfg['raw_log_root'])/(log+'.pkl')),'records':rows}
    atomic_json(report_path,report);return report


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('split-manifest','raw-log-root','observation-root','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    partition=json.loads(Path(a.split_manifest).read_text());train=partition['train_tokens'];dev=partition['dev_tokens'];logs=partition['token_logs']
    if set(train)&set(dev) or {logs[t] for t in train}&{logs[t] for t in dev}:raise ValueError('Split leakage')
    cfg={**vars(a),'schema':SCHEMA,'split_hash':file_sha256(a.split_manifest),'graph':asdict(VehicleGraphConfig(max_vehicles=16,max_context=0)),
         'time_tolerance_s':.05,'time_points_s':[.5*(i+1) for i in range(8)],'source_code_hash':file_sha256(__file__),
         'observation_identity':json.loads((Path(a.observation_root)/'identity.json').read_text())['identity']}
    cfg.pop('resume');cfg.pop('workers');cfg['identity']=identity_hash(cfg)
    out=Path(a.output)
    if out.exists():
        if not a.resume or json.loads((out/'identity.json').read_text())['identity']!=cfg['identity']:raise ValueError('Cache exists/different identity')
    else:
        (out/'records').mkdir(parents=True);(out/'logs').mkdir();atomic_json(out/'identity.json',cfg)
    selected=train[:a.limit] if a.limit else train+dev
    groups=defaultdict(list)
    for t in selected:groups[logs[t]].append(t)
    rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as executor:
        for report in executor.map(process_log,[(log,t,cfg) for log,t in groups.items()]):
            rows.extend(report['records']);atomic_json(out/'progress.json',{'completed':len(rows),'requested':len(selected)})
    for name,tokens in [('train',train),('dev',dev)]:
        keep=set(selected);atomic_json(out/(name+'_index.json'),[{'token':t,'log':logs[t]} for t in tokens if t in keep])
    numeric=('source_current_objects','source_vehicles','selected_vehicles','selected_valid_future_vehicles','ego_only','vehicle_future_points')
    atomic_json(out/'audit.json',{'identity':cfg['identity'],'scenes':len(rows),'failures':0,'population':{k:sum(r[k] for r in rows) for k in numeric}})


if __name__=='__main__':main()
