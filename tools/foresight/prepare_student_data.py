"""Split current-only observations and ego labels into independently hashed files."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
import torch
from tools.ddpolicy_vehicle.prepare_data import load_trusted,atomic_json
from tools.ddpolicy_vehicle.evaluate_ego import relative_ego_target
from starVLA.dataloader.foresight_dataset import encode_ego
from starVLA.model.modules.vehicle_joint.initialization import identity_hash,file_sha256


def process(task):
    log,tokens,cfg=task;out=Path(cfg['output']);frames=load_trusted(Path(cfg['raw_log_root'])/(log+'.pkl'))
    locations={f['token']:i for i,f in enumerate(frames)};rows=[]
    for token in tokens:
        at=locations[token];frame=frames[at]
        if at<3:raise ValueError('Missing allowed ego history')
        obs_path=out/'current'/(token+'.json');label_path=out/'ego'/(token+'.pt')
        if obs_path.exists() and label_path.exists():
            if json.loads(obs_path.read_text())['identity']!=cfg['current_identity']:raise ValueError('Current cache changed')
            rows.append({'token':token,'log':log});continue
        with np.load(Path(cfg['observation_root'])/'observations'/(token+'.npz')) as obs:
            if int(obs['timestamp'])!=frame['timestamp']:raise ValueError('Current image/calibration time mismatch')
            paths=obs['image_paths'].tolist();calibration={k:obs[k].tolist() for k in ('intrinsics','extrinsics','distortion')}
        poses=[]
        for current in frames[at-3:at+1]:
            if current['timestamp']>frame['timestamp']:raise ValueError('Future history')
            xy=current['ego2global_translation'][:2];yaw=Quaternion(*current['ego2global_rotation']).yaw_pitch_roll[0]
            poses.append([float(xy[0]),float(xy[1]),float(yaw)])
        payload={'identity':cfg['current_identity'],'token':token,'log':log,'timestamp':int(frame['timestamp']),
                 'image_paths':paths,'global_pose_history':poses,'navigation':int(np.asarray(frame['driving_command']).argmax()),
                 'ego_speed':float(np.linalg.norm(frame['ego_dynamic_state'][:2])),'current_calibration':calibration}
        atomic_json(obs_path,payload)
        # This is a separate LABEL-side read/write; none enters payload above.
        raw=load_trusted(Path(cfg['processed_root'])/(token+'.pkl'))
        relative=relative_ego_target(np.asarray(raw['glo_status']['global_poses'])[:12])
        ego=torch.from_numpy(encode_ego(relative))
        temporary=label_path.with_suffix('.tmp');torch.save({'identity':cfg['ego_identity'],'token':token,'ego':ego},temporary);temporary.replace(label_path)
        rows.append({'token':token,'log':log})
    return rows


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('split-manifest','raw-log-root','processed-root','observation-root','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--split',choices=('train','dev'),required=True);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    split=json.loads(Path(a.split_manifest).read_text());tokens=split[a.split+'_tokens'];logs=split['token_logs']
    if {logs[t] for t in split['train_tokens']}&{logs[t] for t in split['dev_tokens']}:raise ValueError('Log leakage')
    index=[{'token':t,'log':logs[t]} for t in tokens]
    current={'schema':'ddpolicy_current_cameras_v1','split':a.split,'index_sha256':identity_hash(index),
             'partition_sha256':file_sha256(a.split_manifest),'writer_sha256':file_sha256(__file__),
             'source_observation_identity':json.loads((Path(a.observation_root)/'identity.json').read_text())['identity'],
             'cameras':['CAM_F0','CAM_L0','CAM_R0'],'current_only':True,'history_images':0}
    current['identity']=identity_hash(current)
    ego={'schema':'foresight_ego_labels_v1','split':a.split,'index_sha256':identity_hash(index),
         'normalization':'original DDP1225 xy absolute ego(t0),sincos yaw','processed_root':a.processed_root}
    ego['identity']=identity_hash(ego);out=Path(a.output)
    if out.exists():
        if not a.resume or json.loads((out/'identity.json').read_text())!=current:raise ValueError('Existing student data changed')
    else:
        (out/'current').mkdir(parents=True);(out/'ego').mkdir();atomic_json(out/'identity.json',current)
        atomic_json(out/'ego_identity.json',ego);atomic_json(out/'index.json',index)
    cfg={**vars(a),'current_identity':current['identity'],'ego_identity':ego['identity']};groups=defaultdict(list)
    for t in tokens:groups[logs[t]].append(t)
    completed=0
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for rows in pool.map(process,[(log,t,cfg) for log,t in groups.items()]):
            completed+=len(rows);atomic_json(out/'progress.json',{'completed':completed,'requested':len(tokens)})
    atomic_json(out/'COMPLETE.json',{'scenes':completed,'logs':len(groups),'failures':0,'current_identity':current['identity'],'ego_identity':ego['identity']})


if __name__=='__main__':main()
