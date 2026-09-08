"""Read-only train-log frame check. Differences are a diagnostic, NEVER WM conditions."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import torch
from navsim.common.dataclasses import Scene
from navsim.agents.EpisodeDrive.planreg_v2.motion import GTLogMotionBuilder


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('logs','manifest','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    if Path(args.output).exists():raise FileExistsError('New coordinate audit required')
    manifest=json.loads(Path(args.manifest).read_text())
    if manifest['split'] not in ('train','trainval_final_fit'):raise ValueError('Train-log audit only')
    errors={'ego':[],'global':[]};audit=[];loaded={}
    for record in manifest['records'][:64]:
        log=record['log']
        if log not in loaded:
            with (Path(args.logs)/(log+'.pkl')).open('rb') as stream:raw=pickle.load(stream)
            loaded[log]=(raw,{row['token']:i for i,row in enumerate(raw)})
        raw,indices=loaded[log];start=indices[record['token']];frames=raw[start:start+9]
        states=[Scene._build_ego_status(row) for row in frames]
        values=dict(poses_global=np.stack([s.ego_pose for s in states]),
            logged_velocity=np.stack([s.ego_velocity for s in states]),
            logged_acceleration=np.stack([s.ego_acceleration for s in states]),
            timestamps=np.array([r['timestamp'] for r in frames])/1e6,
            valid_mask=np.ones(len(frames),dtype=bool))
        for frame in ('ego','global'):
            built=GTLogMotionBuilder(frame).build(**values)
            motion,times=built['motion_sequence'],built['timestamps']
            interval=(motion[1:,:2]-motion[:-1,:2])/(times[1:]-times[:-1])[:,None]
            endpoint_average=(motion[1:,4:6]+motion[:-1,4:6])*.5
            moving=interval.norm(dim=-1)>1.
            errors[frame].extend((endpoint_average[moving]-interval[moving]).square().sum(-1).tolist())
        audit.append(dict(token=record['token'],log=log,timestamps=times.tolist()))
    result=dict(scenes=len(audit),log_segments=len(loaded),source_frame_declared=manifest['source_vector_frame'],
        velocity_vector_rmse_mps={name:float(np.sqrt(np.mean(value))) if value else None for name,value in errors.items()},
        moving_intervals=len(errors['ego']),records=audit,
        coordinate_contract='NAVSIM Scene._build_ego_status -> NavsimScenario.get_ego_state -> rear-axle EgoState; no extra reference-point shift',
        interpretation='Empirical source-frame consistency on these logs, not a proof for every future dataset; no finite-difference conditions used in training',
        manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest())
    Path(args.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':main()
