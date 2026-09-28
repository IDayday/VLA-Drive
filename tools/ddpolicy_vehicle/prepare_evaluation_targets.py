"""Offline vehicle GT for frozen camera exports, never prediction inputs.

This permits complete Navtest vehicle coverage/motion analysis without requiring
training metadata or reusing a driving model's intermediate cache. Navtest label
construction is deferred until the final-model lock exists.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import torch
from pyquaternion import Quaternion
from .prepare_data import load_trusted, atomic_json
from .evaluate_ego import relative_ego_target
from starVLA.model.modules.structured_world.geometry import geometric_fov
from starVLA.model.modules.vehicle_joint.targets import make_vehicle_targets
from starVLA.model.modules.vehicle_joint.initialization import identity_hash, file_sha256


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('current-root','raw-log-root','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--final-lock');p.add_argument('--resume',action='store_true')
    a=p.parse_args();current=Path(a.current_root);out=Path(a.output)
    info=json.loads((current/'identity.json').read_text())
    lock_sha=None
    if info['split']=='navtest':
        if not a.final_lock:raise ValueError('Freeze formal models before Navtest label-side evaluation')
        lock=json.loads(Path(a.final_lock).read_text())
        if not lock.get('models') or lock['navtest_results_used_for_selection']:
            raise ValueError('Invalid formal model lock')
        lock_sha=file_sha256(a.final_lock)
    source_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain']):raise ValueError('Commit label evaluation source first')
    metadata={'schema':'ddpolicy_vehicle_eval_targets_v1','current_identity':info['identity'],
        'source_sha':source_sha,'builder_sha256':file_sha256(__file__),'capacity':32,
        'raw_log_root':str(Path(a.raw_log_root).resolve()),'final_lock_sha256':lock_sha,
        'scope':'evaluation labels only; forbidden as deployment graph input',
        'class_name':'vehicle','steps':8,'time_step':.5,'roi':[1,-20,50,20]}
    identity=identity_hash(metadata)
    if out.exists():
        if not a.resume or json.loads((out/'identity.json').read_text())['identity']!=identity:
            raise FileExistsError('New evaluation label directory or exact resume required')
    else:
        (out/'targets').mkdir(parents=True);(out/'logs').mkdir()
        atomic_json(out/'identity.json',{'identity':identity,**metadata})
    index=json.loads((current/'index.json').read_text());groups=defaultdict(list)
    if len({r['token'] for r in index})!=len(index):raise ValueError('Duplicate scene identity')
    for row in index:groups[row['log']].append(row['token'])
    counts=Counter();failures=[];completed=0
    for log,tokens in groups.items():
        report_path=out/'logs'/(log+'.json')
        if report_path.exists():
            report=json.loads(report_path.read_text())
            if report['identity']!=identity:raise ValueError('Changed evaluation label identity')
            for row in report['records']:
                if 'error' not in row and file_sha256(out/'targets'/(row['token']+'.pt'))!=row['sha256']:
                    raise ValueError('Previously completed target changed')
        else:
            raw=Path(a.raw_log_root)/(log+'.pkl');frames=load_trusted(raw)
            lookup={f['token']:i for i,f in enumerate(frames)}
            if len(lookup)!=len(frames):raise ValueError('Duplicate raw timestamp token')
            records=[]
            for token in tokens:
                row={'token':token,'log':log}
                try:
                    at=lookup[token];frame=frames[at]
                    obs=json.loads((current/'current'/(token+'.json')).read_text())
                    if obs['identity']!=info['identity'] or obs['timestamp']!=frame['timestamp']:
                        raise ValueError('Current observation/label timestamp mismatch')
                    if not np.allclose(frame['lidar2ego'],np.eye(4),atol=1e-6):
                        raise ValueError('Nonidentity raw box frame needs explicit adapter')
                    calibration=obs['current_calibration'];anns=frame.get('anns')
                    boxes=np.asarray(anns['gt_boxes']) if anns else np.empty((0,7))
                    support=geometric_fov(boxes[:,:3],np.asarray(calibration['intrinsics']),
                        np.asarray(calibration['extrinsics']),np.asarray(calibration['distortion']))
                    target,population=make_vehicle_targets(frame,frames[at+1:at+9],capacity=32,current_eligibility=support)
                    window=frames[at-3:at+9]
                    if len(window)!=12 or any(abs((f['timestamp']-frame['timestamp'])/1e6-.5*(i-3))>.05 for i,f in enumerate(window)):
                        raise ValueError('Incomplete regular ego label horizon')
                    poses=[[float(f['ego2global_translation'][0]),float(f['ego2global_translation'][1]),
                            Quaternion(*f['ego2global_rotation']).yaw_pitch_roll[0]] for f in window]
                    ego_target=relative_ego_target(poses)
                    mask=np.asarray(anns['gt_names'])=='vehicle' if anns else np.zeros(0,dtype=bool)
                    payload={'schema':metadata['schema'],'identity':identity,'targets':asdict(target),'counts':population,
                        'ego_future_xyyaw':torch.from_numpy(ego_target.astype(np.float32)),
                        'source_vehicle_boxes':torch.from_numpy(boxes[mask].copy()),
                        'source_vehicle_support':torch.from_numpy(support[mask].copy())}
                    path=out/'targets'/(token+'.pt');temp=path.with_suffix('.tmp')
                    torch.save(payload,temp);os.replace(temp,path)
                    row.update(counts=population,sha256=file_sha256(path))
                except Exception as error:row['error']=repr(error)
                records.append(row)
            report={'identity':identity,'raw_log_sha256':file_sha256(raw),'records':records}
            atomic_json(report_path,report)
        for row in report['records']:
            if 'error' in row:failures.append(row)
            else:completed+=1;counts.update(row['counts'])
        atomic_json(out/'progress.json',{'completed':completed,'requested':len(index),'failures':len(failures)})
    atomic_json(out/'audit.json',{'identity':identity,'requested':len(index),'completed':completed,
        'failures':failures,'counts':dict(counts),'raw_log_population_available':True,
        'future_annotations_used_only_for_evaluation':True})
    if failures:raise RuntimeError('Failed label scenes retained, not dropped')


if __name__=='__main__':main()
