"""Build compact raw-Scene labels and legal current-observation paths, no hidden cache."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import time
import numpy as np
from starVLA.rl.qwen_sourceflow.contracts import POPULATION, UPSTREAM, ROOT, file_sha, fingerprint, require_dispatch


def process_log(job):
    log, tokens, split, raw = job
    sys.path.insert(0, str(UPSTREAM/'navsim'))
    os.environ.setdefault('NUPLAN_MAPS_ROOT', '/mnt/navsim/maps')
    from navsim.common.dataclasses import Scene, SceneMetadata
    path = Path(raw)/'navsim_logs'/split/(log+'.pkl')
    with path.open('rb') as f:
        frames = pickle.load(f)
    wanted = set(tokens); rows = {}
    for i, current in enumerate(frames):
        token = current['token']
        if token not in wanted:
            continue
        future = frames[i:i+9]
        if len(future) != 9:
            raise ValueError('Missing future GT: '+token)
        # Use official raw Scene conversion. Maps/objects/images are unnecessary
        # for this GT-only method and never enter the observation record.
        scene = Scene(SceneMetadata(log, current['scene_token'], current['map_location'], token, 1, 8),
            None, [SimpleNamespace(ego_status=Scene._build_ego_status(f)) for f in future])
        poses = scene.get_future_trajectory(8).poses
        dt = np.diff([float(f['timestamp']) for f in future]) / 1e6
        if not np.isfinite(dt).all() or not (dt > 0).all():
            raise ValueError('Invalid nonmonotonic raw timestamps: '+repr(dt))
        ego = scene.frames[0].ego_status
        proprio = np.concatenate([ego.ego_velocity, ego.ego_acceleration, ego.driving_command])
        if poses.shape != (8,3) or proprio.shape != (8,) or not np.isfinite(poses).all():
            raise ValueError('Native action/state contract: '+token)
        images = {}
        for camera in ['cam_l0','cam_f0','cam_r0']:
            c = next((v for k,v in current['cams'].items() if k.lower()==camera), None)
            image = Path(raw)/'sensor_blobs'/split/c['data_path'] if c else None
            if image is not None and not image.is_file():
                candidates = [Path('/mnt/navsim')/(split+'_sensor_blobs')/split/c['data_path']]
                image = next((p for p in candidates if p.is_file()), image)
                if not image.is_file():raise FileNotFoundError(image)
            images[camera] = str(image) if image else None
        labels = ['turn left','go straight','turn right','unknown']
        rows[token] = dict(token=token, log=log, requested_GT=poses.astype(np.float32).tolist(),
            proprio=proprio.tolist(), navigation=labels[int(np.argmax(ego.driving_command))], images=images,
            timestamp=int(current['timestamp']), raw_future_dt_seconds=dt.tolist(),
            timestamp_gap=bool(not np.allclose(dt,.5,atol=.02)))
    if set(rows) != wanted:
        raise ValueError('Raw log missing registered scene tokens: '+log)
    return log, rows, file_sha(path)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--cohort',choices=['TRAIN','DEV','NAVTEST'],required=True)
    p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--raw-root',default='/mnt/project/DriveDreamer-Policy/navsim_raw')
    a=p.parse_args();host=require_dispatch();started=time.monotonic();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    pop=json.loads(POPULATION.read_text());key={'TRAIN':'train_tokens','DEV':'dev_tokens','NAVTEST':'navtest_tokens'}[a.cohort]
    groups={}
    for token in pop[key]:groups.setdefault(pop['token_logs'][token],[]).append(token)
    jobs=[(log,tokens,'test' if a.cohort=='NAVTEST' else 'trainval',a.raw_root) for log,tokens in groups.items()]
    records={};hashes={}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for log, rows, sha in ex.map(process_log,jobs):
            records.update(rows);hashes[log]=sha
    with (out/'records.pkl').open('wb') as f:pickle.dump(records,f,protocol=5)
    manifest=dict(schema='native_qwen_sourceflow_raw_scene_v1',cohort=a.cohort,tokens=pop[key],
        population_sha256=file_sha(POPULATION),raw_logs_sha256=hashes,records_sha256=file_sha(out/'records.pkl'),
        action_contract='Scene.get_future_trajectory(8); rear-axle x-forward/y-left/radian; 0.5..4s',
        condition_contract='current L/F/R images, current vx/vy/ax/ay/command; no map/agents/future',
        source_dataclasses_sha256=file_sha(UPSTREAM/'navsim/navsim/common/dataclasses.py'))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    allposes=np.array([r['requested_GT'] for r in records.values()]);norm=2*(allposes+np.array([1.57,19.68,1.67]))/np.array([66.74,42,3.53])-1
    receipt=dict(status='COMPLETE',host=host,manifest_sha256=file_sha(out/'manifest.json'),scenes=len(records),logs=len(hashes),
        out_of_normalizer_range_fraction=(np.abs(norm)>1).mean((0,1)).tolist(),
        heading_minmax=[float(allposes[...,2].min()),float(allposes[...,2].max())],
        scenes_with_raw_timestamp_gaps=sum(r['timestamp_gap'] for r in records.values()),
        time_contract='Official Scene indexed future with nominal 0.5s spacing, including source raw timestamp gaps; no interpolation/filtering',wall_seconds=time.monotonic()-started)
    (out/'COMPLETE').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
