"""Append real historical image paths to a sealed native population/GT dataset."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import pickle
import time
from starVLA.rl.qwen_sourceflow.contracts import require_dispatch, file_sha
from starVLA.rl.qwen_sourceflow.data import RawSceneData
from starVLA.rl.flow_grpo.transactions import atomic_json

CAMERAS = ('cam_l0', 'cam_f0', 'cam_r0')


def images_at(frame, raw, split):
    images = {}
    for camera in CAMERAS:
        value = next((v for k, v in frame['cams'].items() if k.lower() == camera), None)
        if value is None:
            images[camera] = None
            continue
        candidates = [Path(raw)/'sensor_blobs'/split/value['data_path'],
            Path('/mnt/navsim')/(split+'_sensor_blobs')/split/value['data_path']]
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            raise FileNotFoundError(candidates[0])
        images[camera] = str(found)
    return images


def one_log(job):
    log, rows, expected_sha, raw, split = job
    path = Path(raw)/'navsim_logs'/split/(log+'.pkl')
    if file_sha(path) != expected_sha:
        raise ValueError('Historical frames must come from the same locked raw log: '+log)
    with path.open('rb') as stream:
        frames = pickle.load(stream)
    found = {}
    for i, current in enumerate(frames):
        token = current['token']
        if token not in rows:
            continue
        if i < 2:
            raise ValueError('Missing two actual historical frames; no repeated-current padding: '+token)
        row = dict(rows[token])
        if int(current['timestamp']) != row['timestamp']:
            raise ValueError('Current scene/time association changed')
        observations = []
        for offset in [-2, -1, 0]:
            frame = frames[i+offset]
            delta = (int(frame['timestamp'])-row['timestamp'])/1e6
            if (offset < 0 and delta >= 0.) or (offset == 0 and delta != 0.):
                raise ValueError('Historical observation is not strictly in the past')
            observations.append(dict(offset=offset, timestamp_us=int(frame['timestamp']),
                relative_seconds=delta, images=images_at(frame, raw, split)))
        for camera in CAMERAS:
            old, new = row['images'][camera], observations[-1]['images'][camera]
            if old == new:
                continue
            if old is None or new is None or (not os.path.samefile(old, new) and file_sha(old) != file_sha(new)):
                raise ValueError('Current image bytes/field differ from the baseline: '+token+' '+camera+
                    ' '+repr((old,new)))
        # Preserve the original ingress paths after verifying storage aliases.
        observations[-1]['images'] = dict(row['images'])
        if observations[0]['timestamp_us'] >= observations[1]['timestamp_us']:
            raise ValueError('Nonmonotonic history; never interpolate or silently replace frames')
        row['observation_frames'] = observations
        found[token] = row
    if set(found) != set(rows):
        raise ValueError('Registered scene missing from raw log: '+log)
    return found


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--data', required=True);p.add_argument('--cohort', choices=['TRAIN', 'DEV', 'NAVTEST'], required=True)
    p.add_argument('--output', required=True);p.add_argument('--workers', type=int, default=8)
    p.add_argument('--raw-root', default='/mnt/project/DriveDreamer-Policy/navsim_raw')
    a = p.parse_args();host = require_dispatch();start = time.monotonic()
    data = RawSceneData(a.data, cohort=a.cohort)
    out = Path(a.output);out.mkdir(parents=True, exist_ok=False)
    groups = {}
    for token in data.tokens:
        row = data.records[token];groups.setdefault(row['log'], {})[token] = row
    split = 'test' if a.cohort == 'NAVTEST' else 'trainval'
    jobs = [(log, rows, data.manifest['raw_logs_sha256'][log], a.raw_root, split) for log, rows in groups.items()]
    records = {}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for rows in pool.map(one_log, jobs):
            records.update(rows)
    if set(records) != set(data.tokens):
        raise ValueError('Temporal input cannot change population eligibility')
    with (out/'records.pkl').open('wb') as stream:
        pickle.dump(records, stream, protocol=5)
    manifest = dict(data.manifest, schema='native_qwen_sourceflow_temporal_scene_v1',
        parent_manifest_sha256=data.sha, records_sha256=file_sha(out/'records.pkl'), frame_offsets=[-2,-1,0],
        condition_contract='three actual chronological frames, each L/F/R; current legal ego/navigation only; no future',
        history_time_contract='raw frame indices and actual microsecond timestamps; no interpolation/repeated-current padding',
        writer_sha256=file_sha(__file__))
    atomic_json(out/'manifest.json', manifest)
    receipt = dict(status='COMPLETE', host=host, manifest_sha256=file_sha(out/'manifest.json'),
        scenes=len(records), logs=len(groups), parent_manifest_sha256=data.sha,
        labels_and_scene_order_unchanged=True, temporal_inputs_only=True,
        history_gap_scenes=sum(any(abs(f['relative_seconds']-.5*f['offset'])>.02 for f in r['observation_frames']) for r in records.values()),
        wall_seconds=time.monotonic()-start)
    atomic_json(out/'COMPLETE', receipt);print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
