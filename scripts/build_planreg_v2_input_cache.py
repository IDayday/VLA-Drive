"""Build a NEW input-only cache from explicit training tokens; never overwrite V1."""
import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
import torch
from navsim.common.dataclasses import Scene,SensorConfig
from navsim.agents.EpisodeDrive.drivevla_features import DriveVLAFeatureBuilder
from navsim.agents.EpisodeDrive.layers.world_model.future_image_io import encode_path_tensor,decode_path_tensor
from navsim.agents.EpisodeDrive.planreg_v2.targets import V2TrajectoryTargetBuilder
from navsim.agents.EpisodeDrive.planreg_v2 import CACHE_SCHEMA,LONG_TARGET_VERSION
from navsim.agents.EpisodeDrive.planreg_v2.normalizers import measured_statistics


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ('logs','sensors','metric-metadata','output'): p.add_argument('--'+name,required=True)
    p.add_argument('--tokens',help='Explicit JSON training token array; required outside bounded smoke')
    p.add_argument('--smoke-scenes',type=int,default=0)
    p.add_argument('--exclude-recorded-drives',help='JSON list; disjoint fixed probe selection, not a new unseen claim for old models')
    p.add_argument('--source-vector-frame',choices=['ego','global'],required=True)
    p.add_argument('--split',choices=['train','trainval_final_fit','development','navtest'],default='train')
    args = p.parse_args()
    if args.split in ('train','trainval_final_fit') and (Path(args.logs).name in ('test','navtest') or 'navtest' in args.metric_metadata.lower()):
        raise ValueError('Navtest may not be relabeled as training/statistics data')
    output = Path(args.output)
    if output.exists(): raise FileExistsError('New cache root required; never overwrite older caches')
    if not args.tokens and not 0 < args.smoke_scenes <= 64:
        raise ValueError('Explicit training tokens or bounded <=64-scene smoke required')
    authorized = set(json.loads(Path(args.tokens).read_text())) if args.tokens else None
    rows = list(csv.DictReader(open(args.metric_metadata)))
    metric = {Path(r['file_name']).parent.name:r['file_name'] for r in rows}
    per_log = {}
    for token,path in metric.items():
        if authorized is None or token in authorized:
            per_log.setdefault(Path(path).parents[2].name,[]).append(token)
    sensor = SensorConfig(cam_f0=[3,4,6,11],cam_l0=[],cam_l1=[],cam_l2=[],cam_r0=[],cam_r1=[],cam_r2=[],cam_b0=[],lidar_pc=[])
    builder = V2TrajectoryTargetBuilder(args.source_vector_frame)
    feature_builder = DriveVLAFeatureBuilder(cache_hidden_state=False)
    records,statistics = [],[]
    excluded_drives=set(json.loads(Path(args.exclude_recorded_drives).read_text())) if args.exclude_recorded_drives else set()
    drive_counts={}
    output.mkdir(parents=True)
    for log,tokens in sorted(per_log.items()):
        drive=log.rsplit('_',2)[0]
        if drive in excluded_drives or (args.smoke_scenes and drive_counts.get(drive,0)>=4): continue
        path = Path(args.logs)/(log+'.pkl')
        if not path.is_file(): continue
        with path.open('rb') as stream: raw = pickle.load(stream)
        by_token = {r['token']:i for i,r in enumerate(raw)}
        # Bounded smoke samples at most four per log, across several logs.
        selected = tokens[:4] if args.smoke_scenes else tokens
        for token in selected:
            if token not in by_token: continue
            i = by_token[token]
            if i < 3: raise ValueError('Declared scene lacks numeric history: '+token)
            frame_list = raw[i-3:i+11]
            if any(r['log_name'] != log for r in frame_list): raise ValueError('Cross-log future data')
            scene = Scene.from_scene_dict_list(frame_list,Path(args.sensors),4,len(frame_list)-4,sensor,load_image_path=True)
            target = builder.compute_targets(scene)
            feature = feature_builder.compute_features(scene.get_agent_input())
            feature['image_path_tensor'],feature['image_path_length'] = encode_path_tensor(str(scene.frames[3].cameras.cam_f0.image))
            for h in range(3):
                if target['future_valid_mask'][h]:
                    future = decode_path_tensor(target['future_image_paths'][h],target['future_image_path_lengths'][h])
                    if not Path(future).is_file(): target['future_valid_mask'][h] = False
            cache_path = output/(token+'.pt')
            torch.save(dict(schema=CACHE_SCHEMA,long_target_version=LONG_TARGET_VERSION,features=feature,targets=target),cache_path)
            velocity=scene.frames[3].ego_status.ego_velocity
            heading=target['trajectory'][-1,2].item()
            category='stop' if float(sum(v*v for v in velocity))<.25 else 'turn' if abs(heading)>.3 else 'straight'
            records.append(dict(token=token,log=log,recorded_drive=drive,category=category,
                cache_path=str(cache_path.resolve()),metric_cache_path=metric[token],long_target_version=LONG_TARGET_VERSION))
            drive_counts[drive]=drive_counts.get(drive,0)+1
            statistics.append((token,target['trajectory'],target['trajectory_valid']))
            if args.smoke_scenes and len(records) >= args.smoke_scenes: break
        if args.smoke_scenes and len(records) >= args.smoke_scenes: break
    found = {r['token'] for r in records}
    if authorized is not None and not args.smoke_scenes and found != authorized:
        raise ValueError('Missing authorized scene tokens; cache remains incomplete: '+str(len(authorized-found)))
    if not records: raise ValueError('No requested records matched logs and metric cache')
    manifest = dict(schema=CACHE_SCHEMA,long_target_version=LONG_TARGET_VERSION,split=args.split,records=records,source_vector_frame=args.source_vector_frame,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        target_source_sha256=hashlib.sha256(Path(__import__('inspect').getfile(V2TrajectoryTargetBuilder)).read_bytes()).hexdigest(),
        source_vector_contract='Explicit caller declaration; verify against source database before formal training',
        token_sha256=hashlib.sha256('\n'.join(sorted(found)).encode()).hexdigest(),
        future_offsets=[1,3,8],future_seconds=[.5,1.5,4.],long_seconds=5.,sensor_camera_count=1,
        smoke=bool(args.smoke_scenes))
    if args.split in ('train','trainval_final_fit'):
        n = measured_statistics(statistics,args.split,'navsim-raw-log:'+manifest['token_sha256'])
        n.save(output/'normalizer.json')
        manifest['raw_gt_sha256']=n.metadata['raw_gt_sha256']
        manifest['statistics_contract']=n.metadata['statistics_contract']
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'recorded_drives.json').write_text(json.dumps(sorted(drive_counts),indent=2))
    print(json.dumps(dict(scenes=len(records),logs=len({r['log'] for r in records}),manifest=str(output/'manifest.json'))))


if __name__ == '__main__': main()
