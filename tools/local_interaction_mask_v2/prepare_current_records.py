"""Extract ONLY allowed ego history and current three-camera observations from logs."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from starVLA.model.modules.structured_world.geometry import crop_resize_intrinsics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('index','raw-log-root','sensor-root','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    for name in ('observations','current_records'):(out/name).mkdir()
    index=json.loads(Path(a.index).read_text());grouped=defaultdict(list)
    # The index contains fixed-protocol tokens/log names, not scores or labels.
    if isinstance(index,dict):index=index['records']
    for row in index:grouped[row['log']].append(row['token'])
    records=[];failures=[]
    for log,tokens in grouped.items():
        path=Path(a.raw_log_root)/(log+'.pkl')
        with path.open('rb') as f:frames=pickle.load(f)
        positions={f['token']:i for i,f in enumerate(frames)}
        for token in tokens:
            try:
                i=positions[token]
                if i<3:raise ValueError('Missing allowed four-state history')
                current=frames[i];history=frames[i-3:i+1] # Never access later frames.
                time=int(current['timestamp'])
                if any(int(f['timestamp'])>time for f in history):raise ValueError('Future history state')
                poses=[[float(f['ego2global_translation'][0]),float(f['ego2global_translation'][1]),float(Quaternion(f['ego2global_rotation']).yaw_pitch_roll[0])] for f in history]
                velocity=np.asarray(current['ego_dynamic_state'])[:2]
                command=np.asarray(current['driving_command']);navigation=['left','straight','right','unknown'][int(command.argmax())]
                record={'schema_version':2,'token':token,'ego_history_poses':poses,'ego_speed_mps':float(np.linalg.norm(velocity)),
                        'navigation':navigation,'source':'Raw NAVSIM allowed ego<=t0 whitelist; no annotations or future values accessed'}
                cameras=('CAM_F0','CAM_L0','CAM_R0');ks=[];extrinsics=[];affines=[];distortion=[];images=[]
                for camera in cameras:
                    cam=current['cams'][camera];relative=Path(cam['data_path'])
                    image=Path(a.sensor_root)/relative
                    if not image.is_file():
                        image=Path(a.sensor_root)/Path(*relative.parts[-3:])
                    if not image.is_file():raise FileNotFoundError(image)
                    with Image.open(image) as im:k,affine=crop_resize_intrinsics(cam['cam_intrinsic'],im.size,(1024,576))
                    ext=np.eye(4);ext[:3,:3]=cam['sensor2lidar_rotation'];ext[:3,3]=cam['sensor2lidar_translation']
                    ks.append(k);extrinsics.append(np.asarray(current['lidar2ego'])@ext);affines.append(affine);distortion.append(cam['distortion']);images.append(str(image))
                np.savez(out/'observations'/(token+'.npz'),intrinsics=np.array(ks),extrinsics=np.array(extrinsics),image_transforms=np.array(affines),
                    distortion=np.array(distortion),timestamp=np.array(time),camera_names=np.array(cameras),image_paths=np.array(images),schema_version=np.array(1))
                record_path=out/'current_records'/(token+'.json');record_path.write_text(json.dumps(record,indent=2)+'\n')
                records.append({'token':token,'log':log,'decision_time':time,'record_sha256':hashlib.sha256(record_path.read_bytes()).hexdigest()})
            except Exception as e:failures.append({'token':token,'log':log,'error':repr(e)})
        print(json.dumps({'logs_finished':len(set(r['log'] for r in records)),'records':len(records),'failures':len(failures)}),flush=True)
    (out/'manifest.json').write_text(json.dumps({'requested':len(index),'completed':len(records),'failures':failures,'records':records,'labels_stored':False,
        'future_frames_accessed':False,'annotation_fields_accessed':False,'sensor_contract':'current CAM_F0,CAM_L0,CAM_R0 plus4 permitted ego states; no neighboring speed/history'},indent=2)+'\n')
    if failures:raise RuntimeError('Current record failures retained; cannot silently omit scenes')


if __name__=='__main__':main()
