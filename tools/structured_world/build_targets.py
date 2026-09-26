"""Build separate observation-calibration and target caches from trusted NAVSIM logs."""
import argparse
import dataclasses
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
import torch
from starVLA.model.modules.structured_world.targets import make_targets
from starVLA.model.modules.structured_world.geometry import crop_resize_intrinsics, geometric_fov


def main():
    p=argparse.ArgumentParser()
    for name in ['manifest','processed-root','raw-log-root','sensor-root','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--limit',type=int,default=64);p.add_argument('--capacity',type=int,default=32)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError('Use a fresh versioned cache directory')
    out.mkdir(parents=True);(out/'targets').mkdir();(out/'observations').mkdir()
    tokens=json.loads(Path(a.manifest).read_text())[:a.limit]
    grouped=defaultdict(list)
    for token in tokens:
        raw=pickle.load((Path(a.processed_root)/(token+'.pkl')).open('rb'))
        paths=[raw['glo_images'][cam]['image_paths'][3] for cam in ['cam_f0','cam_l0','cam_r0']]
        log=Path(paths[0]).parts[-3]
        grouped[log].append((token,paths))
    records=[];failures=[]
    for log,items in grouped.items():
        frames=pickle.load((Path(a.raw_log_root)/(log+'.pkl')).open('rb'))
        index={f['token']:i for i,f in enumerate(frames)}
        for token,processed_paths in items:
            try:
                i=index[token];f=frames[i]
                if not np.allclose(f['lidar2ego'],np.eye(4),atol=1e-6):
                    raise ValueError('Nonidentity lidar2ego requires an explicit annotation-frame adapter')
                # Observations use current frame only. No annotation, track or future field here.
                cameras=('CAM_F0','CAM_L0','CAM_R0');ks=[];es=[];affines=[];ds=[];paths=[]
                for cam,processed in zip(cameras,processed_paths):
                    c=f['cams'][cam]
                    if Path(c['data_path']).name != Path(processed).name:
                        raise ValueError('Decision-frame image mismatch')
                    path=Path(a.sensor_root)/c['data_path']
                    if not path.is_file() and Path(processed).is_file():path=Path(processed)
                    with Image.open(path) as image:size=image.size
                    k,affine=crop_resize_intrinsics(c['cam_intrinsic'],size,(1024,576))
                    ext=np.eye(4);ext[:3,:3]=c['sensor2lidar_rotation'];ext[:3,3]=c['sensor2lidar_translation']
                    ks.append(k);es.append(f['lidar2ego']@ext);affines.append(affine);ds.append(c['distortion']);paths.append(str(path))
                np.savez(out/'observations'/f'{token}.npz',intrinsics=np.array(ks),extrinsics=np.array(es),image_transforms=np.array(affines),distortion=np.array(ds),timestamp=np.array(f['timestamp']),camera_names=np.array(cameras),image_paths=np.array(paths),schema_version=np.array(1))
                # Restrict current supervision and no-object loss to the same calibrated FOV.
                raw_boxes=np.asarray(f['anns']['gt_boxes']) if f.get('anns') is not None else np.empty((0,7))
                supported=geometric_fov(raw_boxes[:,:3],np.array(ks),np.array(es),np.array(ds))
                targets=make_targets(f,frames[i+1:i+9],capacity=a.capacity,current_eligibility=supported)
                xx,yy=np.meshgrid(np.arange(1.5,50.),np.arange(-19.5,20.),indexing='ij')
                grid=np.zeros(xx.shape,dtype=bool)
                for height in [0.,1.,2.]:
                    points=np.stack([xx,yy,np.full_like(xx,height)],-1).reshape(-1,3)
                    grid |= geometric_fov(points,np.array(ks),np.array(es),np.array(ds)).reshape(xx.shape)
                targets.supervision_grid=torch.from_numpy(grid)
                torch.save(asdict(targets),out/'targets'/f'{token}.pt')
                records.append({'token':token,'log':log,'timestamp':f['timestamp'],'targets':len(targets.track_ids),'overflow':targets.overflow,'valid_future_points':int(targets.future_valid_mask.sum()),'annotation_valid':bool(targets.annotation_valid_mask)})
            except Exception as error:
                failures.append({'token':token,'log':log,'error':repr(error)})
    report={'requested':len(tokens),'completed':len(records),'failures':failures,'records':records,
            'capacity':a.capacity,'overflow_scene_fraction':sum(r['overflow']>0 for r in records)/max(1,len(records)),
            'roi':[1,-20,50,20],'time_step_s':.5,'future_steps':8,'frame':'ego_t0_full_SE3',
            'visibility_limitation':'Calibrated multi-height FOV within ROI; no occlusion reasoning. Annotation coverage outside logged objects is assumed only inside this support.'}
    (out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='records'}))
    if failures:raise RuntimeError('Target build has failures; see audit.json, never silently drop them')

if __name__=='__main__':main()
