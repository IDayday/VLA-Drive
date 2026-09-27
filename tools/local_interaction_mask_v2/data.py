"""Current-only file boundary; labels and original full metadata never enter a graph."""
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
from PIL import Image
import torch
from starVLA.model.modules.joint_world.observability import CurrentObservation
from starVLA.model.modules.structured_world.geometry import crop_resize_intrinsics

OBSERVATION_FIELDS={'intrinsics','extrinsics','image_transforms','distortion','timestamp','camera_names','image_paths','schema_version'}
CURRENT_FIELDS={'schema_version','token','ego_history_poses','ego_speed_mps','navigation','source'}
CURRENT_FIELDS_WITHOUT_SOURCE=CURRENT_FIELDS-{'source'}


def current_metadata_from_training_pickle(path,token):
    """One-time whitelist conversion; accesses only ego<=t0, never neighboring labels."""
    with Path(path).open('rb') as f:raw=pickle.load(f)
    status=raw['glo_status']
    poses=np.asarray(status['global_poses'][:4],dtype=np.float64)
    if poses.shape!=(4,3):raise ValueError('Need four allowed ego states')
    command=np.asarray(status['commands'][3],dtype=np.float64)
    velocity=np.asarray(status['velocities'][3],dtype=np.float64)
    return {'schema_version':2,'token':token,'ego_history_poses':poses.tolist(),
            'ego_speed_mps':float(np.linalg.norm(velocity)),
            'navigation':['left','straight','right','unknown'][int(command.argmax())],
            'source':'explicit current ego whitelist from NAVSIM metadata; future entries not accessed'}


def observation_from_files(observation_path,current_record,load_images=False,verify_transform=True):
    if set(current_record)!=CURRENT_FIELDS or current_record['schema_version']!=2:
        raise ValueError('Current ego record includes unsupported fields/version')
    with np.load(observation_path,allow_pickle=False) as z:
        if set(z.files)!=OBSERVATION_FIELDS or int(z['schema_version'])!=1:
            raise ValueError('Observation whitelist/version mismatch')
        d={k:z[k] for k in z.files}
    images=[];hashes=[]
    for v,name in enumerate(d['image_paths']):
        path=Path(str(name))
        if verify_transform or load_images:
            with Image.open(path) as image:
                w,h=image.size
                _,affine=crop_resize_intrinsics(np.eye(3),(w,h),(1024,576))
                if not np.allclose(affine,d['image_transforms'][v],atol=1e-7,rtol=1e-6):
                    raise ValueError('Stored calibration transform differs from actual image crop/resize')
                if load_images:
                    image=image.convert('RGB')
                    if w/h>16/9:
                        cw=int(h*16/9);image=image.crop(((w-cw)//2,0,(w+cw)//2,h))
                    elif w/h<16/9:
                        ch=int(w*9/16);image=image.crop((0,(h-ch)//2,w,(h+ch)//2))
                    images.append(image.resize((1024,576),Image.Resampling.LANCZOS))
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    fingerprint=hashlib.sha256(Path(observation_path).read_bytes()+json.dumps(current_record,sort_keys=True).encode()+repr(hashes).encode()).hexdigest()
    obs=CurrentObservation(current_record['token'],d['intrinsics'],d['extrinsics'],d['image_transforms'],d['distortion'],
                           tuple(d['camera_names'].tolist()),(1024,576),int(d['timestamp']),(int(d['timestamp']),)*3,
                           current_record['ego_speed_mps'],current_record['navigation'],fingerprint)
    obs.validate()
    return obs,images


def current_example(observation_path,current_record):
    observation,images=observation_from_files(observation_path,current_record,load_images=True)
    poses=np.asarray(current_record['ego_history_poses'],dtype=np.float64)
    if poses.shape!=(4,3) or not np.isfinite(poses).all():raise ValueError('Invalid allowed ego history')
    yaw=poses[3,2];rot=np.array([[np.cos(yaw),-np.sin(yaw)],[np.sin(yaw),np.cos(yaw)]])
    delta=(poses[3,:2]-poses[2,:2])@rot
    dyaw=(poses[3,2]-poses[2,2]+np.pi)%(2*np.pi)-np.pi
    # Original NAVSIM ver1225 act_norm=1 constants, preserved without future reads.
    state=np.array([[(delta[0]-10.172484)/8.805105,(delta[1]-.360762)/2.277741,np.sin(dyaw),np.cos(dyaw)]],np.float32)
    command={'left':'turn left','straight':'keep straight','right':'turn right','unknown':'unknown'}[observation.navigation]
    language=f'You are an autonomous driving agent. The navigation command for the current timestep is {command}. Your task is to plan future actions based on the understanding of the driving scene.'
    return {'image':images,'state':state,'lang':language,'token':observation.token},observation


def load_current_cache(root,record,manifest,purpose='formal_public_origin'):
    from tools.joint_world.extract_conditions import CACHE_FIELDS
    path=Path(root)/(record['token']+'.pt');blob=path.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=record['sha256']:raise ValueError('Current feature bytes changed')
    data=torch.load(path,map_location='cpu',weights_only=True)
    if set(data)!=CACHE_FIELDS or data['schema_version']!=1 or data['token']!=record['token']:
        raise ValueError('Invalid inherited current-feature contract')
    if data['identity_sha256']!=record.get('identity_sha256',manifest['identity_sha256']):raise ValueError('Current feature identity mismatch')
    identity=manifest['identity']
    if not identity['frozen_upstream'] or identity['targets_loaded']:raise ValueError('Only frozen current features accepted')
    if purpose=='formal_public_origin':
        if not identity.get('public_origin_verified') or identity.get('private_driving_weights_loaded',True):
            raise ValueError('Private/unknown-origin current features forbidden for formal independent algorithm')
    elif purpose!='historical_engineering_audit':raise ValueError('Unknown cache usage scope')
    return data
