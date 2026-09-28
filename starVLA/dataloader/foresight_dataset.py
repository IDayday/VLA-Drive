"""Current cameras and labels are distinct objects; no video/depth dependencies."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Original DDP version1225 constants; no statistics fit on dev/Navtest.
X_MEAN,X_STD,Y_MEAN,Y_STD=10.172484,8.805105,.360762,2.277741


def encode_ego(xy_yaw):
    p=np.asarray(xy_yaw,dtype=np.float32)
    return np.column_stack(((p[:,0]-X_MEAN)/X_STD,(p[:,1]-Y_MEAN)/Y_STD,np.sin(p[:,2]),np.cos(p[:,2]))).astype(np.float32)


def decode_ego(encoded):
    xy=encoded[...,:2]*encoded.new_tensor([X_STD,Y_STD])+encoded.new_tensor([X_MEAN,Y_MEAN])
    return torch.cat((xy,torch.atan2(encoded[...,2],encoded[...,3])[...,None]),-1)


def current_observation(record):
    required={'token','image_paths','global_pose_history','navigation'}
    if not required.issubset(record):raise ValueError('Missing current observation fields')
    poses=np.asarray(record['global_pose_history'],dtype=np.float64)
    if poses.shape!=(4,3) or not np.isfinite(poses).all():raise ValueError('Invalid current history')
    prev,now=poses[2:4]; c,s=np.cos(now[2]),np.sin(now[2])
    xy=(now[:2]-prev[:2])@np.array([[c,-s],[s,c]])
    yaw=(now[2]-prev[2]+np.pi)%(2*np.pi)-np.pi
    state=encode_ego(np.array([[*xy,yaw]]))
    command=record['navigation']
    if command not in (0,1,2,3):raise ValueError('Invalid navigation')
    if len(record['image_paths'])!=3:raise ValueError('Expected front,left-front,right-front')
    images=[]
    for path in record['image_paths']:
        with Image.open(path) as src:
            img=src.convert('RGB');w,h=img.size
            if w/h>16/9:
                cw=int(h*16/9);img=img.crop(((w-cw)//2,0,(w+cw)//2,h))
            elif w/h<16/9:
                ch=int(w*9/16);img=img.crop((0,(h-ch)//2,w,(h+ch)//2))
            images.append(img.resize((1024,576),Image.Resampling.LANCZOS))
    text=('turn left','keep straight','turn right','unknown')[command]
    prompt=f'You are an autonomous driving agent. The navigation command for the current timestep is {text}. Your task is to plan future actions based on the understanding of the driving scene.'
    return {'image':images,'state':state,'lang':prompt,'token':record['token']}


class ForesightCurrentDataset(Dataset):
    def __init__(self,root):
        from starVLA.model.modules.vehicle_joint.initialization import identity_hash
        self.root=Path(root);self.identity=json.loads((self.root/'identity.json').read_text())
        self.index=json.loads((self.root/'index.json').read_text())
        if self.identity['schema']!='ddpolicy_current_cameras_v1' or identity_hash(self.index)!=self.identity['index_sha256']:
            raise ValueError('Current-only schema/index mismatch')
    def __len__(self):return len(self.index)
    def __getitem__(self,i):
        r=json.loads((self.root/'current'/(self.index[i]['token']+'.json')).read_text())
        if r['identity']!=self.identity['identity']:raise ValueError('Current record identity mismatch')
        return current_observation(r)


class ForesightTrainingDataset(ForesightCurrentDataset):
    """Every arm keeps the current/ego index; missing auxiliary labels are masks."""
    def __init__(self,root,*,future_root=None,interaction_root=None,expected_future=None,expected_interaction=None):
        super().__init__(root)
        self.ego_identity=json.loads((self.root/'ego_identity.json').read_text())
        if self.ego_identity['schema']!='foresight_ego_labels_v1' or self.ego_identity['index_sha256']!=self.identity['index_sha256']:
            raise ValueError('Ego/current manifest mismatch')
        self.future_root=Path(future_root) if future_root else None
        self.interaction_root=Path(interaction_root) if interaction_root else None
        self.future_shards={};self.future_identity=None;self.interaction_identity=None
        if self.future_root:
            if not expected_future:raise ValueError('Explicit future identity required')
            for folder in sorted(self.future_root.glob('shard_*')):
                ident=json.loads((folder/'identity.json').read_text())
                if ident['schema']!='foresight_flux_targets_v1' or ident['identity']!=expected_future or ident['partition_hash']!=self.identity['partition_sha256'] or ident['split']!=self.identity['split']:
                    raise ValueError('VAE/time/preprocessing/split cache identity mismatch')
                if not (folder/'COMPLETE.json').exists():raise ValueError('Future cache shard incomplete')
                self.future_shards[int(folder.name.split('_')[-1])]=folder
                self.future_identity=ident
            if self.future_identity is None or len(self.future_shards)!=self.future_identity['shards']:raise ValueError('Missing future shards')
        if self.interaction_root:
            ident=json.loads((self.interaction_root/'identity.json').read_text())
            if ident['schema']!='foresight_interaction_latent_v1' or ident['identity']!=expected_interaction or ident['index_hash']!=self.identity['index_sha256'] or ident['split']!=self.identity['split']:
                raise ValueError('Teacher/schema/graph/coordinate cache identity mismatch')
            if not (self.interaction_root/'COMPLETE.json').exists():raise ValueError('Teacher target cache incomplete')
            self.interaction_identity=ident
    def __getitem__(self,i):
        observation=super().__getitem__(i);token=self.index[i]['token']
        label=torch.load(self.root/'ego'/(token+'.pt'),weights_only=True,map_location='cpu')
        if label['identity']!=self.ego_identity['identity'] or label['token']!=token:raise ValueError('Ego label identity mismatch')
        targets={'ego':label['ego']}
        if self.future_root:
            folder=self.future_shards[i%self.future_identity['shards']]
            label=torch.load(folder/'targets'/(token+'.pt'),weights_only=True,map_location='cpu')
            if label['identity']!=self.future_identity['identity'] or label['token']!=token:raise ValueError('Future label identity mismatch')
            targets.update(future_latent=label['latent'],future_valid=label['valid'])
        if self.interaction_root:
            label=torch.load(self.interaction_root/'targets'/(token+'.pt'),weights_only=True,map_location='cpu')
            if label['identity']!=self.interaction_identity['identity'] or label['token']!=token:raise ValueError('Interaction label identity mismatch')
            targets.update(interaction_latent=label['latent'],interaction_valid=torch.tensor(label['interaction_target_valid'],dtype=torch.bool))
        return observation,targets


def collate_training(rows):
    observations,targets=zip(*rows)
    return list(observations),{k:torch.stack([t[k] for t in targets]) for k in targets[0]}
