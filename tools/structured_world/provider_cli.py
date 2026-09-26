"""Offline extraction / single-request online CLI; only current observation records required."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from starVLA.model.modules.structured_world.contracts import ModelInputs
from starVLA.model.modules.structured_world.providers import GeometricBEVProvider,state_fingerprint


def load_provider(weights,device):
    payload=torch.load(weights,map_location='cpu',weights_only=False)
    cfg=payload.get('world_config',{})
    provider=GeometricBEVProvider(channels=cfg.get('bev_channels',64)).to(device)
    state={k.removeprefix('provider.'):v for k,v in payload['delta'].items() if k.startswith('provider.')}
    provider.load_state_dict(state,strict=True);provider.eval();provider.requires_grad_(False)
    return provider


def current_observation(path,sensor_root,device):
    scene_token=Path(path).stem
    with np.load(path,allow_pickle=False) as record:
        d={k:record[k] for k in record.files}
    allowed={'intrinsics','extrinsics','image_transforms','distortion','timestamp','camera_names','image_paths','schema_version'}
    if set(d)!=allowed or int(d['schema_version'])!=1:raise ValueError('Invalid observation fields/version')
    images=[];digests=[]
    for relative in d['image_paths']:
        path=Path(sensor_root)/str(relative)
        digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
        with Image.open(path) as image:
            image=image.convert('RGB');w,h=image.size
            if w/h>16/9:
                cw=int(h*16/9);image=image.crop(((w-cw)//2,0,(w+cw)//2,h))
            elif w/h<16/9:
                ch=int(w*9/16);image=image.crop((0,(h-ch)//2,w,(h+ch)//2))
            images.append(np.asarray(image.resize((1024,576),Image.Resampling.LANCZOS),dtype=np.float32)/255.)
    tensor=lambda x:torch.as_tensor(x,device=device,dtype=torch.float32).unsqueeze(0)
    time=torch.tensor([int(d['timestamp'])],device=device,dtype=torch.int64)
    inputs=ModelInputs(tensor(np.array(images)).permute(0,1,4,2,3).contiguous(),tensor(d['intrinsics']),tensor(d['extrinsics']),tensor(d['image_transforms']),tuple(d['camera_names'].tolist()),time[:,None].expand(-1,len(images)),time,distortion=tensor(d['distortion']),scene_tokens=(scene_token,))
    return inputs,digests


def main():
    p=argparse.ArgumentParser()
    for name in ['observations','sensor-root','weights','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--token',help='single current request; omit to extract directory')
    a=p.parse_args();provider=load_provider(a.weights,a.device);identity=state_fingerprint(provider)
    paths=[Path(a.observations)/(a.token+'.npz')] if a.token else sorted(Path(a.observations).glob('*.npz'))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for path in paths:
        inputs,digests=current_observation(path,a.sensor_root,a.device)
        with torch.no_grad():f,xyz,support,meta=provider(inputs)
        meta=dict(meta,scene_token=path.stem,decision_time=int(inputs.decision_time[0]),provider_weights_sha256=identity,
                  provider_source_sha256=hashlib.sha256(Path(__import__(provider.__module__,fromlist=['__file__']).__file__).read_bytes()).hexdigest(),
                  input_tensor_sha256=hashlib.sha256(inputs.current_images.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),image_sha256=digests,image_transforms=inputs.image_transforms.cpu().tolist(),dtype=str(f.dtype))
        torch.save({'metadata':meta,'features':f.cpu(),'coordinates':xyz.cpu(),'observation_support':support.cpu()},out/(path.stem+'.pt'))
    print(json.dumps({'completed':len(paths),'provider_weights_sha256':identity}))

if __name__=='__main__':main()
