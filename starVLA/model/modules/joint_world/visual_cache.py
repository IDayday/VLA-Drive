"""Verified frozen-vision cache; never stores final Qwen hidden states or targets."""
import hashlib
import json
import os
from pathlib import Path
import torch


def tensor_digest(value):
    value=value.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(value.shape),str(value.dtype))).encode()+value.view(torch.uint8).numpy().tobytes()).hexdigest()


def visual_identity(base, code_sha):
    return {'schema_version':2,'kind':'public_frozen_current_vision_only',
            'public_repo':base.public_origin['public_repo'],'public_revision':base.public_origin['public_revision'],
            'public_weights_sha256':base.public_origin['public_weights_sha256'],
            'processor_files':base.public_origin['public_source_files'],
            'source_files':{name:hashlib.sha256((Path(__file__).resolve().parents[4]/name).read_bytes()).hexdigest() for name in ('starVLA/model/modules/vlm/QWen3.py','starVLA/model/modules/vlm/qwen3_vl/modeling_qwen3_vl.py','starVLA/model/modules/joint_world/visual_cache.py','tools/local_interaction_mask_v2/data.py')},'cameras':['CAM_F0','CAM_L0','CAM_R0'],
            'time':'decision_current_only','image_size':[1024,576],
            'augmentation':'none','private_driving_weights_loaded':False,
            'dtype':'torch.bfloat16','frozen_upstream':True,'targets_loaded':False}


class FrozenVisualCache:
    def __init__(self,root,identity,write=False,require=False):
        self.root=Path(root);self.identity=identity;self.write=write;self.require=require
        self.signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        self.hits=self.misses=0
        if write:self.root.mkdir(parents=True,exist_ok=True)
        manifest=self.root/'identity.json'
        if manifest.exists():
            if json.loads(manifest.read_text())!=identity:raise ValueError('Visual cache provider/sensor/processor/code version mismatch')
        elif write:
            try:
                with manifest.open('x') as f:json.dump(identity,f,sort_keys=True,indent=2)
            except FileExistsError:
                if json.loads(manifest.read_text())!=identity:raise ValueError('Concurrent visual cache identity mismatch')
        elif require:raise FileNotFoundError(manifest)

    def get(self,example,q,model):
        if any(p.requires_grad for p in model.model.visual.parameters()):raise ValueError('Cannot cache trainable vision')
        if set(example)-{'image','lang','state','token'}:raise ValueError('Current-only visual cache input whitelist violated')
        token=example['token']
        if not token.isalnum():raise ValueError('Unsafe scene token')
        pixels,grid=q['pixel_values'],q['image_grid_thw']
        key={'token':token,'signature':self.signature,'pixels':tensor_digest(pixels),'grid':tensor_digest(grid)}
        path=self.root/(token+'.pt')
        if path.exists():
            data=torch.load(path,map_location='cpu',weights_only=True)
            if set(data)!={'key','parts','deepstack','tensor_hashes'} or data['key']!=key:raise ValueError('Visual cache identity/current observation mismatch')
            tensors=data['parts']+data['deepstack']
            if [tensor_digest(t) for t in tensors]!=data['tensor_hashes']:raise ValueError('Visual cache tensor corruption')
            if any(t.dtype!=torch.bfloat16 or not torch.isfinite(t).all() for t in tensors):raise ValueError('Invalid visual cache tensor')
            self.hits+=1
            return [t.to(pixels.device) for t in data['parts']],[t.to(pixels.device) for t in data['deepstack']]
        if self.require:raise FileNotFoundError(path)
        with torch.no_grad():parts,deepstack=model.model.get_image_features(pixels,grid)
        self.misses+=1
        if self.write:
            data={'key':key,'parts':[t.detach().cpu() for t in parts],'deepstack':[t.detach().cpu() for t in deepstack]}
            data['tensor_hashes']=[tensor_digest(t) for t in data['parts']+data['deepstack']]
            temporary=path.with_suffix('.'+str(os.getpid())+'.tmp');torch.save(data,temporary);temporary.replace(path)
        return parts,deepstack
