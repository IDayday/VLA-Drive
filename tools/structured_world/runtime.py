"""Shared real-data setup used by training, evaluation, and cache generation."""
import copy
from dataclasses import fields
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from starVLA.model.modules.structured_world.contracts import ModelInputs,WorldTargets


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def load_baseline(checkpoint,vlm,device='cuda'):
    os.environ['BASE_VLM']=str(vlm)
    os.environ['VLM_ATTN_IMPLEMENTATION']='sdpa'
    os.environ['NAVSIM_USE_FEATURE_CACHE']='0'
    os.environ.pop('NAVSIM_FEATURE_CACHE_ROOT',None)
    from infer import VLAAgent
    agent=VLAAgent(checkpoint,device=device,qwen_forward_mode='legacy')
    state=torch.load(Path(checkpoint)/'pytorch_model.pt',map_location='cpu',mmap=True,weights_only=True)
    actual=agent.model.state_dict()
    missing=set(actual)-set(state);unexpected=set(state)-set(actual)
    if missing or any(not k.startswith('rgb_model.') for k in unexpected):
        raise ValueError(f'Baseline keys mismatch: {sorted(missing)}, {sorted(unexpected)}')
    return agent


def load_dataset(agent,manifest,data_root,limit=None,split="train"):
    from infer import NavSimDataset
    cfg=copy.deepcopy(agent.model_config)
    cfg.datasets.video_data.load_2d_data=0;cfg.datasets.gs_data.load_3d_data=0;cfg.w_depth=0;cfg.enable_image_aug=0
    return NavSimDataset(manifest,split=split,video_data_cfg=cfg.datasets.video_data,gs_data_cfg=cfg.datasets.gs_data,
                         reward_data_cfg=cfg.datasets.reward_data,ver_1225=cfg.ver_1225,dataset_cfg=cfg.datasets.vla_data,
                         all_cfg=cfg,data_root=data_root,max_samples=limit)


def load_world_batch(examples,cache_root,device='cuda',load_targets=True):
    root=Path(cache_root)
    observations=[np.load(root/'observations'/f'{e["token"]}.npz',allow_pickle=False) for e in examples]
    expected={'intrinsics','extrinsics','image_transforms','distortion','timestamp','camera_names','image_paths','schema_version'}
    for obs in observations:
        if set(obs.files)!=expected or int(obs['schema_version'])!=1:
            raise ValueError('Observation cache schema mismatch')
    tensor=lambda name:torch.as_tensor(np.stack([o[name] for o in observations]),device=device,dtype=torch.float32)
    images=torch.as_tensor(np.stack([[np.asarray(im,dtype=np.float32)/255. for im in e['image']] for e in examples]),device=device).permute(0,1,4,2,3).contiguous()
    times=torch.tensor([int(o['timestamp']) for o in observations],device=device,dtype=torch.int64)
    inputs=ModelInputs(images,tensor('intrinsics'),tensor('extrinsics'),tensor('image_transforms'),tuple(observations[0]['camera_names'].tolist()),times[:,None].expand(-1,3),times,distortion=tensor('distortion'),scene_tokens=tuple(e['token'] for e in examples))
    targets=[]
    if load_targets:
        for e in examples:
            values=torch.load(root/'targets'/f'{e["token"]}.pt',map_location=device,weights_only=True)
            if set(values) != {f.name for f in fields(WorldTargets)}:
                raise ValueError('Target schema mismatch')
            targets.append(WorldTargets(**values))
    return inputs,targets
