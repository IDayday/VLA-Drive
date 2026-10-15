"""Frozen SFT projection cache, distinct from the old trainable-qwen_proj cache."""
import fcntl
import json
import os
from pathlib import Path
import torch
from starVLA.rl.constrained_flow_noise.condition_cache import ConditionStore
from starVLA.rl.flow_grpo.transactions import atomic_json
from .contracts import RAW_CONDITION, CONDITION_SHA, SFT_SHA256, fingerprint, file_sha


class ProjectedConditions:
    def __init__(self, root, head=None, *, readonly=False):
        self.root=Path(root); self.head=head; self.readonly=readonly
        self.raw=ConditionStore(RAW_CONDITION)
        if self.raw.identity_sha!=CONDITION_SHA: raise ValueError('Original deployment cache changed')
        self.identity=dict(schema='simwam_frozen_SFT_projection_v1',SFT_sha256=SFT_SHA256,
            raw_identity=CONDITION_SHA,boundary='after_frozen_SFT_action_model.qwen_proj',
            qwen_proj_frozen=True,projection_storage_dtype='float32',projection_compute_dtype='float32',
            autocast=False,scene_batch=1,shape=[1,8,1536],no_actor_intermediates=True)
        self.sha=fingerprint(self.identity)
        self.root.mkdir(parents=True,exist_ok=True)
        with (self.root/'identity.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            path=self.root/'identity.json'
            if path.exists() and json.loads(path.read_text())!=self.identity: raise ValueError('Projection identity mismatch')
            if not path.exists():
                if readonly: raise FileNotFoundError(path)
                atomic_json(path,self.identity)

    def get(self, token, device='cuda'):
        path=self.root/token[:2]/(token+'.pt'); seal=path.with_suffix('.json')
        if seal.exists():
            s=json.loads(seal.read_text())
            if s['identity_sha256']!=self.sha or s['sha256']!=file_sha(path): raise ValueError('Projection cache changed')
            row=torch.load(path,map_location='cpu',weights_only=True)
            if set(row)!={'token','condition'} or row['token']!=token: raise ValueError('Unexpected cached fields')
            value=row['condition']
        else:
            if self.readonly or self.head is None: raise FileNotFoundError(path)
            if any(p.requires_grad for p in self.head.qwen_proj.parameters()): raise ValueError('Projection must be frozen')
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                value=self.head._project_condition(self.raw.get(token).to(device=device,dtype=torch.float32)).cpu()
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if seal.exists(): return self.get(token,device)
                temporary=path.with_suffix('.incomplete-'+str(os.getpid()))
                torch.save(dict(token=token,condition=value),temporary);os.replace(temporary,path)
                atomic_json(seal,dict(token=token,identity_sha256=self.sha,sha256=file_sha(path)))
        if value.shape!=(1,8,1536) or value.dtype!=torch.float32 or value.requires_grad or not torch.isfinite(value).all():
            raise ValueError('Invalid frozen projection')
        return value.to(device)
