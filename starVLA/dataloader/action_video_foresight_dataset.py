"""Pure-current inputs; native clip targets with strict schema and common population."""
import json
from pathlib import Path
from safetensors import safe_open
from .full_foresight_dataset import FullForesightDataset
from starVLA.model.modules.vehicle_joint.initialization import identity_hash


class ActionVideoForesightDataset(FullForesightDataset):
    def __init__(self, *args, clip_root, expected_clip, future_type, future, **kwargs):
        super().__init__(*args, future=False, **kwargs)
        self.clip_root = Path(clip_root) if clip_root else None
        self.clip_identity = None
        if not future:
            if clip_root:
                raise ValueError('Unexpected disabled future cache')
            return
        if self.clip_root is None:
            raise ValueError('Complete configured clip cache required')
        identity = json.loads((self.clip_root/'identity.json').read_text())
        if identity['schema'] != 'action_video_clip_targets_v1' or identity['future_target_type'] != future_type or identity['identity'] != expected_clip or identity_hash({k:v for k,v in identity.items() if k!='identity'}) != expected_clip:
            raise ValueError('Foreign video/DINO sequence target cache')
        if identity['scene_index_hash'] != self.identity['index_sha256'] or identity['split'] != self.identity['split'] or identity['partition_sha256'] != self.identity['partition_sha256']:
            raise ValueError('Clip targets may not change the ego scene/log population')
        if not kwargs.get('allow_partial', False):
            done = json.loads((self.clip_root/'COMPLETE.json').read_text())
            if done['identity'] != expected_clip or done['scenes'] != len(self.index):
                raise ValueError('Incomplete formal clip cache')
        self.clip_identity = identity

    def __getitem__(self, i):
        observation, targets = super().__getitem__(i)
        if self.clip_identity is None:
            return observation, targets
        chunk, at = divmod(i, self.clip_identity['chunk_size'])
        path = self.clip_root/f'chunk_{chunk:06d}.safetensors'
        with safe_open(str(path), framework='pt', device='cpu') as f:
            if f.metadata()['identity'] != self.clip_identity['identity']:
                raise ValueError('Foreign clip target shard')
            values, mask = f.get_slice('features')[at], f.get_slice('valid')[at]
        if list(values.shape) != self.clip_identity['target_shape'] or mask.shape != values.shape[:-1]:
            raise ValueError('Native clip shape mismatch')
        targets.update(future_clip=values, future_clip_valid=mask)
        return observation, targets
