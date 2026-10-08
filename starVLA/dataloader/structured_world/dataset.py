"""Verified structured cache plus unchanged current/ego/C1 DINO populations."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from starVLA.dataloader.full_foresight_dataset import FullForesightDataset


class StructuredNAVSIMDataset(Dataset):
    def __init__(self, cache_root, current_root, *, dino_root, dino_index, expected_dino, image_root=None,
                 allow_debug=False):
        self.root = Path(cache_root)
        self.identity = json.loads((self.root/'identity.json').read_text())
        complete = json.loads((self.root/'COMPLETE.json').read_text())
        if complete['identity'] != self.identity['identity'] or complete['scenes'] != complete['expected_scenes']:
            raise ValueError('Incomplete or mismatched structured cache')
        if not allow_debug and self.identity['population_kind'] != 'full_train_population':
            raise ValueError('A fixed debug subset is not a formal experiment population')
        self.index = json.loads((self.root/'index.json').read_text())
        self.base = FullForesightDataset(current_root, candidate='C1', current=True, future=False,
            dino_root=dino_root, dino_index=dino_index, expected_dino=expected_dino,
            image_root=image_root, allow_partial=False)
        if self.base.identity != self.identity['source_current_identity']:
            raise ValueError('Historical GT/current source population changed')
        lookup = {row['token']: i for i, row in enumerate(self.base.index)}
        self.global_indices = [lookup[row['token']] for row in self.index]
        self._checked = set()

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        token = self.index[index]['token']
        metadata = json.loads((self.root/'records'/(token+'.json')).read_text())
        path = self.root/'labels'/(token+'.npz')
        if metadata['cache_identity'] != self.identity['identity']:
            raise ValueError('Foreign scene labels')
        if token not in self._checked:
            if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['label_sha256']:
                raise ValueError('Structured label hash mismatch')
            self._checked.add(token)
        observation, targets = self.base[self.global_indices[index]]
        with np.load(path, allow_pickle=False) as arrays:
            rgb = torch.from_numpy(arrays['geometry_rgb'].copy()).float()/255.
            observation['geometry_images'] = (rgb-rgb.new_tensor([.485, .456, .406])[None, :, None, None])/rgb.new_tensor([.229, .224, .225])[None, :, None, None]
            observation['geometry_pixel_valid'] = torch.from_numpy(arrays['geometry_pixel_valid'].copy())
            observation['calibration'] = {k: torch.from_numpy(arrays['calibration_'+k].copy())
                                          for k in ('sensor2ego', 'intrinsics', 'post_rots', 'post_trans')}
            for key in ('road_distance', 'road_valid', 'occupancy', 'occupancy_valid', 'depth', 'future_valid'):
                targets[key] = torch.from_numpy(arrays[key].copy())
            # Native NAVSIM GT stays authoritative, with a numerical adapter check.
            from starVLA.dataloader.foresight_dataset import decode_ego
            physical = decode_ego(targets['ego'])
            if not np.allclose(physical.numpy(), arrays['ego_physical'], atol=2e-4, rtol=1e-5):
                raise ValueError('NAVSIM adapter differs from original GT ego coordinates')
        if not targets['future_valid'].all():
            raise ValueError('Original NAVSIM formal population unexpectedly lacks a complete horizon')
        targets['ego_bodies'] = metadata['ego_body']
        return observation, targets


def collate_structured(rows):
    observations, targets = zip(*rows)
    return list(observations), {key: torch.stack([value[key] for value in targets])
                               if torch.is_tensor(targets[0][key]) else [value[key] for value in targets]
                               for key in targets[0]}
