"""Verified structured cache plus unchanged current/ego/C1 DINO populations."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from starVLA.dataloader.full_foresight_dataset import FullForesightDataset
from starVLA.dataloader.foresight_dataset import ForesightTrainingDataset


def validate_native_ego_contract(encoded, auxiliary_physical, auxiliary_time_valid):
    """Native complete ego labels survive gaps in physical-time scene labels."""
    from starVLA.dataloader.foresight_dataset import decode_ego
    if encoded.shape != (8, 4) or not torch.isfinite(encoded).all():
        raise ValueError('Native NAVSIM eight-point ego population changed')
    valid = np.asarray(auxiliary_time_valid, dtype=bool)
    if valid.shape != (8,) or np.asarray(auxiliary_physical).shape != (8, 3):
        raise ValueError('Auxiliary ego/time contract changed')
    physical = decode_ego(encoded).numpy()
    if not np.allclose(physical[valid], np.asarray(auxiliary_physical)[valid], atol=2e-4, rtol=1e-5):
        raise ValueError('NAVSIM adapter differs from original native GT coordinates')
    return torch.ones(8, dtype=torch.bool)


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
        self.dino_root = Path(dino_root)
        self.dino_identity = json.loads((self.dino_root/'identity.json').read_text())
        self.current_teacher_only = self.dino_identity['schema'] == 'structured_world_current_C1_v1'
        if self.current_teacher_only:
            done = json.loads((self.dino_root/'COMPLETE.json').read_text())
            if self.dino_identity['identity'] != expected_dino or done['identity'] != expected_dino:
                raise ValueError('Wrong current-only C1 teacher')
            if self.dino_identity['structured_cache_identity'] != self.identity['identity'] or done['scenes'] != len(self.index):
                raise ValueError('Current-only teacher population changed')
            if self.dino_identity['cameras'] != 3 or self.dino_identity['grid_hw'] != [6, 8] or self.dino_identity['feature_dim'] != 1024:
                raise ValueError('Current C1 teacher layout changed')
            self.base = ForesightTrainingDataset(current_root)
            self.base.dino_identity = self.dino_identity
            self.base.local_image_root = image_root
            if image_root:
                stage = json.loads((Path(current_root)/'local_stage.json').read_text())
                if stage['source_identity'] != self.base.identity['identity'] or Path(stage['image_root']) != Path(image_root) or stage['scenes'] != len(self.base.index):
                    raise ValueError('Incomplete or foreign current image residency')
        else:
            if not allow_debug: raise ValueError('Formal runs require a dedicated current-only C1 cache, without future teacher dependencies')
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
        if self.current_teacher_only:
            teacher_file = self.dino_root/'targets'/(token+'.npz')
            teacher_record = json.loads((self.dino_root/'records'/(token+'.json')).read_text())
            if teacher_record['identity'] != self.dino_identity['identity'] or hashlib.sha256(teacher_file.read_bytes()).hexdigest() != teacher_record['sha256']:
                raise ValueError('Current-only teacher target changed')
            with np.load(teacher_file, allow_pickle=False) as teacher:
                targets['current_dino'] = torch.from_numpy(teacher['current_dino'].copy()).float()
                targets['current_dino_valid'] = torch.from_numpy(teacher['valid'].copy())
        with np.load(path, allow_pickle=False) as arrays:
            rgb = torch.from_numpy(arrays['geometry_rgb'].copy()).float()/255.
            observation['geometry_images'] = (rgb-rgb.new_tensor([.485, .456, .406])[None, :, None, None])/rgb.new_tensor([.229, .224, .225])[None, :, None, None]
            observation['geometry_pixel_valid'] = torch.from_numpy(arrays['geometry_pixel_valid'].copy())
            observation['calibration'] = {k: torch.from_numpy(arrays['calibration_'+k].copy())
                                          for k in ('sensor2ego', 'intrinsics', 'post_rots', 'post_trans')}
            for key in ('road_distance', 'road_valid', 'occupancy', 'occupancy_valid', 'depth'):
                targets[key] = torch.from_numpy(arrays[key].copy())
            # Source logs can skip a sensor keyframe. They retain native eight
            # planning annotations in the canonical protocol, but a mismatched
            # future scene label is unknown at its nominal physical time.
            targets['auxiliary_future_time_valid'] = torch.from_numpy(arrays['future_valid'].copy())
            targets['future_valid'] = validate_native_ego_contract(targets['ego'], arrays['ego_physical'], arrays['future_valid'])
        targets['ego_bodies'] = metadata['ego_body']
        return observation, targets


def collate_structured(rows):
    observations, targets = zip(*rows)
    return list(observations), {key: torch.stack([value[key] for value in targets])
                               if torch.is_tensor(targets[0][key]) else [value[key] for value in targets]
                               for key in targets[0]}
