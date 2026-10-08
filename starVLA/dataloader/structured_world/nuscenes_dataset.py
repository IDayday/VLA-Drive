"""Six current cameras, legal past state, common C1 and separate GT labels."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from starVLA.dataloader.foresight_dataset import encode_ego
from .cameras import crop_16_9, NUSCENES_CAMERAS


def current_observation(record, *, image_root=None):
    if record['camera_order'] != list(NUSCENES_CAMERAS) or len(record['image_paths']) != 6:
        raise ValueError('Locked nuScenes six-camera order changed')
    images = []
    for name in record['image_paths']:
        path = Path(name) if image_root is None else Path(image_root)/Path(name).relative_to(record['input_root'])
        with Image.open(path) as source:
            image = source.convert('RGB')
            images.append(image.crop(crop_16_9(*image.size)).resize((1024, 576), Image.Resampling.LANCZOS))
    state = np.asarray(record['state'], dtype=np.float32)
    if state.shape != (1, 4) or not np.isfinite(state).all():
        raise ValueError('Illegal current state')
    return {'image': images, 'state': state, 'lang': record['lang'], 'token': record['token']}


class StructuredNuScenesDataset(Dataset):
    def __init__(self, cache_root, *, dino_root, expected_dino, allow_debug=False, image_root=None):
        self.root = Path(cache_root); self.dino_root = Path(dino_root); self.image_root = image_root
        self.identity = json.loads((self.root/'identity.json').read_text())
        complete = json.loads((self.root/'COMPLETE.json').read_text())
        if complete['identity'] != self.identity['identity'] or complete['scenes'] != complete['expected_scenes']:
            raise ValueError('Incomplete structured cache')
        if self.identity['dataset'] != 'nuscenes': raise ValueError('Wrong dataset')
        if not allow_debug and self.identity['population_kind'] != 'full_train_population':
            raise ValueError('Debug population cannot initialize a formal experiment')
        self.index = json.loads((self.root/'index.json').read_text())
        self.dino_identity = json.loads((self.dino_root/'identity.json').read_text())
        dino_complete = json.loads((self.dino_root/'COMPLETE.json').read_text())
        if self.dino_identity['identity'] != expected_dino or dino_complete['identity'] != expected_dino:
            raise ValueError('Wrong current DINO identity')
        if self.dino_identity['structured_cache_identity'] != self.identity['identity'] or dino_complete['scenes'] != len(self.index):
            raise ValueError('Current DINO changed the training population')
        self.checked = set()

    def __len__(self): return len(self.index)

    def __getitem__(self, index):
        token = self.index[index]['token']
        metadata = json.loads((self.root/'records'/(token+'.json')).read_text())
        path = self.root/'labels'/(token+'.npz')
        dino_meta = json.loads((self.dino_root/'records'/(token+'.json')).read_text())
        dino_path = self.dino_root/'targets'/(token+'.npz')
        if metadata['cache_identity'] != self.identity['identity'] or dino_meta['identity'] != self.dino_identity['identity']:
            raise ValueError('Foreign scene record')
        if token not in self.checked:
            for p, expected in ((path, metadata['label_sha256']), (dino_path, dino_meta['sha256'])):
                if hashlib.sha256(p.read_bytes()).hexdigest() != expected: raise ValueError('Scene hash mismatch')
            self.checked.add(token)
        current = {**metadata['current_record'], 'input_root': self.identity['input_root']}
        observation = current_observation(current, image_root=self.image_root)
        with np.load(path, allow_pickle=False) as arrays:
            rgb = torch.from_numpy(arrays['geometry_rgb'].copy()).float()/255.
            observation['geometry_images'] = (rgb-rgb.new_tensor([.485, .456, .406])[None, :, None, None])/rgb.new_tensor([.229, .224, .225])[None, :, None, None]
            observation['geometry_pixel_valid'] = torch.from_numpy(arrays['geometry_pixel_valid'].copy())
            observation['calibration'] = {k: torch.from_numpy(arrays['calibration_'+k].copy())
                                          for k in ('sensor2ego', 'intrinsics', 'post_rots', 'post_trans')}
            targets = {key: torch.from_numpy(arrays[key].copy()) for key in
                       ('road_distance', 'road_valid', 'occupancy', 'occupancy_valid', 'depth', 'future_valid')}
            targets['ego'] = torch.from_numpy(encode_ego(arrays['ego_physical']))
        with np.load(dino_path, allow_pickle=False) as arrays:
            targets['current_dino'] = torch.from_numpy(arrays['current_dino'].copy()).float()
            targets['current_dino_valid'] = torch.from_numpy(arrays['valid'].copy())
        if targets['ego'].shape != (6, 4) or targets['current_dino'].shape != (6, 1024, 6, 8) or not targets['future_valid'].all():
            raise ValueError('nuScenes 6-step / 6-camera target contract changed')
        targets['ego_bodies'] = metadata['ego_body']
        return observation, targets
