"""Deployment reader has no dependency on maps, boxes, LiDAR or GT labels."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from .cameras import crop_16_9


class CurrentInputs(Dataset):
    def __init__(self, root):
        self.root = Path(root)
        self.identity = json.loads((self.root/'identity.json').read_text())
        self.index = json.loads((self.root/'index.json').read_text())
        done = json.loads((self.root/'COMPLETE.json').read_text())
        if done['identity'] != self.identity['identity'] or done['scenes'] != len(self.index):
            raise ValueError('Incomplete current-only inputs')
        self.checked = set()

    def __len__(self): return len(self.index)

    def __getitem__(self, index):
        token = self.index[index]['token']
        record = json.loads((self.root/'current'/(token+'.json')).read_text())
        if record['identity'] != self.identity['identity']: raise ValueError('Foreign current record')
        allowed_record = {'identity', 'token', 'image_paths', 'image_sha256', 'state', 'lang', 'pixels_sha256'}
        if set(record)-allowed_record: raise ValueError('Non-current data entered deployment record')
        path = self.root/'pixels'/(token+'.npz')
        if token not in self.checked:
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['pixels_sha256']:
                raise ValueError('Current geometry pixels/calibration changed')
            if self.identity.get('Qwen_image_hashes_verified') and 'image_sha256' not in record:
                raise ValueError('Unbound Qwen current image')
            if 'image_sha256' in record:
                if len(record['image_sha256']) != self.identity['cameras']: raise ValueError('Incomplete image hash list')
                for name, expected in zip(record['image_paths'], record['image_sha256']):
                    if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
                        raise ValueError('Qwen current image changed')
            self.checked.add(token)
        images = []
        for name in record['image_paths']:
            with Image.open(name) as source:
                image = source.convert('RGB')
                images.append(image.crop(crop_16_9(*image.size)).resize((1024, 576), Image.Resampling.LANCZOS))
        if len(images) != self.identity['cameras']: raise ValueError('Current camera population changed')
        with np.load(path, allow_pickle=False) as arrays:
            allowed = {'geometry_rgb', 'geometry_pixel_valid'} | {'calibration_'+key for key in
                       ('sensor2ego', 'intrinsics', 'post_rots', 'post_trans')}
            if set(arrays.files) != allowed: raise ValueError('Non-input data in deployment pixels file')
            rgb = torch.from_numpy(arrays['geometry_rgb'].copy()).float()/255.
            geometry = (rgb-rgb.new_tensor([.485, .456, .406])[None, :, None, None])/rgb.new_tensor([.229, .224, .225])[None, :, None, None]
            calibration = {key: torch.from_numpy(arrays['calibration_'+key].copy()) for key in
                           ('sensor2ego', 'intrinsics', 'post_rots', 'post_trans')}
            valid = torch.from_numpy(arrays['geometry_pixel_valid'].copy())
        return {'token': token, 'image': images, 'state': np.asarray(record['state'], dtype=np.float32),
                'lang': record['lang'], 'geometry_images': geometry, 'calibration': calibration,
                'geometry_pixel_valid': valid}
