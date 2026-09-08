"""Input-only V2 cache and fixed-current-layout RGB preprocessing."""
import hashlib
import json
from pathlib import Path
import time
import torch
from torch.utils.data import Dataset
from PIL import Image
from . import CACHE_SCHEMA
from .backbone import V2_SYSTEM_PROMPT
from ..utils.internvl_preprocess import tile_metadata_from_image_size, build_transform
from ..utils.internvl_tokenize import build_internvl_model_inputs
from ..utils.utils import build_drivevla_questions
from ..layers.world_model.future_image_io import decode_path_tensor

FORBIDDEN = {'last_hidden_state','patch_features','semantic_tokens','planning_registers',
             'future_registers','ema_registers','visual_content','semantic_queries'}


def reject_cached_representations(obj):
    if isinstance(obj,dict):
        bad = FORBIDDEN.intersection(obj)
        if bad:
            raise ValueError('Dynamic VLM/EMA feature cache changes the training method: '+str(sorted(bad)))
        for value in obj.values(): reject_cached_representations(value)
    elif isinstance(obj,(list,tuple)):
        for value in obj: reject_cached_representations(value)


def preprocess_fixed_layout(path,metadata=None):
    with (path.copy() if isinstance(path,Image.Image) else Image.open(path)) as source:
        image = source.convert('RGB')
        if metadata is None:
            metadata = tile_metadata_from_image_size(*image.size)
        metadata = torch.as_tensor(metadata).float()
        crop = metadata[metadata[:,4] == 0]
        if len(crop) == 0:
            raise ValueError('Current tile layout needs at least one actual image crop')
        grid_w,grid_h = int(round(1/float(crop[0,2]))),int(round(1/float(crop[0,3])))
        resized = image.resize((grid_w*448,grid_h*448))
        transform = build_transform(448)
        pixels = []
        for cx,cy,w,h,thumbnail in metadata.tolist():
            if thumbnail:
                tile = image.resize((448,448))
            else:
                x,y = round((cx-w/2)*grid_w*448),round((cy-h/2)*grid_h*448)
                tile = resized.crop((x,y,x+448,y+448))
            pixels.append(transform(tile))
    return torch.stack(pixels),metadata


def prepare_current(features,tokenizer):
    feature = dict(features)
    path = decode_path_tensor(feature['image_path_tensor'],feature['image_path_length'])
    pixels,metadata = preprocess_fixed_layout(path,feature.get('tile_metadata'))
    question = build_drivevla_questions(feature['history_trajectory'],feature['high_command_one_hot'])[0]
    inputs = build_internvl_model_inputs(tokenizer,[question],[len(pixels)],V2_SYSTEM_PROMPT)
    valid = inputs['attention_mask'][0].bool()
    feature.update(pixel_values=pixels.bfloat16(),tile_metadata=metadata,
        input_ids=inputs['input_ids'][0][valid],attention_mask=inputs['attention_mask'][0][valid])
    return feature


class InputOnlyV2Dataset(Dataset):
    def __init__(self,manifest,vlm_path,world_model=True):
        self.manifest_path = Path(manifest)
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get('schema') != CACHE_SCHEMA:
            raise ValueError('Stale cache: V2 needs actual offsets [1,3,8] and logged motion')
        if self.manifest.get('split') not in ('train','trainval_final_fit','development','navtest'):
            raise ValueError('Explicit data split provenance required')
        self.records = self.manifest['records']
        tokens = [r['token'] for r in self.records]
        if len(set(tokens)) != len(tokens):
            raise ValueError('Input cache contains duplicate scene tokens')
        self.vlm_path,self.world_model = vlm_path,world_model
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.vlm_path,trust_remote_code=True,use_fast=False)
        return self._tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self,index):
        record = self.records[index]
        cached = torch.load(record['cache_path'],map_location='cpu',weights_only=False)
        if cached['schema'] != CACHE_SCHEMA:
            raise ValueError('Stale per-scene input cache schema')
        reject_cached_representations(cached)
        feature,target = dict(cached['features']),dict(cached['targets'])
        start = time.perf_counter()
        feature = prepare_current(feature,self.tokenizer)
        if self.world_model:
            future = []
            for h in range(3):
                if bool(target['future_valid_mask'][h]):
                    path = decode_path_tensor(target['future_image_paths'][h],target['future_image_path_lengths'][h])
                    pixels,metadata = preprocess_fixed_layout(path,feature['tile_metadata'])
                    if not torch.equal(metadata,feature['tile_metadata']):
                        raise ValueError('Teacher layout differs from current frame')
                    future.append(pixels.bfloat16())
                else:
                    # Placeholder is explicitly invalid, never a repeated-current target.
                    future.append(torch.zeros_like(feature['pixel_values']))
            feature['future_pixel_values'] = future
        feature['input_processing_seconds'] = torch.tensor(time.perf_counter()-start)
        target['metric_cache_path'] = record['metric_cache_path']
        return feature,target


def v2_collate(batch):
    from torch.utils.data._utils.collate import default_collate
    features,targets = zip(*batch)
    lists = {'pixel_values','tile_metadata','future_pixel_values'}
    result = {key:[f[key] for f in features] for key in lists if key in features[0]}
    result['input_ids'] = torch.nn.utils.rnn.pad_sequence([f['input_ids'] for f in features],batch_first=True,padding_value=0)
    result['attention_mask'] = torch.nn.utils.rnn.pad_sequence([f['attention_mask'] for f in features],batch_first=True,padding_value=0)
    excluded = lists | {'input_ids','attention_mask'}
    result.update(default_collate([{k:v for k,v in f.items() if k not in excluded} for f in features]))
    return result,default_collate(targets)
