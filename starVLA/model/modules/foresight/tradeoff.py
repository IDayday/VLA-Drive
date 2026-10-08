"""Registered current-only resolution/query experiment; no future task inputs."""
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import nn
from PIL import Image


@dataclass(frozen=True)
class Candidate:
    name: str
    width: int
    height: int
    pool: int

    @property
    def native_hw(self): return self.height // 16, self.width // 16

    @property
    def grid_hw(self): return tuple(x // self.pool for x in self.native_hw)

    @property
    def tokens_per_view(self): return self.grid_hw[0] * self.grid_hw[1]

    @property
    def num_queries(self): return 3 * self.tokens_per_view

    def record(self):
        return {**asdict(self), 'native_hw':list(self.native_hw), 'grid_hw':list(self.grid_hw),
                'tokens_per_view':self.tokens_per_view, 'num_queries':self.num_queries}


CANDIDATES = {c.name:c for c in [Candidate('C0',128,96,1), Candidate('C1',256,192,2),
    Candidate('C2',256,192,1), Candidate('C3',384,288,2), Candidate('C4',512,384,4), Candidate('C5',512,384,2)]}


def preprocess_current(path, width, height):
    """Direct original RGB -> full-field 4:3 anisotropic resize; no crop/pad.

    This explicit NEW baseline does not claim to reproduce the historical89.41.
    All resolutions read the original file, never another resized target.
    """
    if (width,height) not in {(c.width,c.height) for c in CANDIDATES.values()}:
        raise ValueError('Unregistered teacher resolution')
    with Image.open(path) as source:
        image=source.convert('RGB');original=image.size
        if abs(original[0]/original[1]-16/9)>.001:
            raise ValueError('Source field differs from fixed Qwen full16:9 field')
        image=image.resize((width,height),Image.Resampling.BICUBIC)
        pixels=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255
    pixels=(pixels-pixels.new_tensor([.485,.456,.406])[:,None,None])/pixels.new_tensor([.229,.224,.225])[:,None,None]
    return pixels, {'original_wh':list(original), 'actual_tensor_chw':list(pixels.shape),
                    'crop_xyxy':[0,0,*original], 'padding_ltrb':[0,0,0,0],
                    'source_range':'full original RGB', 'resize':'PIL bicubic to4:3; anisotropic, no crop'}


def pool_patches(features, candidate):
    """B(or independent images),C,H,W. Never combine cameras or CLS/register."""
    if features.ndim!=4 or tuple(features.shape[-2:])!=candidate.native_hw or not torch.isfinite(features).all():
        raise ValueError('Native patch grid/value mismatch')
    return torch.nn.functional.avg_pool2d(features.float(),candidate.pool,candidate.pool)


class TokenProjectionHead(nn.Module):
    """Same affine projection for every view/position and every candidate."""
    def __init__(self, hidden, channels):
        super().__init__();self.projection=nn.Linear(hidden,channels)

    def forward(self, queries, horizon_s, grid_hw):
        h,w=grid_hw
        if queries.ndim!=3 or queries.shape[1]!=3*h*w or torch.any(horizon_s!=0):
            raise ValueError('Current-only view-major query/grid contract')
        return self.projection(queries).reshape(len(queries),3,h,w,-1).permute(0,1,4,2,3)
