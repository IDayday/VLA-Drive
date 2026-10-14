"""Offline-only frozen generic DINOv3. Not imported by the student/deployment."""
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image

REPOSITORY = 'timm/vit_large_patch16_dinov3.lvd1689m'
REVISION = '30c1109559f65dea34316b0d4842d35c5771fe11'
WEIGHT_SHA256 = '45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941'
CONFIG_SHA256 = 'a71f705b0074e173540d0bdbd3aa940fa8d7d3c6c7f020a683004c46ca605b24'


def file_hash(path):
    sha = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            sha.update(block)
    return sha.hexdigest()


def preprocess_image(path, patch_size=(16,16), short_side=256):
    """Full RGB field, aspect-preserving resize, normalized-zero right/bottom pad.

    Fully contained patches only contribute to the loss. Partial border patches
    are excluded (their content still participates in teacher self-attention).
    No square center crop; different DDP16:9 fields must be explicitly adapted.
    """
    if short_side < 128 or len(patch_size) != 2 or min(patch_size) < 1:
        raise ValueError('Invalid registered image/patch dimensions')
    with Image.open(path) as source:
        image = source.convert('RGB'); width, height = image.size
        if abs(width / height - 16 / 9) > .001:
            raise ValueError('DINO/Qwen field mismatch: register common image range before non16:9 input')
        scale = short_side / min(width, height)
        w, h = int(round(width * scale)), int(round(height * scale))
        image = image.resize((w, h), Image.Resampling.BICUBIC)
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float() / 255
    mean = pixels.new_tensor([.485,.456,.406])[:,None,None]
    std = pixels.new_tensor([.229,.224,.225])[:,None,None]
    ph, pw = patch_size; pad_h, pad_w = (-h) % ph, (-w) % pw
    pixels = torch.nn.functional.pad((pixels - mean) / std, (0,pad_w,0,pad_h), value=0)
    gh, gw = (h+pad_h)//ph, (w+pad_w)//pw
    valid = ((torch.arange(gh)[:,None]+1)*ph <= h) & ((torch.arange(gw)[None,:]+1)*pw <= w)
    return pixels, valid, {'original_wh':[width,height], 'resized_wh':[w,h], 'padding_ltrb':[0,0,pad_w,pad_h],
                           'grid_hw':[gh,gw], 'crop_xyxy':[0,0,width,height]}


class DINOv3TargetEncoder(nn.Module):
    def __init__(self, root, device='cpu'):
        super().__init__()
        # Lazy imports keep the pure-current student independent of timm.
        import timm
        from safetensors.torch import load_file
        root = Path(root)
        identity = json.loads((root/'IDENTITY.json').read_text())
        if identity['repository'] != REPOSITORY or identity['revision'] != REVISION:
            raise ValueError('Unregistered DINO source')
        for name, expected in [('model.safetensors',WEIGHT_SHA256),('config.json',CONFIG_SHA256)]:
            if file_hash(root/name) != expected:
                raise ValueError('DINO artifact hash mismatch: '+name)
        with torch.random.fork_rng(devices=[]):
            self.model = timm.create_model('vit_large_patch16_dinov3', pretrained=False, num_classes=0)
        self.model.load_state_dict(load_file(str(root/'model.safetensors')), strict=True)
        self.model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        self.patch_size = tuple(self.model.patch_embed.patch_size)
        self.feature_dim = self.model.num_features
        self.prefix = self.model.num_prefix_tokens
        self.recipe = {'schema':'dinov3_dense_target_recipe_v1', 'repository':REPOSITORY, 'revision':REVISION,
            'weight_sha256':WEIGHT_SHA256, 'config_sha256':CONFIG_SHA256, 'timm':timm.__version__, 'torch':torch.__version__,
            'feature':'last_block_post_norm_dense_patch', 'prefix_tokens_excluded':self.prefix,
            'feature_dim':self.feature_dim, 'patch_size':list(self.patch_size), 'extra_norm':'none', 'pooling':'none',
            'source_rgb_range':'original full16:9 field, same as DDP', 'short_side':256, 'interpolation':'PIL bicubic antialias',
            'padding':'right/bottom normalized zero; fully contained patches only', 'mean':[.485,.456,.406], 'std':[.229,.224,.225],
            'precision':'FP32 model/input/output; TF32 disabled; FP16 storage', 'rope':'timm default FP32 periods',
            'forward_source_sha256':hashlib.sha256(inspect.getsource(type(self.model).forward_features).encode()).hexdigest(),
            'implementation_sha256':file_hash(__file__), 'original_user_recipe':'UNVERIFIED; explicit fallback'}
        self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(self, pixels):
        if pixels.ndim != 4 or pixels.shape[1] != 3 or not torch.isfinite(pixels).all():
            raise ValueError('Invalid teacher RGB input')
        ph, pw = self.patch_size; h, w = pixels.shape[-2:]
        if h % ph or w % pw:
            raise ValueError('Teacher input not patch-aligned')
        with torch.autocast(pixels.device.type, enabled=False):
            full = self.model.forward_features(pixels.float())
        patches = full[:, self.prefix:]
        if patches.shape[1:] != (h//ph*(w//pw), self.feature_dim) or not torch.isfinite(patches).all():
            raise ValueError('DINO prefix/grid/feature contract')
        return patches.reshape(len(pixels),h//ph,w//pw,self.feature_dim).permute(0,3,1,2).contiguous()
