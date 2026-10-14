"""Verified pretrained visual backbone + NEW calibrated BEV construction.

Not a pretrained BEV model. Geometry is the same current-camera multi-height
inverse projection as V1; learned CNN/fusion is replaced by frozen DAV2 patches.
"""
import hashlib
from pathlib import Path
import torch
from torch import nn
from .providers import GeometricBEVProvider


class DAV2Patches(nn.Module):
    def __init__(self,weights,expected_sha256):
        super().__init__()
        from starVLA.model.modules.depth_model.models.depth_anything_v2.dpt import DepthAnythingV2
        actual=hashlib.sha256(Path(weights).read_bytes()).hexdigest()
        if actual!=expected_sha256:raise ValueError('Pretrained weight identity mismatch')
        self.backbone=DepthAnythingV2(encoder='vitl')
        state=torch.load(weights,map_location='cpu',weights_only=True)
        unexpected=[k for k in state if not k.startswith(('pretrained.','depth_head.'))]
        if unexpected:raise ValueError('Unrecognised pretrained checkpoint keys: '+repr(unexpected))
        self.excluded_depth_head_keys=sorted(k for k in state if k.startswith('depth_head.'))
        self.backbone.load_state_dict({k:v for k,v in state.items() if k.startswith('pretrained.')},strict=True)
        self.requires_grad_(False);self.eval();self.weight_sha256=actual

    def train(self,mode=True):
        super().train(False);return self

    def forward(self,image):
        h,w=image.shape[-2:]
        with torch.no_grad():patches=self.backbone.forward_semantics(image)
        return patches.transpose(1,2).reshape(len(image),1024,h//16,w//16)


class PretrainedVisualBEVProvider(GeometricBEVProvider):
    def __init__(self,weights,expected_sha256):
        super().__init__(channels=1024)
        # Delete V1 random feature/fusion modules. The new BEV itself has no
        # randomly initialized CNN; adaptation is measured in a separate Reader.
        self.encoder=DAV2Patches(weights,expected_sha256)
        self.fuse=nn.Identity()
        self.requires_grad_(False);self.eval()

    def encoder_output_shape(self,h,w):return h//16,w//16

    def metadata(self):
        meta=super().metadata()
        meta.update(provider='DAV2_ViTL_visual_plus_new_calibrated_BEV_v1',
                    pretrained_bev=False,backbone_weights_sha256=self.encoder.weight_sha256,
                    backbone_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    backbone_internal_resize='(H//16*14,W//16*14), bicubic align_corners=False',
                    backbone_normalization={'mean':[.485,.456,.406],'std':[.229,.224,.225]},
                    learned_bev_fusion=False)
        return meta
