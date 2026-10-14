"""Strict official V-JEPA 2.1 encoder only; offline labels, no Hub downloading."""
import importlib
from pathlib import Path
import subprocess
import sys
import torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
import numpy as np
from starVLA.model.modules.vehicle_joint.initialization import file_sha256


class VideoTargetEncoder(nn.Module):
    def __init__(self, source_root, checkpoint, *, source_sha, weight_sha256, pool=2):
        super().__init__()
        source_root = Path(source_root)
        actual_source = subprocess.check_output(['git', '-C', str(source_root), 'rev-parse', 'HEAD'], text=True).strip()
        if actual_source != source_sha or file_sha256(checkpoint) != weight_sha256:
            raise ValueError('Official video teacher identity mismatch')
        if subprocess.check_output(['git', '-C', str(source_root), 'status', '--porcelain']).strip():
            raise ValueError('Official video teacher source must be immutable')
        sys.path.insert(0, str(source_root))
        module = importlib.import_module('app.vjepa_2_1.models.vision_transformer')
        if not Path(module.__file__).resolve().is_relative_to(source_root.resolve()):
            raise ValueError('Foreign src/app package shadowed the pinned official encoder')
        # Exact official ViT-L loader arguments; dynamic 8-frame rectangular input
        # is handled by official RoPE, not by duplicating/interpolating frames.
        self.encoder = module.vit_large(patch_size=16, img_size=(384, 384), num_frames=64,
            tubelet_size=2, use_sdpa=True, use_SiLU=False, wide_SiLU=True,
            uniform_power=False, use_rope=True, img_temporal_dim_size=1, interpolate_rope=True)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if 'ema_encoder' not in saved:
            raise ValueError('Official 2.1 distilled ViT-L requires ema_encoder')
        state = {}
        for key, value in saved['ema_encoder'].items():
            # Prefix-only cleanup. No unexplained discarded or initialized keys.
            for prefix in ('module.', 'backbone.'):
                if key.startswith(prefix):
                    key = key[len(prefix):]
            if key in state:
                raise ValueError('Duplicate official checkpoint key after prefix cleanup')
            state[key] = value
        self.encoder.load_state_dict(state, strict=True)
        self.encoder.requires_grad_(False).eval()
        self.pool = int(pool)
        if self.pool != 2:
            raise ValueError('This campaign fixes one spatial pool=2 and preserves native time')
        self.identity = {'family': 'V-JEPA 2.1 distilled ViT-L/16', 'source_sha': source_sha,
            'weight_sha256': weight_sha256, 'checkpoint_key': 'ema_encoder',
            'input_axes': 'B,C,T,H,W', 'input_hw': [288,384], 'input_times_s': [.5*i for i in range(1,9)],
            'native_thw': [4,18,24], 'output_thw': [4,9,12], 'tubelet': 2, 'pool': 2,
            'feature_dim': self.encoder.embed_dim, 'output': 'last block, norms_block[-1], time,row,column',
            'temporal_semantics': 'bidirectional full-clip context; pairs [.5,1],[1.5,2],[2.5,3],[3.5,4]',
            'precision': 'FP32, TF32 off', 'preprocess': 'RGB PIL bicubic resize384x288, ImageNet mean/std once; no crop/padding',
            'tokenizer': 'official video PatchEmbed3D, no predictor/robot action weights'}

    def train(self, mode=True):
        super().train(False)
        self.encoder.eval()
        return self

    @staticmethod
    def preprocess(frames):
        if len(frames) != 8:
            raise ValueError('Eight real source frames required; no missing-frame filling')
        values = []
        for frame in frames:
            if not isinstance(frame, Image.Image):
                raise TypeError('Original RGB PIL images required')
            x = np.asarray(frame.convert('RGB').resize((384,288), Image.Resampling.BICUBIC)).copy()
            values.append(torch.from_numpy(x).permute(2,0,1).float()/255.)
        clip = torch.stack(values,1)
        return (clip-clip.new_tensor([.485,.456,.406])[:,None,None,None])/clip.new_tensor([.229,.224,.225])[:,None,None,None]

    @torch.no_grad()
    def forward(self, clips):
        if clips.ndim != 5 or clips.shape[1:] != (3,8,288,384) or not torch.isfinite(clips).all():
            raise ValueError('Expected eight real normalized frames B,C,T,H,W')
        features = self.encoder(clips.float())
        if not isinstance(features, torch.Tensor) or features.shape != (len(clips),4*18*24,1024):
            raise ValueError('Official native video time/space layout changed')
        features = features.reshape(len(clips),4,18,24,1024)
        features = F.avg_pool2d(features.permute(0,1,4,2,3).flatten(0,1),2)
        result = features.reshape(len(clips),4,1024,9,12).permute(0,1,3,4,2).contiguous()
        if not torch.isfinite(result).all():
            raise FloatingPointError('Invalid official video features')
        return result
