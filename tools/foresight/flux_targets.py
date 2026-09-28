"""Frozen deterministic FLUX image VAE, label side only; no diffusion transformer."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from starVLA.model.modules.vehicle_joint.initialization import file_sha256


class FluxTargetEncoder(nn.Module):
    def __init__(self,root,identity,device='cuda',dtype=torch.float32):
        super().__init__()
        # A local name is insufficient evidence: require pinned public hashes.
        if identity['repository']!='black-forest-labs/FLUX.1-schnell' or len(identity['revision'])!=40:
            raise ValueError('This campaign fixes the authorized generic FLUX.1 image VAE')
        root=Path(root)
        for relative,expected in identity['files'].items():
            path=(root/relative).resolve()
            if not path.is_relative_to(root.resolve()) or file_sha256(path)!=expected:raise ValueError('VAE source identity mismatch')
        from diffusers import AutoencoderKL
        self.vae=AutoencoderKL.from_pretrained(str(root/'vae'),local_files_only=True,torch_dtype=dtype).to(device)
        self.vae.requires_grad_(False).eval()
        self.scale=float(self.vae.config.scaling_factor);self.shift=float(self.vae.config.shift_factor)
        self.stride=2**(len(self.vae.config.block_out_channels)-1)
        self.channels=int(self.vae.config.latent_channels)
        self.identity=identity
    def train(self,mode=True):
        super().train(False);self.vae.eval();return self
    def preprocess(self,path,short_side=256):
        with Image.open(path) as src:
            image=src.convert('RGB');w,h=image.size
            # Match the student's fixed DDP crop in the raw image coordinate
            # system, then use the smaller label-encoder resolution.
            cw,ch=(int(h*16/9),h) if w/h>16/9 else (w,int(w*9/16))
            crop=((w-cw)//2,(h-ch)//2,(w-cw)//2+cw,(h-ch)//2+ch)
            image=image.crop(crop)
            ow,oh=round(cw*short_side/min(cw,ch)),round(ch*short_side/min(cw,ch))
            image=image.resize((ow,oh),Image.Resampling.LANCZOS)
            x=torch.from_numpy(np.array(image,copy=True)).permute(2,0,1).float()/127.5-1
        pw=(-ow)%self.stride;ph=(-oh)%self.stride
        x=F.pad(x,(0,pw,0,ph),mode='replicate')
        return x,{'original_wh':[w,h],'resized_wh':[ow,oh],'pad_right_bottom':[pw,ph],
                  'input_range':[-1,1],'crop':list(crop),'resize':'PIL_LANCZOS','short_side':short_side}
    @torch.no_grad()
    def encode_images(self,images):
        self.vae.eval();parameter=next(self.vae.parameters())
        x=images.to(device=parameter.device,dtype=parameter.dtype)
        if not torch.isfinite(x).all():raise ValueError('Invalid VAE image')
        z=self.vae.encode(x).latent_dist.mode()
        return (z-self.shift)*self.scale
    @torch.no_grad()
    def reconstruct(self,latents):
        return self.vae.decode(latents/self.scale+self.shift).sample
