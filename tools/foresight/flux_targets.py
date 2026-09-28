"""Frozen deterministic FLUX image VAE, label side only; no diffusion transformer."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from starVLA.model.modules.vehicle_joint.initialization import file_sha256

FLUX_REPOSITORY='black-forest-labs/FLUX.1-schnell'
FLUX_REVISION='741f7c3ce8b383c54771c7003378a50191e9efe9'
FLUX_WEIGHT='vae/diffusion_pytorch_model.safetensors'
FLUX_WEIGHT_SHA256='f5b59a26851551b67ae1fe58d32e76486e1e812def4696a4bea97f16604d40a3'
FLUX_CONFIG='vae/config.json'
FLUX_CONFIG_GIT_BLOB='b43183d0f5f0274bccd8054cd0069fc1d5f64586'


def verify_flux_source(root,identity):
    """Pin to public file metadata, not a caller's arbitrary self-declared hash."""
    if identity['repository']!=FLUX_REPOSITORY or identity['revision']!=FLUX_REVISION:
        raise ValueError('Wrong registered generic FLUX source/revision')
    files=identity['files']
    if set(files)!={FLUX_CONFIG,FLUX_WEIGHT} or files[FLUX_WEIGHT]!=FLUX_WEIGHT_SHA256:
        raise ValueError('Both exact registered VAE config and public weight hashes are required')
    root=Path(root);config=(root/FLUX_CONFIG).read_bytes()
    git_blob=hashlib.sha1(b'blob '+str(len(config)).encode()+b'\0'+config).hexdigest()
    if git_blob!=FLUX_CONFIG_GIT_BLOB:raise ValueError('VAE config differs from the pinned public Git blob')
    for relative,expected in files.items():
        if file_sha256(root/relative)!=expected:raise ValueError('VAE file differs from registered public identity')


class FluxTargetEncoder(nn.Module):
    def __init__(self,root,identity,device='cuda',dtype=torch.float32):
        super().__init__()
        verify_flux_source(root,identity);root=Path(root)
        from diffusers import AutoencoderKL
        self.vae=AutoencoderKL.from_pretrained(str(root/'vae'),local_files_only=True,use_safetensors=True,torch_dtype=dtype).to(device)
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
