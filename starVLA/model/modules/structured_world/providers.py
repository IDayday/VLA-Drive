"""Current-camera geometric BEV: calibrated multi-height inverse projection.

This is a trainable lightweight provider, NOT a pretrained BEVDet prior. Pixels
are sampled from calibrated ego grid columns, never reshaped into a fake BEV.
"""
import hashlib
import json
import torch
from torch import nn
from torch.nn import functional as F
from .contracts import ModelInputs, SCHEMA_VERSION


class GeometricBEVProvider(nn.Module):
    def __init__(self, channels=64, bounds=(1.,-20.,50.,20.), resolution=1., heights=(0.,1.,2.), cameras=('CAM_F0','CAM_L0','CAM_R0')):
        super().__init__()
        self.cameras = tuple(cameras)
        self.bounds = tuple(bounds)
        self.resolution = resolution
        self.heights = tuple(heights)
        self.channels = channels
        self.encoder = nn.Sequential(nn.Conv2d(3,32,5,stride=2,padding=2),nn.GroupNorm(8,32),nn.GELU(),
                                     nn.Conv2d(32,channels,3,stride=2,padding=1),nn.GroupNorm(8,channels),nn.GELU(),
                                     nn.Conv2d(channels,channels,3,padding=1))
        x = torch.arange(bounds[0]+resolution/2,bounds[2],resolution)
        y = torch.arange(bounds[1]+resolution/2,bounds[3],resolution)
        xx, yy = torch.meshgrid(x,y,indexing='ij')
        xyz = torch.stack([xx,yy,torch.zeros_like(xx)],-1)
        self.register_buffer('coordinates',xyz.reshape(-1,3))
        self.grid_shape = tuple(xx.shape)
        self.fuse = nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),nn.GELU(),nn.Conv2d(channels,channels,3,padding=1))

    def metadata(self):
        return {'schema_version':SCHEMA_VERSION,'provider':'calibrated_multiplane_v1',
                'sensor_contract':{'cameras':list(self.cameras),'time':'current_only'},
                'coordinates':'ego_t0_x_forward_y_left_z_up_metres','extrinsics':'camera_to_ego',
                'grid':{'bounds':self.bounds,'resolution':self.resolution,'shape':self.grid_shape,'heights':self.heights},
                'feature_dimension':self.channels,'support_semantics':'geometric_FOV_not_occlusion_or_confidence'}

    def forward(self, inputs: ModelInputs):
        if not isinstance(inputs,ModelInputs):
            raise TypeError('Provider requires whitelisted ModelInputs, never Scene/WorldTargets')
        inputs.validate(self.cameras)
        b,v,_,h,w = inputs.current_images.shape
        image = inputs.current_images.reshape(b*v,3,h,w)
        feature = self.encoder(image).reshape(b,v,self.channels,-1)
        fh, fw = self.encoder_output_shape(h,w)
        feature = feature.reshape(b*v,self.channels,fh,fw)
        xyz = self.coordinates.to(inputs.camera_extrinsics.dtype)
        columns = xyz[:,None,:].expand(-1,len(self.heights),-1).clone()
        columns[...,2] = columns.new_tensor(self.heights)
        points = columns.reshape(-1,3)
        transform = torch.linalg.inv(inputs.camera_extrinsics.float())
        camera = torch.einsum('bvij,nj->bvni',transform[...,:3,:3],points.float()) + transform[...,:3,3].unsqueeze(-2)
        z = camera[...,2]
        normalized = camera[...,:2] / z.clamp_min(1e-6)[...,None]
        if inputs.distortion is not None:
            coeff = inputs.distortion.float()
            k1,k2,p1,p2,k3 = [coeff[...,i,None] for i in range(5)]
            x,y = normalized.unbind(-1)
            r2 = (x*x+y*y).clamp_max(1e4)
            radial = 1+k1*r2+k2*r2.square()+k3*r2.pow(3)
            normalized = torch.stack([x*radial+2*p1*x*y+p2*(r2+2*x*x), y*radial+p1*(r2+2*y*y)+2*p2*x*y],-1)
        homogeneous = torch.cat([normalized,torch.ones_like(z[...,None])],-1)
        pixels = torch.einsum('bvij,bvnj->bvni',inputs.camera_intrinsics.float(),homogeneous)[...,:2]
        valid = (z > .1) & (pixels[...,0]>=0) & (pixels[...,0]<w) & (pixels[...,1]>=0) & (pixels[...,1]<h)
        # Pixel-centre grid convention, align_corners=False.
        grid = (pixels+.5) / pixels.new_tensor([w,h]) * 2-1
        grid = grid.masked_fill(~valid[...,None],2.)
        sampled = F.grid_sample(feature,grid.reshape(b*v,-1,1,2).to(feature.dtype),align_corners=False)
        sampled = sampled.reshape(b,v,self.channels,-1,len(self.heights))
        support = valid.reshape(b,v,-1,len(self.heights))
        count = support.sum((1,3))
        pooled = (sampled * support[:,:,None]).sum((1,4)) / count[:,None].clamp_min(1)
        bev = self.fuse(pooled.reshape(b,self.channels,*self.grid_shape))
        features = bev.flatten(2).transpose(1,2)
        return features, self.coordinates[None].expand(b,-1,-1), count>0, self.metadata()

    @staticmethod
    def encoder_output_shape(h,w):
        return ((h+1)//2+1)//2, ((w+1)//2+1)//2


def validate_feature_cache(payload, expected_metadata):
    required = {'metadata','features','coordinates','observation_support'}
    if set(payload) != required:
        raise ValueError('Feature cache field whitelist mismatch')
    # Caller supplies complete expected identity, including code/weights/image transforms/token/time.
    if payload['metadata'] != expected_metadata:
        raise ValueError('Stale or incompatible feature cache')
    for name in required-{'metadata'}:
        if not torch.is_tensor(payload[name]) or not torch.isfinite(payload[name]).all():
            raise ValueError(f'Invalid cached tensor {name}')
    return payload['features'],payload['coordinates'],payload['observation_support']


def state_fingerprint(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
