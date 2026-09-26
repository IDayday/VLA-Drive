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
        return {'schema_version':SCHEMA_VERSION,'provider':'calibrated_multiplane_v2',
                'sensor_contract':{'cameras':list(self.cameras),'time':'current_only'},
                'coordinates':'ego_t0_x_forward_y_left_z_up_metres','extrinsics':'camera_to_ego',
                'grid':{'bounds':self.bounds,'resolution':self.resolution,'shape':self.grid_shape,'heights':self.heights},
                'feature_dimension':self.channels,'support_semantics':'geometric_FOV_not_occlusion_or_confidence'}

    def forward(self, inputs: ModelInputs):
        if not isinstance(inputs,ModelInputs):
            raise TypeError('Provider requires whitelisted ModelInputs, never Scene/WorldTargets')
        inputs.validate(self.cameras)
        if inputs.optional_current_feature_cache is not None:
            return self.read_cache(inputs)
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
        distortion_valid = torch.ones_like(z,dtype=torch.bool)
        if inputs.distortion is not None:
            coeff = inputs.distortion.float()
            k1,k2,p1,p2,k3 = [coeff[...,i,None] for i in range(5)]
            x,y = normalized.unbind(-1)
            r2 = (x*x+y*y).clamp_max(1e4)
            radial = 1+k1*r2+k2*r2.square()+k3*r2.pow(3)
            # Do not fold off-axis rays back through the nonphysical polynomial branch.
            distortion_valid = (radial>0) & ((1+3*k1*r2+5*k2*r2.square()+7*k3*r2.pow(3))>0)
            normalized = torch.stack([x*radial+2*p1*x*y+p2*(r2+2*x*x), y*radial+p1*(r2+2*y*y)+2*p2*x*y],-1)
        homogeneous = torch.cat([normalized,torch.ones_like(z[...,None])],-1)
        pixels = torch.einsum('bvij,bvnj->bvni',inputs.camera_intrinsics.float(),homogeneous)[...,:2]
        valid = distortion_valid & (z > .1) & (pixels[...,0]>=0) & (pixels[...,0]<w) & (pixels[...,1]>=0) & (pixels[...,1]<h)
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

    def bind_cache_identity(self):
        if any(p.requires_grad for p in self.parameters()):
            raise ValueError('Feature caching requires a frozen provider')
        self.cache_weight_identity = state_fingerprint(self)

    def read_cache(self, inputs):
        if any(p.requires_grad for p in self.parameters()) or not hasattr(self,'cache_weight_identity'):
            raise ValueError('Bind frozen provider identity before using its cache')
        if inputs.current_images.shape[0] != 1 or not inputs.scene_tokens:
            raise ValueError('Cached provider requests require a single explicit scene token')
        payload=inputs.optional_current_feature_cache
        meta=payload.get('metadata',{})
        expected=self.metadata()
        allowed_metadata=set(expected)|{'scene_token','decision_time','provider_weights_sha256','provider_source_sha256','input_tensor_sha256','image_sha256','image_transforms','dtype','calibration_sha256'}
        if set(meta)!=allowed_metadata:raise ValueError('Provider cache metadata field whitelist mismatch')
        # JSON roundtrip normalizes grid tuples saved by external services.
        for key,value in expected.items():
            if json.dumps(meta.get(key),sort_keys=True) != json.dumps(value,sort_keys=True):
                raise ValueError(f'Provider cache metadata mismatch: {key}')
        checks={'provider_weights_sha256':self.cache_weight_identity,
                'scene_token':inputs.scene_tokens[0], 'decision_time':int(inputs.decision_time[0]),
                'input_tensor_sha256':hashlib.sha256(inputs.current_images.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
                'image_transforms':inputs.image_transforms.cpu().tolist(),
                'calibration_sha256':calibration_fingerprint(inputs)}
        from pathlib import Path
        checks['provider_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        for key,value in checks.items():
            if meta.get(key)!=value:raise ValueError(f'Provider cache identity mismatch: {key}')
        f,xyz,support=validate_feature_cache(payload,meta)
        if f.shape!=(1,len(self.coordinates),self.channels) or xyz.shape!=(1,len(self.coordinates),3) or support.shape!=f.shape[:2]:
            raise ValueError('Provider cache tensor shape mismatch')
        if str(f.dtype)!=meta.get('dtype'):raise ValueError('Provider cache dtype mismatch')
        expected_dtype=torch.get_autocast_dtype('cuda') if inputs.current_images.is_cuda and torch.is_autocast_enabled() else next(self.parameters()).dtype
        if f.dtype!=expected_dtype:raise ValueError('Provider cache compute precision mismatch')
        if support.dtype!=torch.bool:raise ValueError('Observation support must be a binary FOV mask')
        if not torch.equal(xyz.cpu(),self.coordinates[None].cpu()):raise ValueError('Provider cache grid coordinate mismatch')
        device=inputs.current_images.device
        return f.to(device),xyz.to(device),support.to(device),meta

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


class ExternalBEVFeatures(nn.Module):
    """Strict tensor/service boundary for an independently executed current-camera BEV.

    The producer signature is configuration, including code/weights/grid/dtype.
    This class does not pretend to generate features; absent features are an error.
    Use provider_cli.py to supply reproducible real geometric features, or pin an
    independently deployed provider's signature with the same sensor contract.
    """
    def __init__(self, signature):
        super().__init__()
        required={'schema_version','provider','sensor_contract','coordinates','extrinsics','grid',
                  'feature_dimension','support_semantics','provider_weights_sha256','provider_source_sha256','dtype'}
        if set(signature)!=required:raise ValueError('Incomplete external provider signature')
        self.signature=dict(signature)
        self.channels=int(signature['feature_dimension'])
        self.cameras=tuple(signature['sensor_contract']['cameras'])
        if signature['sensor_contract'].get('time')!='current_only':raise ValueError('External provider must be current-camera only')
        if not signature['provider_weights_sha256'] or not signature['provider_source_sha256']:raise ValueError('External provider identity required')

    def forward(self, inputs):
        if not isinstance(inputs,ModelInputs):raise TypeError('External provider accepts ModelInputs only')
        inputs.validate(self.cameras)
        if inputs.optional_current_feature_cache is None:raise ValueError('No external feature payload; run the pinned provider first')
        if inputs.current_images.shape[0]!=1 or not inputs.scene_tokens:raise ValueError('Use explicit singleton external requests')
        payload=inputs.optional_current_feature_cache;meta=payload.get('metadata',{})
        dynamic={'scene_token','decision_time','input_tensor_sha256','image_sha256','image_transforms','calibration_sha256'}
        if set(meta)!=set(self.signature)|dynamic:raise ValueError('External metadata whitelist mismatch')
        for key,value in self.signature.items():
            if json.dumps(meta[key],sort_keys=True)!=json.dumps(value,sort_keys=True):raise ValueError(f'External identity mismatch: {key}')
        if meta['scene_token']!=inputs.scene_tokens[0] or meta['decision_time']!=int(inputs.decision_time[0]):raise ValueError('External scene/time mismatch')
        fingerprint=hashlib.sha256(inputs.current_images.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        if meta['input_tensor_sha256']!=fingerprint or meta['image_transforms']!=inputs.image_transforms.cpu().tolist() or meta['calibration_sha256']!=calibration_fingerprint(inputs):raise ValueError('External image/augmentation mismatch')
        f,xyz,support=validate_feature_cache(payload,meta)
        if f.ndim!=3 or f.shape[0]!=1 or f.shape[-1]!=self.channels or xyz.shape!=(*f.shape[:2],3) or support.shape!=f.shape[:2]:raise ValueError('External feature shape mismatch')
        if support.dtype!=torch.bool or str(f.dtype)!=self.signature['dtype']:raise ValueError('External dtype mismatch')
        if f.shape[1]!=int(np_product(self.signature['grid']['shape'])):raise ValueError('External BEV grid size mismatch')
        device=inputs.current_images.device
        return f.to(device),xyz.to(device),support.to(device),meta


def np_product(values):
    result=1
    for value in values:result*=int(value)
    return result


def calibration_fingerprint(inputs):
    digest=hashlib.sha256()
    for name in ('camera_intrinsics','camera_extrinsics','image_transforms','distortion'):
        value=getattr(inputs,name)
        digest.update(name.encode())
        if value is not None: digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
