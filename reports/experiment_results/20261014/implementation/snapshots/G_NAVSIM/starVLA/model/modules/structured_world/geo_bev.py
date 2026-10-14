"""Single-current-frame wrapper around the pinned ResWorld/GeoBEV modules.

The port corrects rectangular-grid axes and pixel-center coordinates, and masks
camera/depth projection support. It does not accept depth labels or global poses.
"""
import torch
from torch import nn
from torch.nn import functional as F
from third_party.resworld.ported.rcsample import RCSample
from third_party.resworld.ported.resnet import ResNet
from third_party.resworld.ported.custom_module import CustomFPN, CustomResNet, FPN_LSS
from .grid import GridSpec


class CalibratedRCSample(RCSample):
    """Documented coordinate/support fixes; the original depth/frustum core stays."""
    def create_grid_infos(self, x, y, z, **kwargs):
        # Upstream meshes x/y then assigns their names in reverse. Our convention
        # is H=number of y cells and W=number of x cells, including rectangular ROI.
        self.grid = GridSpec(x[0], x[1], y[0], y[1], x[2], y[2])
        self.grid_lower_bound = torch.tensor([x[0], y[0], z[0]])
        self.grid_upper_bound = torch.tensor([x[1], y[1], z[1]])
        self.grid_interval = torch.tensor([x[2], y[2], z[2]])
        self.grid_size = (self.grid_upper_bound-self.grid_lower_bound)/self.grid_interval
        xy = self.grid.centers()
        self.bev_coor = torch.cat((xy, torch.zeros_like(xy[..., :1])), -1)

    def get_sample_coor(self, coor, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda):
        # Reuse upstream projection, then correct its missing feature-pixel center
        # offset and record the full image/depth support before height collapse.
        sample = super().get_sample_coor(coor, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda)
        batch, cameras = sensor2ego.shape[:2]
        xyz = torch.linalg.inv(bda)[:, None, None] @ coor.unsqueeze(-1)
        xyz = xyz[:, None]-sensor2ego[:, :, None, None, :3, 3, None]
        camera = torch.linalg.inv(sensor2ego[..., :3, :3])[:, :, None, None] @ xyz
        projected = cam2imgs[:, :, None, None] @ camera
        z = projected[..., 2:3, :]
        pixels = torch.cat((projected[..., :2, :]/z.clamp_min(1e-8), z), -2)
        pixels = post_rots[:, :, None, None] @ pixels + post_trans[:, :, None, None, :, None]
        px, py = pixels[..., 0, 0], pixels[..., 1, 0]
        height, width = self.input_size
        depth_min, depth_max, _ = self.grid_config['depth']
        valid = ((camera[..., 2, 0] >= depth_min) & (camera[..., 2, 0] < depth_max)
                 & (px >= 0) & (px < width) & (py >= 0) & (py < height)
                 & torch.isfinite(pixels).all(dim=(-2, -1)))
        pixel_valid = getattr(self, 'current_pixel_valid', None)
        if pixel_valid is not None:
            uv = torch.stack((2*(px+.5)/width-1, 2*(py+.5)/height-1), -1)
            support = F.grid_sample(pixel_valid.float().reshape(batch*cameras, 1, height, width),
                uv.reshape(batch*cameras, -1, 1, 2), mode='bilinear',
                padding_mode='zeros', align_corners=False)
            valid &= support.reshape_as(valid) >= 1.-1e-5
        sample = sample.clone()
        # Both frustum axes are sampled with align_corners=False: column and
        # depth-bin index zero must address the first pixel/bin center.
        sample += .5
        # Far outside the frustum yields zero features, with a separate support
        # mask. This zero is a feature value, never a ground-truth free label.
        sample[~valid] = -1e6
        self.last_projection_support = valid.detach()
        return sample


class SingleFrameGeoBEV(nn.Module):
    def __init__(self, grid=None, image_size=(256, 448), *, imagenet_checkpoint=None):
        super().__init__()
        self.grid = grid or GridSpec()
        self.image_size = tuple(image_size)
        self.backbone = ResNet(depth=50, num_stages=4, out_indices=(2, 3),
                              frozen_stages=-1, norm_cfg={'type': 'BN', 'requires_grad': True},
                              norm_eval=False, style='pytorch', with_cp=True)
        self.fpn = CustomFPN([1024, 2048], 512, 1, start_level=0, out_ids=[0])
        self.view = CalibratedRCSample(self.grid.resworld_config(), self.image_size,
                                      scale_num=1, ins_channels=[512], out_channels=80,
                                      downsamples=[16], accelerate=False, with_cp=True,
                                      loss_depth_weight=[1.],
                                      depthnet_cfg={'use_dcn': False, 'aspp_mid_channels': 96})
        self.bev_backbone = CustomResNet(numC_input=80, num_channels=[160, 320, 640])
        self.bev_neck = FPN_LSS(in_channels=800, extra_upsample=2, out_channels=256)
        self.initialization = {'backbone': 'not_loaded', 'geometry_supervised_pretraining': None}
        if imagenet_checkpoint is not None:
            self.load_imagenet(imagenet_checkpoint)

    def load_imagenet(self, path):
        from starVLA.model.modules.vehicle_joint.initialization import file_sha256
        state = torch.load(path, map_location='cpu', weights_only=True)
        removed = {k for k in state if k.startswith('fc.')}
        if removed != {'fc.weight', 'fc.bias'}:
            raise ValueError('Expected public torchvision ResNet50 ImageNet state with only FC omitted')
        selected = {k: v for k, v in state.items() if k not in removed}
        self.backbone.load_state_dict(selected, strict=True)
        self.initialization['backbone'] = {'kind': 'public_ImageNet_ResNet50', 'sha256': file_sha256(path),
                                           'excluded_classifier_keys': sorted(removed)}

    def forward(self, images, calibration, pixel_valid=None):
        if images.ndim != 5 or images.shape[1] not in (3, 6) or images.shape[2:] != (3, *self.image_size):
            raise ValueError('One current timestep with exactly 3/6 registered cameras required')
        batch, cameras = images.shape[:2]
        if pixel_valid is not None and (pixel_valid.shape != (batch, cameras, *self.image_size)
                                       or pixel_valid.dtype != torch.bool):
            raise ValueError('Rectified current-image support mask has an invalid layout')
        self.view.current_pixel_valid = pixel_valid
        allowed = {'sensor2ego', 'intrinsics', 'post_rots', 'post_trans'}
        if set(calibration) != allowed:
            raise ValueError('Calibration whitelist excludes global location and label inputs')
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise RuntimeError('Pinned geometry path requires FP32 parameters')
        with torch.autocast(device_type=images.device.type, enabled=False):
            features = self.fpn(self.backbone(images.float().flatten(0, 1)))
            features = features.reshape(batch, cameras, *features.shape[1:])
            sensor2ego = calibration['sensor2ego'].float()
            intrinsics = calibration['intrinsics'].float()
            post_rots, post_trans = calibration['post_rots'].float(), calibration['post_trans'].float()
            # GeoBEV accepts ego2global but does not need it for a single frame;
            # an identity avoids exposing world position to this driving model.
            global_identity = torch.eye(4, device=images.device).expand(batch, cameras, 4, 4)
            bda = torch.eye(3, device=images.device).expand(batch, 3, 3)
            mlp = self.view.get_mlp_input(sensor2ego, global_identity, intrinsics, post_rots, post_trans, bda)
            bev, depth = self.view((features, sensor2ego, global_identity, intrinsics, post_rots, post_trans, bda, mlp))
            output = self.bev_neck(self.bev_backbone(bev))
            if output.shape[-2:] != (self.grid.height, self.grid.width):
                raise RuntimeError('BEV encoder changed the registered spatial extent')
            return {'B0': output, 'depth': depth,
                    'projection_support': self.view.last_projection_support.any(1)}
