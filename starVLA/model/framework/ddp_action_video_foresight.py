"""Current-only execution, GT-action-conditioned auxiliary clip prediction."""
import torch
from .ddp_full_foresight import DDPFullForesight
from starVLA.model.modules.vehicle_joint.initialization import initialization_seed
from starVLA.model.modules.foresight.future_spatiotemporal_head import FutureSpatiotemporalHead, normalized_clip_loss


class DDPActionVideoForesight(DDPFullForesight):
    def __init__(self, config, accelerator=None):
        if config.framework.name != 'DDPActionVideoForesight' or config.foresight.get('future_target_type', 'legacy_single_frame') == 'legacy_single_frame':
            raise ValueError('Explicit action/video configuration required')
        super().__init__(config, accelerator)
        cfg = self.foresight_config
        if cfg.enable_future_dino:
            with initialization_seed(int(config.seed)+2500):
                self.spatiotemporal_head = FutureSpatiotemporalHead(
                    self.qwen_vl_interface.model.config.hidden_size, cfg.future_feature_dim,
                    cfg.future_time_intervals_s, cfg.readout_dim, cfg.readout_layers,
                    action_condition=cfg.future_action_condition)

    def compute_clip_loss(self, world, targets, global_count):
        cfg = self.foresight_config
        if getattr(self, '_deployment_only', False) or not hasattr(self, 'spatiotemporal_head'):
            raise RuntimeError('Missing configured future task')
        device = world.device
        action = None
        if cfg.future_action_condition == 'gt_ego':
            from starVLA.dataloader.foresight_dataset import decode_ego
            # Decode fixed DDP normalization into physical ego(t0) coordinates.
            decoded = decode_ego(targets['ego'].detach().to(device).float())
            action = torch.cat((decoded[..., :2], decoded[..., 2:3].sin(), decoded[..., 2:3].cos()), -1)
        with self.amp():
            prediction = self.spatiotemporal_head(world, (cfg.future_grid_height, cfg.future_grid_width), gt_action=action)
        return normalized_clip_loss(prediction, targets['future_clip'].to(device),
                                    targets['future_clip_valid'].to(device), eps=cfg.normalization_eps,
                                    global_count=global_count)

    def forward(self, *args, **kwargs):
        if hasattr(self, 'spatiotemporal_head') != self.foresight_config.enable_future_dino:
            raise RuntimeError('Future task/module mismatch')
        return super().forward(*args, **kwargs)
