"""Native deterministic-trajectory IL with one stochastic Query sample per observation.

This is T2 (not a lossless scheduling change). All three auxiliary targets,
normalizers, masks, warmup factors and DDP denominators are inherited verbatim.
The protected historical framework and its repeat8 entrypoint remain untouched.
"""
import types
import torch
from ..contracts import require


def native_forward(self, observations, targets, *, completed_updates=0, global_counts=None,
                   noise_generator=None, time_generator=None, horizon_generator=None):
    from starVLA.model.modules.foresight.losses import masked_regression, interaction_loss
    cfg=self.foresight_config
    require(not getattr(self,'_deployment_only',False),'deployment-only model cannot train')
    require(cfg.full_algorithm and cfg.future_target_type!='legacy_single_frame' and not hasattr(self,'future_head'),
            'native Query currently supports the locked full action-video framework only')
    for name,required in [('dino_head',cfg.enable_current_dino or cfg.enable_future_dino),
                          ('interaction_head',cfg.enable_interaction),('spatiotemporal_head',cfg.enable_future_dino)]:
        require(hasattr(self,name)==required,'objective inventory mismatch: '+name)
    self.action_model.sample_valid=targets.get('iqe_sample_valid')
    encoded=self.encode_current(observations)
    action=self.build_planner_condition(encoded);device=action.device
    ego=targets['ego'].to(device=device,dtype=torch.float32)
    require(ego.shape==(len(observations),8,4) and bool(torch.isfinite(ego).all()),'invalid ego target')
    with self.amp():il=self.action_model(action,ego)
    losses={'ego_il':il};metrics={};counts=global_counts or {}
    weights=min(1.,(completed_updates+1)/cfg.auxiliary_warmup)
    if 'ego_scenes' in counts:
        from torch import distributed as dist
        world=dist.get_world_size() if dist.is_initialized() else 1
        losses['ego_il']=il*(len(ego)*world/float(counts['ego_scenes']))
    if hasattr(self,'interaction_head'):
        with self.amp():prediction=self.interaction_head(encoded['W'])
        loss,count=interaction_loss(prediction,targets['interaction_latent'].to(device),
            targets['interaction_valid'].to(device),cfg.normalization_eps,counts.get('interaction'))
        losses['interaction']=loss*cfg.lambda_int*weights
        metrics.update(interaction_raw=loss.detach(),interaction_global_elements=count)
    if cfg.enable_current_dino:
        values,mask=targets['current_dino'],targets['current_dino_valid']
        require(values.shape==(len(ego),3,cfg.dino_feature_dim,cfg.dino_height,cfg.dino_width)
                and mask.shape==(len(ego),3,cfg.dino_height,cfg.dino_width),'DINO target/grid contract')
        seconds=torch.zeros(len(ego),device=device)
        with self.amp():prediction=self.dino_head(encoded['W'],seconds,(cfg.dino_height,cfg.dino_width))
        loss,count=masked_regression(prediction,values.to(device),mask.to(device)[:,:,None],global_count=counts.get('current_dino'))
        losses['current_dino']=loss*cfg.lambda_cur
        metrics.update(current_dino_raw=loss.detach(),current_dino_global_elements=count,current_dino_effective_weight=cfg.lambda_cur)
    if cfg.enable_future_dino:
        loss,count=self.compute_clip_loss(encoded['W'],targets,counts.get('future_clip'))
        losses['future_clip']=loss*cfg.lambda_fut*weights
        metrics.update(future_clip_raw=loss.detach(),future_clip_global_valid_scenes=count,future_clip_effective_weight=cfg.lambda_fut*weights)
    return {'loss':sum(losses.values()),'losses':losses,'metrics':metrics}


def install_native_query(model):
    require(model.__class__.__name__=='QueryFramework','native IL is only for actual Query model')
    model.forward=types.MethodType(native_forward,model)
