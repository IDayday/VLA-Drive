"""DDP Action-Only + shared queries. Auxiliary labels never enter encode_current."""
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
from .QwenOFT import Qwenvl_OFT
from starVLA.cache.navsim_feature_cache import ROBOT_HISTORY_TOKEN, action_query_tokens
from starVLA.model.modules.vehicle_joint.initialization import initialization_seed, add_random_driving_tokens
from starVLA.model.modules.foresight.config import ForesightConfig
from starVLA.model.modules.foresight.future_latent_head import FutureLatentHead
from starVLA.model.modules.foresight.dino_feature_head import DINOFeatureHead
from starVLA.model.modules.foresight.interaction_latent_head import InteractionLatentHead
from starVLA.model.modules.foresight.losses import masked_regression, interaction_loss
from starVLA.model.modules.foresight.tokens import replace_query_embeddings


class DDPForesight(Qwenvl_OFT):
    @classmethod
    def from_pretrained(cls,*args,**kwargs):
        raise ValueError('Use verified generic initialization or an identity-checked THIS-campaign resume')

    def __init__(self, config, accelerator=None):
        options=dict(config.foresight)
        # Environment-backed OmegaConf values are strings until explicitly typed.
        for key in ('lambda_vis','lambda_int','lambda_cur','lambda_fut','normalization_eps'):
            if key in options: options[key]=float(options[key])
        self.foresight_config=ForesightConfig(**options).validate()
        if config.get('from_scratch') is None: raise ValueError('Generic source manifest required')
        flags=[config.datasets.video_data.load_2d_data,config.datasets.gs_data.load_3d_data,
               config.datasets.reward_data.load_reward_data,config.get('w_depth',0),
               config.get('w_video_latent',0),config.get('doing_s2',0),config.get('vit_pre',0),
               config.get('rgb_query_loss',0),config.get('gs_query_loss',0)]
        if any(flags) or config.framework.action_model.mlp_head:
            raise ValueError('Action-Only forbids video/depth/reward and alternate driving branches')
        with initialization_seed(int(config.seed)):
            super().__init__(config,accelerator=accelerator)
        hidden=self.qwen_vl_interface.model.config.hidden_size
        if config.framework.qwenvl.vl_hidden_dim != hidden:
            raise ValueError('Action conditioning dimension must match actual VLM hidden dimension')
        self.foresight_tokens=[f'<foresight_{i}>' for i in range(self.foresight_config.num_queries)]
        tokenizer=self.qwen_vl_interface.processor.tokenizer
        self._special_token_ids={
            'history':(tokenizer.convert_tokens_to_ids(ROBOT_HISTORY_TOKEN),),
            'action':tuple(tokenizer.convert_tokens_to_ids(list(action_query_tokens(self.act_tok))))}
        if self.foresight_tokens:
            self.foresight_initialization=add_random_driving_tokens(self.qwen_vl_interface.model,tokenizer,
                self.foresight_tokens,int(config.seed)+2100)
            self._special_token_ids['foresight']=tuple(self.foresight_initialization['tokens'].values())
            with initialization_seed(int(config.seed)+2200):
                self.foresight_queries=nn.Parameter(torch.randn(len(self.foresight_tokens),hidden)*.02)
        cfg=self.foresight_config
        with initialization_seed(int(config.seed)+2300):
            if cfg.arm in ('B','D'):
                self.future_head=FutureLatentHead(hidden,cfg.latent_channels,cfg.readout_dim,cfg.readout_layers)
            if cfg.enable_current_dino or cfg.enable_future_dino:
                if cfg.is_tradeoff:
                    from starVLA.model.modules.foresight.tradeoff import TokenProjectionHead
                    self.dino_head=TokenProjectionHead(hidden,cfg.dino_feature_dim)
                else:
                    self.dino_head=DINOFeatureHead(hidden,cfg.dino_feature_dim,cfg.readout_dim,cfg.readout_layers)
        with initialization_seed(int(config.seed)+2400):
            if cfg.uses_interaction:
                self.interaction_head=InteractionLatentHead(hidden,cfg.readout_dim,layers=cfg.readout_layers)
        self.qwen_vl_interface.model.model.visual.requires_grad_(False)
        if cfg.gradient_checkpointing:
            self.qwen_vl_interface.model.model.language_model.gradient_checkpointing_enable()
            self.action_model.model.gradient_checkpointing=True

    def train(self,mode=True):
        super().train(mode)
        self.qwen_vl_interface.model.model.visual.eval()
        return self

    def amp(self):
        if getattr(self,'inference_fp32',False):return nullcontext()
        return torch.autocast('cuda' if next(self.parameters()).is_cuda else 'cpu',dtype=torch.bfloat16)

    def query_embeddings(self):
        return self.foresight_queries

    def encode_current(self, observations):
        # A strict whitelist is rebuilt BEFORE either tokenizer or vision model.
        # Labels can be deleted/permuted without changing this computation.
        current=[{k:e[k] for k in ('image','lang','state')} for e in observations]
        if not current or any(len(e['image'])!=3 for e in current): raise ValueError('Exactly three current front views required')
        instructions=[e['lang']+' '+ROBOT_HISTORY_TOKEN+''.join(self.foresight_tokens)+
                      ''.join(self.act_query_tokens) for e in current]
        ids,attention,positions,slots,visual,deepstack=self._build_qwen_batch(current,instructions)
        self.last_sequence_lengths=attention.sum(-1).detach()
        rows=torch.arange(len(current),device=ids.device)
        with self.amp():
            embeddings=self.qwen_vl_interface.model.get_input_embeddings()(ids)
            states=torch.as_tensor(np.asarray([e['state'] for e in current]),device=ids.device,dtype=torch.float32)
            if states.shape!=(len(current),1,4) or not torch.isfinite(states).all():raise ValueError('Invalid allowed ego history state')
            embeddings=replace_query_embeddings(embeddings,slots['history'],self.action_input_model(states))
            if self.foresight_tokens:
                if not (slots['history'][:,-1]<slots['foresight'][:,0]).all() or not (slots['foresight'][:,-1]<slots['action'][:,0]).all():
                    raise ValueError('Causal state→W→action token order violated')
                embeddings=replace_query_embeddings(embeddings,slots['foresight'],self.query_embeddings())
            hidden=self._qwen_language_forward(ids,embeddings,attention,positions,visual,deepstack)
        return {'W':hidden[rows[:,None],slots['foresight']] if self.foresight_tokens else hidden[:,:0],
                'action_queries':hidden[rows[:,None],slots['action']]}

    def forward(self, observations, targets, *, completed_updates=0, noise_generator=None,
                time_generator=None, horizon_generator=None, global_counts=None):
        encoded=self.encode_current(observations)
        action=encoded['action_queries'];device=action.device
        ego=targets['ego'].to(device=device,dtype=torch.float32)
        if ego.shape!=(len(observations),8,4) or not torch.isfinite(ego).all():raise ValueError('Invalid ego target')
        repeat=int(self.config.framework.action_model.repeated_diffusion_steps)
        y=ego.repeat(repeat,1,1)
        cfg=self.config.framework.action_model
        if cfg.noise_beta_beta!=1.:raise ValueError('Explicit independent time RNG currently supports original Beta(alpha,1)')
        noise=torch.randn(y.shape,device=device,dtype=y.dtype,generator=noise_generator)
        u=torch.rand((len(y),),device=device,generator=time_generator)
        times=(cfg.noise_s-u.pow(1./cfg.noise_beta_alpha))/cfg.noise_s
        with self.amp():
            fm=self.action_model(action.repeat(repeat,1,1),y,noise=noise,times=times)
        losses={'ego_fm':fm};metrics={};cfg=self.foresight_config
        weights=min(1.,(completed_updates+1)/cfg.auxiliary_warmup)
        counts=global_counts or {}
        if 'ego_scenes' in counts:
            from torch import distributed as dist
            world=dist.get_world_size() if dist.is_initialized() else 1
            losses['ego_fm']=fm*(len(ego)*world/float(counts['ego_scenes']))
        if hasattr(self,'future_head'):
            values=targets['future_latent'].to(device)
            valid=targets['future_valid'].to(device)
            if values.shape[0:3]!=(len(ego),3,3) or valid.shape!=(len(ego),3,3):raise ValueError('Future horizon/view shape')
            # Labels choose only a LOSS task; encode_current was already completed.
            if 'visual_horizon' in targets:
                horizon=targets['visual_horizon'].to(device)
            else:
                from starVLA.model.modules.foresight.losses import select_horizons
                horizon=select_horizons(valid.cpu(),horizon_generator).to(device)
            rows=torch.arange(len(ego),device=device)
            with self.amp():prediction=self.future_head(encoded['W'],horizon,(cfg.latent_height,cfg.latent_width))
            mask=valid[rows,horizon,:,None,None,None]
            if 'future_spatial_valid' in targets:mask=mask & targets['future_spatial_valid'].to(device)
            loss,count=masked_regression(prediction,values[rows,horizon],mask,global_count=counts.get('visual'))
            losses['visual']=loss*cfg.lambda_vis*weights
            metrics.update(visual_raw=loss.detach(),visual_global_elements=count,horizon=horizon.detach())
        if hasattr(self,'interaction_head'):
            with self.amp():prediction=self.interaction_head(encoded['W'])
            loss,count=interaction_loss(prediction,targets['interaction_latent'].to(device),
                targets['interaction_valid'].to(device),cfg.normalization_eps,counts.get('interaction'))
            losses['interaction']=loss*cfg.lambda_int*weights
            metrics.update(interaction_raw=loss.detach(),interaction_global_elements=count)
        if hasattr(self,'dino_head'):
            from starVLA.model.modules.foresight.losses import request_future_horizons
            # Physical seconds keep old future IDs0/1/2 mapped to1/2/4, even with h=0.
            if cfg.enable_future_dino:
                horizon=targets.get('dino_horizon')
                if horizon is None: horizon=request_future_horizons(len(ego),horizon_generator)
                if horizon.shape!=(len(ego),) or horizon.dtype!=torch.long or ((horizon<0)|(horizon>2)).any():
                    raise ValueError('Invalid requested future horizon')
            for task,enabled,weight in [('current_dino',cfg.enable_current_dino,cfg.lambda_cur),
                                        ('future_dino',cfg.enable_future_dino,cfg.lambda_fut)]:
                if not enabled: continue
                values=targets[task];mask=targets[task+'_valid']
                if task=='future_dino':
                    rows=torch.arange(len(ego),device=values.device);selection=horizon.to(values.device)
                    values=values[rows,selection];mask=mask[rows,selection]
                    seconds=torch.tensor(cfg.future_horizons_s,device=device)[horizon.to(device)]
                    metrics['dino_horizon']=horizon.detach()
                else: seconds=torch.zeros(len(ego),device=device)
                expected=(len(ego),3,cfg.dino_feature_dim,cfg.dino_height,cfg.dino_width)
                if values.shape!=expected or mask.shape!=(len(ego),3,cfg.dino_height,cfg.dino_width):
                    raise ValueError('DINO target/grid contract')
                with self.amp(): prediction=self.dino_head(encoded['W'],seconds,(cfg.dino_height,cfg.dino_width))
                loss,count=masked_regression(prediction,values.to(device),mask.to(device)[:,:,None],global_count=counts.get(task))
                task_warmup = 1. if cfg.full_algorithm and task == 'current_dino' else weights
                losses[task]=loss*weight*task_warmup
                metrics.update({task+'_raw':loss.detach(),task+'_global_elements':count})
                metrics[task+'_effective_weight']=weight*task_warmup
        return {'loss':sum(losses.values()),'losses':losses,'metrics':metrics}

    @torch.no_grad()
    def predict_action(self,observations,*,sampling_seed=42,initial_noise=None):
        encoded=self.encode_current(observations)
        action=encoded['action_queries']
        if initial_noise is None:
            generator=torch.Generator(device=action.device).manual_seed(sampling_seed)
            initial_noise=torch.randn((len(observations),8,4),device=action.device,generator=generator)
        with self.amp():ego=self.action_model.predict_action(action,initial_noise=initial_noise)
        from starVLA.dataloader.foresight_dataset import decode_ego
        return decode_ego(ego.float())

    def strip_auxiliary_heads(self):
        for name in ('future_head','dino_head','interaction_head'):
            if hasattr(self,name):delattr(self,name)
        return self
