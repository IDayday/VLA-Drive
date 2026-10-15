"""DDP hooks around unmodified SimWAM sampler/probability/loss functions.

No C/N cost, noise-head, dynamic sampling, replay or adaptive KL is imported.
Source transition clipping and partial-step estimator remain explicit proxies.
"""
import math
from pathlib import Path
import torch
from torch import nn
from .vendor.action_math import ActionGRPOMixin
from .vendor.trainer_math import SimWAMGRPOTrainer
from .vendor.scheduler_continuous import WanContinuousFlowMatchScheduler
from .lora_adapter import attention_paths,parameter_contract


class SourceLoss(SimWAMGRPOTrainer):
    def __init__(self,cfg):
        t=cfg['grpo']['train']
        self.kl_gate_enabled=False
        self.ppo_clip_range=t['ppo_clip_range'];self.adv_clip_max=t['adv_clip_max']
        self.drop_zero_adv=t['drop_zero_adv'];self.adv_eps=t['adv_eps']
        self.adv_q_lo=t['adv_clip_lower_quantile'];self.adv_q_hi=t['adv_clip_upper_quantile']
        self.adv_global_std=t['adv_global_std']


class RecipeActor(ActionGRPOMixin,nn.Module):
    def __init__(self,head,cfg):
        nn.Module.__init__(self)
        self.action_expert=head
        self.device=next(head.parameters()).device
        self.torch_dtype=torch.float32
        self.grpo_action_dtype=torch.float32
        self.infer_action_scheduler=WanContinuousFlowMatchScheduler(1000,shift=cfg['action_infer_shift'])
        self._velocity_calls=0;self._reference_calls=0
        self._noise_audit=[];self._probability_audit={}
        self._loss=SourceLoss(cfg)
        with torch.random.fork_rng(devices=[self.device.index] if self.device.type=='cuda' else []):
            torch.manual_seed(cfg['seed'])
            self.configure_grpo(cfg['grpo'])
        self.eval()
        self.closure=parameter_contract(self)

    def _resolve_action_lora_target_modules(self, target_names):
        return attention_paths(self.action_expert,target_names)

    def action_velocity(self,latents_action,timestep_action,cond,use_ref=False):
        # Source sigma decreases; original DDP data time and velocity increase.
        sigma=timestep_action.float()/float(self.infer_action_scheduler.num_train_timesteps)
        buckets=((1-sigma)*self.action_expert.num_timestep_buckets).long()
        if (buckets<0).any() or (buckets>1000).any():raise ValueError('DDP time outside original bucket domain')
        head=self.ref_action_expert if use_ref else self.action_expert
        self._reference_calls+=int(use_ref);self._velocity_calls+=int(not use_ref)
        with torch.autocast(self.device.type,enabled=False):
            value=head.predict_velocity(latents_action.float(),buckets,cond['context'].float())
        if not torch.isfinite(value).all():raise ValueError('Nonfinite DDP velocity')
        return -value

    def _randn(self,shape,device,dtype,generator):
        value=super()._randn(shape,device,dtype,generator)
        self._noise_audit.append(dict(shape=list(shape),dtype=str(dtype),
                                     count=value.numel(),outside_source_clip=int((value.abs()>self.grpo_randn_clip).sum())))
        return value

    def action_chain_logprobs(self,*args,return_terms=False,use_ref=False,**kwargs):
        # This calls the unchanged source ordering and cast path. Diagnostics
        # reconstruct raw density from returned means without another DiT pass.
        logp,terms=super().action_chain_logprobs(*args,return_terms=True,use_ref=use_ref,**kwargs)
        chain=args[1] if len(args)>1 else kwargs['chain']
        ts=args[2] if len(args)>2 else kwargs['timesteps']
        steps=self._normalize_stoch_idx(kwargs.get('stoch_idx'),len(ts))
        std=terms['std'][None,:,None,None]
        mean=terms['mean'].to(self.grpo_action_dtype).float()
        nxt=chain[:,torch.tensor(steps,device=chain.device)+1].float()
        raw=torch.distributions.Normal(mean,std).log_prob(nxt)
        # The source computes clamp bounds from a Python-double std before
        # constructing its FP32 Normal. Taking log of the rounded FP32 std
        # instead changes tail values; reproduce the actual source order.
        deltas=args[3] if len(args)>3 else kwargs['deltas']
        bounds=[]
        for k in steps:
            sigma=float(ts[k].item())/self.infer_action_scheduler.num_train_timesteps
            if self.grpo_sde_mode=='rigorous':
                _,scalar_std=self._sde_mean_std(chain[:,k],torch.zeros_like(chain[:,k]),sigma,float(deltas[k].item()))
            else:scalar_std=self._step_std(sigma,self.grpo_logprob_noise_std)
            bounds.append(-math.log(scalar_std)-.5*math.log(2*math.pi))
        clipped=torch.stack([raw[:,i].clamp(min=p-self.grpo_logprob_clamp_width,max=p)
                             for i,p in enumerate(bounds)],dim=1)
        if not torch.equal(clipped.mean(dim=(-1,-2)),logp):raise ValueError('Source probability diagnostic diverged')
        self._probability_audit['reference' if use_ref else 'current']=dict(
            raw_gaussian_logprob=raw.detach(),source_clamped_logprob=clipped.detach(),
            source_clamp_hit_rate=float((raw!=clipped).float().mean()),
            mean=terms['mean'].detach(),velocity=terms['vel'].detach(),std=terms['std'].detach())
        return (logp,terms) if return_terms else logp

    def forward(self,rollout):
        if self.training or self.ref_action_expert.training:raise ValueError('PG must retain deterministic eval mode')
        total,metrics=self._loss._policy_loss(self,rollout)
        if not torch.isfinite(total):raise ValueError('Nonfinite source loss')
        return total,metrics


def make_actor(cfg, *, device='cuda',verify_hash=True):
    from starVLA.rl.constrained_flow_noise.loading import load_action_closure
    # Existing loader validates all original 359 tensors before the new recipe
    # freezes them. Its original full-head assertion is not removed or weakened.
    closure,oldcfg,sft=load_action_closure(dtype=torch.float32,device=device,verify_hash=verify_hash)
    actor=RecipeActor(closure.action_model,cfg)
    return actor,oldcfg,sft
