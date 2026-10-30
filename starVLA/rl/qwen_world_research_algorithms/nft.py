"""Forward-process DiffusionNFT for World clean-x flow policies.

Implements Eq. (5) and Algorithm 1 of https://arxiv.org/abs/2509.16117.
The data/old policy is an EMA of all trainable actor and memory parameters,
distinct from the permanently frozen IL reference. Positive and negative
implicit velocity regressions are both retained. No trajectory likelihood,
PPO ratio, signed MSE, learned critic, or future observation enters this loss.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import torch

from starVLA.rl.simwam_recipe.trainer_adapter import SourceLoss


def trainable_parameters(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


class BehaviorEMA:
    """EMA tensors on the actor device; swapping is only legal outside backward.

    Parameters are restored bit-for-bit, including on exceptions. Only adapter
    and trainable memory tensors are exchanged; frozen Qwen, IL reference and
    action backbone are shared unchanged. Call use() before constructing the
    current-policy graph, never while a live graph awaits backward.
    """
    def __init__(self, model, decay=.9):
        if not math.isfinite(float(decay)) or not 0 <= decay < 1:
            raise ValueError('NFT behavior EMA decay must be finite in [0,1)')
        self.decay = float(decay)
        self.values = {n: p.detach().clone() for n, p in trainable_parameters(model).items()}
        if not self.values:
            raise ValueError('Behavior EMA requires a nonempty trainable scope')
        self.updates = 0

    def _parameters(self, model):
        params = trainable_parameters(model)
        if params.keys() != self.values.keys():
            raise ValueError('NFT behavior EMA trainable closure changed')
        for name,p in params.items():
            value = self.values[name]
            if value.shape != p.shape or value.dtype != p.dtype or value.device != p.device:
                raise ValueError('NFT EMA tensor contract changed: '+name)
        return params

    @torch.no_grad()
    def update(self, model):
        for n,p in self._parameters(model).items():
            self.values[n].mul_(self.decay).add_(p.detach(), alpha=1-self.decay)
        self.updates += 1

    @contextmanager
    def use(self, model):
        params = self._parameters(model)
        if getattr(model,'_prepared_scope',None) is not None:
            raise RuntimeError('EMA exchange must not retain a prepared current-policy graph')
        originals = {n: p.detach().clone() for n,p in params.items()}
        try:
            with torch.no_grad():
                for n,p in params.items():
                    p.copy_(self.values[n])
            yield
        finally:
            with torch.no_grad():
                for n,p in params.items():
                    p.copy_(originals[n])

    def state_dict(self):
        return dict(decay=self.decay,updates=self.updates,
                    values={n:v.detach().cpu().clone() for n,v in self.values.items()})

    @torch.no_grad()
    def load_state_dict(self, state, model):
        if state['decay'] != self.decay or state['values'].keys() != self.values.keys():
            raise ValueError('NFT behavior EMA state identity changed')
        params = self._parameters(model)
        for n,p in params.items():
            value = state['values'][n]
            if value.shape != p.shape or value.dtype != p.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError('Invalid NFT EMA checkpoint tensor: '+n)
            self.values[n].copy_(value.to(p.device))
        if type(state['updates']) is not int or state['updates'] < 0:
            raise ValueError('Invalid behavior EMA update count')
        self.updates = state['updates']


@torch.no_grad()
def optimality_probabilities(rewards, global_std, *, eps=1e-8):
    if rewards.ndim != 2 or not bool(torch.isfinite(rewards).all()):
        raise ValueError('Finite [scene,candidate] rewards required')
    scale = torch.as_tensor(global_std,device=rewards.device,dtype=torch.float64)
    if scale.numel()!=1 or not bool(torch.isfinite(scale)) or bool(scale < 0):
        raise ValueError('Finite nonnegative raw reward population std required')
    r = rewards.detach().double()
    offsets = r-r[:,:1]
    centered = offsets-offsets.mean(1,keepdim=True)
    return (.5+.5*(centered/(scale+eps)).clamp(-1,1)).float()


def nft_objective(current_velocity, old_velocity, target_velocity, optimality, *,
                  beta=1., sigma=None, loss_type='velocity_mse', normalization_floor=1e-5):
    """Nonnegative positive/negative implicit-policy regressions.

    velocity_mse is the direct Eq. (5) with w(t)=1. Equivalently, a clean-x
    model's squared error is weighted by 1/sigma^2. x0_self_normalized is the
    paper's optional detached-MAE weighting; it is explicitly a different
    time weighting and never the original IL L1 objective.
    """
    if current_velocity.shape != old_velocity.shape or current_velocity.shape != target_velocity.shape:
        raise ValueError('Aligned velocity/current/old/target tensors required')
    if current_velocity.ndim != 3 or current_velocity.shape[1:] != (8,3):
        raise ValueError('World NFT uses native [candidate,8,3] flow outputs')
    if optimality.shape != current_velocity.shape[:1]:
        raise ValueError('One optimality probability per candidate required')
    if not math.isfinite(float(beta)) or beta <= 0:
        raise ValueError('NFT implicit guidance beta must be finite and positive')
    if loss_type not in ('velocity_mse','x0_self_normalized'):
        raise ValueError('Unknown NFT flow weighting')
    if not bool(torch.isfinite(optimality).all()) or bool(((optimality<0)|(optimality>1)).any()):
        raise ValueError('NFT optimality probabilities must be in [0,1]')
    old, target = old_velocity.detach(), target_velocity.detach()
    positive = (1-beta)*old+beta*current_velocity
    negative = (1+beta)*old-beta*current_velocity
    positive_error, negative_error = positive-target, negative-target
    if loss_type == 'x0_self_normalized':
        if sigma is None or sigma.shape != current_velocity.shape[:1] or bool((sigma<=0).any()):
            raise ValueError('Positive per-candidate sigma required for clean-x weighting')
        # x0_branch - x0_target = -sigma * (v_branch - v_target).
        positive_error = -sigma[:,None,None]*positive_error
        negative_error = -sigma[:,None,None]*negative_error
        positive_mse = positive_error.square().mean((1,2))/positive_error.abs().mean((1,2)).detach().clamp_min(normalization_floor)
        negative_mse = negative_error.square().mean((1,2))/negative_error.abs().mean((1,2)).detach().clamp_min(normalization_floor)
    else:
        positive_mse = positive_error.square().mean((1,2))
        negative_mse = negative_error.square().mean((1,2))
    r = optimality.detach().to(positive_mse)
    per_candidate = r*positive_mse+(1-r)*negative_mse
    loss = per_candidate.mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite forward-process NFT objective')
    return loss, dict(nft_positive_loss=(r*positive_mse).mean().detach(),
                      nft_negative_loss=((1-r)*negative_mse).mean().detach(),
                      nft_optimality_mean=r.mean().detach(),
                      nft_positive_fraction=(r>.5).float().mean().detach(),
                      nft_negative_fraction=(r<.5).float().mean().detach(),
                      nft_current_old_velocity_mse=(current_velocity.detach()-old).square().mean().detach())


class NFTLoss(SourceLoss):
    """Actor.forward adapter for forward-process mini-batches only."""
    def __init__(self,cfg):
        super().__init__(cfg)
        research = cfg['research']
        if research.get('policy_objective')!='nft' or cfg['grpo']['train']['kl_beta']!=0:
            raise ValueError('NFT is a separately registered family without the IL PG anchor')
        self.beta = float(research.get('nft_beta',1.))
        self.loss_type = research.get('nft_loss','velocity_mse')

    def _policy_loss(self,model,batch):
        if batch.get('schema')!='world_nft_forward_batch_v1':
            raise ValueError('NFT cannot consume an old SDE rollout or likelihood proxy')
        with model._context_call(batch['cond_bg']) as condition:
            current = model.action_velocity(batch['state'],batch['sigma']*1000,condition)
        loss,metrics = nft_objective(current,batch['old_velocity'],batch['target_velocity'],
            batch['optimality'],beta=self.beta,sigma=batch['sigma'],loss_type=self.loss_type)
        metrics.update(nft_loss=loss.detach(),nft_sigma_mean=batch['sigma'].mean().detach(),
                       nft_sigma_min=batch['sigma'].min().detach(),nft_sigma_max=batch['sigma'].max().detach())
        return loss,metrics


@torch.no_grad()
def make_forward_batch(actor,rollout,ema,generator,cfg):
    research = cfg['research']
    minimum = float(research.get('nft_sigma_min',.05))
    if not 0 < minimum < 1:
        raise ValueError('NFT sigma minimum must be strictly between zero and one')
    clean = rollout['clean_actions'].detach().float()
    # Independent persistent CPU RNG, explicit uniform positive sigma. Noise
    # and sigma are newly drawn per inner update and sealed on exact resume.
    noise = torch.randn(clean.shape,generator=generator,dtype=torch.float32).to(actor.device)
    sigma = (minimum+(1-minimum)*torch.rand((len(clean),),generator=generator)).to(actor.device)
    state = (1-sigma[:,None,None])*clean+sigma[:,None,None]*noise
    with ema.use(actor), actor._context_call(rollout['cond_bg']) as condition:
        # Own the output storage before restoring current parameters: a model
        # wrapper may legally return a view of a parameter or mutable cache.
        old = actor.action_velocity(state,sigma*1000,condition).detach().clone()
    return dict(schema='world_nft_forward_batch_v1',cond_bg=rollout['cond_bg'],state=state,
                sigma=sigma,target_velocity=noise-clean,old_velocity=old,
                optimality=rollout['optimality'].detach())
