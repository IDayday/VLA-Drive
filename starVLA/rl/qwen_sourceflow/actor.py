"""Native 3D action expert wired to the audited, unchanged source GRPO math."""
import torch
from torch import nn
from starVLA.rl.simwam_recipe.trainer_adapter import RecipeActor, SourceLoss
from starVLA.rl.simwam_recipe.vendor.scheduler_continuous import WanContinuousFlowMatchScheduler
from starVLA.rl.simwam_recipe.vendor.lora import LoRALinear
from starVLA.model.modules.qwen_sourceflow.cached_action_runner import CachedActionRunner


def attention_paths(head):
    paths=[]
    for i, block in enumerate(head.blocks):
        for attention in ('self_attn', 'cross_attn'):
            for projection in ('q', 'k', 'v', 'o'):
                path=f'blocks.{i}.{attention}.{projection}'
                if not isinstance(head.get_submodule(path), nn.Linear):
                    raise ValueError('Expected native source attention Linear '+path)
                paths.append(path)
    return paths


class NativeRecipeActor(RecipeActor):
    def __init__(self, head, cfg, conditions=None):
        nn.Module.__init__(self)
        self.action_expert=head
        self.device=next(head.parameters()).device
        self.torch_dtype=self.grpo_action_dtype=torch.float32
        self.infer_action_scheduler=WanContinuousFlowMatchScheduler(1000,shift=5.)
        self._velocity_calls=self._reference_calls=0
        self._noise_audit=[];self._probability_audit={}
        self.record_sampling_terms=False
        self.sampling_terms=[]
        self._loss=SourceLoss(cfg)
        paths=attention_paths(head)
        with torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type=='cuda' else []):
            torch.manual_seed(42)
            self.configure_grpo(cfg['grpo'])
        # Runners contain the SAME experts already registered above. Avoid
        # duplicate parameter names/state; no hidden independent expert exists.
        object.__setattr__(self,'current_runner',CachedActionRunner(self.action_expert).eval())
        object.__setattr__(self,'reference_runner',CachedActionRunner(self.ref_action_expert).eval())
        object.__setattr__(self,'conditions',conditions)
        self.eval()
        names={n:p for n,p in self.named_parameters() if p.requires_grad}
        expected={'action_expert.'+p+'.'+suffix for p in paths for suffix in ('lora_A','lora_B')}
        if set(names)!=expected or any(p.requires_grad for p in self.ref_action_expert.parameters()):
            raise ValueError('Native LoRA closure/reference freeze mismatch')
        modules=[]
        for path in paths:
            m=head.get_submodule(path)
            if not isinstance(m,LoRALinear):raise ValueError('Source LoRA wrapper absent '+path)
            modules.append(dict(path=path,kind=path.split('.')[2],in_features=m.base.in_features,
                out_features=m.base.out_features,r=m.r,alpha=m.r*m.scaling,scaling=m.scaling,
                dtype=str(m.lora_A.dtype)))
        self.closure=dict(adapters=modules,trainable_names=sorted(names),
            trainable_numel=sum(p.numel() for p in names.values()),action_shape=[8,3],
            action_layers=len(head.blocks),observation_frozen=True,reference_frozen=True)

    def _resolve_action_lora_target_modules(self, targets):
        if list(targets)!=['q','k','v','o']:raise ValueError('Changed native attention scope')
        return attention_paths(self.action_expert)

    def materialize(self, condition):
        if condition.get('schema')=='native_frozen_observation_v1':
            if self.conditions is None:raise ValueError('No frozen observation provider')
            return self.conditions.materialize(condition,self.device)
        return condition

    def action_velocity(self, latents_action, timestep_action, cond, use_ref=False):
        cond=self.materialize(cond)
        self._reference_calls+=int(use_ref);self._velocity_calls+=int(not use_ref)
        runner=self.reference_runner if use_ref else self.current_runner
        with torch.autocast(self.device.type,enabled=False):
            value=runner(latents_action.float(),timestep_action.float(),cond)
        if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite native source velocity')
        return value

    def action_chain_logprobs(self, cond, *args, **kwargs):
        return super().action_chain_logprobs(self.materialize(cond),*args,**kwargs)

    def sample_action_chain(self, cond, *args, **kwargs):
        if not self.record_sampling_terms:
            return super().sample_action_chain(self.materialize(cond),*args,**kwargs)
        self.sampling_terms=[];original=self._sde_mean_std
        def recorded(*values,**options):
            mean,std=original(*values,**options)
            self.sampling_terms.append((mean.detach().clone(),std))
            return mean,std
        self._sde_mean_std=recorded
        try:return super().sample_action_chain(self.materialize(cond),*args,**kwargs)
        finally:del self._sde_mean_std
