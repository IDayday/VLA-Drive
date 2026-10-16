"""Same source Flow-GRPO kernel, with each new supervised policy as reference."""
import copy
import json
from pathlib import Path
import torch
from torch import nn
from starVLA.rl.simwam_recipe.trainer_adapter import RecipeActor, SourceLoss
from starVLA.rl.simwam_recipe.vendor.scheduler_continuous import WanContinuousFlowMatchScheduler
from starVLA.rl.simwam_recipe.vendor.lora import LoRALinear
from starVLA.rl.simwam_recipe.contracts import file_sha, fingerprint, CONDITION_SHA, RAW_CONDITION
from starVLA.rl.constrained_flow_noise.condition_cache import ConditionStore
from .model import make_policy


def load_supervised_policy(checkpoint, *, device='cuda', qualification=False):
    checkpoint=Path(checkpoint)
    if '.incomplete' in checkpoint.name:raise ValueError('Unpublished SFT checkpoint')
    seal=json.loads((checkpoint/'COMPLETE').read_text())
    files=json.loads((checkpoint/'files.json').read_text())
    identity=json.loads((checkpoint/'identity.json').read_text())
    if (seal['phase']!='READY' or file_sha(checkpoint/'files.json')!=seal['files_sha256'] or
        file_sha(checkpoint/'identity.json')!=files['identity.json'] or
        file_sha(checkpoint/'model.pt')!=files['model.pt']):
        raise ValueError('SFT publication/weight identity changed')
    if not qualification and (identity['qualification'] or seal['updates']!=identity['config']['max_updates']):
        raise ValueError('Formal RL requires the registered complete new SFT endpoint')
    policy=make_policy(identity['config']['architecture'],device=device)
    policy.load_state_dict(torch.load(checkpoint/'model.pt',map_location='cpu',weights_only=True,mmap=True),strict=True)
    return policy.eval(),dict(checkpoint=str(checkpoint),model_sha256=files['model.pt'],
        identity_sha256=identity['sha256'],config=identity['config'],updates=seal['updates'],
        qualification=identity['qualification'])


class SourceHeadRecipeActor(RecipeActor):
    """Native continuous sigma/time interface; source forward is not discretized."""
    def __init__(self,head,cfg):
        nn.Module.__init__(self)
        self.action_expert=head;self.device=next(head.parameters()).device
        self.torch_dtype=self.grpo_action_dtype=torch.float32
        self.infer_action_scheduler=WanContinuousFlowMatchScheduler(1000,shift=cfg['action_infer_shift'])
        self._velocity_calls=0;self._reference_calls=0
        self._noise_audit=[];self._probability_audit={};self._loss=SourceLoss(cfg)
        with torch.random.fork_rng(devices=[self.device.index] if self.device.type=='cuda' else []):
            torch.manual_seed(cfg['seed']);self.configure_grpo(cfg['grpo'])
        self.eval()
        adapters={n:p for n,p in self.named_parameters() if p.requires_grad}
        expected={f'action_expert.blocks.{i}.{kind}.{proj}.{ab}' for i in range(30)
            for kind in ('self_attn','cross_attn') for proj in ('q','k','v','o') for ab in ('lora_A','lora_B')}
        if set(adapters)!=expected:raise ValueError('Only all 240 source action attention projections may train')
        if any(p.requires_grad for p in self.ref_action_expert.parameters()):raise ValueError('SFT reference unfrozen')
        info=[]
        for name,module in self.action_expert.named_modules():
            if isinstance(module,LoRALinear):
                info.append(dict(path='action_model.'+name,in_features=module.base.in_features,
                    out_features=module.base.out_features,r=module.r,scaling=module.scaling,
                    attention='self' if '.self_attn.' in name else 'cross'))
        self.closure=dict(trainable_names=sorted(adapters),trainable_numel=sum(p.numel() for p in adapters.values()),
            adapters=info,source_action_projection_count=240,source_action_adapter_tensors=480,
            qwen_frozen=True,native_text_embedding_frozen=True,reference_frozen=True)

    def _resolve_action_lora_target_modules(self, targets):
        if list(targets)!=['q','k','v','o']:raise ValueError('Changed source LoRA targets')
        paths=[f'blocks.{i}.{kind}.{proj}' for i in range(30)
            for kind in ('self_attn','cross_attn') for proj in targets]
        if any(not isinstance(self.action_expert.get_submodule(p),nn.Linear) for p in paths):
            raise ValueError('Unexpected ActionDiT projection topology')
        return paths

    def action_velocity(self,latents_action,timestep_action,cond,use_ref=False):
        head=self.ref_action_expert if use_ref else self.action_expert
        self._reference_calls+=int(use_ref);self._velocity_calls+=int(not use_ref)
        with torch.autocast(self.device.type,enabled=False):
            value=head(latents_action.float(),timestep_action.float(),cond['context'].float())
        if not torch.isfinite(value).all():raise ValueError('Nonfinite source ActionDiT velocity')
        return value


def make_transfer_actor(checkpoint,cfg,*,device='cuda',qualification=False):
    policy,identity=load_supervised_policy(checkpoint,device=device,qualification=qualification)
    if policy.architecture=='ddp':actor=RecipeActor(policy.head,cfg)
    else:actor=SourceHeadRecipeActor(policy.head,cfg)
    # configure_grpo cloned this exact new SFT BEFORE inserting LoRA.
    return actor,identity


class TransferConditions:
    def __init__(self,actor,sft_identity):
        self.actor=actor;self.raw=ConditionStore(RAW_CONDITION)
        if self.raw.identity_sha!=CONDITION_SHA:raise ValueError('Changed Qwen execution')
        self.architecture=sft_identity['config']['architecture']
        self.sha=fingerprint(dict(raw=CONDITION_SHA,SFT=sft_identity['model_sha256'],
            boundary='raw_Qwen' if self.architecture=='source_action_dit' else 'own_frozen_SFT_qwen_proj',
            mode='eval',projection_dtype='float32',autocast=False))

    @torch.no_grad()
    def get(self,token,device):
        raw=self.raw.get(token).to(device=device,dtype=torch.float32)
        if self.architecture=='source_action_dit':return raw
        with torch.autocast(torch.device(device).type,enabled=False):
            return self.actor.action_expert._project_condition(raw)
