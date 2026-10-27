"""Trainable memory or targeted time LoRA; inherited source GRPO mathematics."""
import copy
from contextlib import contextmanager
import torch
from torch import nn

from .rl_actor import CleanXRecipeActor,RAW_CONTEXT_SCHEMA,OBSERVATION_SCHEMA
from .scope_contracts import MEMORY,TIME,MEMORY_OBSERVATION,time_paths
from starVLA.rl.qwen_sourceflow.actor import attention_paths
from starVLA.rl.simwam_recipe.trainer_adapter import SourceLoss
from starVLA.rl.simwam_recipe.vendor.scheduler_continuous import WanContinuousFlowMatchScheduler
from starVLA.rl.simwam_recipe.vendor.lora import LoRALinear


class ScopeRecipeActor(CleanXRecipeActor):
    def __init__(self,head,cfg,*,scene_provider=None,predictive_provider=None,conditions=None):
        nn.Module.__init__(self)
        self.adaptation=cfg['adaptation']
        if head.prediction_type!='x0' or head.action_dim!=3:
            raise ValueError('Scope experiments require the native clean-x actor')
        self.action_expert=head;self.device=next(head.parameters()).device
        if any(p.dtype!=torch.float32 for p in head.parameters()):
            raise ValueError('Store original and trainable parameters in FP32')
        self.torch_dtype=self.grpo_action_dtype=torch.float32
        self.infer_action_scheduler=WanContinuousFlowMatchScheduler(1000,shift=float(cfg['action_infer_shift']))
        self._velocity_calls=self._reference_calls=0
        self._noise_audit=[];self._probability_audit={};self.record_sampling_terms=False;self.sampling_terms=[]
        self._loss=SourceLoss(cfg)
        paths=time_paths(head) if self.adaptation==TIME else attention_paths(head)
        self._scope_paths=tuple(paths)
        with torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type=='cuda' else []):
            torch.manual_seed(int(cfg['seed']));self.configure_grpo(cfg['grpo'])
        expected={'action_expert.'+p+'.'+s for p in paths for s in ('lora_A','lora_B')}
        if self.adaptation==MEMORY:
            if scene_provider is None or predictive_provider is None:
                raise ValueError('Memory experiment requires both complete IL providers')
            self.ref_scene_provider=copy.deepcopy(scene_provider).eval().requires_grad_(False)
            self.ref_predictive_provider=copy.deepcopy(predictive_provider).eval().requires_grad_(False)
            self.scene_provider=scene_provider.requires_grad_(True)
            self.predictive_provider=predictive_provider.requires_grad_(True)
            expected|={'scene_provider.'+n for n,_ in self.scene_provider.named_parameters()}
            expected|={'predictive_provider.'+n for n,_ in self.predictive_provider.named_parameters()}
        elif self.adaptation!=TIME:raise ValueError('Unknown scope')
        object.__setattr__(self,'conditions',conditions);object.__setattr__(self,'_prepared_scope',None)
        self.prepared_context_builds=dict(current=0,reference=0);self.prepared_context_source_batches=[]
        self.current_memory_builds=0;self.eval()
        trainable={n:p for n,p in self.named_parameters() if p.requires_grad}
        if set(trainable)!=expected:raise ValueError('Trainable parameter closure differs from exact scope')
        if any(p.requires_grad for n,p in self.named_parameters() if n.startswith('ref_')):
            raise ValueError('Permanent IL reference is not frozen')
        modules=[]
        for path in paths:
            m=head.get_submodule(path)
            if not isinstance(m,LoRALinear):raise ValueError('A selected LoRA was bypassed')
            modules.append(dict(path=path,in_features=m.base.in_features,out_features=m.base.out_features,
                r=m.r,alpha=m.r*m.scaling,scaling=m.scaling,dtype=str(m.lora_A.dtype)))
        self.closure=dict(adaptation=self.adaptation,adapters=modules,trainable_names=sorted(trainable),
            trainable_numel=sum(p.numel() for p in trainable.values()),
            action_shape=[8,3],action_layers=len(head.blocks),reference_frozen=True,qwen_frozen=True,
            observation_frozen=self.adaptation!=MEMORY,prediction_type='x0',
            checkpoint_trainable_state='native adapters field contains ALL named trainable tensors, including memory',
            time_adaptation='TimeMLP + per-block and output AdaLN projections; fused time+ego; ego MLP frozen' if self.adaptation==TIME else None)

    def _resolve_action_lora_target_modules(self,targets):
        wanted=['time_modulation_exact_paths'] if self.adaptation==TIME else ['q','k','v','o']
        if list(targets)!=wanted:raise ValueError('Unregistered LoRA target selection')
        return list(self._scope_paths)

    def materialize(self,condition):
        if self.adaptation!=MEMORY:return super().materialize(condition)
        if condition.get('schema')==MEMORY_OBSERVATION:
            if self.conditions is None:raise ValueError('Frozen Qwen feature provider not bound')
            return self.conditions.materialize(condition,self.device)
        expected={'schema','frozen_features','base_memory','base_valid','base_ego','context','valid','ego'}
        if set(condition)!=expected or condition['schema']!=RAW_CONTEXT_SCHEMA:
            raise ValueError('Trainable memory accepts only legal current frozen-Qwen feature inputs')
        if condition['base_memory'].shape[0]!=1 or condition['base_ego'].shape[0]!=1:
            raise ValueError('Exactly one genuine scene per rank; G candidates share its condition')
        return condition

    def _prepare(self,condition,use_ref):
        if self.adaptation!=MEMORY or use_ref:return super()._prepare(condition,use_ref)
        key='current'
        if self._prepared_scope is not None and key in self._prepared_scope:return self._prepared_scope[key]
        with torch.autocast(self.device.type,dtype=torch.bfloat16):
            memory=self.predictive_provider(self.scene_provider(condition['frozen_features'],condition['base_ego']))
            # Keep the entire differentiable current-memory path. The frozen
            # reference uses condition.base_memory, produced by its OWN providers.
            prepared=self.action_expert.prepare_context(memory.actor_memory,memory.actor_valid,condition['base_ego'])
        if any(v.dtype!=torch.bfloat16 for pair in prepared.layer_kv for v in pair):
            raise ValueError('Current projected memory KV changed its BF16 execution contract')
        self.current_memory_builds+=1;self.prepared_context_builds[key]+=1
        self.prepared_context_source_batches.append(1);del self.prepared_context_source_batches[:-64]
        batch=condition['context'].shape[0]
        if batch!=1:prepared=prepared.index_select(torch.zeros(batch,device=self.device,dtype=torch.long))
        if self._prepared_scope is not None:self._prepared_scope[key]=prepared
        return prepared
