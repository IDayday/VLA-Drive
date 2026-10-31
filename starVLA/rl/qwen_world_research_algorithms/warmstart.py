"""Explicit RL weights initialization with fresh phase optimizer and RNG."""
import copy
import json
from pathlib import Path

import torch

from starVLA.rl.qwen_world_postcot.contracts import file_sha
from starVLA.rl.simwam_recipe.lora_adapter import restore_adapters


SCHEMA='world_verified_RL_weights_only_warmstart_v1'
ARMS=('PPO_RESCUE_REPLAY','PPO_RESCUE_CFM')
INFERENCE_FILES=(
    'starVLA/rl/qwen_world_postcot/scope_actor.py',
    'starVLA/rl/qwen_world_postcot/rl_actor.py',
    'starVLA/rl/qwen_world_postcot/scope_conditions.py',
    'starVLA/rl/simwam_recipe/vendor/action_math.py',
    'starVLA/model/modules/qwen_world_postcot/flow.py',
)


def restore_current(actor,adapters,expected_names):
    params={n:p for n,p in actor.named_parameters() if p.requires_grad}
    if set(params)!=set(adapters) or set(params)!=set(expected_names):
        raise ValueError('Warmstart requires the complete current trainable closure')
    for name,param in params.items():
        value=adapters[name]
        if (value.shape!=param.shape or value.dtype!=param.dtype or
                value.dtype!=torch.float32 or not torch.isfinite(value).all()):
            raise ValueError('Warmstart shape/dtype/nonfinite current tensor: '+name)
    restore_adapters(actor,adapters)


class WeightsOnlyWarmstartMixin:
    def load_recipe(self,args):
        cfg=super().load_recipe(args);r=cfg['research'];w=r.get('warmstart',{})
        if (r.get('arm')not in ARMS or w.get('schema')!=SCHEMA
                or w.get('expected_updates')!=10000 or w.get('optimizer')!='fresh_AdamW'
                or w.get('phase_seed')!=42 or w.get('reference')!='original_frozen_IL70'):
            raise ValueError('Explicit weights-only Safety10k initialization required')
        cp=Path(w['checkpoint']).resolve();seal=json.loads((cp/'COMPLETE').read_text())
        if (seal['updates']!=w['expected_updates'] or seal['next_inner']!=0
                or seal.get('world')!=8 or seal['files_sha256']!=w['checkpoint_files_sha256']
                or file_sha(cp/'files.json')!=w['checkpoint_files_sha256']):
            raise ValueError('Warmstart must be the sealed complete eight-rank10k boundary')
        files=json.loads((cp/'files.json').read_text())
        if (files['identity.json']!=w['checkpoint_identity_sha256']
                or files['optimizer.pt']!=w['optimizer_file_sha256']
                or file_sha(cp/'identity.json')!=w['checkpoint_identity_sha256']):
            raise ValueError('Warmstart metadata/payload registration changed')
        identity=json.loads((cp/'identity.json').read_text())
        if (identity.get('purpose')!='formal'
                or identity['recipe']['research']['arm']!='PPO_SAFETY_EP'
                or identity['recipe']['adaptation']!=cfg['adaptation']
                or identity['IL_parent']['model_sha256']!=self.parent['model_sha256']
                or w['expected_parent_model_sha256']!=self.parent['model_sha256']):
            raise ValueError('Warmstart RL provenance or frozen IL ancestry changed')
        if any(file_sha(path)!=identity['source_files'].get(path) for path in INFERENCE_FILES):
            raise ValueError('Warmstart inference model/sampler source changed')
        self.warmstart_checkpoint=cp;self.warmstart_identity=identity
        self.warmstart_protocol=dict(copy.deepcopy(w),checkpoint=str(cp),
            reference_reset=False,optimizer_state_inherited=False,RNG_state_inherited=False,
            phase_updates_start=0,initial_RL_knowledge_updates=10000,
            interpretation='current LoRA+scene/predictive weights only; original IL70 remains the frozen reference; fresh AdamW and phase seed42')
        return cfg

    def make_actor(self,cfg,args,device):
        actor,old_cfg,sft=super().make_actor(cfg,args,device)
        if file_sha(self.warmstart_checkpoint/'optimizer.pt')!=cfg['research']['warmstart']['optimizer_file_sha256']:
            raise ValueError('Warmstart native payload bytes changed')
        payload=torch.load(self.warmstart_checkpoint/'optimizer.pt',map_location='cpu',mmap=True,weights_only=False)
        restore_current(actor,payload['adapters'],self.warmstart_identity['parameter_contract']['trainable_names'])
        del payload
        return actor,old_cfg,sft

    def recipe_identity(self,cfg,args,condition_sha):
        value=super().recipe_identity(cfg,args,condition_sha)
        value['explicit_RL_weights_only_warmstart']=copy.deepcopy(self.warmstart_protocol)
        return value
