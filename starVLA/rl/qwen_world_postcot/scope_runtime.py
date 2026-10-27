"""Two scope variants using the SAME real-PDM GRPO engine and native resume."""
from pathlib import Path
import json
import torch

from scripts.analysis.simwam_recipe.train import parameter_hash
from .contracts import ROOT,file_sha
from .checkpoint import atomic_json
from .loading import inspect_checkpoint,load_deployment_checkpoint,validate_rl_parent_selection
from .rl_runtime import CleanXRuntime,source_files as base_source_files
from .rl_conditions import FrozenWorldConditions
from .scope_conditions import MemoryLearningConditions
from .scope_actor import ScopeRecipeActor
from .scope_contracts import MEMORY,TIME,SCOPE_SCHEMA,load_scope_recipe


class ScopeRuntime(CleanXRuntime):
    @staticmethod
    def source_files():
        files=base_source_files()
        for name in ['tools/qwen_world_postcot/train_scope_rl.py','tools/qwen_world_postcot/start_scope_rl.py']:
            files[name]=file_sha(name)
        return dict(sorted(files.items()))

    def load_recipe(self,args):
        from starVLA.rl.flow_grpo.reproducibility import configure_numerics
        configure_numerics();cfg=load_scope_recipe(args.config);self.cfg=cfg;self.args=args
        self.parent=inspect_checkpoint(args.il_checkpoint)
        identity=self.parent['identity']
        if (identity.get('uses_rl') is not False or identity.get('uses_metric_supervision') is not True
                or identity.get('qualification') is not False or self.parent['seal']['updates']!=40000):
            raise ValueError('Both scope experiments start from the same complete formal IL40k')
        self.selection=None
        if not args.qualification:
            if not args.parent_selection:raise ValueError('Explicit sealed user IL40k authorization required')
            self.selection=validate_rl_parent_selection(args.parent_selection,self.parent)
            if (self.selection['training_recipe_sha256']!=file_sha(args.config)
                    or self.selection['hard_manifest_sha256']!=file_sha(args.manifest)
                    or self.selection.get('adaptation')!=cfg['adaptation']):
                raise ValueError('User parent/config/data/scope binding changed')
        return cfg

    def make_actor(self,cfg,args,device):
        from .data import WorldSceneData
        model,info=load_deployment_checkpoint(args.il_checkpoint,device=device)
        if info['model_sha256']!=self.parent['model_sha256']:raise ValueError('IL40k parent changed')
        self.parent=info;head=model.actor
        if (head.hidden_dim,head.num_layers,head.action_dim)!=(1024,20,3):
            raise ValueError('Use the complete20-layer clean-x World actor')
        actor=ScopeRecipeActor(head,cfg,scene_provider=model.scene_provider,predictive_provider=model.predictive_provider)
        data=WorldSceneData(args.data,role='TRAIN')
        conditions=(MemoryLearningConditions(model,data,info['model_sha256'],actor) if cfg['adaptation']==MEMORY
                    else FrozenWorldConditions(model,data,info['model_sha256']))
        object.__setattr__(actor,'conditions',conditions)
        self.actor=actor;self.conditions=conditions
        if args.qualification:
            actor.record_sampling_terms=True
            self.observation_before=self._frozen_observation_hashes()
            self.memory_before=self._current_memory_hashes()
        return actor,None,None

    def _frozen_observation_hashes(self):
        modules={'qwen':self.conditions.qwen}
        if self.cfg['adaptation']==MEMORY:
            modules.update(reference_scene=self.actor.ref_scene_provider,reference_predictive=self.actor.ref_predictive_provider)
        else:modules.update(scene=self.conditions.scene_provider,predictive=self.conditions.predictive_provider)
        return {name:parameter_hash(module,trainable=False) for name,module in modules.items()}

    def _current_memory_hashes(self):
        if self.cfg['adaptation']!=MEMORY:return {}
        return {name:parameter_hash(getattr(self.actor,name),trainable=True)
                for name in ['scene_provider','predictive_provider']}

    def make_optimizer(self,actor,cfg):
        named={name:p for name,p in actor.named_parameters() if p.requires_grad}
        action=[p for name,p in named.items() if name.startswith('action_expert.')]
        groups=[dict(params=action,lr=cfg['learning_rate'],name='action_lora')]
        if cfg['adaptation']==MEMORY:
            memory=[p for name,p in named.items() if name.startswith(('scene_provider.','predictive_provider.'))]
            groups.append(dict(params=memory,lr=cfg['memory_learning_rate'],name='scene_predictive_memory'))
        if sum(len(g['params']) for g in groups)!=len(named):raise ValueError('Optimizer omitted a trainable scope tensor')
        return torch.optim.AdamW(groups,lr=cfg['learning_rate'],betas=tuple(cfg['optimizer_betas']),
            eps=cfg['optimizer_eps'],weight_decay=cfg['weight_decay'])

    def recipe_identity(self,cfg,args,condition_sha):
        value=super().recipe_identity(cfg,args,condition_sha)
        value.update(schema=SCOPE_SCHEMA,adaptation=cfg['adaptation'],
            trainable_scope=self.actor.closure,
            reference_observation='original IL40k frozen Qwen + original IL40k frozen scene/predictive providers',
            current_observation='frozen Qwen + trainable scene/predictive memory' if cfg['adaptation']==MEMORY else 'same frozen IL40k Qwen/memory')
        return value

    @torch.no_grad()
    def validate_collection(self,rollout):
        super().validate_collection(rollout)
        if self.args.qualification and rollout['behavior_version']['update']==0:
            noise=rollout['chain'][:,0]
            rng=torch.get_rng_state().clone()
            current,*_=self.actor.sample_action_chain(rollout['cond_bg'],8,deterministic=True,init_actions=noise)
            reference,*_=self.actor.sample_action_chain(rollout['cond_bg'],8,deterministic=True,init_actions=noise,velocity_use_ref=True)
            error=float((current-reference).abs().max())
            if error!=0. or not torch.equal(rng,torch.get_rng_state()):
                raise ValueError('Zero-initialized scope changes IL ODE or consumes sampling RNG: '+str(error))
            rank=torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            atomic_json(Path(self.args.output)/f'initial_ODE_parity_rank{rank}.json',dict(status='PASS',
                normalized_full_chain_max_abs=error,root_and_innovations_RNG_unchanged=True,
                parent_model_sha256=self.parent['model_sha256'],adaptation=self.cfg['adaptation']))

    def finish_qualification(self,result):
        after=self._frozen_observation_hashes()
        if after!=self.observation_before:raise ValueError('Frozen Qwen/IL reference providers changed')
        memory_after=self._current_memory_hashes()
        if self.cfg['adaptation']==MEMORY and any(self.memory_before[k]==v for k,v in memory_after.items()):
            raise ValueError('One of the current memory providers failed to update')
        result.update(frozen_observation_before=self.observation_before,frozen_observation_after=after,
            current_memory_before=self.memory_before,current_memory_after=memory_after,
            prefill_calls_this_invocation=self.conditions.prefill_calls,prefill_seconds_this_invocation=self.conditions.prefill_seconds,
            actual_frozen_memory_dtype=self.conditions.observed_memory_dtype,
            prepared_context_builds_this_invocation=dict(self.actor.prepared_context_builds),
            current_memory_builds_this_invocation=self.actor.current_memory_builds,
            adaptation=self.cfg['adaptation'],prediction_type='x0',teachers_or_post_head_used_in_RL=False)
