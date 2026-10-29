"""Four architectures on the unchanged source SDE/logprob/PPO/anchor math.

Only model scope, immutable observation data and differentiable prefix lifetime
are adapted.  A policy-call prefix is never detached/reused across inner updates.
The reference is a separate permanent IL copy, not a mutable global LoRA switch.
"""
import copy
from contextlib import contextmanager
from dataclasses import replace

import torch
from torch import nn

from starVLA.model.modules.qwen_world_compare.contracts import CurrentContext
from starVLA.model.modules.qwen_world_postcot.flow import clean_x_to_velocity
from starVLA.rl.qwen_sourceflow.actor import NativeRecipeActor
from starVLA.rl.simwam_recipe.trainer_adapter import SourceLoss
from starVLA.rl.simwam_recipe.vendor.scheduler_continuous import WanContinuousFlowMatchScheduler
from .rl_scope import scope_targets, install_lora, trainable_manifest


CONDITION_SCHEMA = 'qwen_world_four_frameworks_frozen_current_v1'
DESCRIPTOR_SCHEMA = 'qwen_world_compare_current_descriptor_v1'


class FourFrameworkRecipeActor(NativeRecipeActor):
    """Canonical native source policy math with no fixed memory shape/parent."""
    def __init__(self, model, cfg, conditions=None):
        nn.Module.__init__(self)
        if model.architecture_id not in ('A', 'B', 'C', 'D'):
            raise ValueError('Unknown four-framework policy')
        if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
            raise ValueError('Four-framework RL stores model/base/LoRA/reference in FP32')
        if cfg.get('action_forward_dtype') != 'bfloat16':
            raise ValueError('Formal RL must explicitly select BF16 action compute')
        self.action_expert = model
        # Source setup queries this public codec size; it must not infer it from
        # a historical actor path or an independently learned score/readout.
        self.action_expert.action_dim = 3
        self.device = next(model.parameters()).device
        self.torch_dtype = self.grpo_action_dtype = torch.float32
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            1000, shift=float(cfg['action_infer_shift']))
        self._velocity_calls = self._reference_calls = 0
        self._noise_audit, self._probability_audit = [], {}
        self.record_sampling_terms, self.sampling_terms = False, []
        self._loss = SourceLoss(cfg)
        self._scope_paths = scope_targets(model)
        with torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type == 'cuda' else []):
            torch.manual_seed(int(cfg['seed']))
            self.configure_grpo(cfg['grpo'])
        object.__setattr__(self, '_prepared_scope', None)
        object.__setattr__(self, 'conditions', conditions)
        self.prepared_context_builds = dict(current=0, reference=0)
        self._policy_version = 'initial_IL_plus_zero_LoRA'
        self.eval()
        self.closure = trainable_manifest(self)

    def _resolve_action_lora_target_modules(self, targets):
        if list(targets) != ['q', 'k', 'v', 'o']:
            raise ValueError('Four-framework RL uses the fixed architecture attention scope')
        return self._scope_paths

    def _configure_action_grpo(self, grpo_cfg):
        """Source freeze/copy/install lifecycle, with shared-object-safe install."""
        self.eval().requires_grad_(False)
        self.ref_action_expert = copy.deepcopy(self.action_expert).eval().requires_grad_(False)
        if (grpo_cfg.get('finetune') or 'lora') != 'lora':
            raise ValueError('Four-framework GRPO requires ordinary attention LoRA')
        lora = dict(grpo_cfg.get('lora', {}))
        if not lora.get('enabled', True):
            raise ValueError('Four-framework LoRA must be enabled')
        self.action_finetune_mode, self.action_lora_enabled = 'lora', True
        self.lora_r, self.lora_alpha = int(lora.get('r', 16)), float(lora.get('alpha', 32.))
        self.lora_dropout = float(lora.get('dropout', 0.))
        self.lora_target_modules = self._resolve_action_lora_target_modules(
            list(lora.get('target_modules', ['q', 'k', 'v', 'o'])))
        self.scope_manifest = install_lora(self.action_expert, paths=self.lora_target_modules,
            r=self.lora_r, alpha=self.lora_alpha, dropout=self.lora_dropout)

    def set_policy_version(self, version):
        self._policy_version = str(version)

    def materialize(self, condition):
        if condition.get('schema') == DESCRIPTOR_SCHEMA:
            if self.conditions is None:
                raise ValueError('No frozen current-only observation provider is bound')
            condition = self.conditions.materialize(condition, self.device)
        allowed = {'schema', 'context', 'current', 'ref_current', 'candidate_to_scene'}
        if set(condition)-allowed or condition.get('schema') != CONDITION_SCHEMA:
            raise ValueError('Only current deployment inputs may enter the policy condition')
        if not isinstance(condition.get('current'), CurrentContext):
            raise TypeError('Four-framework condition requires a CurrentContext')
        batch = condition.get('context')
        if not isinstance(batch, torch.Tensor) or batch.ndim != 3 or not len(batch):
            raise ValueError('Source context carrier must expose the real candidate batch')
        current = condition['current']
        if current.ego.device != self.device:
            raise ValueError('Current context is not on the policy device')
        indices = condition.get('candidate_to_scene')
        if indices is None:
            if len(current.ego) != 1:
                raise ValueError('Multi-scene condition needs explicit candidate_to_scene')
        elif (indices.shape != (len(batch),) or indices.dtype != torch.long
              or indices.device != self.device or int(indices.min()) < 0
              or int(indices.max()) >= len(current.ego)):
            raise ValueError('Candidate/scene indexing differs from the source batch')
        ref = condition.get('ref_current', current)
        if not isinstance(ref, CurrentContext) or ref.scene_identity != current.scene_identity:
            raise ValueError('Reference and current observations must identify the same scenes')
        # A-C C is produced by entirely frozen Qwen+scene providers. D encoded
        # data stops before text C/P. Neither field contains a policy prefix.
        if self.action_expert.architecture_id != 'D' and current.scene.requires_grad:
            raise ValueError('RL C must come from the frozen current encoder, not an old graph')
        if self.action_expert.architecture_id == 'D':
            if current.encoded.inputs_embeds.requires_grad or any(x.requires_grad for x in current.encoded.deepstack):
                raise ValueError('RL D caches only frozen visual/base-embedding inputs, never an old graph')
        return condition

    @contextmanager
    def _context_call(self, condition):
        previous = self._prepared_scope
        object.__setattr__(self, '_prepared_scope', {})
        try:
            yield self.materialize(condition)
        finally:
            object.__setattr__(self, '_prepared_scope', previous)

    def _prepare(self, condition, use_ref):
        key = 'reference' if use_ref else 'current'
        if self._prepared_scope is not None and key in self._prepared_scope:
            return self._prepared_scope[key]
        policy = self.ref_action_expert if use_ref else self.action_expert
        current = condition.get('ref_current', condition['current']) if use_ref else condition['current']
        current = replace(current, policy_version='permanent_IL_REFERENCE' if use_ref else self._policy_version)
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            prefix = policy.prepare_prefix(current, auxiliary=False)
        indices = condition.get('candidate_to_scene')
        if indices is None:
            indices = torch.zeros(len(condition['context']), dtype=torch.long, device=self.device)
        prefix = prefix.index_select(indices)
        self.prepared_context_builds[key] += 1
        if self._prepared_scope is not None:
            self._prepared_scope[key] = prefix
        return prefix

    def action_velocity(self, latents_action, timestep_action, cond, use_ref=False):
        condition = self.materialize(cond)
        sigma = timestep_action.float().contiguous()/float(self.infer_action_scheduler.num_train_timesteps)
        if sigma.shape != (len(latents_action),) or not torch.isfinite(sigma).all() or (sigma <= 0).any():
            raise ValueError('Unique clean-x conversion requires one strictly positive sigma per candidate')
        policy = self.ref_action_expert if use_ref else self.action_expert
        prefix = self._prepare(condition, use_ref)
        self._reference_calls += int(use_ref)
        self._velocity_calls += int(not use_ref)
        state = latents_action.float().contiguous()
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            predicted_x0 = policy.denoise_x0(state, sigma, prefix)
        # The only x0->velocity conversion; native source math is decreasing sigma.
        with torch.autocast(self.device.type, enabled=False):
            velocity = clean_x_to_velocity(state, predicted_x0.float(), sigma)
        if not torch.isfinite(velocity).all():
            raise FloatingPointError('Nonfinite four-framework source velocity')
        return velocity

    def action_chain_logprobs(self, cond, *args, **kwargs):
        with self._context_call(cond) as condition:
            return super().action_chain_logprobs(condition, *args, **kwargs)

    def sample_action_chain(self, cond, *args, **kwargs):
        with self._context_call(cond) as condition:
            return super().sample_action_chain(condition, *args, **kwargs)


ArchitectureRecipeActor = FourFrameworkRecipeActor
