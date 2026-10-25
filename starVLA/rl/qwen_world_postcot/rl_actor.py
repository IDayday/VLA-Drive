"""Canonical source GRPO with a clean-x actor and one positive-sigma conversion.

The proven source SDE, density clamp, advantage and PPO/IL-anchor implementations
are inherited. Only model invocation and time-independent context preparation
are adapted. No learned post-score substitutes for the real PDM reward.
"""
from contextlib import contextmanager

import torch
from torch import nn

from starVLA.model.modules.qwen_world_postcot.flow import clean_x_to_velocity
from starVLA.rl.qwen_sourceflow.actor import NativeRecipeActor, attention_paths
from starVLA.rl.simwam_recipe.trainer_adapter import SourceLoss
from starVLA.rl.simwam_recipe.vendor.lora import LoRALinear, apply_lora_to_module, merged_state_dict
from starVLA.rl.simwam_recipe.vendor.scheduler_continuous import WanContinuousFlowMatchScheduler


RAW_CONTEXT_SCHEMA = 'world_postcot_frozen_context_v1'
OBSERVATION_SCHEMA = 'world_postcot_frozen_observation_v1'


def install_deployment_lora(head, grpo_cfg):
    """Install only current-policy adapters, with no reference actor or teacher.

    Keep wrappers for exact BF16 execution parity. Merging is a separately
    measured numerical execution change, not an exact algebraic export.
    """
    lora = grpo_cfg['lora']
    if not lora['enabled'] or list(lora['target_modules']) != ['q', 'k', 'v', 'o']:
        raise ValueError('Deployment LoRA must match the trained attention projection scope')
    paths = attention_paths(head)
    count = apply_lora_to_module(head, target_names=paths, r=int(lora['r']),
                                alpha=float(lora['alpha']), dropout=float(lora['dropout']))
    if count != len(paths):
        raise ValueError('Incomplete deployment attention LoRA installation')
    head.requires_grad_(False).eval()
    return head


class CleanXRecipeActor(NativeRecipeActor):
    """FP32 stored parameters/Adam; BF16 actor, FP32 source probability math."""
    def __init__(self, head, cfg, conditions=None):
        nn.Module.__init__(self)
        if getattr(head, 'prediction_type', None) != 'x0' or head.action_dim != 3:
            raise ValueError('RL requires the clean-x native8x3 actor, never an old velocity head')
        if any(parameter.dtype != torch.float32 for parameter in head.parameters()):
            raise ValueError('RL stores actor/reference parameters and optimizer states in FP32')
        if cfg.get('action_forward_dtype') != 'bfloat16':
            raise ValueError('Clean-x RL action BF16 compute must be explicitly configured')
        self.action_expert = head
        self.device = next(head.parameters()).device
        self.torch_dtype = self.grpo_action_dtype = torch.float32
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(1000, shift=float(cfg['action_infer_shift']))
        self._velocity_calls = self._reference_calls = 0
        self._noise_audit, self._probability_audit = [], {}
        self.record_sampling_terms = False
        self.sampling_terms = []
        self._loss = SourceLoss(cfg)
        paths = attention_paths(head)
        with torch.random.fork_rng(devices=[self.device.index or 0] if self.device.type == 'cuda' else []):
            torch.manual_seed(int(cfg['seed']))
            # This snapshots the actual loaded IL actor, freezes it permanently,
            # and adds exactly the existing source LoRALinear implementation.
            self.configure_grpo(cfg['grpo'])
        object.__setattr__(self, 'conditions', conditions)
        object.__setattr__(self, '_prepared_scope', None)
        self.prepared_context_builds = dict(current=0, reference=0)
        self.prepared_context_source_batches = []
        self.eval()
        trainable = {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}
        expected = {'action_expert.'+path+'.'+suffix for path in paths for suffix in ('lora_A', 'lora_B')}
        if set(trainable) != expected or any(parameter.requires_grad for parameter in self.ref_action_expert.parameters()):
            raise ValueError('Clean-x attention LoRA/reference parameter closure differs')
        modules = []
        for path in paths:
            value = head.get_submodule(path)
            if not isinstance(value, LoRALinear):
                raise ValueError('A source attention LoRA wrapper was not installed: '+path)
            modules.append(dict(path=path, kind=path.split('.')[2],
                in_features=value.base.in_features, out_features=value.base.out_features,
                r=value.r, alpha=value.r*value.scaling, scaling=value.scaling,
                dtype=str(value.lora_A.dtype)))
        self.closure = dict(adapters=modules, trainable_names=sorted(trainable),
            trainable_numel=sum(parameter.numel() for parameter in trainable.values()),
            action_shape=[8, 3], action_layers=len(head.blocks),
            observation_frozen=True, reference_frozen=True, prediction_type='x0',
            action_forward_dtype='bfloat16', source_velocity_conversion='(x-pred_x0)/sigma; once; sigma>0',
            context_cache='compact frozen memory; current KV graph scoped to one sampler/logprob invocation',
            deployed_lora='preserve wrappers for BF16 parity; merged weights require numerical audit')

    def materialize(self, condition):
        if condition.get('schema') == OBSERVATION_SCHEMA:
            if self.conditions is None:
                raise ValueError('No frozen current-observation encoder was bound')
            return self.conditions.materialize(condition, self.device)
        allowed = {'schema', 'context', 'valid', 'ego', 'base_memory', 'base_valid', 'base_ego'}
        if set(condition)-allowed:
            raise ValueError('Unexpected non-deployment fields in RL context: '+repr(sorted(set(condition)-allowed)))
        if condition.get('schema', RAW_CONTEXT_SCHEMA) != RAW_CONTEXT_SCHEMA:
            raise ValueError('A different observation/teacher context cannot enter clean-x RL')
        required = {'context', 'valid', 'ego'}
        if not required.issubset(condition):
            raise ValueError('RL context requires current student memory, validity and ego')
        if (condition['context'].ndim != 3 or condition['valid'].shape != condition['context'].shape[:2]
                or condition['valid'].dtype != torch.bool
                or condition['ego'].shape != (condition['context'].shape[0], self.action_expert.ego_dim)):
            raise ValueError('Raw RL context layout is invalid')
        return condition

    @contextmanager
    def _context_call(self, condition):
        previous = self._prepared_scope
        object.__setattr__(self, '_prepared_scope', {})
        try:
            yield self.materialize(condition)
        finally:
            # Graph references needed by autograd remain in the returned loss.
            # No prepared student KV survives to the next inner optimizer step.
            object.__setattr__(self, '_prepared_scope', previous)

    def _prepare(self, condition, use_ref):
        key = 'reference' if use_ref else 'current'
        if self._prepared_scope is not None and key in self._prepared_scope:
            return self._prepared_scope[key]
        head = self.ref_action_expert if use_ref else self.action_expert
        if 'base_memory' in condition:
            memory, valid, ego = (condition[name] for name in ('base_memory', 'base_valid', 'base_ego'))
            if memory.shape[0] != 1:
                raise ValueError('A compact per-rank source rollout must bind exactly one scene')
        else:
            memory, valid, ego = (condition[name] for name in ('context', 'valid', 'ego'))
        if memory.device != self.device or ego.device != self.device or valid.device != self.device:
            raise ValueError('Frozen current-memory inputs must be materialized on the actor device')
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            prepared = head.prepare_context(memory.detach(), valid.detach(), ego.detach())
        if any(value.dtype!=torch.bfloat16 for pair in prepared.layer_kv for value in pair):
            raise ValueError('Qualified clean-x RL projected KV must use BF16 compute values')
        self.prepared_context_builds[key] += 1
        self.prepared_context_source_batches.append(int(memory.shape[0]))
        del self.prepared_context_source_batches[:-64]
        batch = condition['context'].shape[0]
        if memory.shape[0] == 1 and batch != 1:
            indices = torch.zeros(batch, device=self.device, dtype=torch.long)
            prepared = prepared.index_select(indices)
        if self._prepared_scope is not None:
            self._prepared_scope[key] = prepared
        return prepared

    def action_velocity(self, latents_action, timestep_action, cond, use_ref=False):
        condition = self.materialize(cond)
        sigma = timestep_action.float().contiguous() / float(self.infer_action_scheduler.num_train_timesteps)
        if sigma.shape != (latents_action.shape[0],) or bool((sigma <= 0).any()):
            raise ValueError('Clean-x RL velocity conversion requires one strictly positive sigma per candidate')
        head = self.ref_action_expert if use_ref else self.action_expert
        self._reference_calls += int(use_ref)
        self._velocity_calls += int(not use_ref)
        prepared = self._prepare(condition, use_ref)
        state = latents_action.float().contiguous()
        with torch.autocast(self.device.type, dtype=torch.bfloat16):
            predicted_x0 = head(state, sigma, prepared)
        # Exactly one conversion, outside autocast. The inherited source SDE
        # and log probabilities consume a velocity in the decreasing-sigma
        # convention; a kl_type named 'x' does not perform this conversion.
        velocity = clean_x_to_velocity(state, predicted_x0.float(), sigma)
        if not bool(torch.isfinite(velocity).all()):
            raise FloatingPointError('Nonfinite clean-x source velocity')
        return velocity

    def action_chain_logprobs(self, cond, *args, **kwargs):
        with self._context_call(cond) as condition:
            return super().action_chain_logprobs(condition, *args, **kwargs)

    def sample_action_chain(self, cond, *args, **kwargs):
        with self._context_call(cond) as condition:
            return super().sample_action_chain(condition, *args, **kwargs)

    def merged_action_state(self):
        """Optional FP32 merge; BF16 deployment equivalence must be measured."""
        return merged_state_dict(self.action_expert)


NewCleanXRecipeActor = CleanXRecipeActor
