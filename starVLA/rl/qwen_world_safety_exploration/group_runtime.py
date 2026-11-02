"""Fresh-IL70 SafetyEP with independently registered G8/G16 collection shapes."""
import copy
import json
import time
from pathlib import Path

import torch
import yaml

from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from starVLA.rl.qwen_world_postcot.contracts import file_sha
from starVLA.rl.qwen_world_postcot.loading import inspect_checkpoint, load_deployment_checkpoint
from starVLA.rl.qwen_world_postcot.scope_actor import ScopeRecipeActor
from starVLA.rl.qwen_world_postcot.scope_contracts import MEMORY, expected_scope_recipe
from starVLA.rl.qwen_world_research.runtime import ResearchRuntime, read_sealed
from starVLA.rl.qwen_world_research_algorithms.policy import ResearchLoss
from .conditions import GroupMemoryConditions
from .credit import GroupCredit, STRUCTURED_SCHEMA
from . import recovery

CAMPAIGN = Path('/mnt/project/qwen-world-safety-exploration-20261031')
SCHEMA = 'world_Safety_exploration_experiment_v1'
IL70_SHA256 = '7dda5b14b1ea67fc631b06c7fa336950fa07a0b84c969f4e714bdeb5b5436d40'


def validate_recipe(cfg):
    r = cfg.get('research', {})
    g = cfg.get('grpo', {}).get('sample', {}).get('group_size')
    if (type(g) is not int or g not in (8, 16)
            or r.get('schema') != SCHEMA or r.get('campaign_root') != str(CAMPAIGN)
            or not r.get('arm') or r.get('policy_objective') != 'ppo'
            or r.get('data_policy') not in ('uniform_full', 'online_priority')
            or not isinstance(r.get('structured_advantage'), str)):
        raise ValueError('Unregistered Safety exploration recipe/G/data/credit')
    if any(r.get(k) is not None for k in ('advantage_normalization', 'reward_transform',
            'anchor_ablation', 'memory_lr_ablation', 'noise_level_ablation',
            'sampler_ablation', 'warmstart', 'native_fork', 'lr_schedule', 'preference_auxiliary')):
        raise ValueError('Fresh group study must not silently bundle old research ablations')
    base = expected_scope_recipe(MEMORY)
    expected_grpo = copy.deepcopy(base['grpo']); expected_grpo['sample']['group_size'] = g
    if cfg['grpo'] != expected_grpo:
        raise ValueError('Only group size may change the source sampler/PPO/anchor')
    metadata = {'research', 'evaluate_updates', 'save_every_optimizer_updates',
                'reference_observation', 'data_order', 'logical_budget'}
    if set(cfg) != set(base) | {'research'}:
        raise ValueError('Unexpected top-level scientific fields')
    for k, expected in base.items():
        if k not in metadata and k != 'grpo' and cfg[k] != expected:
            raise ValueError('Changed inherited SafetyEP science: '+k)
    if (cfg['max_steps'] != 50000 or cfg['lr_warmup_steps'] != 100
            or cfg['lr_scheduler_type'] != 'constant' or cfg['learning_rate'] != 5e-5
            or cfg['memory_learning_rate'] != 5e-6
            or cfg['world_size'] != 8 or cfg['seed'] != 42
            or cfg['save_every_optimizer_updates'] <= 0
            or any(type(u) is not int or not 0 <= u <= 50000 for u in cfg['evaluate_updates'])):
        raise ValueError('Preserve 50k scheduler identity, warmup100, LR, world8 and seed42')
    budget = dict(fresh_global_rollouts=12500, scenes=100000, candidates=100000*g)
    if cfg['logical_budget'] != budget:
        raise ValueError('Logical scheduler budget must reflect actual G: '+repr(budget))
    return cfg


@torch.no_grad()
def collect_group(actor, condition, token, log, service, generator, *, group_size,
                  variant, version, progress=None):
    """Same source W3 chain and scoring; horizon8 is independent of G."""
    if group_size not in (8, 16, 32) or variant != 'R_CANONICAL':
        raise ValueError('Registered G and canonical W3 collection required')
    actor._noise_audit = []
    begin = time.monotonic()
    chain, times, deltas, indices = actor.sample_action_chain(condition, 8, generator=generator)
    if (tuple(chain.shape) != (group_size, 11, 8, 3) or indices.tolist() != [7, 8, 9]
            or not torch.isfinite(chain).all()):
        raise ValueError('Invalid native G-chain; no replacement candidate')
    if chain.is_cuda: torch.cuda.synchronize()
    rollout_seconds = time.monotonic()-begin
    if progress: progress('generated', rollout_seconds, group_size)
    requested = ActionCodec.denorm_odo(chain[:, -1]).cpu().numpy()
    begin = time.monotonic(); records = service.score((token, log, requested))
    scoring_seconds = time.monotonic()-begin
    if len(records) != group_size:
        raise ValueError('Scoring must retain exactly G original candidates')
    if progress: progress('scored', scoring_seconds, group_size)
    rewards = torch.tensor([[r['readouts']['canonical_PDMS'] for r in records]],
                           device=actor.device, dtype=torch.float32)
    begin = time.monotonic()
    old_logp = actor.action_chain_logprobs(condition, chain, times, deltas, stoch_idx=indices).detach()
    old = copy.deepcopy(actor._probability_audit['current'])
    sampling_check = None
    if actor.record_sampling_terms:
        if len(actor.sampling_terms) != 3:
            raise ValueError('Actual W3 sampling terms changed')
        errors = []
        for i, (mean, std) in enumerate(actor.sampling_terms):
            errors.append(dict(step=int(indices[i]), mean_max_abs=float((mean-old['mean'][:, i]).abs().max()),
                std_abs=float((torch.tensor(std, device=old['std'].device, dtype=torch.float32)-old['std'][i]).abs())))
        sampling_check = dict(status='PASS' if all(e['mean_max_abs'] == 0 and e['std_abs'] == 0 for e in errors) else 'FAIL',
            actual_sample_vs_recompute_mean_std='exact_required', steps=errors)
    if chain.is_cuda: torch.cuda.synchronize()
    return dict(cond_bg=condition, chain=chain.detach(), timesteps=times, deltas=deltas, stoch_idx=indices,
        old_logp=old_logp, old_probability=old, rewards=rewards, records=records, token=token, log=log,
        behavior_version=version, group_size=group_size, noise_audit=copy.deepcopy(actor._noise_audit),
        qualification_sampling_check=sampling_check,
        qualification_sample_terms=copy.deepcopy(actor.sampling_terms) if actor.record_sampling_terms else None,
        seconds=dict(rollout=rollout_seconds, scoring_wait=scoring_seconds,
            old_probability=time.monotonic()-begin, scoring_cpu=sum(r['scoring_seconds'] for r in records)),
        protocol_sha256=service.protocol['sha256'])


class GroupRuntime(ResearchRuntime):
    """Subclass make_credit or the explicit stream/state hooks for new arms."""
    def require_dispatch(self):
        from .guard_return import native_dispatch
        return native_dispatch(self)

    @staticmethod
    def source_files():
        files = ResearchRuntime.source_files()
        for root in ('starVLA/rl/qwen_world_safety_exploration', 'tools/qwen_world_safety_exploration'):
            for p in Path(root).rglob('*.py'):
                files[str(p)] = file_sha(p)
        return dict(sorted(files.items()))

    def make_credit(self, cfg):
        if cfg['research']['structured_advantage'] != STRUCTURED_SCHEMA:
            raise ValueError('A different credit schema requires an explicit runtime/credit subclass')
        return GroupCredit(cfg['grpo']['sample']['group_size'])

    def load_recipe(self, args):
        configure_numerics()
        recipe_validator = getattr(self, 'recipe_validator', validate_recipe)
        cfg = recipe_validator(yaml.safe_load(Path(args.config).read_text()))
        self.cfg, self.args = cfg, args
        self.group_size = cfg['grpo']['sample']['group_size']
        self.credit = self.make_credit(cfg)
        self.parent = inspect_checkpoint(args.il_checkpoint, verify_files='model')
        identity = self.parent['identity']
        if (self.parent['model_sha256'] != IL70_SHA256 or self.parent['seal']['updates'] != 70000
                or identity.get('uses_rl') is not False or identity.get('qualification') is not False
                or identity.get('uses_metric_supervision') is not True):
            raise ValueError('Use the same complete formal IL70 parent; fresh RL only')
        selected = read_sealed(args.parent_selection)
        if (selected.get('schema') != 'world_RL_optimization_pinned_IL_parent_v1'
                or selected['selected']['model_sha256'] != self.parent['model_sha256']
                or selected['selected']['updates'] != self.parent['seal']['updates']
                or selected['selected']['files_sha256'] != self.parent['seal']['files_sha256']
                or Path(selected['selected']['checkpoint']).resolve() != Path(self.parent['path'])
                or selected.get('Test_observed') is not True):
            raise ValueError('Pinned IL70 selection/ancestry disclosure changed')
        r = cfg['research']
        if file_sha(args.manifest) != r['manifest_sha256'] or str(Path(args.data).resolve()) != r['observation_data']:
            raise ValueError('Registered complete scene stream or observations changed')
        self.selection = None if args.qualification else selected
        return cfg

    def make_scene_stream(self, cfg, args, world):
        stream = super().make_scene_stream(cfg, args, world)
        if (world != 8 or stream.mode != 'uniform' or len(stream.tokens) != 103288
                or stream.manifest.get('seed') != 42):
            raise ValueError('Baseline retains uniform complete103288 population/source seed42 order')
        return stream

    def make_actor(self, cfg, args, device):
        from starVLA.rl.qwen_world_research_data import ResearchSceneData
        model, info = load_deployment_checkpoint(args.il_checkpoint, device=device)
        if info['model_sha256'] != self.parent['model_sha256']:
            raise ValueError('Pinned IL70 changed between verification and loading')
        head = model.actor
        if (head.hidden_dim, head.num_layers, head.action_dim) != (1024, 20, 3):
            raise ValueError('Complete original World clean-x actor required')
        self.parent = info
        actor = ScopeRecipeActor(head, cfg, scene_provider=model.scene_provider,
                                 predictive_provider=model.predictive_provider)
        actor._loss = ResearchLoss(cfg)
        if actor.closure['trainable_numel'] != 52707840:
            raise ValueError('Exact original52.7M action+memory scope required')
        data = ResearchSceneData(args.data, role='NAVTRAIN')
        self.conditions = GroupMemoryConditions(model, data, info['model_sha256'], actor,
                                                 group_size=self.group_size)
        object.__setattr__(actor, 'conditions', self.conditions)
        self.actor = actor
        if args.qualification:
            actor.record_sampling_terms = True
            self.observation_before = self._frozen_observation_hashes()
            self.memory_before = self._current_memory_hashes()
        return actor, None, None

    def collect(self, *args, **kwargs):
        return collect_group(*args, group_size=self.group_size, **kwargs)

    def finalize_advantages(self, rollout):
        return self.credit.finalize(rollout)

    def validate_boundary(self, *args, **kwargs):
        return recovery.validate_boundary(*args, credit=self.credit, recipe_validator=validate_recipe, **kwargs)

    def validate_ranks(self, summaries):
        return recovery.validate_ranks(summaries, credit=self.credit)

    def bind_training_state(self, state, rollout, identity):
        self.training_state, self.training_identity = state, identity

    def on_collection_committed(self, rollout, state):
        """Curriculum overrides after finalized A and exposure/cursor commit."""

    def public_training_state(self, state):
        value = dict(state)
        if 'curriculum' in value:
            value['curriculum'] = {k:v for k,v in value['curriculum'].items()
                                   if k not in ('bank', 'selection_plan', 'rng_state', 'private_rng_state')}
        return value

    def recipe_identity(self, cfg, args, condition_sha):
        value = super().recipe_identity(cfg, args, condition_sha)
        value.update(exploration_protocol=dict(schema=SCHEMA, group_size=self.group_size,
            world=8, horizon=8, stochastic_steps=[7, 8, 9],
            global_normalization_candidates=8*self.group_size,
            native_resume='same recipe/G/source and complete rank state only; no G8-to-G16 resume'),
            structured_advantage_protocol=copy.deepcopy(self.credit.protocol),
            fresh_IL70=True)
        return value
