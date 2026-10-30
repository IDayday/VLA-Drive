"""Identified population/algorithm experiments on a pinned complete World IL."""
import copy
import json
from pathlib import Path

import torch
import yaml

from starVLA.rl.qwen_world_postcot.contracts import file_sha, fingerprint
from starVLA.rl.qwen_world_postcot.loading import inspect_checkpoint, load_deployment_checkpoint
from starVLA.rl.qwen_world_postcot.scope_contracts import MEMORY, expected_scope_recipe
from starVLA.rl.qwen_world_postcot.scope_runtime import ScopeRuntime
from starVLA.rl.qwen_world_postcot.scope_actor import ScopeRecipeActor
from starVLA.rl.qwen_world_postcot.scope_conditions import MemoryLearningConditions
from starVLA.rl.qwen_world_postcot.rl_runtime import training_dispatch
from starVLA.rl.flow_grpo.reproducibility import configure_numerics
from scripts.analysis.simwam_recipe.train import parameter_hash

CAMPAIGN = Path('/mnt/project/qwen-world-rl-optimization-20261030')
SCHEMA = 'world_RL_optimization_experiment_v1'


def read_sealed(path):
    value = json.loads(Path(path).read_text())
    body = dict(value)
    if body.pop('sha256', None) != fingerprint(body):
        raise ValueError('Changed research registration: '+str(path))
    return value


def validate_recipe(cfg):
    research = cfg.get('research', {})
    if (research.get('schema') != SCHEMA or research.get('campaign_root') != str(CAMPAIGN)
            or research.get('policy_objective') not in ('ppo', 'cispo', 'nft')
            or research.get('data_policy') not in ('hard', 'uniform_full', 'uniform_world_train', 'adaptive')
            or cfg.get('adaptation') != MEMORY or cfg.get('seed') != 42
            or cfg.get('world_size') != 8 or cfg.get('action_shape') != [8, 3]):
        raise ValueError('Unregistered World research scope, algorithm, population, or seed')
    base = expected_scope_recipe(MEMORY)
    permitted_top = {'research', 'max_steps', 'evaluate_updates', 'save_every_optimizer_updates',
                     'learning_rate', 'memory_learning_rate', 'lr_scheduler_type', 'reference_observation', 'data_order'}
    for key, expected in base.items():
        if key in permitted_top:
            continue
        if key == 'grpo':
            expected = copy.deepcopy(expected)
            if research['policy_objective'] == 'nft':
                expected['train']['kl_beta'] = 0
            if research.get('anchor_ablation') == 'raw_batchstd_beta40_v1':
                if (research.get('arm') != 'PPO_ANCHOR40' or research['policy_objective'] != 'ppo'
                        or research['data_policy'] != 'uniform_full'
                        or research.get('advantage_normalization') is not None
                        or research.get('reward_transform') is not None):
                    raise ValueError('Register beta40 as its own RAW-batchstd canonical PPO/full arm')
                expected['train']['kl_beta'] = 40
        if cfg.get(key) != expected:
            raise ValueError('Unregistered inherited scientific change: '+key)
    if set(cfg) != set(base)|{'research'}:
        raise ValueError('Unexpected research config keys')
    memory_lr_limit=5e-6
    if research.get('memory_lr_ablation') == 'memory_lr2e-5_v1':
        if (research.get('arm')!='PPO_MEMORYLR20' or research['policy_objective']!='ppo'
                or research['data_policy']!='uniform_full'
                or research.get('advantage_normalization') is not None
                or research.get('reward_transform') is not None
                or research.get('anchor_ablation') is not None
                or cfg['memory_learning_rate']!=2e-5):
            raise ValueError('Register memory LR2e-5 as its own canonical full PPO arm')
        memory_lr_limit=2e-5
    if (not 0 < cfg['learning_rate'] <= 5e-5 or not 0 < cfg['memory_learning_rate'] <= memory_lr_limit
            or cfg['lr_scheduler_type'] not in ('constant', 'cosine') or cfg['max_steps'] <= 0
            or cfg['max_steps'] % 4 or cfg['save_every_optimizer_updates'] <= 0):
        raise ValueError('Invalid bounded optimizer schedule')
    return cfg


class ResearchRuntime(ScopeRuntime):
    require_dispatch = staticmethod(training_dispatch)

    @staticmethod
    def source_files():
        files = ScopeRuntime.source_files()
        for root in ('starVLA/rl/qwen_world_research', 'starVLA/rl/qwen_world_research_data',
                     'starVLA/rl/qwen_world_research_algorithms', 'tools/qwen_world_research'):
            for p in Path(root).rglob('*.py'):
                files[str(p)] = file_sha(p)
        return dict(sorted(files.items()))

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--il-checkpoint', required=True)
        parser.add_argument('--parent-selection', required=True)
        parser.add_argument('--data', required=True)

    def load_recipe(self, args):
        configure_numerics()
        cfg = validate_recipe(yaml.safe_load(Path(args.config).read_text()))
        self.cfg, self.args = cfg, args
        self.parent = inspect_checkpoint(args.il_checkpoint, verify_files='model')
        identity = self.parent['identity']
        if (identity.get('uses_rl') is not False or identity.get('qualification') is not False
                or identity.get('uses_metric_supervision') is not True):
            raise ValueError('Research parent must be a complete formal World IL')
        selected = read_sealed(args.parent_selection)
        if (selected.get('schema') != 'world_RL_optimization_pinned_IL_parent_v1'
                or selected['selected']['model_sha256'] != self.parent['model_sha256']
                or selected['selected']['updates'] != self.parent['seal']['updates']
                or selected['selected']['files_sha256'] != self.parent['seal']['files_sha256']
                or Path(selected['selected']['checkpoint']).resolve() != Path(self.parent['path'])
                or selected.get('Test_observed') is not True):
            raise ValueError('Pinned selected IL ancestry or observed-Test disclosure changed')
        r = cfg['research']
        if (file_sha(args.manifest) != r['manifest_sha256']
                or str(Path(args.data).resolve()) != r['observation_data']):
            raise ValueError('Registered scene stream or observation population changed')
        self.selection = None if args.qualification else selected
        return cfg

    def make_scene_stream(self, cfg, args, world):
        from starVLA.rl.qwen_world_research_data import SceneStream
        self.stream = SceneStream(args.manifest, world)
        return self.stream

    def make_reward_service(self):
        from starVLA.rl.simwam_recipe.reward_readouts import RewardService
        # The third existing canonical cache is the original 1696-scene DEV
        # context. Full NAVTRAIN now trains those observations explicitly.
        return RewardService(Path.cwd(), workers=1, cache_roots=self.cfg['research']['scoring_cache_roots'])

    def make_actor(self, cfg, args, device):
        from starVLA.rl.qwen_world_research_data import ResearchSceneData
        model, info = load_deployment_checkpoint(args.il_checkpoint, device=device)
        if info['model_sha256'] != self.parent['model_sha256']:
            raise ValueError('Pinned IL changed between verification and loading')
        head = model.actor
        if (head.hidden_dim, head.num_layers, head.action_dim) != (1024, 20, 3):
            raise ValueError('Use the complete original World clean-x actor')
        self.parent = info
        actor = ScopeRecipeActor(head, cfg, scene_provider=model.scene_provider,
                                 predictive_provider=model.predictive_provider)
        if cfg['research']['policy_objective'] != 'nft':
            from starVLA.rl.qwen_world_research_algorithms import ResearchLoss
            actor._loss = ResearchLoss(cfg)
        data = ResearchSceneData(args.data, role='NAVTRAIN')
        self.conditions = MemoryLearningConditions(model, data, info['model_sha256'], actor)
        object.__setattr__(actor, 'conditions', self.conditions)
        self.actor = actor
        if args.qualification:
            actor.record_sampling_terms = True
            self.observation_before = self._frozen_observation_hashes()
            self.memory_before = self._current_memory_hashes()
        return actor, None, None

    def recipe_identity(self, cfg, args, condition_sha):
        value = super().recipe_identity(cfg, args, condition_sha)
        value.update(research_protocol=copy.deepcopy(cfg['research']),
                     reference_observation='permanently frozen pinned IL parent Qwen and memory',
                     trainable_scope=self.actor.closure,
                     data_role='explicit NAVTRAIN population; old DEV may be training diagnostic; NAVTEST never trained',
                     data_population_manifest_sha256=self.conditions.data.sha,
                     selected_parent_updates=self.parent['seal']['updates'],
                     Test_used_for_user_requested_parent_selection=True,
                     policy_objective=cfg['research']['policy_objective'],
                     execution_changes_vs_old_native='new explicit population/algorithm protocol; see research_protocol')
        return value
